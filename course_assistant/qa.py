"""Question answering over retrieved evidence.

Flow: retrieve evidence -> build a grounded prompt (text excerpts + slide images) ->
vision LLM in JSON mode -> validate `{answer, sources, missing_information}` ->
resolve source ids to documents/pages/images -> check the sources support the answer.
The assistant never invents citations: only evidence ids that were actually supplied
can appear in the output, and anything else is dropped.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from .index import Evidence, HybridRetriever, RetrievalResult
from .services import ChatMessage, LLMClient, ServiceUnavailable

MAX_IMAGES = 4           # slide images attached to the LLM request
MAX_EXCERPT_CHARS = 1200
SUPPORT_THRESHOLD = 0.25 # fraction of answer content words that must appear in cited sources
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set((
    "a an the of to in on for and or is are was were be by with as at from this that it its "
    "based materials material course slide slides page says state states according provided "
    "which who what when where how not no yes can will may also than then into about"
).split())

SYSTEM_PROMPT = """You are a course assistant. Answer the student's question using ONLY the evidence blocks provided.
Each block is tagged [S<n>] and comes from a specific document and page/slide; some include the slide image.
Rules:
- Use the text AND the images. When asked to describe or explain a slide, diagram, chart or meme, describe what the image actually shows.
- Cite the blocks you relied on by their numbers. Never cite a block you did not use and never invent sources.
- If the evidence does not contain the information needed, say so plainly, set "missing_information" to true and cite nothing. Do not guess or use outside knowledge.
- Be concise and specific; quote key phrases from the evidence where useful.
Respond with a single JSON object and nothing else:
{"answer": "<your answer>", "sources": [<block numbers you used>], "missing_information": <true|false>}"""


class AnswerPayload(BaseModel):
    answer: str = Field(min_length=1)
    sources: list[int] = Field(default_factory=list)
    missing_information: bool = False

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, v):
        if v is None:
            return []
        out = []
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                m = re.search(r"\d+", item)
                if m:
                    out.append(int(m.group()))
            elif isinstance(item, dict) and "id" in item:
                out.append(int(item["id"]))
        return out


@dataclass
class Source:
    id: int
    doc_id: str
    doc_name: str
    page_number: int
    excerpt: str
    image_path: Path | None
    channels: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def label(self) -> str:
        kind = "slide" if self.doc_name.lower().endswith((".pptx", ".ppt")) else "page"
        return f"{self.doc_name}, {kind} {self.page_number}"


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]
    missing_information: bool
    supported: bool                 # did the support check pass
    support_score: float
    warnings: list[str]
    error: str | None = None
    latency_s: float = 0.0
    reranked: bool = False
    n_candidates: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {"id": s.id, "document": s.doc_name, "page": s.page_number, "excerpt": s.excerpt,
                 "image": str(s.image_path) if s.image_path else None}
                for s in self.sources
            ],
            "missing_information": self.missing_information,
            "supported": self.supported,
            "support_score": round(self.support_score, 3),
            "warnings": self.warnings,
            "error": self.error,
            "latency_s": round(self.latency_s, 2),
            "reranked": self.reranked,
        }


# ----------------------------------------------------------------------------

def _excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_messages(question: str, evidence: list[Evidence], *, max_images: int = MAX_IMAGES) -> list[ChatMessage]:
    blocks, images = [], []
    for i, ev in enumerate(evidence, start=1):
        c = ev.chunk
        header = f"[S{i}] {c.doc_name} — page/slide {c.page_number}"
        if ev.image_path and len(images) < max_images and ev.image_path.exists():
            images.append(ev.image_path)
            header += f" (slide image #{len(images)} attached)"
        blocks.append(f"{header}\n{_excerpt(c.text)}")
    user = f"Question: {question}\n\nEvidence:\n\n" + "\n\n".join(blocks)
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user, images=images)]


def parse_payload(raw: str) -> AnswerPayload:
    """Parse the LLM reply into a validated payload. Tolerates code fences and prose around the JSON."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return AnswerPayload.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return AnswerPayload.model_validate(json.loads(text[start:end + 1]))
        raise


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


def support_score(answer: str, sources: list[Source]) -> float:
    """Fraction of the answer's content words that appear in the cited sources' text."""
    words = _content_words(answer)
    if not words:
        return 1.0
    pool = set()
    for s in sources:
        pool |= _content_words(s.excerpt)
    return len(words & pool) / len(words)


def _resolve_sources(ids: list[int], evidence: list[Evidence]) -> tuple[list[Source], list[int]]:
    valid, dropped, seen = [], [], set()
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        if 1 <= sid <= len(evidence):
            ev = evidence[sid - 1]
            valid.append(Source(
                id=sid, doc_id=ev.chunk.doc_id, doc_name=ev.chunk.doc_name, page_number=ev.chunk.page_number,
                excerpt=_excerpt(ev.chunk.text), image_path=ev.image_path, channels=list(ev.channels), score=ev.score,
            ))
        else:
            dropped.append(sid)
    return valid, dropped


MISSING_TEXT = ("The course materials I have do not contain information that answers this question. "
                "Try rephrasing, or add the relevant document.")


class QuestionAnswerer:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient, *, top_k: int | None = None):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str, *, doc_ids: set[str] | None = None, use_reranker: bool | None = None,
               top_k: int | None = None) -> Answer:
        t0 = time.time()
        question = question.strip()
        if not question:
            return Answer(question, "Please enter a question.", [], False, True, 1.0, [], latency_s=0.0)

        retrieval: RetrievalResult = self.retriever.retrieve(
            question, top_k=top_k or self.top_k, doc_ids=doc_ids, use_reranker=use_reranker)
        warnings = list(retrieval.warnings)
        evidence = retrieval.evidence
        if not evidence:
            return Answer(question, MISSING_TEXT, [], True, True, 1.0, warnings,
                          latency_s=time.time() - t0, reranked=retrieval.reranked, n_candidates=0)

        messages = build_messages(question, evidence)
        try:
            payload = self._ask_llm(messages)
        except ServiceUnavailable as e:
            return Answer(question, "The language model service is currently unavailable, so I cannot compose an "
                          "answer. The most relevant passages are listed below.", self._fallback_sources(evidence),
                          False, False, 0.0, warnings, error=str(e), latency_s=time.time() - t0,
                          reranked=retrieval.reranked, n_candidates=len(evidence))
        except (json.JSONDecodeError, ValidationError):
            return Answer(question, "The language model returned an unreadable response. Please try again.",
                          self._fallback_sources(evidence), False, False, 0.0, warnings,
                          error="invalid model output", latency_s=time.time() - t0,
                          reranked=retrieval.reranked, n_candidates=len(evidence))

        if payload.missing_information:
            text = payload.answer if "not" in payload.answer.lower() else MISSING_TEXT
            return Answer(question, text, [], True, True, 1.0, warnings, latency_s=time.time() - t0,
                          reranked=retrieval.reranked, n_candidates=len(evidence))

        sources, dropped = _resolve_sources(payload.sources, evidence)
        if dropped:
            warnings.append(f"Model cited non-existent evidence ids {dropped}; they were removed.")
        if not sources:
            warnings.append("The model gave an answer without citing any supplied evidence; treat it with caution.")
            return Answer(question, payload.answer, [], False, False, 0.0, warnings,
                          latency_s=time.time() - t0, reranked=retrieval.reranked, n_candidates=len(evidence))

        score = support_score(payload.answer, sources)
        has_image = any(s.image_path is not None for s in sources)
        supported = score >= SUPPORT_THRESHOLD or (has_image and score > 0)
        if not supported:
            warnings.append("Low overlap between the answer and its cited sources; verify against the excerpts shown.")
        return Answer(question, payload.answer, sources, False, supported, score, warnings,
                      latency_s=time.time() - t0, reranked=retrieval.reranked, n_candidates=len(evidence))

    def _ask_llm(self, messages: list[ChatMessage]) -> AnswerPayload:
        raw = self.llm.chat(messages, json_mode=True)
        try:
            return parse_payload(raw)
        except (json.JSONDecodeError, ValidationError):
            retry = list(messages) + [
                ChatMessage("assistant", raw),
                ChatMessage("user", "That was not valid JSON. Reply with ONLY the JSON object described in the instructions."),
            ]
            return parse_payload(self.llm.chat(retry, json_mode=True))

    @staticmethod
    def _fallback_sources(evidence: list[Evidence], n: int = 3) -> list[Source]:
        srcs, _ = _resolve_sources(list(range(1, min(n, len(evidence)) + 1)), evidence)
        return srcs

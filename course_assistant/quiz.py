"""Practice-quiz generation and grading.

- Students pick course material (one or more documents) and an optional topic.
- Evidence is gathered from those documents (topic-driven retrieval, or an even
  sample of pages when no topic is given) and sent to the LLM with the same
  [S<n>] tagging used for answers, so every question carries real sources.
- The answer key is fixed when the quiz is created (options are shuffled once with a
  seed derived from the quiz id) and stored with the quiz. Student views never
  contain the key; it is revealed only by grading or an explicit reveal.
"""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from .index import Evidence, HybridRetriever
from .ingest import Chunk
from .qa import Source, _excerpt, _resolve_sources, evidence_header, rescue_citations, support_score
from .services import ChatMessage, LLMClient, ServiceUnavailable

MAX_EVIDENCE = 12
MAX_IMAGES = 4
OPTION_LETTERS = "ABCDEF"

QUIZ_SYSTEM_PROMPT = """You are a course assistant writing a PRACTICE QUIZ.
Write multiple-choice questions that can be answered ONLY from the evidence blocks provided (each starts with a marker [S<n>] followed by its document and its slide or page number; some include the slide image).
Rules:
- Every question must be answerable from the evidence; do not use outside knowledge.
- Exactly 4 options per question, one clearly correct, the others plausible but wrong according to the evidence.
- Give a short explanation of why the correct option is right, grounded in the evidence.
- Put the numbers of the blocks each question is based on (the n in [S<n>]) in "sources". A block number is NOT a slide or page number: block [S2] may come from slide 17; cite 2, not 17. Never invent sources.
- In questions, options and explanations, never write the word "tag" and never write block markers like [S2]. If you name a slide or page, use the slide or page number from that block's header (for example "slide 17"), not its block number.
- Vary question types (definitions, comparisons, numbers/dates, interpretation of diagrams or charts shown in images).
Respond with a single JSON object and nothing else:
{"questions": [{"question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "...", "sources": [1]}]}"""


class QuestionPayload(BaseModel):
    question: str = Field(min_length=5)
    options: list[str] = Field(min_length=2, max_length=6)
    answer_index: int
    explanation: str = ""
    sources: list[int] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def _unique_options(cls, v):
        cleaned = [str(o).strip() for o in v if str(o).strip()]
        if len({o.lower() for o in cleaned}) != len(cleaned):
            raise ValueError("duplicate options")
        return cleaned

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, v):
        out = []
        for item in (v or []):
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                m = re.search(r"\d+", item)
                if m:
                    out.append(int(m.group()))
        return out


class QuizPayload(BaseModel):
    questions: list[QuestionPayload] = Field(min_length=1)


class QuizError(RuntimeError):
    """User-facing quiz failure (message is safe to show)."""


@dataclass
class QuizQuestion:
    number: int
    question: str
    options: list[str]
    answer_index: int           # fixed at creation; never exposed in student_view
    explanation: str
    sources: list[Source]
    verified: bool              # support check between question+answer and sources

    def student_view(self) -> dict:
        return {"number": self.number, "question": self.question, "options": list(self.options)}

    @property
    def answer_letter(self) -> str:
        return OPTION_LETTERS[self.answer_index]

    def to_dict(self) -> dict:
        return {
            "number": self.number, "question": self.question, "options": self.options,
            "answer_index": self.answer_index, "explanation": self.explanation, "verified": self.verified,
            "sources": [{"id": s.id, "doc_id": s.doc_id, "document": s.doc_name, "page": s.page_number,
                         "excerpt": s.excerpt, "image": str(s.image_path) if s.image_path else None} for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuizQuestion":
        return cls(
            number=d["number"], question=d["question"], options=list(d["options"]), answer_index=int(d["answer_index"]),
            explanation=d.get("explanation", ""), verified=bool(d.get("verified", False)),
            sources=[Source(id=s["id"], doc_id=s["doc_id"], doc_name=s["document"], page_number=int(s["page"]),
                            excerpt=s["excerpt"], image_path=Path(s["image"]) if s.get("image") else None)
                     for s in d.get("sources", [])],
        )


@dataclass
class Quiz:
    quiz_id: str
    created_at: str
    doc_names: list[str]
    topic: str | None
    questions: list[QuizQuestion]
    warnings: list[str] = field(default_factory=list)
    latency_s: float = 0.0

    def student_view(self) -> list[dict]:
        return [q.student_view() for q in self.questions]

    def answer_key(self) -> dict[int, int]:
        return {q.number: q.answer_index for q in self.questions}

    def to_dict(self) -> dict:
        return {
            "quiz_id": self.quiz_id, "created_at": self.created_at, "doc_names": self.doc_names, "topic": self.topic,
            "questions": [q.to_dict() for q in self.questions], "warnings": self.warnings,
            "latency_s": round(self.latency_s, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Quiz":
        return cls(quiz_id=d["quiz_id"], created_at=d["created_at"], doc_names=list(d["doc_names"]), topic=d.get("topic"),
                   questions=[QuizQuestion.from_dict(q) for q in d["questions"]], warnings=list(d.get("warnings", [])),
                   latency_s=float(d.get("latency_s", 0.0)))


@dataclass
class QuestionFeedback:
    number: int
    question: str
    selected_index: int | None
    correct_index: int
    correct: bool
    explanation: str
    sources: list[Source]
    options: list[str]

    @property
    def selected_letter(self) -> str | None:
        return OPTION_LETTERS[self.selected_index] if self.selected_index is not None else None

    @property
    def correct_letter(self) -> str:
        return OPTION_LETTERS[self.correct_index]


@dataclass
class QuizResult:
    quiz_id: str
    score: int
    total: int
    feedback: list[QuestionFeedback]

    @property
    def percent(self) -> float:
        return 100.0 * self.score / self.total if self.total else 0.0


# ----------------------------------------------------------------------------

def _sample_chunks(chunks: list[Chunk], n: int, seed: int = 0) -> list[Chunk]:
    """Pick up to n chunks spread evenly across documents and pages."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for lst in by_doc.values():
        lst.sort(key=lambda c: (c.page_number, c.chunk_index))
    picked: list[Chunk] = []
    docs = list(by_doc.values())
    if not docs:
        return picked
    per_doc = max(1, n // len(docs))
    for lst in docs:
        if len(lst) <= per_doc:
            picked.extend(lst)
        else:
            step = len(lst) / per_doc
            picked.extend(lst[int(i * step)] for i in range(per_doc))
    rnd = random.Random(seed)
    rnd.shuffle(picked)
    return picked[:n]


def build_quiz_messages(evidence: list[Evidence], n_questions: int, topic: str | None) -> list[ChatMessage]:
    blocks, images = [], []
    for i, ev in enumerate(evidence, start=1):
        c = ev.chunk
        header = evidence_header(i, c)
        if ev.image_path and len(images) < MAX_IMAGES and ev.image_path.exists():
            images.append(ev.image_path)
            header += f" (slide image #{len(images)} attached)"
        blocks.append(f"{header}\n{_excerpt(c.text)}")
    focus = f" Focus on the topic: {topic}." if topic else ""
    user = (f"Generate {n_questions} question(s) for a practice quiz.{focus}\n\nEvidence:\n\n" + "\n\n".join(blocks))
    return [ChatMessage("system", QUIZ_SYSTEM_PROMPT), ChatMessage("user", user, images=images)]


def parse_quiz_payload(raw: str) -> QuizPayload:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return QuizPayload.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return QuizPayload.model_validate(json.loads(text[start:end + 1]))
        raise


class QuizGenerator:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient, chunks_for_docs, save_dir: Path | None = None):
        """`chunks_for_docs(doc_ids: set[str] | None) -> list[Chunk]` supplies material when no topic is given."""
        self.retriever = retriever
        self.llm = llm
        self.chunks_for_docs = chunks_for_docs
        self.save_dir = save_dir

    def generate(self, *, doc_ids: set[str] | None = None, topic: str | None = None, n_questions: int = 5,
                 doc_names: list[str] | None = None, use_reranker: bool | None = None) -> Quiz:
        t0 = time.time()
        n_questions = max(1, min(int(n_questions), 15))
        topic = (topic or "").strip() or None
        warnings: list[str] = []

        evidence = self._gather_evidence(doc_ids, topic, warnings, use_reranker)
        if not evidence:
            raise QuizError("No course material is available for this selection. Add documents first, or widen the topic.")

        messages = build_quiz_messages(evidence, n_questions, topic)
        try:
            payload = self._ask_llm(messages)
        except ServiceUnavailable as e:
            raise QuizError(f"The language model service is unavailable, so a quiz cannot be generated right now. ({e})") from None
        except (json.JSONDecodeError, ValidationError):
            raise QuizError("The language model returned an unreadable quiz. Please try again.") from None

        quiz_id = uuid.uuid4().hex[:12]
        rnd = random.Random(int(quiz_id, 16))
        questions: list[QuizQuestion] = []
        for qp in payload.questions:
            if not (0 <= qp.answer_index < len(qp.options)):
                warnings.append(f"Dropped a question with an out-of-range answer index: {qp.question[:60]}…")
                continue
            correct = qp.options[qp.answer_index]
            ids, notes = rescue_citations(f"{qp.question} {correct} {qp.explanation}", qp.sources, evidence)
            warnings.extend(notes)
            sources, dropped = _resolve_sources(ids, evidence)
            if dropped:
                warnings.append(f"Removed non-existent source ids {dropped} from a question.")
            if not sources:
                warnings.append(f"Dropped a question with no valid sources: {qp.question[:60]}…")
                continue
            # Shuffle options once; the key is fixed from here on.
            options = list(qp.options)
            rnd.shuffle(options)
            answer_index = options.index(correct)
            # Same rule as answers: text overlap, or any overlap when the source is a slide image
            # (questions about charts/memes legitimately use words that are only in the picture).
            score = support_score(f"{qp.question} {correct} {qp.explanation}", sources)
            verified = score >= 0.2 or (any(src.image_path is not None for src in sources) and score > 0)
            questions.append(QuizQuestion(
                number=len(questions) + 1, question=qp.question.strip(), options=options, answer_index=answer_index,
                explanation=qp.explanation.strip(), sources=sources, verified=verified,
            ))
            if len(questions) >= n_questions:
                break
        if not questions:
            raise QuizError("The model did not produce any usable questions for this material. Try again or pick a different topic.")
        if len(questions) < n_questions:
            warnings.append(f"Only {len(questions)} of {n_questions} requested questions were usable.")

        quiz = Quiz(
            quiz_id=quiz_id, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            doc_names=doc_names or sorted({s.doc_name for q in questions for s in q.sources}), topic=topic,
            questions=questions, warnings=warnings, latency_s=time.time() - t0,
        )
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / f"{quiz_id}.json").write_text(json.dumps(quiz.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return quiz

    def _gather_evidence(self, doc_ids, topic, warnings, use_reranker) -> list[Evidence]:
        if topic:
            res = self.retriever.retrieve(topic, top_k=MAX_EVIDENCE, doc_ids=doc_ids, use_reranker=use_reranker)
            warnings.extend(res.warnings)
            if res.evidence:
                return res.evidence
            warnings.append(f"Nothing in the selected material matched the topic '{topic}'; using a general sample instead.")
        chunks = self.chunks_for_docs(doc_ids)
        sample = _sample_chunks(chunks, MAX_EVIDENCE, seed=len(chunks))
        return [Evidence(chunk=c, score=0.0, fused_score=0.0, channels=["sample"]) for c in sample]

    def _ask_llm(self, messages: list[ChatMessage]) -> QuizPayload:
        raw = self.llm.chat(messages, json_mode=True, temperature=0.4)
        try:
            return parse_quiz_payload(raw)
        except (json.JSONDecodeError, ValidationError):
            retry = list(messages) + [
                ChatMessage("assistant", raw),
                ChatMessage("user", "That was not valid JSON matching the required schema. Reply with ONLY the JSON object."),
            ]
            return parse_quiz_payload(self.llm.chat(retry, json_mode=True, temperature=0.0))

    def load(self, quiz_id: str) -> Quiz | None:
        if self.save_dir is None:
            return None
        p = self.save_dir / f"{quiz_id}.json"
        return Quiz.from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def grade_quiz(quiz: Quiz, responses: dict[int, int | None]) -> QuizResult:
    """Score student responses {question_number: selected_index or None} against the fixed key."""
    feedback = []
    score = 0
    for q in quiz.questions:
        sel = responses.get(q.number)
        if sel is not None and not (0 <= int(sel) < len(q.options)):
            sel = None
        correct = sel is not None and int(sel) == q.answer_index
        score += int(correct)
        feedback.append(QuestionFeedback(
            number=q.number, question=q.question, selected_index=None if sel is None else int(sel),
            correct_index=q.answer_index, correct=correct, explanation=q.explanation, sources=q.sources, options=q.options,
        ))
    return QuizResult(quiz_id=quiz.quiz_id, score=score, total=len(quiz.questions), feedback=feedback)

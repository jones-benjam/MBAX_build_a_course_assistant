"""Deterministic offline stand-ins for the model services.

They let the full pipeline (ingest → index → retrieve → answer → quiz) run and be
tested without network access or API keys. They are intentionally simple:
- embeddings are hashed bag-of-words vectors (so lexical overlap ⇒ similarity),
- the reranker scores token overlap,
- the LLM returns well-formed JSON built from the evidence it was given.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Sequence

from .base import ChatMessage, LLMClient, RerankClient, TextEmbedClient, VisualEmbedClient

_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def hashed_embedding(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class MockTextEmbed(TextEmbedClient):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [hashed_embedding(t) for t in texts]


class MockVisualEmbed(VisualEmbedClient):
    """Uses a sidecar `.txt` next to the image (page text) when present, so that
    text→image search behaves plausibly offline. Falls back to the filename."""

    def embed_images(self, images: Sequence[Path]) -> list[list[float]]:
        out = []
        for img in images:
            sidecar = Path(img).with_suffix(".txt")
            text = sidecar.read_text(encoding="utf-8", errors="ignore") if sidecar.exists() else Path(img).stem
            out.append(hashed_embedding(text))
        return out

    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        return [hashed_embedding(t) for t in texts]


class MockReranker(RerankClient):
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[float]:
        q = set(_tokens(query))
        scores = []
        for c in candidates:
            text = c.get("text") or ""
            if not text and c.get("image"):
                sidecar = Path(c["image"]).with_suffix(".txt")
                if sidecar.exists():
                    text = sidecar.read_text(encoding="utf-8", errors="ignore")
            toks = set(_tokens(text))
            scores.append(len(q & toks) / (len(q) or 1))
        return scores


class MockLLM(LLMClient):
    """Returns JSON matching the task hinted in the system prompt.

    The real prompts (added in later steps) embed evidence blocks tagged with
    `[S<n>]`. The mock cites every tag it sees, and declares the information
    missing when no evidence block is present.
    """

    def chat(self, messages: Sequence[ChatMessage], *, json_mode: bool = False, temperature: float = 0.0) -> str:
        system = " ".join(m.text for m in messages if m.role == "system")
        user = "\n".join(m.text for m in messages if m.role == "user")
        tags = re.findall(r"\[S(\d+)\]", user)
        if "QUIZ" in system.upper():
            return json.dumps(self._quiz(user, tags))
        if not json_mode:
            return "Mock reply."
        if not tags:
            return json.dumps({
                "answer": "The course materials provided do not contain information to answer this question.",
                "sources": [],
                "missing_information": True,
            })
        first = user.split(f"[S{tags[0]}]", 1)[1][:200].strip().replace("\n", " ")
        return json.dumps({
            "answer": f"Based on the materials: {first}",
            "sources": sorted({int(t) for t in tags})[:3],
            "missing_information": False,
        })

    @staticmethod
    def _quiz(user: str, tags: list[str]) -> dict:
        n = 3
        m = re.search(r"(\d+)\s+question", user)
        if m:
            n = int(m.group(1))
        questions = []
        uniq = sorted({int(t) for t in tags}) or [1]
        for i in range(n):
            src = uniq[i % len(uniq)]
            questions.append({
                "question": f"Mock question {i + 1} about the selected material?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer_index": i % 4,
                "explanation": f"Mock explanation grounded in source [S{src}].",
                "sources": [src],
            })
        return {"questions": questions}

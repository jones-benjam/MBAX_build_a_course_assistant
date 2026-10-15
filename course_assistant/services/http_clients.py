"""HTTP clients for the class endpoints (vLLM OpenAI-compatible servers).

Request shapes were checked against each server's `/openapi.json` (vLLM) and the
model cards on 2026-10-15:

- Vision LLM (9001, Qwen3.6-VL): `POST /v1/chat/completions`, images as `image_url`
  data URLs, `response_format: {"type": "json_object"}`, thinking disabled through
  `chat_template_kwargs`; any `<think>…</think>` block is stripped defensively.
- Text embeddings (9002, Nemotron-3-Embed): `POST /v1/embeddings` with plain strings.
  The model card requires manual `query: ` / `passage: ` prefixes on this route.
- Visual embeddings (9003, Qwen3-VL-Embedding): `/v1/embeddings` only accepts strings
  in `input`, so images go through the chat form (`messages` with `image_url` parts;
  the instruction is the system message). One conversation per request.
- Reranker (9004, Qwen3-VL-Reranker): `POST /v1/rerank`; each document is a string or
  `{"content": [typed parts]}`; the response lists `results[{index, relevance_score}]`.

The class list gives base URLs without `/v1`; `_api_base` adds it when missing.
Keys are read from config and sent only in the Authorization header; they are never
logged and never appear in error messages.
"""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Sequence

import httpx

from ..config import ServiceConfig
from .base import ChatMessage, LLMClient, RerankClient, ServiceUnavailable, TextEmbedClient, VisualEmbedClient

_TIMEOUT = httpx.Timeout(180.0, connect=15.0)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.S)

# Instructions recommended by the model cards (used as the system message).
VISUAL_DOC_INSTRUCTION = "Represent the user's input."
VISUAL_QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
RERANK_INSTRUCTION = "Retrieve images or text relevant to the user's query."
TEXT_QUERY_PREFIX = "query: "
TEXT_PASSAGE_PREFIX = "passage: "
MAX_COMPLETION_TOKENS = 2048


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _api_base(base_url: str) -> str:
    """Normalise a configured base URL to the OpenAI-compatible `/v1` root."""
    url = base_url.strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def _error_detail(response: httpx.Response, api_key: str = "") -> str:
    """Redact the configured key before shortening an untrusted error body."""
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("detail") or err)
            else:
                detail = str(body.get("detail") or err)
        else:
            detail = str(body)
    except ValueError:
        detail = response.text
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    return detail[:200]


class _Http:
    def __init__(self, cfg: ServiceConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.base = _api_base(cfg.base_url)
        self._client = httpx.Client(timeout=_TIMEOUT, transport=transport)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _post(self, path: str, payload: dict) -> dict:
        name = self.cfg.model or "service"
        try:
            r = self._client.post(self.base + path, json=payload, headers=self._headers())
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = _error_detail(e.response, self.cfg.api_key)
            raise ServiceUnavailable(f"{name} returned HTTP {e.response.status_code}: {detail}") from None
        except (httpx.HTTPError, ValueError) as e:
            raise ServiceUnavailable(f"{name} unreachable ({type(e).__name__})") from None


def _embeddings_from(data: dict, what: str) -> list[list[float]]:
    try:
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]
    except (KeyError, TypeError):
        raise ServiceUnavailable(f"{what} returned an unexpected response shape") from None


class HttpLLM(_Http, LLMClient):
    def chat(self, messages: Sequence[ChatMessage], *, json_mode: bool = False, temperature: float = 0.0) -> str:
        out = []
        for m in messages:
            if m.images:
                content: list[dict] = [{"type": "text", "text": m.text}]
                for img in m.images:
                    content.append({"type": "image_url", "image_url": {"url": _image_data_url(img)}})
                out.append({"role": m.role, "content": content})
            else:
                out.append({"role": m.role, "content": m.text})
        payload: dict = {
            "model": self.cfg.model,
            "messages": out,
            "temperature": temperature,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            # Qwen3 models think by default; the answer must be the plain JSON reply.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ServiceUnavailable("LLM returned an unexpected response shape") from None
        if not isinstance(text, str):
            raise ServiceUnavailable("LLM returned an unexpected response shape")
        return _THINK_BLOCK.sub("", text).strip()


class HttpTextEmbed(_Http, TextEmbedClient):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([TEXT_PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([TEXT_QUERY_PREFIX + t for t in texts])

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.cfg.model, "input": inputs})
        return _embeddings_from(data, "Embedding service")


class HttpVisualEmbed(_Http, VisualEmbedClient):
    """Qwen3-VL-Embedding through vLLM's chat-style embedding request."""

    def embed_images(self, images: Sequence[Path]) -> list[list[float]]:
        return [self._embed_one(VISUAL_DOC_INSTRUCTION,
                                [{"type": "image_url", "image_url": {"url": _image_data_url(p)}}])
                for p in images]

    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(VISUAL_QUERY_INSTRUCTION, [{"type": "text", "text": t}]) for t in texts]

    def _embed_one(self, instruction: str, content: list[dict]) -> list[float]:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
        }
        data = self._post("/embeddings", payload)
        vecs = _embeddings_from(data, "Visual embedding service")
        if not vecs:
            raise ServiceUnavailable("Visual embedding service returned no embedding")
        return vecs[0]


class HttpReranker(_Http, RerankClient):
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[float]:
        docs: list = []
        for c in candidates:
            parts: list[dict] = []
            if c.get("text"):
                parts.append({"type": "text", "text": c["text"]})
            if c.get("image"):
                parts.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(c["image"]))}})
            if not parts:
                parts.append({"type": "text", "text": ""})
            # Text-only candidates can be plain strings; anything with an image needs typed parts.
            docs.append(parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else {"content": parts})
        payload = {
            "model": self.cfg.model,
            "query": query,
            "documents": docs,
            "top_n": len(docs),
            "instruction": RERANK_INSTRUCTION,
        }
        data = self._post("/rerank", payload)
        try:
            scores = [0.0] * len(candidates)
            for r in data["results"]:
                scores[r["index"]] = float(r["relevance_score"])
            return scores
        except (KeyError, TypeError, IndexError, ValueError):
            raise ServiceUnavailable("Reranker returned an unexpected response shape") from None

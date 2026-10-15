"""HTTP clients for the class endpoints.

The request/response shapes below follow the OpenAI-compatible conventions that
most self-hosted model servers (vLLM, TEI, etc.) expose. Adjust `_payload`/`_parse`
methods once the class team shares each endpoint's documented format.
Keys are read from config and sent only in the Authorization header; they are
never logged.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Sequence

import httpx

from ..config import ServiceConfig
from .base import ChatMessage, LLMClient, RerankClient, ServiceUnavailable, TextEmbedClient, VisualEmbedClient

_TIMEOUT = httpx.Timeout(120.0, connect=15.0)


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


class _Http:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self._client = httpx.Client(timeout=_TIMEOUT)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _post(self, path: str, payload: dict) -> dict:
        url = self.cfg.base_url.rstrip("/") + path
        try:
            r = self._client.post(url, json=payload, headers=self._headers())
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            raise ServiceUnavailable(
                f"{self.cfg.model or 'service'} returned HTTP {e.response.status_code}"
            ) from None
        except (httpx.HTTPError, ValueError) as e:
            raise ServiceUnavailable(f"{self.cfg.model or 'service'} unreachable ({type(e).__name__})") from None


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
        payload: dict = {"model": self.cfg.model, "messages": out, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ServiceUnavailable("LLM returned an unexpected response shape") from None


class HttpTextEmbed(_Http, TextEmbedClient):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.cfg.model, "input": list(texts)})
        try:
            return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]
        except (KeyError, TypeError):
            raise ServiceUnavailable("Embedding service returned an unexpected response shape") from None


class HttpVisualEmbed(_Http, VisualEmbedClient):
    def embed_images(self, images: Sequence[Path]) -> list[list[float]]:
        payload = {"model": self.cfg.model, "input": [{"image": _image_data_url(p)} for p in images]}
        data = self._post("/embeddings", payload)
        try:
            return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]
        except (KeyError, TypeError):
            raise ServiceUnavailable("Visual embedding service returned an unexpected response shape") from None

    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.cfg.model, "input": [{"text": t} for t in texts]})
        try:
            return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]
        except (KeyError, TypeError):
            raise ServiceUnavailable("Visual embedding service returned an unexpected response shape") from None


class HttpReranker(_Http, RerankClient):
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[float]:
        docs = []
        for c in candidates:
            item: dict = {}
            if c.get("text"):
                item["text"] = c["text"]
            if c.get("image"):
                item["image"] = _image_data_url(Path(c["image"]))
            docs.append(item)
        data = self._post("/rerank", {"model": self.cfg.model, "query": query, "documents": docs})
        try:
            scores = [0.0] * len(candidates)
            for r in data["results"]:
                scores[r["index"]] = float(r["relevance_score"])
            return scores
        except (KeyError, TypeError, IndexError, ValueError):
            raise ServiceUnavailable("Reranker returned an unexpected response shape") from None

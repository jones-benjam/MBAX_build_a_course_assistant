"""Hybrid retrieval: keyword (BM25) + text embeddings + visual embeddings, fused with
reciprocal rank fusion and optionally reranked by a multimodal reranker.

The three indexes are kept separate (as the assignment requires) and combined only at
query time. All indexes implement the Library's IndexListener protocol, so adding or
removing a document keeps every index in sync.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import bm25s
import chromadb

from .config import Settings
from .ingest import Chunk
from .services import Services, ServiceUnavailable

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are was were be by with as at from this that it its".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _batched(items: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ----------------------------------------------------------------------------
# Keyword index
# ----------------------------------------------------------------------------

class KeywordIndex:
    """BM25 over chunk text. bm25s indexes are immutable, so the index is rebuilt
    from the in-memory chunk table on every change (cheap at course scale)."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._retriever: bm25s.BM25 | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._rebuild()

    def remove_document(self, doc_id: str) -> None:
        self._chunks = {k: v for k, v in self._chunks.items() if v.doc_id != doc_id}
        self._rebuild()

    def _rebuild(self) -> None:
        self._ids = list(self._chunks)
        if not self._ids:
            self._retriever = None
            return
        corpus = [_tokenize(self._chunks[i].text) for i in self._ids]
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus, show_progress=False)

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int, doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if self._retriever is None:
            return []
        q = _tokenize(query)
        if not q:
            return []
        n = min(len(self._ids), max(k * 4, k))
        results, scores = self._retriever.retrieve([q], k=n, show_progress=False)
        out = []
        for idx, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            cid = self._ids[int(idx)]
            if doc_ids and self._chunks[cid].doc_id not in doc_ids:
                continue
            out.append((cid, float(score)))
            if len(out) >= k:
                break
        return out


# ----------------------------------------------------------------------------
# Vector indexes (ChromaDB, persistent, one collection each)
# ----------------------------------------------------------------------------

class _ChromaBase:
    def __init__(self, path: Path, name: str):
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def remove_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._col.count()

    def _query(self, embedding: list[float], k: int, doc_ids: set[str] | None) -> list[tuple[str, float, dict]]:
        total = self._col.count()
        if total == 0:
            return []
        where = None
        if doc_ids:
            where = {"doc_id": {"$in": sorted(doc_ids)}}
        res = self._col.query(query_embeddings=[embedding], n_results=min(k, total), where=where,
                              include=["distances", "metadatas"])
        ids, dists, metas = res["ids"][0], res["distances"][0], res["metadatas"][0]
        return [(i, 1.0 - float(d), m) for i, d, m in zip(ids, dists, metas)]


class TextVectorIndex(_ChromaBase):
    def __init__(self, path: Path, services: Services, batch_size: int = 32):
        super().__init__(path, "text")
        self._svc = services
        self._batch = batch_size

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for batch in _batched(chunks, self._batch):
            vecs = self._svc.text_embed.embed([c.text for c in batch])
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vecs,
                metadatas=[{"doc_id": c.doc_id, "page_number": c.page_number} for c in batch],
            )

    def search(self, query: str, k: int, doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        emb = self._svc.text_embed.embed_query([query])[0]
        return [(cid, score) for cid, score, _ in self._query(emb, k, doc_ids)]


class VisualIndex(_ChromaBase):
    """One entry per page/slide image, embedded with the multimodal model. Queries
    embed the question text into the same space (text -> image retrieval)."""

    def __init__(self, path: Path, services: Services, batch_size: int = 8):
        super().__init__(path, "visual")
        self._svc = services
        self._batch = batch_size

    @staticmethod
    def page_key(doc_id: str, page_number: int) -> str:
        return f"{doc_id}:p{page_number}"

    def add_chunks(self, chunks: list[Chunk]) -> None:
        pages: dict[str, Chunk] = {}
        for c in chunks:
            if c.image_path is not None:
                pages.setdefault(self.page_key(c.doc_id, c.page_number), c)
        items = list(pages.items())
        for batch in _batched(items, self._batch):
            vecs = self._svc.visual_embed.embed_images([c.image_path for _, c in batch])
            self._col.upsert(
                ids=[key for key, _ in batch],
                embeddings=vecs,
                metadatas=[{"doc_id": c.doc_id, "page_number": c.page_number, "image_path": str(c.image_path)}
                           for _, c in batch],
            )

    def search(self, query: str, k: int, doc_ids: set[str] | None = None) -> list[tuple[str, float, dict]]:
        emb = self._svc.visual_embed.embed_text([query])[0]
        return self._query(emb, k, doc_ids)


# ----------------------------------------------------------------------------
# Hybrid retriever
# ----------------------------------------------------------------------------

@dataclass
class Evidence:
    chunk: Chunk
    score: float                      # final score (reranker if used, else fused RRF score)
    fused_score: float
    channels: list[str] = field(default_factory=list)  # which indexes surfaced it
    rerank_score: float | None = None

    @property
    def image_path(self) -> Path | None:
        return self.chunk.image_path


@dataclass
class RetrievalResult:
    evidence: list[Evidence]
    warnings: list[str] = field(default_factory=list)   # e.g. a channel that was unavailable
    reranked: bool = False


class HybridRetriever:
    RRF_K = 60

    def __init__(self, settings: Settings, services: Services, chunk_lookup):
        """`chunk_lookup(chunk_id) -> Chunk | None` and `chunk_lookup.page(doc_id, page) -> list[Chunk]`
        are provided by the Library-backed store (see ChunkStore)."""
        self.settings = settings
        self.services = services
        self.store = chunk_lookup
        self.keyword = KeywordIndex()
        chroma_path = settings.index_dir / "chroma"
        self.text = TextVectorIndex(chroma_path, services)
        self.visual = VisualIndex(chroma_path, services)

    # IndexListener protocol — fan out to the three indexes
    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.keyword.add_chunks(chunks)
        self.text.add_chunks(chunks)
        self.visual.add_chunks(chunks)

    def remove_document(self, doc_id: str) -> None:
        self.keyword.remove_document(doc_id)
        self.text.remove_document(doc_id)
        self.visual.remove_document(doc_id)

    def rebuild_keyword_index(self, chunks: list[Chunk]) -> None:
        """Keyword index lives in memory; rebuild from the chunk store on startup."""
        self.keyword = KeywordIndex()
        if chunks:
            self.keyword.add_chunks(chunks)

    def retrieve(self, query: str, *, top_k: int | None = None, doc_ids: set[str] | None = None,
                 use_reranker: bool | None = None, candidates_per_channel: int | None = None) -> RetrievalResult:
        top_k = top_k or self.settings.top_k
        use_reranker = self.settings.use_reranker if use_reranker is None else use_reranker
        n = candidates_per_channel or max(top_k * 2, 10)
        warnings: list[str] = []
        ranked: dict[str, list[str]] = {}

        # Keyword
        try:
            ranked["keyword"] = [cid for cid, _ in self.keyword.search(query, n, doc_ids)]
        except Exception as e:  # pragma: no cover - defensive
            warnings.append(f"Keyword search failed: {e}")
        # Text embeddings
        try:
            ranked["text"] = [cid for cid, _ in self.text.search(query, n, doc_ids)]
        except ServiceUnavailable as e:
            warnings.append(f"Text embedding service unavailable; using remaining channels. ({e})")
        # Visual embeddings -> map page hits to that page's chunks
        try:
            visual_hits = self.visual.search(query, n, doc_ids)
            vis_ids: list[str] = []
            for _, _, meta in visual_hits:
                page_chunks = self.store.page(meta["doc_id"], int(meta["page_number"]))
                for c in page_chunks[:1]:  # the page's first chunk represents the slide
                    if c.chunk_id not in vis_ids:
                        vis_ids.append(c.chunk_id)
            ranked["visual"] = vis_ids
        except ServiceUnavailable as e:
            warnings.append(f"Visual embedding service unavailable; using remaining channels. ({e})")

        if not any(ranked.values()):
            return RetrievalResult([], warnings)

        # Reciprocal rank fusion
        fused: dict[str, float] = {}
        channels: dict[str, list[str]] = {}
        for name, ids in ranked.items():
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank + 1)
                channels.setdefault(cid, []).append(name)

        candidates: list[Evidence] = []
        for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            chunk = self.store.get(cid)
            if chunk is None:
                continue
            candidates.append(Evidence(chunk=chunk, score=score, fused_score=score, channels=channels[cid]))
        candidates = candidates[: max(top_k * 2, top_k)]

        reranked = False
        if use_reranker and len(candidates) > 1:
            try:
                payload = [{"text": e.chunk.text, "image": str(e.image_path) if e.image_path else None}
                           for e in candidates]
                scores = self.services.rerank.rerank(query, payload)
                for e, s in zip(candidates, scores):
                    e.rerank_score = float(s)
                    # Fused score as a small tie-breaker so reranker ties keep retrieval order.
                    e.score = float(s) + e.fused_score * 1e-3
                candidates.sort(key=lambda e: e.score, reverse=True)
                reranked = True
            except ServiceUnavailable as e:
                warnings.append(f"Reranker unavailable; showing fused retrieval order. ({e})")

        return RetrievalResult(candidates[:top_k], warnings, reranked)


class ChunkStore:
    """In-memory lookup of chunks by id and by page, kept in sync with the Library."""

    def __init__(self, chunks: Iterable[Chunk] = ()):
        self._by_id: dict[str, Chunk] = {}
        self._by_page: dict[str, list[Chunk]] = {}
        self.add_chunks(list(chunks))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._by_id[c.chunk_id] = c
            self._by_page.setdefault(VisualIndex.page_key(c.doc_id, c.page_number), []).append(c)
        for lst in self._by_page.values():
            lst.sort(key=lambda c: c.chunk_index)

    def remove_document(self, doc_id: str) -> None:
        self._by_id = {k: v for k, v in self._by_id.items() if v.doc_id != doc_id}
        self._by_page = {k: v for k, v in self._by_page.items() if not k.startswith(doc_id + ":")}

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def page(self, doc_id: str, page_number: int) -> list[Chunk]:
        return self._by_page.get(VisualIndex.page_key(doc_id, page_number), [])

    def __len__(self) -> int:
        return len(self._by_id)

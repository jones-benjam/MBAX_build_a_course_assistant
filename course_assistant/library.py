"""Document library: add/remove course materials with content-hash de-duplication.

The library owns the on-disk copies (uploads/), rendered page images (pages/<doc_id>/),
and the chunk store (index/<doc_id>.chunks.json). Index backends (keyword/text/visual,
added in Step 3) register as listeners and are told to add or purge a document's
chunks, so removing a document also removes everything searchable about it.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import Settings
from .ingest import Chunk, SUPPORTED_EXTENSIONS, UnsupportedFormat, chunk_pages, describe_supported_formats, file_sha256, parse_document


class IndexListener(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...
    def remove_document(self, doc_id: str) -> None: ...


@dataclass
class DocumentRecord:
    doc_id: str
    name: str
    ext: str
    sha256: str
    stored_path: str
    n_pages: int
    n_chunks: int
    added_at: str


class Library:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_dirs()
        self.manifest_path = settings.data_dir / "library.json"
        self._docs: dict[str, DocumentRecord] = {}
        self._listeners: list[IndexListener] = []
        self._load()

    # -- persistence -----------------------------------------------------------
    def _load(self) -> None:
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._docs = {d["doc_id"]: DocumentRecord(**d) for d in data.get("documents", [])}

    def _save(self) -> None:
        payload = {"documents": [asdict(d) for d in self._docs.values()]}
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _chunks_path(self, doc_id: str) -> Path:
        return self.settings.index_dir / f"{doc_id}.chunks.json"

    def add_listener(self, listener: IndexListener) -> None:
        self._listeners.append(listener)

    # -- queries ---------------------------------------------------------------
    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self._docs.values(), key=lambda d: d.name.lower())

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._docs.get(doc_id)

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        p = self._chunks_path(doc_id)
        if not p.exists():
            return []
        return [Chunk.from_dict(d) for d in json.loads(p.read_text(encoding="utf-8"))]

    def all_chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        for doc_id in self._docs:
            out.extend(self.chunks_for(doc_id))
        return out

    # -- mutations -------------------------------------------------------------
    def add_document(self, src: str | Path, display_name: str | None = None) -> tuple[DocumentRecord, str]:
        """Add a file. Returns (record, status) with status 'added' or 'duplicate'.

        Duplicates are detected by SHA-256 of the file content, so re-uploading the
        same file (even under a different name) never creates a second copy.
        """
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        ext = src.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormat(f"Unsupported file type '{ext}'. {describe_supported_formats()}")
        name = display_name or src.name
        digest = file_sha256(src)
        doc_id = digest[:16]
        if doc_id in self._docs:
            return self._docs[doc_id], "duplicate"

        stored = self.settings.uploads_dir / f"{doc_id}{ext}"
        shutil.copyfile(src, stored)
        pages_dir = self.settings.pages_dir / doc_id
        try:
            pages = parse_document(stored, doc_id, name, pages_dir)
            chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
            self._chunks_path(doc_id).write_text(
                json.dumps([c.to_dict() for c in chunks], ensure_ascii=False), encoding="utf-8"
            )
            for listener in self._listeners:
                listener.add_chunks(chunks)
        except Exception:
            # Roll back partial state so a failed parse leaves nothing behind.
            self._purge_files(doc_id, stored)
            raise

        record = DocumentRecord(
            doc_id=doc_id, name=name, ext=ext, sha256=digest, stored_path=str(stored),
            n_pages=len(pages), n_chunks=len(chunks),
            added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._docs[doc_id] = record
        self._save()
        return record, "added"

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and purge its chunks, page images and index entries."""
        record = self._docs.pop(doc_id, None)
        if record is None:
            return False
        for listener in self._listeners:
            listener.remove_document(doc_id)
        self._purge_files(doc_id, Path(record.stored_path))
        self._save()
        return True

    def _purge_files(self, doc_id: str, stored: Path) -> None:
        if stored.exists():
            stored.unlink()
        shutil.rmtree(self.settings.pages_dir / doc_id, ignore_errors=True)
        cp = self._chunks_path(doc_id)
        if cp.exists():
            cp.unlink()

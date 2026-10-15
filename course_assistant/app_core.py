"""Wires Library + indexes + services into one object the UI and tests use."""
from __future__ import annotations

from .config import Settings, load_settings
from .index import ChunkStore, HybridRetriever
from .library import Library
from .services import Services, get_services


class Assistant:
    def __init__(self, settings: Settings | None = None, services: Services | None = None):
        self.settings = settings or load_settings()
        self.settings.ensure_dirs()
        self.services = services or get_services(self.settings)
        self.library = Library(self.settings)
        self.store = ChunkStore(self.library.all_chunks())
        self.retriever = HybridRetriever(self.settings, self.services, self.store)
        self.retriever.rebuild_keyword_index(list(self.store._by_id.values()))
        # Order matters: the store must know a chunk before the retriever maps hits to it.
        self.library.add_listener(self.store)
        self.library.add_listener(self.retriever)

    def add_document(self, path, display_name: str | None = None):
        return self.library.add_document(path, display_name)

    def remove_document(self, doc_id: str) -> bool:
        return self.library.remove_document(doc_id)

    def retrieve(self, query: str, **kw):
        return self.retriever.retrieve(query, **kw)

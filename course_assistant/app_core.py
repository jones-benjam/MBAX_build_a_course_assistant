"""Wires Library + indexes + services into one object the UI and tests use."""
from __future__ import annotations

from .config import Settings, load_settings
from .index import ChunkStore, HybridRetriever
from .library import Library
from .qa import Answer, QuestionAnswerer
from .quiz import Quiz, QuizGenerator, QuizResult, grade_quiz
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
        self.qa = QuestionAnswerer(self.retriever, self.services.llm, top_k=self.settings.top_k)
        self.quizzes = QuizGenerator(self.retriever, self.services.llm, self._chunks_for_docs,
                                     save_dir=self.settings.data_dir / "quizzes")

    def add_document(self, path, display_name: str | None = None):
        return self.library.add_document(path, display_name)

    def remove_document(self, doc_id: str) -> bool:
        return self.library.remove_document(doc_id)

    def retrieve(self, query: str, **kw):
        return self.retriever.retrieve(query, **kw)

    def ask(self, question: str, **kw) -> Answer:
        return self.qa.answer(question, **kw)

    def _chunks_for_docs(self, doc_ids: set[str] | None):
        docs = self.library.list_documents()
        if doc_ids:
            docs = [d for d in docs if d.doc_id in doc_ids]
        out = []
        for d in docs:
            out.extend(self.library.chunks_for(d.doc_id))
        return out

    def generate_quiz(self, *, doc_ids: set[str] | None = None, topic: str | None = None, n_questions: int = 5, **kw) -> Quiz:
        names = [d.name for d in self.library.list_documents() if not doc_ids or d.doc_id in doc_ids]
        return self.quizzes.generate(doc_ids=doc_ids, topic=topic, n_questions=n_questions, doc_names=names, **kw)

    def grade_quiz(self, quiz: Quiz, responses: dict[int, int | None]) -> QuizResult:
        return grade_quiz(quiz, responses)

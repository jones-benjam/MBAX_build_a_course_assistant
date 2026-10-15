import pytest

from course_assistant.app_core import Assistant
from course_assistant.index import KeywordIndex
from course_assistant.ingest import Chunk
from course_assistant.services import ServiceUnavailable
from tests.fixtures import make_pdf


def _chunk(cid, text, doc="d1", page=1):
    return Chunk(chunk_id=cid, doc_id=doc, doc_name=f"{doc}.pdf", page_number=page, text=text)


def test_keyword_index_ranks_and_filters():
    ki = KeywordIndex()
    ki.add_chunks([
        _chunk("a", "vibe coding on prod is a meme about shipping untested code"),
        _chunk("b", "the grading policy weights exams and projects", doc="d2"),
        _chunk("c", "prompt engineering: few-shot examples and system prompts"),
    ])
    hits = ki.search("vibe coding prod", k=3)
    assert hits and hits[0][0] == "a"
    assert ki.search("grading", k=3, doc_ids={"d1"}) == []
    ki.remove_document("d1")
    assert len(ki) == 1 and ki.search("vibe coding", k=3) == []


@pytest.fixture
def assistant(settings, tmp_path):
    a = Assistant(settings)
    make_pdf(tmp_path / "syllabus.pdf", [
        "Course grading: 40% exams, 60% team projects. Late work loses 10% per day.",
        "Office hours are Tuesday at 2pm in the finance lab.",
    ])
    make_pdf(tmp_path / "week2.pdf", [
        "Vibe Coding on Prod meme: 'it works on my machine' then ship it to production.",
        "Agents use tools, memory and planning loops.",
    ])
    a.add_document(tmp_path / "syllabus.pdf")
    a.add_document(tmp_path / "week2.pdf")
    return a


def test_indexes_populated_and_hybrid_retrieval_finds_right_slide(assistant):
    assert len(assistant.retriever.keyword) == 4
    assert assistant.retriever.text.count() == 4
    assert assistant.retriever.visual.count() == 4  # one entry per page image
    res = assistant.retrieve("find the meme about vibe coding on prod", top_k=3)
    assert res.warnings == []
    top = res.evidence[0]
    assert top.chunk.doc_name == "week2.pdf" and top.chunk.page_number == 1
    assert top.image_path is not None and top.image_path.exists()
    assert {"keyword", "text", "visual"} <= set(top.channels)
    assert res.reranked is True and top.rerank_score is not None


def test_doc_filter_restricts_results(assistant):
    syllabus = next(d for d in assistant.library.list_documents() if d.name == "syllabus.pdf")
    res = assistant.retrieve("vibe coding meme", top_k=3, doc_ids={syllabus.doc_id})
    assert all(e.chunk.doc_id == syllabus.doc_id for e in res.evidence)


def test_removing_document_purges_all_indexes(assistant):
    week2 = next(d for d in assistant.library.list_documents() if d.name == "week2.pdf")
    assert assistant.remove_document(week2.doc_id)
    assert len(assistant.retriever.keyword) == 2
    assert assistant.retriever.text.count() == 2 and assistant.retriever.visual.count() == 2
    res = assistant.retrieve("vibe coding on prod meme", top_k=5)
    assert all(e.chunk.doc_id != week2.doc_id for e in res.evidence)
    # Indexes and manifest persist: a fresh Assistant sees the same state.
    again = Assistant(assistant.settings)
    assert len(again.store) == 2 and again.retriever.text.count() == 2
    hits = again.retrieve("office hours", top_k=1).evidence
    assert hits and hits[0].chunk.page_number == 2


def test_reranker_toggle_off(assistant):
    res = assistant.retrieve("office hours tuesday", top_k=2, use_reranker=False)
    assert res.reranked is False and all(e.rerank_score is None for e in res.evidence)
    assert res.evidence[0].chunk.doc_name == "syllabus.pdf"


class _Down:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ServiceUnavailable("service down")
        return fail


def test_retrieval_degrades_gracefully_when_services_down(assistant):
    svc = assistant.services
    svc.text_embed, svc.visual_embed, svc.rerank = _Down(), _Down(), _Down()
    res = assistant.retrieve("grading policy exams office hours", top_k=2)
    assert len(res.evidence) == 2 and all(e.chunk.doc_name == "syllabus.pdf" for e in res.evidence)
    assert len(res.warnings) == 3 and res.reranked is False
    assert all(e.channels == ["keyword"] for e in res.evidence)


def test_add_fails_cleanly_when_embedding_service_down(settings, tmp_path):
    a = Assistant(settings)
    a.services.text_embed = _Down()
    make_pdf(tmp_path / "x.pdf", ["some text"])
    with pytest.raises(ServiceUnavailable):
        a.add_document(tmp_path / "x.pdf")
    assert a.library.list_documents() == [] and a.retriever.text.count() == 0 and len(a.retriever.keyword) == 0

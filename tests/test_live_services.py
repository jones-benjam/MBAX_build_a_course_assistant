"""Live checks against the real class endpoints and course files.

Skipped unless LIVE_SERVICES=1 is set and a .env with real endpoints exists, so the
normal `./run_tests.sh` stays offline. Run with:

    LIVE_SERVICES=1 MOCK_SERVICES=0 .venv/bin/python -m pytest tests/test_live_services.py -v -s

Set COURSE_MATERIALS_DIR (default ../course_materials) to run the end-to-end checks
on the syllabus and the Week 2 deck (the "Vibe Coding on Prod" meme is slide 33).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

LIVE = os.environ.get("LIVE_SERVICES") == "1" and Path(".env").exists()
pytestmark = pytest.mark.skipif(not LIVE, reason="set LIVE_SERVICES=1 with a real .env to run")

MATERIALS = Path(os.environ.get("COURSE_MATERIALS_DIR", "../course_materials"))


def _settings(tmp_path):
    os.environ["MOCK_SERVICES"] = "0"
    os.environ["DATA_DIR"] = str(tmp_path / "data")
    from course_assistant.config import load_settings
    s = load_settings(dotenv=".env")
    assert not s.mock_services, "MOCK_SERVICES must be 0 for live tests"
    s.ensure_dirs()
    return s


@pytest.fixture
def services(tmp_path):
    from course_assistant.services import get_services
    return get_services(_settings(tmp_path))


@pytest.fixture
def slide_png(tmp_path):
    """A rendered page with real text, so the visual services get a realistic image."""
    from tests.fixtures import make_pdf
    from course_assistant.ingest import parse_document
    pdf = make_pdf(tmp_path / "one.pdf", ["Vibe Coding on Prod: it works on my machine, ship it"])
    pages = parse_document(pdf, "d1", "one.pdf", tmp_path / "pages")
    return pages[0].image_path


def test_llm_returns_json_and_sees_the_image(services, slide_png):
    from course_assistant.services import ChatMessage
    from course_assistant.qa import parse_payload
    raw = services.llm.chat([
        ChatMessage("system", 'Reply with a JSON object {"answer": str, "sources": [1], "missing_information": false}.'),
        ChatMessage("user", "[S1] (image attached) What phrase is written on the attached slide image?", images=[slide_png]),
    ], json_mode=True)
    payload = parse_payload(raw)
    assert "<think>" not in raw
    assert "vibe" in payload.answer.lower() or "prod" in payload.answer.lower()


def test_text_embeddings_are_normalised_and_asymmetric(services):
    import math
    docs = services.text_embed.embed(["The final exam is worth 40% of the grade.", "Office hours are on Tuesday."])
    q = services.text_embed.embed_query(["how much is the exam worth"])[0]
    assert len(docs) == 2 and len(docs[0]) == len(q) > 100
    assert abs(math.sqrt(sum(x * x for x in q)) - 1.0) < 0.05
    sims = [sum(a * b for a, b in zip(q, d)) for d in docs]
    assert sims[0] > sims[1]


def test_visual_embeddings_align_text_query_to_matching_image(services, slide_png, tmp_path):
    from tests.fixtures import make_pdf
    from course_assistant.ingest import parse_document
    other = parse_document(make_pdf(tmp_path / "two.pdf", ["Grading policy and office hours"]),
                           "d2", "two.pdf", tmp_path / "pages2")[0].image_path
    imgs = services.visual_embed.embed_images([slide_png, other])
    q = services.visual_embed.embed_text(["vibe coding on prod"])[0]
    assert len(imgs) == 2 and len(imgs[0]) == len(q)
    sims = [sum(a * b for a, b in zip(q, v)) for v in imgs]
    assert sims[0] > sims[1]


def test_reranker_scores_relevant_text_and_image_higher(services, slide_png):
    scores = services.rerank.rerank("vibe coding on prod", [
        {"text": "Office hours are Tuesday at 2pm."},
        {"text": "Vibe coding on prod meme", "image": str(slide_png)},
        {"text": "", "image": str(slide_png)},
    ])
    assert len(scores) == 3 and scores[1] > scores[0] and scores[2] > scores[0]


# ---------------------------------------------------------------- end to end on course files

needs_materials = pytest.mark.skipif(not MATERIALS.is_dir(), reason=f"no course materials at {MATERIALS}")


def _find(pattern: str) -> Path:
    hits = sorted(p for p in MATERIALS.iterdir() if pattern.lower() in p.name.lower() and p.suffix.lower() in {".pdf", ".pptx"})
    if not hits:
        pytest.skip(f"no file matching {pattern!r} in {MATERIALS}")
    return hits[0]


@pytest.fixture(scope="module")
def assistant(tmp_path_factory):
    from course_assistant.app_core import Assistant
    s = _settings(tmp_path_factory.mktemp("live"))
    a = Assistant(s)
    a.add_document(_find("syllabus"))
    a.add_document(_find("week 2") if any("week 2" in p.name.lower() for p in MATERIALS.iterdir()) else _find("week2"))
    return a


@needs_materials
def test_duplicate_upload_is_detected(assistant):
    rec, status = assistant.add_document(_find("syllabus"))
    assert status == "duplicate" and len(assistant.library.list_documents()) == 2


@needs_materials
def test_syllabus_question_is_answered_with_sources(assistant):
    ans = assistant.ask("How is the final grade calculated in this course?")
    print("\nANSWER:", ans.answer, "\nSOURCES:", [s.label for s in ans.sources], "latency", ans.latency_s)
    assert not ans.error and ans.sources and not ans.missing_information
    assert any("syllabus" in s.doc_name.lower() for s in ans.sources)


@needs_materials
def test_meme_slide_is_found_through_the_visual_path(assistant):
    ans = assistant.ask("What does the 'Vibe Coding on Prod' meme show, and what point is it making?")
    print("\nANSWER:", ans.answer, "\nSOURCES:", [s.label for s in ans.sources], "latency", ans.latency_s)
    assert not ans.error and not ans.missing_information
    assert any(s.page_number == 33 for s in ans.sources), [s.label for s in ans.sources]
    assert any(s.image_path is not None for s in ans.sources)


@needs_materials
def test_unanswerable_question_is_declared_missing(assistant):
    ans = assistant.ask("What is the capital of Mongolia?")
    print("\nANSWER:", ans.answer)
    assert ans.missing_information and ans.sources == []


@needs_materials
def test_quiz_has_fixed_key_and_sources(assistant):
    quiz = assistant.generate_quiz(n_questions=3)
    assert len(quiz.questions) == 3
    assert all(q.sources for q in quiz.questions)
    key = quiz.answer_key()
    result = assistant.grade_quiz(quiz, dict(key))
    assert result.score == len(quiz.questions)


@needs_materials
def test_removal_cleans_indexes(assistant):
    before = len(assistant.library.list_documents())
    doc = next(d for d in assistant.library.list_documents() if "syllabus" in d.name.lower())
    assert assistant.remove_document(doc.doc_id)
    assert len(assistant.library.list_documents()) == before - 1
    ans = assistant.ask("How is the final grade calculated in this course?")
    assert not any("syllabus" in s.doc_name.lower() for s in ans.sources)

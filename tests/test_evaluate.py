"""Offline checks for the evaluation script's scoring and reporting helpers."""
import importlib.util
import json
from pathlib import Path

import pytest

from course_assistant.app_core import Assistant
from tests.fixtures import make_pdf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evaluate():
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_keyword_groups_any_alternative_counts(evaluate):
    hits, total = evaluate.keyword_groups_hit("Exams are 40% and projects 60%", [["40%"], ["sixty", "60%"], ["late"]])
    assert (hits, total) == (2, 3)


def test_run_question_and_summary_with_mock_services(evaluate, settings, tmp_path):
    a = Assistant(settings)
    make_pdf(tmp_path / "Syllabus.pdf", ["Course grading: 40% exams, 60% team projects.", "Office hours Tuesday 2pm."])
    a.add_document(tmp_path / "Syllabus.pdf")
    questions = [
        {"id": "grading", "type": "syllabus", "question": "What is the course grading?", "expected_doc": "Syllabus",
         "expected_pages": [1], "expected_keywords": [["40%"], ["60%"]]},
        {"id": "none", "type": "unanswerable", "question": "zzzz qqqq", "expected_doc": None,
         "expected_pages": [], "expected_keywords": []},
    ]
    rows = [evaluate.run_question(a, q, rr) for rr in (True, False) for q in questions]
    grading_on = rows[0]
    assert grading_on["correct"] and grading_on["expected_page_cited"] and grading_on["expected_rank_in_retrieval"] == 1
    assert grading_on["image_shown"] and "Syllabus.pdf p.1" in grading_on["sources"]
    summary = evaluate.summarise(rows)
    assert set(summary) == {"rerank_on", "rerank_off"} and summary["rerank_on"]["n"] == 2
    md = evaluate.to_markdown({"timestamp": "t", "files": ["Syllabus.pdf"], "n_questions": 2, "top_k": 8, "mock": True},
                              summary, rows)
    assert "| Correct answers |" in md and "### grading (rerank on)" in md
    json.dumps(rows)  # everything recorded must be serialisable

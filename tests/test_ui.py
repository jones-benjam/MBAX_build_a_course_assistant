import pytest

from course_assistant.app_core import Assistant
from course_assistant import ui
from tests.fixtures import make_pdf


@pytest.fixture
def assistant(settings, tmp_path):
    a = Assistant(settings)
    make_pdf(tmp_path / "syllabus.pdf", ["Course grading: 40% exams, 60% team projects.", "Office hours Tuesday 2pm."])
    return a


def test_build_app_constructs(assistant):
    demo = ui.build_app(assistant)
    assert demo is not None


def test_add_duplicate_unsupported_and_remove_flow(assistant, tmp_path):
    msg, table, dd, ask_docs, quiz_docs = ui.handle_add(assistant, [str(tmp_path / "syllabus.pdf")])
    assert "Added" in msg and len(table) == 1 and table[0][0] == "syllabus.pdf" and table[0][2] == 2
    assert len(ask_docs["choices"]) == 1 and quiz_docs["choices"] == ask_docs["choices"]

    msg, table, *_ = ui.handle_add(assistant, [str(tmp_path / "syllabus.pdf")])
    assert "already in the library" in msg and len(table) == 1

    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"x")
    msg, table, *_ = ui.handle_add(assistant, [str(bad)])
    assert "Accepted: PDF" in msg and len(table) == 1

    doc_id = assistant.library.list_documents()[0].doc_id
    msg, table, dd, ask_docs, quiz_docs = ui.handle_remove(assistant, doc_id)
    assert "Removed" in msg and table == [] and ask_docs["choices"] == [] and dd["value"] is None
    msg, *_ = ui.handle_remove(assistant, None)
    assert "Select a document" in msg


def test_ask_flow_shows_answer_gallery_and_sources(assistant, tmp_path):
    md, gallery, src = ui.handle_ask(assistant, "How is grading weighted?", None, True)
    assert "No course materials yet" in md and gallery == []
    ui.handle_add(assistant, [str(tmp_path / "syllabus.pdf")])
    md, gallery, src = ui.handle_ask(assistant, "How is grading weighted between exams and projects?", None, True)
    assert "40% exams" in md and "Sources support" in md
    assert gallery and gallery[0][0].endswith("page_0001.png") and "syllabus.pdf, page 1" in gallery[0][1]
    assert "[S1] syllabus.pdf, page 1" in src
    # doc filter arrives as a list of ids; reranker off is honoured
    doc_id = assistant.library.list_documents()[0].doc_id
    md, gallery, src = ui.handle_ask(assistant, "office hours", [doc_id], False)
    assert "reranked" not in md and gallery


def test_quiz_flow_hides_then_reveals_answers(assistant, tmp_path):
    ui.handle_add(assistant, [str(tmp_path / "syllabus.pdf")])
    out = ui.handle_generate_quiz(assistant, None, "", 3, True)
    state, status, radios, feedback, gallery = out[0], out[1], out[2:2 + ui.MAX_QUIZ_QUESTIONS], out[-2], out[-1]
    assert state and "3 questions" in status and feedback == "" and gallery == []
    visible = [r for r in radios if r["visible"]]
    assert len(visible) == 3 and all(len(r["choices"]) == 4 for r in visible) and all(r["value"] is None for r in visible)
    assert "answer" not in status.lower().replace("answers stay hidden", "")

    key = {q["number"]: q["answer_index"] for q in state["questions"]}
    answers = [visible[i]["choices"][key[i + 1]] for i in range(3)] + [None] * (ui.MAX_QUIZ_QUESTIONS - 3)
    fb, fgal = ui.handle_submit_quiz(assistant, state, *answers)
    assert "Score: 3 / 3" in fb and "✅ Q1" in fb and fgal
    answers[0] = visible[0]["choices"][(key[1] + 1) % 4]
    answers[1] = None
    fb, _ = ui.handle_submit_quiz(assistant, state, *answers)
    assert "Score: 1 / 3" in fb and "❌ Q1" in fb and "⬜ Q2" in fb and "unanswered" in fb

    rev, rgal = ui.handle_reveal_quiz(assistant, state)
    assert "Answer key" in rev and rgal
    assert ui.handle_submit_quiz(assistant, None)[0] == "Generate a quiz first."


def test_quiz_error_is_shown_not_raised(settings):
    a = Assistant(settings)
    out = ui.handle_generate_quiz(a, None, "", 2, True)
    assert out[0] is None and "Add documents first" in out[1]

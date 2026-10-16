import re
import json

import pytest

from course_assistant.app_core import Assistant
from course_assistant.quiz import Quiz, QuizError, grade_quiz, parse_quiz_payload
from course_assistant.services import ChatMessage, ServiceUnavailable
from tests.fixtures import make_pdf


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages, *, json_mode=False, temperature=0.0):
        self.calls.append(list(messages))
        if not self.replies:
            raise ServiceUnavailable("llm down")
        return self.replies.pop(0)


@pytest.fixture
def assistant(settings, tmp_path):
    a = Assistant(settings)
    make_pdf(tmp_path / "syllabus.pdf", [
        "Course grading: 40% exams, 60% team projects. Late work loses 10% per day.",
        "Office hours are Tuesday at 2pm in the finance lab.",
        "The final project is a team-based deliverable worth 45% of the grade.",
    ])
    make_pdf(tmp_path / "week5.pdf", [
        "Retrieval-augmented generation (RAG) retrieves relevant documents and adds them to the prompt.",
        "Reranking reorders retrieved candidates with a cross-encoder for higher precision.",
    ])
    a.add_document(tmp_path / "syllabus.pdf")
    a.add_document(tmp_path / "week5.pdf")
    return a


def _good_payload(n=2, sources=(1,)):
    qs = []
    for i in range(n):
        qs.append({
            "question": f"According to the syllabus, what share of the grade do exams carry? ({i})",
            "options": ["40%", "60%", "45%", "10%"],
            "answer_index": 0,
            "explanation": "The grading section states exams are 40% and team projects 60%.",
            "sources": list(sources),
        })
    return json.dumps({"questions": qs})


def test_parse_quiz_payload_validates_shape():
    p = parse_quiz_payload("```json\n" + _good_payload() + "\n```")
    assert len(p.questions) == 2 and p.questions[0].sources == [1]
    with pytest.raises(Exception):
        parse_quiz_payload(json.dumps({"questions": []}))
    with pytest.raises(Exception):
        parse_quiz_payload(json.dumps({"questions": [{"question": "dup options?", "options": ["a", "A"], "answer_index": 0}]}))


def test_generate_quiz_with_mock_llm_has_sources_and_hidden_key(assistant):
    quiz = assistant.generate_quiz(n_questions=3)
    assert len(quiz.questions) == 3
    for q in quiz.questions:
        assert len(q.options) == 4 and 0 <= q.answer_index < 4
        assert q.sources and all(s.doc_name in {"syllabus.pdf", "week5.pdf"} for s in q.sources)
        assert all(s.image_path and s.image_path.exists() for s in q.sources)
    view = quiz.student_view()
    assert all(set(v) == {"number", "question", "options"} for v in view)
    assert "answer" not in json.dumps(view).lower()
    # Persisted and reloadable with the same key.
    loaded = assistant.quizzes.load(quiz.quiz_id)
    assert loaded is not None and loaded.answer_key() == quiz.answer_key()


def test_answer_key_is_fixed_and_grading_matches_key(assistant):
    assistant.quizzes.llm = ScriptedLLM([_good_payload(n=3)])
    quiz = assistant.generate_quiz(doc_ids={d.doc_id for d in assistant.library.list_documents() if d.name == "syllabus.pdf"},
                                   topic="grading exams", n_questions=3)
    key = quiz.answer_key()
    assert quiz.answer_key() == key and Quiz.from_dict(quiz.to_dict()).answer_key() == key
    # The correct option text survives the shuffle.
    assert all(q.options[q.answer_index] == "40%" for q in quiz.questions)
    all_right = {n: idx for n, idx in key.items()}
    res = grade_quiz(quiz, all_right)
    assert res.score == 3 and res.total == 3 and res.percent == 100.0
    wrong = {n: (idx + 1) % 4 for n, idx in key.items()}
    wrong[3] = None  # unanswered
    res2 = grade_quiz(quiz, wrong)
    assert res2.score == 0 and [f.correct for f in res2.feedback] == [False, False, False]
    assert res2.feedback[2].selected_letter is None and res2.feedback[2].correct_letter == quiz.questions[2].answer_letter
    fb = res2.feedback[0]
    assert "40%" in fb.explanation and fb.sources[0].doc_name == "syllabus.pdf" and fb.sources[0].page_number == 1
    assert quiz.questions[0].verified is True


def test_topic_and_doc_filter_restrict_evidence(assistant):
    week5 = {d.doc_id for d in assistant.library.list_documents() if d.name == "week5.pdf"}
    quiz = assistant.generate_quiz(doc_ids=week5, topic="reranking", n_questions=2)
    assert all(s.doc_id in week5 for q in quiz.questions for s in q.sources)
    assert quiz.doc_names == ["week5.pdf"] and quiz.topic == "reranking"
    user_msg = None  # confirm the topic reached the prompt
    llm = ScriptedLLM([_good_payload(n=1)])
    assistant.quizzes.llm = llm
    assistant.generate_quiz(doc_ids=week5, topic="reranking", n_questions=1)
    user_msg = llm.calls[0][1].text
    assert "Focus on the topic: reranking" in user_msg and "[S1] week5.pdf, page " in user_msg
    assert not re.search(r"\btag\b", user_msg, re.I)


def test_unmatched_topic_falls_back_to_sample_with_warning(assistant, monkeypatch):
    from course_assistant.index import RetrievalResult
    # Dense search always returns nearest neighbours, so an empty result only happens when
    # every channel fails or the filter excludes everything; simulate that case.
    monkeypatch.setattr(assistant.retriever, "retrieve", lambda *a, **k: RetrievalResult([], ["Text embedding service unavailable"]))
    quiz = assistant.generate_quiz(topic="reranking", n_questions=2)
    assert quiz.questions and any("using a general sample" in w for w in quiz.warnings)
    assert all(q.sources for q in quiz.questions)


def test_bad_questions_are_dropped_with_warnings(assistant):
    bad = json.dumps({"questions": [
        {"question": "Out of range index question?", "options": ["a", "b", "c", "d"], "answer_index": 7, "sources": [1]},
        {"question": "No valid sources question?", "options": ["a", "b", "c", "d"], "answer_index": 0, "sources": [99]},
        {"question": "Good question about exams share?", "options": ["40%", "60%", "45%", "10%"], "answer_index": 0,
         "explanation": "Exams are 40%.", "sources": [1, 42]},
    ]})
    assistant.quizzes.llm = ScriptedLLM([bad])
    quiz = assistant.generate_quiz(topic="grading", n_questions=3)
    assert len(quiz.questions) == 1 and quiz.questions[0].number == 1
    joined = " ".join(quiz.warnings)
    assert "out-of-range" in joined and "no valid sources" in joined and "[42]" in joined and "Only 1 of 3" in joined


def test_retry_on_malformed_then_errors(assistant):
    llm = ScriptedLLM(["not json", _good_payload(n=1)])
    assistant.quizzes.llm = llm
    quiz = assistant.generate_quiz(n_questions=1)
    assert len(quiz.questions) == 1 and len(llm.calls) == 2
    assistant.quizzes.llm = ScriptedLLM(["garbage", "garbage"])
    with pytest.raises(QuizError, match="unreadable"):
        assistant.generate_quiz(n_questions=1)
    assistant.quizzes.llm = ScriptedLLM([])
    with pytest.raises(QuizError, match="unavailable"):
        assistant.generate_quiz(n_questions=1)


def test_empty_library_gives_clear_error(settings):
    a = Assistant(settings)
    with pytest.raises(QuizError, match="Add documents"):
        a.generate_quiz(n_questions=2)


def test_quiz_rescues_slide_number_citations(assistant):
    """The model cites a page number instead of a block tag; the question ends up citing the
    page that actually supports it, with a warning explaining the correction."""
    from course_assistant.qa import rescue_citations
    from course_assistant.index import Evidence
    from course_assistant.ingest import Chunk

    def chunk(page, text):
        return Chunk(doc_id="d", doc_name="w.pptx", page_number=page, chunk_index=0,
                     chunk_id=f"d:{page}:0", text=text, image_path=None)
    evidence = [Evidence(chunk(3, "Office hours Tuesday"), 0, 0),
                Evidence(chunk(1, "Exams are 40% and team projects 60% of the grade"), 0, 0)]
    ids, notes = rescue_citations("Exams carry 40% of the grade per the grading section", [1], evidence)
    assert ids == [2] and notes

    # End to end through the generator: a payload citing page 1 of the syllabus by its page number.
    import json as _json
    syllabus = next(d for d in assistant.library.list_documents() if d.name == "syllabus.pdf")
    assistant.quizzes.llm = ScriptedLLM([_json.dumps({"questions": [{
        "question": "What share of the grade do exams carry?",
        "options": ["40%", "60%", "45%", "10%"], "answer_index": 0,
        "explanation": "The grading section states exams are 40% and team projects 60%.",
        "sources": [1],  # page-number style citation (page 1); block 1 may be a different page
    }]})])
    quiz = assistant.generate_quiz(doc_ids={syllabus.doc_id}, n_questions=1)
    q = quiz.questions[0]
    assert q.sources and q.sources[0].page_number == 1 and "40% exams" in q.sources[0].excerpt
    assert any("looked like a page/slide number" in w for w in quiz.warnings) or q.verified

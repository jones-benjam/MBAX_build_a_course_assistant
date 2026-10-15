import json

import pytest

from course_assistant.app_core import Assistant
from course_assistant.qa import AnswerPayload, build_messages, parse_payload, support_score, Source
from course_assistant.services import ChatMessage, ServiceUnavailable
from tests.fixtures import make_pdf


class ScriptedLLM:
    """Returns canned replies in order; records what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages, *, json_mode=False, temperature=0.0):
        self.calls.append(list(messages))
        if not self.replies:
            raise ServiceUnavailable("llm down")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def assistant(settings, tmp_path):
    a = Assistant(settings)
    make_pdf(tmp_path / "syllabus.pdf", [
        "Course grading: 40% exams, 60% team projects. Late work loses 10% per day.",
        "Office hours are Tuesday at 2pm in the finance lab.",
    ])
    make_pdf(tmp_path / "week2.pdf", [
        "Vibe Coding on Prod meme: 'it works on my machine' then ship it to production.",
    ])
    a.add_document(tmp_path / "syllabus.pdf")
    a.add_document(tmp_path / "week2.pdf")
    return a


def test_parse_payload_tolerates_fences_and_string_ids():
    raw = '```json\n{"answer": "x", "sources": ["S1", 2, "[S3]"], "missing_information": false}\n```'
    p = parse_payload(raw)
    assert p.sources == [1, 2, 3]
    p2 = parse_payload('Sure! {"answer": "y", "sources": []} thanks')
    assert p2.answer == "y" and p2.missing_information is False
    with pytest.raises(Exception):
        parse_payload("no json here")
    with pytest.raises(Exception):
        AnswerPayload.model_validate({"answer": ""})


def test_build_messages_tags_blocks_and_attaches_images(assistant):
    res = assistant.retrieve("grading exams projects", top_k=3)
    msgs = build_messages("How is grading weighted?", res.evidence, max_images=2)
    assert msgs[0].role == "system" and "missing_information" in msgs[0].text
    user = msgs[1]
    assert "[S1]" in user.text and "page/slide" in user.text and "Question: How is grading weighted?" in user.text
    assert 1 <= len(user.images) <= 2 and all(p.exists() for p in user.images)


def test_support_score():
    src = [Source(1, "d", "syllabus.pdf", 1, "Course grading: 40% exams, 60% team projects.", None)]
    assert support_score("Grading is 40% exams and 60% team projects.", src) > 0.5
    assert support_score("The capital of France is Paris.", src) == 0.0


def test_answer_with_mock_llm_cites_real_sources(assistant):
    ans = assistant.ask("How is the course grade weighted between exams and projects?")
    assert ans.error is None and ans.missing_information is False
    assert ans.sources and ans.sources[0].doc_name == "syllabus.pdf" and ans.sources[0].page_number == 1
    assert ans.sources[0].image_path.exists() and "40% exams" in ans.sources[0].excerpt
    assert ans.supported and ans.support_score > 0
    assert ans.latency_s >= 0 and ans.reranked
    d = ans.to_dict()
    assert d["sources"][0]["document"] == "syllabus.pdf" and d["sources"][0]["page"] == 1
    json.dumps(d)


def test_missing_information_when_nothing_retrieved(settings):
    a = Assistant(settings)  # empty library
    ans = a.ask("What is the late policy?")
    assert ans.missing_information is True and ans.sources == [] and "do not contain" in ans.answer
    assert ans.error is None


def test_model_declared_missing_information_is_respected(assistant):
    llm = ScriptedLLM([json.dumps({"answer": "This is not covered in the materials.", "sources": [1], "missing_information": True})])
    assistant.qa.llm = llm
    ans = assistant.ask("Who is the teaching assistant?")
    assert ans.missing_information is True and ans.sources == [] and "not covered" in ans.answer


def test_invalid_source_ids_are_dropped_and_flagged(assistant):
    llm = ScriptedLLM([json.dumps({"answer": "Grading is 40% exams and 60% team projects.", "sources": [1, 99], "missing_information": False})])
    assistant.qa.llm = llm
    ans = assistant.ask("How is grading weighted?")
    assert [s.id for s in ans.sources] == [1]
    assert any("non-existent" in w for w in ans.warnings)


def test_uncited_answer_is_marked_unsupported(assistant):
    llm = ScriptedLLM([json.dumps({"answer": "Grading is pass/fail.", "sources": [], "missing_information": False})])
    assistant.qa.llm = llm
    ans = assistant.ask("How is grading weighted?")
    assert ans.supported is False and ans.sources == [] and any("caution" in w for w in ans.warnings)


def test_low_overlap_answer_is_flagged(assistant):
    llm = ScriptedLLM([json.dumps({"answer": "Students receive unlimited extensions and bonus credit for attendance.", "sources": [1], "missing_information": False})])
    assistant.qa.llm = llm
    ans = assistant.ask("How is grading weighted?")
    assert ans.supported is False and ans.support_score < 0.25
    assert any("Low overlap" in w for w in ans.warnings)


def test_retries_once_on_malformed_json(assistant):
    llm = ScriptedLLM(["I think the answer is 40% exams.", json.dumps({"answer": "40% exams, 60% team projects.", "sources": [1]})])
    assistant.qa.llm = llm
    ans = assistant.ask("How is grading weighted?")
    assert ans.error is None and ans.sources and len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1][-1].text


def test_unreadable_output_twice_returns_clean_error(assistant):
    assistant.qa.llm = ScriptedLLM(["garbage", "more garbage"])
    ans = assistant.ask("How is grading weighted?")
    assert ans.error == "invalid model output" and ans.sources  # fallback passages still shown
    assert ans.supported is False


def test_llm_down_returns_clean_error_with_passages(assistant):
    assistant.qa.llm = ScriptedLLM([])
    ans = assistant.ask("How is grading weighted?")
    assert ans.error and "unavailable" in ans.answer and ans.sources
    assert "6418" not in json.dumps(ans.to_dict())


def test_doc_filter_and_reranker_toggle_pass_through(assistant):
    week2 = next(d for d in assistant.library.list_documents() if d.name == "week2.pdf")
    ans = assistant.ask("vibe coding meme", doc_ids={week2.doc_id}, use_reranker=False)
    assert ans.reranked is False and all(s.doc_id == week2.doc_id for s in ans.sources)

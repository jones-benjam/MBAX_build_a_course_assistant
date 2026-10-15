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
    assert "[S1] syllabus.pdf, page " in user.text and "Question: How is grading weighted?" in user.text
    assert "tag" not in user.text.lower()  # the model echoed "(Tag 5)" into prose when headers said "tag"
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


def test_slide_number_cited_as_block_tag_is_rescued(assistant):
    """Real-service finding: the model reasons 'slide N says...' and cites N, but N is a page
    number, not a block tag. The citation is moved to the block from that page when that
    block clearly supports the answer."""
    from course_assistant.index import Evidence
    from course_assistant.qa import rescue_citations
    from course_assistant.ingest import Chunk

    def chunk(page, text):
        return Chunk(doc_id="d", doc_name="week2.pptx", page_number=page, chunk_index=0,
                     chunk_id=f"d:{page}:0", text=text, image_path=None)
    evidence = [
        Evidence(chunk(17, "Model types: instruct and mixture of experts models"), 0, 0),
        Evidence(chunk(11, "The KV cache avoids reprocessing prior tokens during decoding, reducing TTFT"), 0, 0),
        Evidence(chunk(13, "Speculative decoding guesses candidate tokens ahead"), 0, 0),
    ]
    answer = "Slide 11 explains the KV cache avoids reprocessing prior tokens, reducing TTFT."
    ids, notes = rescue_citations(answer, [11], evidence)
    assert ids == [2] and len(notes) == 1 and "[S11]" in notes[0] and "[S2]" in notes[0]
    # A correct tag is left alone, and a page-number collision with a well-supported block is kept.
    assert rescue_citations(answer, [2], evidence) == ([2], [])
    assert rescue_citations("instruct and mixture of experts", [1], evidence) == ([1], [])


def test_evidence_headers_name_slides_and_prompts_forbid_tag_in_prose():
    """Regression for quiz explanations like "Slide 5 (Tag 5)": headers name the real
    slide or page, and both prompts tell the model never to write "tag" in prose."""
    from course_assistant.ingest import Chunk
    from course_assistant.qa import SYSTEM_PROMPT, evidence_header
    from course_assistant.quiz import QUIZ_SYSTEM_PROMPT

    deck = Chunk(chunk_id="d:11:0", doc_id="d", doc_name="Week 5.pptx", page_number=11, text="x")
    pdf = Chunk(chunk_id="s:2:0", doc_id="s", doc_name="Syllabus.pdf", page_number=2, text="x")
    assert evidence_header(5, deck) == "[S5] Week 5.pptx, slide 11"
    assert evidence_header(1, pdf) == "[S1] Syllabus.pdf, page 2"
    for prompt in (SYSTEM_PROMPT, QUIZ_SYSTEM_PROMPT):
        assert 'Never write the word "tag"' in prompt or 'never write the word "tag"' in prompt
        assert "TAG numbers" not in prompt


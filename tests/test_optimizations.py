"""Offline checks for the speed and evaluation improvements:
page-level reranking, the rerank candidate cap, parallel retrieval channels,
query-embedding caching, control-token escaping, reusing a retrieval when
answering, and the evaluation's retrieval metrics."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from course_assistant.app_core import Assistant
from course_assistant.index import _QueryCache
from course_assistant.qa import build_messages, escape_control_tokens, restore_control_tokens
from course_assistant.services import ServiceUnavailable
from tests.fixtures import make_pdf

ROOT = Path(__file__).resolve().parents[1]


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


class SpyReranker:
    """Wraps a reranker and records every payload it receives."""

    def __init__(self, inner):
        self.inner = inner
        self.payloads: list[list[dict]] = []

    def rerank(self, query, candidates):
        self.payloads.append(list(candidates))
        return self.inner.rerank(query, candidates)


class CountingEmbed:
    """Wraps an embedding client and counts query-embedding calls."""

    def __init__(self, inner):
        self.inner = inner
        self.query_calls = 0

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if name in ("embed_query", "embed_text"):
            def counted(*a, **k):
                self.query_calls += 1
                return attr(*a, **k)
            return counted
        return attr


# ---------------------------------------------------------------- reranking

def test_reranker_scores_each_page_once(assistant):
    spy = SpyReranker(assistant.services.rerank)
    assistant.services.rerank = spy
    res = assistant.retrieve("vibe coding meme office hours grading agents", top_k=8)
    assert res.reranked and len(spy.payloads) == 1
    payload = spy.payloads[0]
    # One entry per distinct page among the candidates, never one per chunk.
    pages = {(e.chunk.doc_id, e.chunk.page_number) for e in res.evidence}
    assert len(payload) == len(pages)
    images = [p["image"] for p in payload if p["image"]]
    assert len(images) == len(set(images))
    # Every chunk carries its page's score.
    by_page = {}
    for e in res.evidence:
        by_page.setdefault((e.chunk.doc_id, e.chunk.page_number), set()).add(e.rerank_score)
    assert all(len(scores) == 1 for scores in by_page.values())


def test_reranker_candidate_cap_limits_pages_sent(assistant):
    assistant.retriever.settings = dataclasses.replace(assistant.settings, rerank_candidates=2)
    spy = SpyReranker(assistant.services.rerank)
    assistant.services.rerank = spy
    res = assistant.retrieve("vibe coding meme office hours grading agents", top_k=4)
    assert res.reranked and len(spy.payloads[0]) == 2
    scored = [e for e in res.evidence if e.rerank_score is not None]
    unscored = [e for e in res.evidence if e.rerank_score is None]
    # Pages beyond the cap keep their fused order behind every reranked page.
    assert scored and res.evidence[: len(scored)] == scored
    assert all(e.score < min(s.score for s in scored) for e in unscored)


def test_reranker_wrong_score_count_degrades_to_fused_order(assistant):
    class Short:
        def rerank(self, query, candidates):
            return [1.0]
    assistant.services.rerank = Short()
    res = assistant.retrieve("vibe coding meme office hours grading agents", top_k=4)
    assert res.reranked is False and any("Reranker unavailable" in w for w in res.warnings)
    assert res.evidence


# ---------------------------------------------------------------- parallel channels

def test_parallel_channels_give_same_results_and_warnings(assistant):
    a = assistant.retrieve("find the meme about vibe coding on prod", top_k=3)
    b = assistant.retrieve("find the meme about vibe coding on prod", top_k=3)
    assert [e.chunk.chunk_id for e in a.evidence] == [e.chunk.chunk_id for e in b.evidence]
    assert {"keyword", "text", "visual"} <= set(a.evidence[0].channels)


def test_one_channel_down_still_returns_others(assistant):
    class Down:
        def __getattr__(self, name):
            def fail(*a, **k):
                raise ServiceUnavailable("visual down")
            return fail
    assistant.services.visual_embed = Down()
    res = assistant.retrieve("office hours tuesday", top_k=2, use_reranker=False)
    assert res.evidence and len(res.warnings) == 1 and "Visual embedding" in res.warnings[0]


# ---------------------------------------------------------------- query cache

def test_query_embeddings_are_cached(assistant):
    text_spy = CountingEmbed(assistant.services.text_embed)
    vis_spy = CountingEmbed(assistant.services.visual_embed)
    assistant.services.text_embed, assistant.services.visual_embed = text_spy, vis_spy
    q = "a question asked twice about grading"
    first = assistant.retrieve(q, top_k=3, use_reranker=False)
    second = assistant.retrieve(q, top_k=3, use_reranker=False)
    assert text_spy.query_calls == 1 and vis_spy.query_calls == 1
    assert [e.chunk.chunk_id for e in first.evidence] == [e.chunk.chunk_id for e in second.evidence]
    assistant.retrieve("a different question", top_k=3, use_reranker=False)
    assert text_spy.query_calls == 2 and vis_spy.query_calls == 2


def test_query_cache_evicts_oldest_and_skips_failures():
    cache = _QueryCache(maxsize=2)
    cache.get_or_compute("a", lambda: [1.0])
    cache.get_or_compute("b", lambda: [2.0])
    cache.get_or_compute("a", lambda: [9.0])          # hit; "a" becomes most recent
    cache.get_or_compute("c", lambda: [3.0])          # evicts "b"
    assert len(cache) == 2
    assert cache.get_or_compute("a", lambda: [9.0]) == [1.0]
    assert cache.get_or_compute("b", lambda: [7.0]) == [7.0]

    def boom():
        raise ServiceUnavailable("down")
    with pytest.raises(ServiceUnavailable):
        cache.get_or_compute("d", boom)
    assert cache.get_or_compute("d", lambda: [4.0]) == [4.0]


# ---------------------------------------------------------------- control tokens

def test_control_tokens_round_trip():
    text = "ChatML uses <|im_start|> and <|im_end|>; Llama uses <|eot_id|>."
    escaped = escape_control_tokens(text)
    assert "<|" not in escaped and "< |im_start| >" in escaped
    assert restore_control_tokens(escaped) == text
    # The model may drop or add spaces; restoring still works.
    assert restore_control_tokens("<|im_end |> and < | im_start|>") == "<|im_end|> and <|im_start|>"
    # Ordinary text with angle brackets or pipes is left alone.
    plain = "a < b | c > d and x || y"
    assert escape_control_tokens(plain) == plain and restore_control_tokens(plain) == plain


def test_prompt_escapes_tokens_and_answer_restores_them(settings, tmp_path):
    a = Assistant(settings)
    make_pdf(tmp_path / "week3.pdf", ["ChatML marks messages with <|im_start|> and <|im_end|> tokens."])
    a.add_document(tmp_path / "week3.pdf")
    res = a.retrieve("ChatML special tokens", top_k=2)
    msgs = build_messages("Which tokens does ChatML use, like <|im_end|>?", res.evidence)
    assert "<|" not in msgs[1].text and "< |im_end| >" in msgs[1].text

    class SpacedLLM:
        def chat(self, messages, *, json_mode=False, temperature=0.0):
            return json.dumps({"answer": "It uses < |im_start| > and < |im_end| > to mark messages.",
                               "sources": [1], "missing_information": False})
    a.qa.llm = SpacedLLM()
    ans = a.ask("Which tokens does ChatML use?")
    assert "<|im_start|>" in ans.answer and "<|im_end|>" in ans.answer
    # Stored source excerpts keep the original tokens.
    assert any("<|im_end|>" in s.excerpt for s in ans.sources)


# ---------------------------------------------------------------- reuse retrieval

def test_answer_reuses_supplied_retrieval(assistant):
    res = assistant.retrieve("office hours tuesday", top_k=2)
    calls = []
    real = assistant.retriever.retrieve
    assistant.retriever.retrieve = lambda *a, **k: calls.append(1) or real(*a, **k)
    ans = assistant.ask("When are office hours?", retrieval=res)
    assert calls == [] and ans.sources
    assistant.ask("When are office hours?")
    assert calls == [1]


# ---------------------------------------------------------------- evaluation

@pytest.fixture(scope="module")
def evaluate():
    spec = importlib.util.spec_from_file_location("evaluate_opt", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retrieval_metrics(evaluate):
    m = evaluate.retrieval_metrics([1, 2, None, 3])
    assert m["mrr"] == round((1 + 0.5 + 0 + 1 / 3) / 4, 3)
    assert m["recall_at_1"] == 0.25 and m["recall_at_3"] == 0.75
    assert evaluate.retrieval_metrics([]) == {"mrr": 0.0, "recall_at_1": 0.0, "recall_at_3": 0.0}


def test_run_question_retrieves_once(evaluate, assistant):
    calls = []
    real = assistant.retriever.retrieve
    assistant.retriever.retrieve = lambda *a, **k: calls.append(1) or real(*a, **k)
    q = {"id": "oh", "type": "syllabus", "question": "When are office hours?", "expected_doc": "syllabus",
         "expected_pages": [2], "expected_keywords": [["tuesday"]]}
    row = evaluate.run_question(assistant, q, True)
    assert calls == [1]
    assert row["answer_s"] >= row["retrieval_s"]
    summary = evaluate.summarise([row, {**row, "rerank": False}])
    assert {"mrr", "recall_at_1", "recall_at_3"} <= set(summary["rerank_on"])

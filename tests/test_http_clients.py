"""Offline tests for the HTTP clients: request shapes match the class endpoints'
vLLM OpenAPI specs and model cards, responses are parsed, and failures surface as
ServiceUnavailable without leaking the API key. Uses an httpx mock transport, so no
network access is needed."""
import json

import httpx
import pytest

from course_assistant.config import ServiceConfig
from course_assistant.services import ChatMessage, ServiceUnavailable
from course_assistant.services.http_clients import (
    HttpLLM, HttpReranker, HttpTextEmbed, HttpVisualEmbed, _api_base,
)

KEY = "sk-TEST-SECRET"


class Recorder:
    """Mock transport that records requests and replies with scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            r = self.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            status, body = r
            return httpx.Response(status, json=body)
        return httpx.MockTransport(handler)

    def payload(self, i: int = 0) -> dict:
        return json.loads(self.requests[i].content)


def cfg(base="http://host:9001", model="m"):
    return ServiceConfig(base_url=base, api_key=KEY, model=model)


def embedding_response(vectors):
    return 200, {"object": "list", "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "slide.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return p


# ---------------------------------------------------------------- base URL

@pytest.mark.parametrize("given", ["http://host:9002", "http://host:9002/", "http://host:9002/v1", "http://host:9002/v1/"])
def test_api_base_always_ends_with_v1(given):
    assert _api_base(given) == "http://host:9002/v1"


def test_authorization_header_sent_and_path_under_v1():
    rec = Recorder([embedding_response([[0.1, 0.2]])])
    HttpTextEmbed(cfg("http://host:9002"), transport=rec.transport()).embed(["x"])
    req = rec.requests[0]
    assert str(req.url) == "http://host:9002/v1/embeddings"
    assert req.headers["authorization"] == f"Bearer {KEY}"


# ---------------------------------------------------------------- vision LLM

def test_llm_chat_payload_matches_openai_chat_schema(png):
    rec = Recorder([(200, {"choices": [{"message": {"role": "assistant", "content": '{"answer": "ok"}'}}]})])
    llm = HttpLLM(cfg(), transport=rec.transport())
    out = llm.chat([ChatMessage("system", "rules"), ChatMessage("user", "What is on the slide?", images=[png])],
                   json_mode=True, temperature=0.0)
    assert out == '{"answer": "ok"}'
    p = rec.payload()
    assert p["model"] == "m" and p["temperature"] == 0.0
    assert p["response_format"] == {"type": "json_object"}
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["max_completion_tokens"] > 0
    sys_msg, user_msg = p["messages"]
    assert sys_msg == {"role": "system", "content": "rules"}
    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    assert parts[0] == {"type": "text", "text": "What is on the slide?"}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_llm_strips_think_block_and_omits_response_format_without_json_mode():
    rec = Recorder([(200, {"choices": [{"message": {"content": "<think>hmm\nmore</think>\nFinal answer"}}]})])
    out = HttpLLM(cfg(), transport=rec.transport()).chat([ChatMessage("user", "hi")])
    assert out == "Final answer"
    assert "response_format" not in rec.payload()


def test_llm_unexpected_shape_is_service_unavailable():
    rec = Recorder([(200, {"choices": []})])
    with pytest.raises(ServiceUnavailable):
        HttpLLM(cfg(), transport=rec.transport()).chat([ChatMessage("user", "hi")])


# ---------------------------------------------------------------- text embeddings

def test_text_embed_adds_passage_and_query_prefixes():
    rec = Recorder([embedding_response([[1.0], [2.0]]), embedding_response([[3.0]])])
    client = HttpTextEmbed(cfg("http://host:9002"), transport=rec.transport())
    assert client.embed(["alpha", "beta"]) == [[1.0], [2.0]]
    assert client.embed_query(["who?"]) == [[3.0]]
    assert rec.payload(0) == {"model": "m", "input": ["passage: alpha", "passage: beta"]}
    assert rec.payload(1) == {"model": "m", "input": ["query: who?"]}


def test_text_embed_orders_vectors_by_index():
    rec = Recorder([(200, {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})])
    assert HttpTextEmbed(cfg(), transport=rec.transport()).embed(["a", "b"]) == [[1.0], [2.0]]


# ---------------------------------------------------------------- visual embeddings

def test_visual_embed_images_use_chat_messages_with_image_parts(png):
    rec = Recorder([embedding_response([[0.5]]), embedding_response([[0.6]])])
    client = HttpVisualEmbed(cfg("http://host:9003"), transport=rec.transport())
    assert client.embed_images([png, png]) == [[0.5], [0.6]]
    assert len(rec.requests) == 2  # one conversation per request
    p = rec.payload(0)
    assert "input" not in p
    system, user = p["messages"]
    assert system["role"] == "system" and isinstance(system["content"], str)
    assert user["role"] == "user"
    assert user["content"] == [{"type": "image_url", "image_url": {"url": user["content"][0]["image_url"]["url"]}}]
    assert user["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_visual_embed_text_query_uses_text_part():
    rec = Recorder([embedding_response([[0.7]])])
    client = HttpVisualEmbed(cfg("http://host:9003"), transport=rec.transport())
    assert client.embed_text(["vibe coding meme"]) == [[0.7]]
    user = rec.payload()["messages"][1]
    assert user["content"] == [{"type": "text", "text": "vibe coding meme"}]


# ---------------------------------------------------------------- reranker

def test_rerank_documents_are_strings_or_typed_content_parts(png):
    rec = Recorder([(200, {"results": [
        {"index": 1, "relevance_score": 0.9, "document": {"text": "b"}},
        {"index": 0, "relevance_score": 0.1, "document": {"text": "a"}},
    ]})])
    client = HttpReranker(cfg("http://host:9004"), transport=rec.transport())
    scores = client.rerank("q", [{"text": "a"}, {"text": "b", "image": str(png)}])
    assert scores == [0.1, 0.9]
    assert str(rec.requests[0].url) == "http://host:9004/v1/rerank"
    p = rec.payload()
    assert p["query"] == "q" and p["top_n"] == 2 and isinstance(p["instruction"], str)
    assert p["documents"][0] == "a"
    parts = p["documents"][1]["content"]
    assert parts[0] == {"type": "text", "text": "b"}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_rerank_bad_shape_is_service_unavailable():
    rec = Recorder([(200, {"results": [{"index": 5, "relevance_score": 1.0}]})])
    with pytest.raises(ServiceUnavailable):
        HttpReranker(cfg(), transport=rec.transport()).rerank("q", [{"text": "a"}])


# ---------------------------------------------------------------- failures

@pytest.mark.parametrize("response", [
    (401, {"error": "Unauthorized"}),
    (422, {"detail": [{"loc": ["body", "input"], "msg": "field required"}]}),
    (500, {"error": {"message": "engine died", "type": "InternalServerError"}}),
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("slow"),
])
def test_failures_become_service_unavailable_without_key(response):
    rec = Recorder([response])
    client = HttpTextEmbed(cfg(), transport=rec.transport())
    with pytest.raises(ServiceUnavailable) as e:
        client.embed(["x"])
    msg = str(e.value)
    assert KEY not in msg
    if isinstance(response, tuple):
        assert str(response[0]) in msg


@pytest.mark.parametrize("as_json", [True, False])
@pytest.mark.parametrize("padding", [0, 190])
def test_echoed_api_key_is_redacted_before_error_truncation(as_json, padding):
    secret = "synthetic-credential-for-redaction-test"
    detail = "x" * padding + secret + " authentication rejected"

    def handler(request):
        if as_json:
            return httpx.Response(401, json={"error": {"message": detail}})
        return httpx.Response(401, text=detail)

    client = HttpTextEmbed(
        ServiceConfig("https://example.invalid", secret, "test-model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ServiceUnavailable) as error:
            client.embed(["test"])
        message = str(error.value)
        assert secret not in message
        assert "synthetic-" not in message  # catch a partially truncated key too
        assert "[REDACTED]" in message
        assert "HTTP 401" in message
        assert len(message.split(": ", 1)[1]) <= 200
    finally:
        client._client.close()


def test_http_error_message_includes_server_detail_for_debugging():
    rec = Recorder([(400, {"error": {"message": "Unsupported content part: image", "type": "BadRequest"}})])
    with pytest.raises(ServiceUnavailable, match="Unsupported content part"):
        HttpVisualEmbed(cfg(), transport=rec.transport()).embed_text(["x"])

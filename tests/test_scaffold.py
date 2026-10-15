import json

from course_assistant.config import load_settings
from course_assistant.services import ChatMessage, ServiceUnavailable, get_services
from course_assistant.services.http_clients import HttpLLM
from course_assistant.config import ServiceConfig


def test_settings_default_to_mock_and_create_dirs(settings):
    assert settings.mock_services is True
    assert settings.uploads_dir.is_dir() and settings.index_dir.is_dir()
    assert "***" in json.dumps(settings.llm.redacted())


def test_dotenv_loader_does_not_override_existing_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=from-file\nTOP_K=3\n")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    monkeypatch.delenv("TOP_K", raising=False)
    s = load_settings(dotenv=env)
    assert s.llm.model == "from-env"
    assert s.top_k == 3


def test_mock_services_roundtrip(settings):
    svc = get_services(settings)
    assert svc.is_mock
    vecs = svc.text_embed.embed(["vibe coding on prod", "vibe coding in prod", "syllabus grading policy"])
    assert len(vecs) == 3 and len(vecs[0]) == 256

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cos(vecs[0], vecs[1]) > cos(vecs[0], vecs[2])
    scores = svc.rerank.rerank("vibe coding", [{"text": "vibe coding meme"}, {"text": "grading"}])
    assert scores[0] > scores[1]


def test_mock_llm_declares_missing_information_without_evidence(settings):
    svc = get_services(settings)
    reply = json.loads(svc.llm.chat([ChatMessage("system", "answer"), ChatMessage("user", "What is X?")], json_mode=True))
    assert reply["missing_information"] is True and reply["sources"] == []


def test_mock_llm_cites_evidence_tags(settings):
    svc = get_services(settings)
    user = "Question: when is the exam?\n[S1] The final exam is in week 12.\n[S2] Office hours Tuesday."
    reply = json.loads(svc.llm.chat([ChatMessage("system", "answer"), ChatMessage("user", user)], json_mode=True))
    assert reply["sources"] == [1, 2] and "week 12" in reply["answer"]


def test_http_client_reports_unavailable_without_leaking_key():
    cfg = ServiceConfig(base_url="http://127.0.0.1:9", api_key="sk-SECRET", model="m")
    try:
        HttpLLM(cfg).chat([ChatMessage("user", "hi")])
    except ServiceUnavailable as e:
        assert "sk-SECRET" not in str(e)
    else:
        raise AssertionError("expected ServiceUnavailable")

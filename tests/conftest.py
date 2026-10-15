import os
import pytest

# Tests must never hit real services or read a developer's .env.
os.environ["MOCK_SERVICES"] = "1"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from course_assistant.config import load_settings
    s = load_settings(dotenv=None)
    s.ensure_dirs()
    return s

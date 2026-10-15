"""Configuration loaded from environment variables (and an optional .env file).

Secrets are never logged or echoed. Use `.env.example` as the template.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency). Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceConfig:
    base_url: str
    api_key: str
    model: str

    def redacted(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "api_key": "***"}


@dataclass(frozen=True)
class Settings:
    mock_services: bool
    llm: ServiceConfig
    text_embed: ServiceConfig
    visual_embed: ServiceConfig
    rerank: ServiceConfig
    parser: ServiceConfig
    data_dir: Path
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    use_reranker: bool = True
    supported_extensions: tuple[str, ...] = field(
        default=(".pdf", ".pptx", ".docx", ".txt", ".md")
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.pages_dir, self.index_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


def _svc(prefix: str, default_port: int, default_model: str) -> ServiceConfig:
    return ServiceConfig(
        base_url=os.environ.get(f"{prefix}_BASE_URL", f"http://dobolyi.com:{default_port}"),
        api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        model=os.environ.get(f"{prefix}_MODEL", default_model),
    )


def load_settings(dotenv: str | Path | None = ".env") -> Settings:
    if dotenv is not None:
        _load_dotenv(Path(dotenv))
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    return Settings(
        mock_services=_env_bool("MOCK_SERVICES", default=True),
        llm=_svc("LLM", 9001, "class-vision-llm"),
        text_embed=_svc("TEXT_EMBED", 9002, "class-text-embed"),
        visual_embed=_svc("VISUAL_EMBED", 9003, "class-visual-embed"),
        rerank=_svc("RERANK", 9004, "class-reranker"),
        parser=_svc("PARSER", 9005, "class-parser"),
        data_dir=data_dir,
        chunk_size=int(os.environ.get("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "120")),
        top_k=int(os.environ.get("TOP_K", "8")),
        use_reranker=_env_bool("USE_RERANKER", default=True),
    )

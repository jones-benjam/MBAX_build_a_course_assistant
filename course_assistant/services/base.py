"""Abstract interfaces for the model services the assistant depends on."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


class ServiceUnavailable(RuntimeError):
    """Raised when a model service cannot be reached or returns an error.

    The message is safe to show to users: it never contains keys or URLs with secrets.
    """


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    text: str
    images: list[Path] = field(default_factory=list)  # local image files attached to this turn


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], *, json_mode: bool = False, temperature: float = 0.0) -> str:
        """Return the assistant's reply text. With json_mode the reply should be a JSON object."""


class TextEmbedClient(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, same order."""


class VisualEmbedClient(ABC):
    @abstractmethod
    def embed_images(self, images: Sequence[Path]) -> list[list[float]]:
        """Embed a batch of image files into the shared image/text space."""

    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed query text into the same space as `embed_images` (for text→image search)."""


class RerankClient(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[dict]) -> list[float]:
        """Score each candidate for the query. Candidates are dicts with optional
        'text' and 'image' (Path) keys. Higher is more relevant. Same order as input."""

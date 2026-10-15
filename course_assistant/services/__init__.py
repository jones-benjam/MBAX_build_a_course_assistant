"""Model-service clients.

`get_services(settings)` returns a `Services` bundle. With `MOCK_SERVICES=1` the
bundle uses deterministic local stand-ins so the app and tests run offline; otherwise
it uses HTTP clients for the class endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .base import (
    ChatMessage,
    LLMClient,
    RerankClient,
    ServiceUnavailable,
    TextEmbedClient,
    VisualEmbedClient,
)
from .mock import MockLLM, MockReranker, MockTextEmbed, MockVisualEmbed
from .http_clients import HttpLLM, HttpReranker, HttpTextEmbed, HttpVisualEmbed


@dataclass
class Services:
    llm: LLMClient
    text_embed: TextEmbedClient
    visual_embed: VisualEmbedClient
    rerank: RerankClient
    is_mock: bool


def get_services(settings: Settings) -> Services:
    if settings.mock_services:
        return Services(
            llm=MockLLM(),
            text_embed=MockTextEmbed(),
            visual_embed=MockVisualEmbed(),
            rerank=MockReranker(),
            is_mock=True,
        )
    return Services(
        llm=HttpLLM(settings.llm),
        text_embed=HttpTextEmbed(settings.text_embed),
        visual_embed=HttpVisualEmbed(settings.visual_embed),
        rerank=HttpReranker(settings.rerank),
        is_mock=False,
    )


__all__ = [
    "ChatMessage",
    "LLMClient",
    "RerankClient",
    "ServiceUnavailable",
    "Services",
    "TextEmbedClient",
    "VisualEmbedClient",
    "get_services",
]

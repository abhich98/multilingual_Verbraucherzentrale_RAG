"""Provider-neutral chat completion types."""

from __future__ import annotations

from typing import Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMProvider(Protocol):
    """Minimal interface consumed by the RAG application layer."""

    def generate(self, messages: list[ChatMessage]) -> str: ...


class ProviderError(RuntimeError):
    """Raised when a configured model provider cannot return an answer."""
"""Ground answers in retrieved chunks and return citation metadata."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from multilingual_rag.providers.base import ChatMessage, LLMProvider
from multilingual_rag.retrieval.faiss_store import SearchResult


class Retriever(Protocol):
    def __call__(self, query: str, *, top_k: int) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[SearchResult]


SYSTEM_PROMPT = (
    "Du bist ein hilfreicher, präziser Verbraucher-Ratgeber. "
    "Beantworte die Frage ausschließlich anhand der bereitgestellten Quellen. "
    "Zitiere verwendete Quellen inline mit ihrer Kennung, zum Beispiel [S1]. "
    "Wenn die Quellen keine ausreichende Antwort enthalten, sage das offen und "
    "rate nicht. Antworte in der Sprache der Frage."
)


def _source_block(sources: Sequence[SearchResult], max_characters: int) -> str:
    blocks: list[str] = []
    remaining = max_characters
    for index, result in enumerate(sources, start=1):
        chunk = result.chunk
        source_path = str(chunk.get("source_path", "Unbekannte Quelle"))
        title = str(chunk.get("title") or source_path)
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end", page_start)
        page_label = f", S. {page_start}"
        if page_end and page_end != page_start:
            page_label += f"–{page_end}"
        url = str(chunk.get("url", ""))
        citation = f"[S{index}] {title} ({source_path}{page_label})"
        if url:
            citation += f"\nURL: {url}"
        text = str(chunk.get("text", ""))
        available = remaining - len(citation) - 1
        if available <= 0:
            break
        text = text[:available]
        block = f"{citation}\n{text}"
        blocks.append(block)
        remaining -= len(block) + 2
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    *,
    retrieve: Retriever,
    provider: LLMProvider,
    top_k: int = 5,
    max_context_characters: int = 18000,
    history: Sequence[dict[str, Any]] = (),
) -> RAGResponse:
    """Retrieve relevant sources and ask the configured provider for a grounded answer."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if max_context_characters < 1:
        raise ValueError("max_context_characters must be positive")

    sources = retrieve(question, top_k=top_k)
    if not sources:
        return RAGResponse(
            answer="Ich habe in den verfügbaren Quellen keine passenden Informationen gefunden.",
            sources=[],
        )

    messages: list[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Pass only recent conversational turns; source evidence is always retrieved afresh.
    for message in history[-8:]:
        role, content = message.get("role"), message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    context = _source_block(sources, max_context_characters)
    messages.append(
        {
            "role": "user",
            "content": f"Frage:\n{question}\n\nVerfügbare Quellen:\n{context}",
        }
    )
    return RAGResponse(answer=provider.generate(messages), sources=sources)
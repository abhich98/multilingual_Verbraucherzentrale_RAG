"""Environment-backed configuration for the local RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from multilingual_rag.retrieval.embeddings import DEFAULT_MODEL_NAME

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ConfigurationError(ValueError):
    """Raised when a required application setting is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_url: str
    request_timeout_seconds: float
    index_root: Path
    embedding_model: str
    embedding_revision: str | None
    embedding_cache_dir: str | None
    top_k: int
    max_context_characters: int

    @classmethod
    def from_environment(
        cls, *, api_key_override: str | None = None
    ) -> AppConfig:
        api_key = api_key_override or os.getenv("OPENROUTER_API_KEY", "").strip()
        model = os.getenv("OPENROUTER_MODEL", "").strip()
        missing = []
        if not api_key:
            missing.append("OPENROUTER_API_KEY")
        if not model:
            missing.append("OPENROUTER_MODEL")
        if missing:
            raise ConfigurationError(
                "Set the following environment variable(s): " + ", ".join(missing)
            )

        timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "45"))
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        max_context = int(os.getenv("RAG_MAX_CONTEXT_CHARACTERS", "18000"))
        if timeout <= 0:
            raise ConfigurationError("OPENROUTER_TIMEOUT_SECONDS must be positive")
        if top_k < 1:
            raise ConfigurationError("RAG_TOP_K must be greater than zero")
        if max_context < 1:
            raise ConfigurationError("RAG_MAX_CONTEXT_CHARACTERS must be positive")

        revision = os.getenv("EMBEDDING_REVISION", "").strip() or None
        return cls(
            openrouter_api_key=api_key,
            openrouter_model=model,
            openrouter_url=os.getenv("OPENROUTER_URL", DEFAULT_OPENROUTER_URL),
            request_timeout_seconds=timeout,
            index_root=Path(os.getenv("RAG_INDEX_ROOT", "artifacts/indexes")),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
            embedding_revision=revision,
            embedding_cache_dir=os.getenv("EMBEDDING_CACHE_DIR", ".cache/"),
            top_k=top_k,
            max_context_characters=max_context,
        )
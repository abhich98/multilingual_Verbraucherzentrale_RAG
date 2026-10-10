"""Local multilingual-E5 embeddings for passages and search queries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbeddingProvider(Protocol):
    """Model-independent contract for passage and query vectorization."""

    def encode_passages(self, texts: Sequence[str], *, batch_size: int = 32) -> Any: ...

    def encode_query(self, text: str) -> Any: ...


class E5Embedder:
    """Encode E5 passages and queries into normalized CPU vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "cpu",
        revision: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.revision = revision
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device, revision=revision)
        self._model = model

    def encode_passages(
        self, texts: Sequence[str], *, batch_size: int = 32
    ) -> Any:
        """Encode document chunks using E5's passage task prefix."""
        return self._encode(texts, prefix="passage: ", batch_size=batch_size)

    def encode_query(self, text: str) -> Any:
        """Encode a user search query using E5's query task prefix."""
        return self._encode([text], prefix="query: ", batch_size=1)[0]

    def _encode(
        self, texts: Sequence[str], *, prefix: str, batch_size: int
    ) -> Any:
        # E5 expects task prefixes, and unit-length vectors allow inner product to rank cosine similarity.
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        return self._model.encode(
            prefixed_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
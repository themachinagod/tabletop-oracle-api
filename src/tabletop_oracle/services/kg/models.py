"""Data models and exceptions for the KG embedding service.

Contains value objects for embedding results and domain-specific
exceptions. Follows the same pattern as the VisionService models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of an embedding API call.

    Attributes:
        embeddings: List of embedding vectors returned by the API.
        model_id: The model identifier that produced these embeddings.
        input_tokens: Number of input tokens consumed.
    """

    embeddings: list[list[float]]
    model_id: str
    input_tokens: int = 0


class EmbeddingError(Exception):
    """Raised when the embedding API call fails after retries.

    Attributes:
        provider: The model provider that failed.
        model_id: The model identifier that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model_id: str = "",
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        super().__init__(message)


class EmbeddingUnavailableError(Exception):
    """Raised when no embedding model slot is configured.

    Used for graceful degradation -- callers can catch this to skip
    embedding generation when no model is available.
    """

    def __init__(self, message: str = "Embedding model not configured") -> None:
        super().__init__(message)

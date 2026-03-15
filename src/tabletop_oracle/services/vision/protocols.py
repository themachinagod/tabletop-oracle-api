"""Protocol definitions for vision service abstractions.

Defines the VisionServiceProtocol that both the V1 implementation and
any future implementations must satisfy. Also defines the ModelSlotRepository
and TokenUsageLogger protocols for dependency injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tabletop_oracle.services.vision.models import (
        ImageDescription,
        TokenAttribution,
    )


@runtime_checkable
class VisionServiceProtocol(Protocol):
    """Interface for vision processing services.

    Implementations convert images to text descriptions using a
    configured vision model. Used by the ingestion pipeline and
    AI workflow.
    """

    async def describe_image(
        self,
        image_data: bytes,
        image_format: str,
        context: str | None = None,
        attribution: TokenAttribution | None = None,
    ) -> ImageDescription:
        """Process an image via a vision model and return a text description.

        Args:
            image_data: Raw image bytes.
            image_format: Image format (png, jpeg, webp, gif).
            context: Optional context to guide description
                (e.g., "game board diagram").
            attribution: Token usage attribution for tracking.

        Returns:
            ImageDescription with text content and confidence metadata.

        Raises:
            VisionProcessingError: If the model call fails after retries.
            VisionUnavailableError: If no vision model is configured.
        """
        ...

    async def is_available(self) -> bool:
        """Check whether a vision model is configured and reachable.

        Returns:
            True if vision processing can be performed.
        """
        ...


@runtime_checkable
class ModelSlotRepository(Protocol):
    """Repository for reading model slot configuration.

    Provides access to the model_slots table. Only the read path
    is needed by the vision service.
    """

    async def get_by_capability(self, capability: str) -> dict[str, Any] | None:
        """Fetch the model slot configuration for a given capability.

        Args:
            capability: The model capability identifier
                (e.g., "vision_processing").

        Returns:
            Dict with keys: provider, model_id, max_tokens_per_call,
            temperature, fallback_provider, fallback_model_id.
            None if no slot is configured for the capability.
        """
        ...


@runtime_checkable
class TokenUsageLogger(Protocol):
    """Logger for recording AI model token consumption.

    Writes to the token_usage_log table. Append-only.
    """

    async def log_usage(
        self,
        capability: str,
        provider: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        attribution: TokenAttribution | None = None,
    ) -> None:
        """Record a token usage event.

        Args:
            capability: The AI capability that consumed tokens.
            provider: Model provider used.
            model_id: Specific model used.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            attribution: Optional attribution context (user, document,
                session, message).
        """
        ...

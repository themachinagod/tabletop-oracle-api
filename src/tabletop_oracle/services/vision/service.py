"""V1 VisionService implementation using configured model slot.

Calls the configured vision_processing model slot to convert images
to text descriptions. Supports primary + fallback model, token usage
tracking, and graceful degradation when no model is configured.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

import httpx

from tabletop_oracle.services.vision.models import (
    ImageDescription,
    VisionProcessingError,
    VisionUnavailableError,
)

if TYPE_CHECKING:
    from tabletop_oracle.services.vision.models import TokenAttribution
    from tabletop_oracle.services.vision.protocols import (
        ModelSlotRepository,
        TokenUsageLogger,
    )

logger = logging.getLogger(__name__)

_CAPABILITY = "vision_processing"

_SUPPORTED_FORMATS = frozenset({"png", "jpeg", "jpg", "webp", "gif"})

_DEFAULT_PROMPT = (
    "Describe this image in detail, focusing on any text, diagrams, tables, "
    "or game-relevant content visible. Be thorough but concise."
)

_MAX_RETRIES = 2
_REQUEST_TIMEOUT_SECONDS = 60.0


class VisionService:
    """Shared vision processing service for image-to-text conversion.

    Uses the vision_processing model slot from the model_slots configuration
    table (PRD-007). Supports primary and fallback model providers.

    The service is designed as a shared singleton — both the ingestion
    pipeline and the AI workflow use the same service instance.

    Args:
        model_slot_repo: Repository for model slot configuration.
        token_usage_logger: Logger for tracking token consumption.
        http_client: Optional pre-configured httpx.AsyncClient for API calls.
    """

    def __init__(
        self,
        model_slot_repo: ModelSlotRepository,
        token_usage_logger: TokenUsageLogger,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_slot_repo = model_slot_repo
        self._token_usage_logger = token_usage_logger
        self._http_client = http_client
        self._owns_client = http_client is None

    async def describe_image(
        self,
        image_data: bytes,
        image_format: str,
        context: str | None = None,
        attribution: TokenAttribution | None = None,
    ) -> ImageDescription:
        """Process an image via the configured vision model.

        Attempts the primary model first, then falls back to the fallback
        model if configured. Logs token usage on success.

        Args:
            image_data: Raw image bytes.
            image_format: Image format (png, jpeg, webp, gif).
            context: Optional context to guide description.
            attribution: Token usage attribution for tracking.

        Returns:
            ImageDescription with text content and confidence metadata.

        Raises:
            VisionProcessingError: If both primary and fallback models fail.
            VisionUnavailableError: If no vision model slot is configured.
            ValueError: If image_data is empty or format is unsupported.
        """
        self._validate_input(image_data, image_format)

        slot = await self._get_model_slot()
        prompt = self._build_prompt(context)
        normalised_format = self._normalise_format(image_format)

        # Try primary model
        primary_error: Exception | None = None
        try:
            result = await self._call_model(
                provider=slot["provider"],
                model_id=slot["model_id"],
                image_data=image_data,
                image_format=normalised_format,
                prompt=prompt,
                max_tokens=slot.get("max_tokens_per_call"),
                temperature=slot.get("temperature"),
            )
            await self._log_token_usage(
                provider=slot["provider"],
                model_id=slot["model_id"],
                result=result,
                attribution=attribution,
            )
            return result
        except (httpx.HTTPError, VisionProcessingError) as exc:
            primary_error = exc
            logger.warning(
                "Primary vision model failed, attempting fallback",
                extra={
                    "provider": slot["provider"],
                    "model_id": slot["model_id"],
                    "error": str(exc),
                },
            )

        # Try fallback model if configured
        fallback_provider = slot.get("fallback_provider")
        fallback_model_id = slot.get("fallback_model_id")
        if fallback_provider and fallback_model_id:
            try:
                result = await self._call_model(
                    provider=fallback_provider,
                    model_id=fallback_model_id,
                    image_data=image_data,
                    image_format=normalised_format,
                    prompt=prompt,
                    max_tokens=slot.get("max_tokens_per_call"),
                    temperature=slot.get("temperature"),
                )
                await self._log_token_usage(
                    provider=fallback_provider,
                    model_id=fallback_model_id,
                    result=result,
                    attribution=attribution,
                )
                return result
            except (httpx.HTTPError, VisionProcessingError) as exc:
                logger.error(
                    "Fallback vision model also failed",
                    extra={
                        "provider": fallback_provider,
                        "model_id": fallback_model_id,
                        "error": str(exc),
                    },
                )

        raise VisionProcessingError(
            f"Vision processing failed: {primary_error}",
            provider=slot["provider"],
            model_id=slot["model_id"],
        )

    async def is_available(self) -> bool:
        """Check whether a vision model slot is configured.

        Returns:
            True if a vision_processing model slot exists.
        """
        slot = await self._model_slot_repo.get_by_capability(_CAPABILITY)
        return slot is not None

    async def _get_model_slot(self) -> dict[str, Any]:
        """Fetch the vision_processing model slot or raise.

        Returns:
            Model slot configuration dict.

        Raises:
            VisionUnavailableError: If no vision model slot is configured.
        """
        slot = await self._model_slot_repo.get_by_capability(_CAPABILITY)
        if slot is None:
            raise VisionUnavailableError()
        return slot

    async def _call_model(
        self,
        *,
        provider: str,
        model_id: str,
        image_data: bytes,
        image_format: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ImageDescription:
        """Call a vision model API and return the result.

        Currently supports OpenAI-compatible and Anthropic-compatible
        API formats. Retries up to _MAX_RETRIES times on transient errors.

        Args:
            provider: Model provider identifier.
            model_id: Provider-specific model identifier.
            image_data: Raw image bytes.
            image_format: Normalised image format.
            prompt: Text prompt for the vision model.
            max_tokens: Maximum tokens per call (optional).
            temperature: Sampling temperature (optional).

        Returns:
            ImageDescription with the model's response.

        Raises:
            VisionProcessingError: If the model call fails after retries.
        """
        image_b64 = base64.b64encode(image_data).decode("ascii")
        media_type = f"image/{image_format}"

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                if provider == "anthropic":
                    return await self._call_anthropic(
                        model_id=model_id,
                        image_b64=image_b64,
                        media_type=media_type,
                        prompt=prompt,
                        max_tokens=max_tokens or 1024,
                        temperature=temperature,
                    )
                else:
                    # OpenAI-compatible API (default)
                    return await self._call_openai_compatible(
                        model_id=model_id,
                        image_b64=image_b64,
                        media_type=media_type,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    break  # Non-retryable client error
                logger.warning(
                    "Vision API call failed (attempt %d/%d)",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    extra={"status": exc.response.status_code, "provider": provider},
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "Vision API call failed (attempt %d/%d)",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    extra={"error": str(exc), "provider": provider},
                )

        raise VisionProcessingError(
            f"Vision model call failed after {_MAX_RETRIES + 1} attempts: {last_error}",
            provider=provider,
            model_id=model_id,
        )

    async def _call_openai_compatible(
        self,
        *,
        model_id: str,
        image_b64: str,
        media_type: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ImageDescription:
        """Call an OpenAI-compatible vision API.

        Args:
            model_id: Model identifier (e.g., "gpt-4o").
            image_b64: Base64-encoded image data.
            media_type: MIME type of the image.
            prompt: Text prompt for the model.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            ImageDescription parsed from the API response.
        """
        client = self._get_http_client()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_openai_response(data)

    async def _call_anthropic(
        self,
        *,
        model_id: str,
        image_b64: str,
        media_type: str,
        prompt: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> ImageDescription:
        """Call the Anthropic Messages API with vision.

        Args:
            model_id: Model identifier (e.g., "claude-sonnet-4-20250514").
            image_b64: Base64-encoded image data.
            media_type: MIME type of the image.
            prompt: Text prompt for the model.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            ImageDescription parsed from the API response.
        """
        client = self._get_http_client()
        payload: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = await client.post(
            "/v1/messages",
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_anthropic_response(data)

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client for API calls.

        Returns:
            An httpx.AsyncClient instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
            self._owns_client = True
        return self._http_client

    async def _log_token_usage(
        self,
        *,
        provider: str,
        model_id: str,
        result: ImageDescription,
        attribution: TokenAttribution | None,
    ) -> None:
        """Log token consumption for a successful vision call.

        Failures in logging are caught and logged — they should not
        prevent the vision result from being returned.

        Args:
            provider: Model provider used.
            model_id: Model identifier used.
            result: The ImageDescription containing token counts.
            attribution: Optional attribution context.
        """
        try:
            await self._token_usage_logger.log_usage(
                capability=_CAPABILITY,
                provider=provider,
                model_id=model_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                attribution=attribution,
            )
        except Exception:
            logger.exception(
                "Failed to log token usage for vision call",
                extra={"provider": provider, "model_id": model_id},
            )

    @staticmethod
    def _validate_input(image_data: bytes, image_format: str) -> None:
        """Validate image data and format.

        Args:
            image_data: Raw image bytes.
            image_format: Image format string.

        Raises:
            ValueError: If image_data is empty or format is unsupported.
        """
        if not image_data:
            raise ValueError("image_data must not be empty")
        normalised = image_format.lower().strip().lstrip(".")
        if normalised not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format '{image_format}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"
            )

    @staticmethod
    def _normalise_format(image_format: str) -> str:
        """Normalise image format string to a standard MIME sub-type.

        Args:
            image_format: Raw format string (e.g., "jpg", ".PNG", "JPEG").

        Returns:
            Normalised format string (e.g., "jpeg", "png").
        """
        normalised = image_format.lower().strip().lstrip(".")
        if normalised == "jpg":
            return "jpeg"
        return normalised

    @staticmethod
    def _build_prompt(context: str | None) -> str:
        """Build the vision model prompt, incorporating optional context.

        Args:
            context: Optional context string to guide the model.

        Returns:
            Complete prompt string.
        """
        if context:
            return f"{_DEFAULT_PROMPT}\n\nContext: This image is from a {context}."
        return _DEFAULT_PROMPT

    @staticmethod
    def _parse_openai_response(data: dict[str, Any]) -> ImageDescription:
        """Parse an OpenAI-compatible API response into an ImageDescription.

        Args:
            data: Raw JSON response from the API.

        Returns:
            ImageDescription with extracted text and token counts.

        Raises:
            VisionProcessingError: If the response is malformed.
        """
        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionProcessingError(
                f"Malformed OpenAI response: {exc}",
                provider="openai",
            ) from exc

        return ImageDescription(
            text=text,
            confidence=_estimate_confidence(text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _parse_anthropic_response(data: dict[str, Any]) -> ImageDescription:
        """Parse an Anthropic Messages API response into an ImageDescription.

        Args:
            data: Raw JSON response from the API.

        Returns:
            ImageDescription with extracted text and token counts.

        Raises:
            VisionProcessingError: If the response is malformed.
        """
        try:
            content_blocks = data["content"]
            text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
            text = "\n".join(text_parts)
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionProcessingError(
                f"Malformed Anthropic response: {exc}",
                provider="anthropic",
            ) from exc

        return ImageDescription(
            text=text,
            confidence=_estimate_confidence(text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _estimate_confidence(text: str) -> str:
    """Estimate description confidence based on response length.

    A simple heuristic: longer, more detailed responses tend to indicate
    the model found meaningful content in the image.

    Args:
        text: The generated description text.

    Returns:
        Confidence level: "high", "medium", or "low".
    """
    word_count = len(text.split())
    if word_count >= 50:
        return "high"
    if word_count >= 15:
        return "medium"
    return "low"

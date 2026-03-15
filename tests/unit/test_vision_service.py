"""Unit tests for the shared VisionService.

Tests all service methods with mocked dependencies (ModelSlotRepository,
TokenUsageLogger, HTTP client). Covers happy paths, fallback model behavior,
graceful degradation, input validation, token tracking, and error cases.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tabletop_oracle.services.vision.models import (
    ImageDescription,
    TokenAttribution,
    VisionProcessingError,
    VisionUnavailableError,
)
from tabletop_oracle.services.vision.service import VisionService, _estimate_confidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
_SAMPLE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _make_slot(**overrides: Any) -> dict[str, Any]:
    """Build a model slot configuration dict with sensible defaults."""
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model_id": "gpt-4o",
        "max_tokens_per_call": 1024,
        "temperature": 0.2,
        "fallback_provider": None,
        "fallback_model_id": None,
    }
    defaults.update(overrides)
    return defaults


def _make_openai_response(
    text: str = "A game board with hexagonal tiles.",
    prompt_tokens: int = 500,
    completion_tokens: int = 50,
) -> dict[str, Any]:
    """Build a mock OpenAI-compatible API response."""
    return {
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _make_anthropic_response(
    text: str = "A game board with hexagonal tiles.",
    input_tokens: int = 500,
    output_tokens: int = 50,
) -> dict[str, Any]:
    """Build a mock Anthropic Messages API response."""
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _mock_model_slot_repo(slot: dict[str, Any] | None = None) -> AsyncMock:
    """Build a mock ModelSlotRepository."""
    repo = AsyncMock()
    repo.get_by_capability = AsyncMock(return_value=slot)
    return repo


def _mock_token_logger() -> AsyncMock:
    """Build a mock TokenUsageLogger."""
    logger = AsyncMock()
    logger.log_usage = AsyncMock()
    return logger


def _mock_http_client(response_json: dict[str, Any], status_code: int = 200) -> AsyncMock:
    """Build a mock httpx.AsyncClient that returns a fixed response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()

    if status_code >= 400:
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(spec=httpx.Request),
            response=mock_response,
        )

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)
    return client


# ---------------------------------------------------------------------------
# describe_image — happy path
# ---------------------------------------------------------------------------


class TestDescribeImageHappyPath:
    """Tests for successful vision processing."""

    @pytest.mark.asyncio
    async def test_describe_image_openai_returns_description(self) -> None:
        """OpenAI-compatible model returns a valid ImageDescription."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_openai_response())

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert isinstance(result, ImageDescription)
        assert result.text == "A game board with hexagonal tiles."
        assert result.input_tokens == 500
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_describe_image_anthropic_returns_description(self) -> None:
        """Anthropic provider returns a valid ImageDescription."""
        slot = _make_slot(provider="anthropic", model_id="claude-sonnet-4-20250514")
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_anthropic_response())

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert isinstance(result, ImageDescription)
        assert result.text == "A game board with hexagonal tiles."
        assert result.input_tokens == 500
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_describe_image_with_context_modifies_prompt(self) -> None:
        """Context parameter is incorporated into the prompt sent to the model."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_openai_response())

        svc = VisionService(repo, logger, http_client=client)
        await svc.describe_image(_SAMPLE_PNG, "png", context="game board diagram")

        call_args = client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        prompt_text = payload["messages"][0]["content"][1]["text"]
        assert "game board diagram" in prompt_text

    @pytest.mark.asyncio
    async def test_describe_image_accepts_various_formats(self) -> None:
        """All supported formats (png, jpeg, jpg, webp, gif) are accepted."""
        for fmt in ("png", "jpeg", "jpg", "webp", "gif"):
            slot = _make_slot()
            repo = _mock_model_slot_repo(slot)
            logger = _mock_token_logger()
            client = _mock_http_client(_make_openai_response())

            svc = VisionService(repo, logger, http_client=client)
            result = await svc.describe_image(_SAMPLE_PNG, fmt)
            assert isinstance(result, ImageDescription)


# ---------------------------------------------------------------------------
# describe_image — token tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Tests for token usage logging."""

    @pytest.mark.asyncio
    async def test_describe_image_logs_token_usage(self) -> None:
        """Token usage is logged after a successful call."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_openai_response(prompt_tokens=200, completion_tokens=30))

        svc = VisionService(repo, logger, http_client=client)
        await svc.describe_image(_SAMPLE_PNG, "png")

        logger.log_usage.assert_called_once_with(
            capability="vision_processing",
            provider="openai",
            model_id="gpt-4o",
            input_tokens=200,
            output_tokens=30,
            attribution=None,
        )

    @pytest.mark.asyncio
    async def test_describe_image_logs_with_attribution(self) -> None:
        """Token usage is logged with the provided attribution context."""
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()
        attribution = TokenAttribution(user_id=user_id, document_id=doc_id)

        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_openai_response())

        svc = VisionService(repo, logger, http_client=client)
        await svc.describe_image(_SAMPLE_PNG, "png", attribution=attribution)

        call_kwargs = logger.log_usage.call_args.kwargs
        assert call_kwargs["attribution"] is attribution

    @pytest.mark.asyncio
    async def test_describe_image_succeeds_when_logging_fails(self) -> None:
        """Vision result is returned even if token logging raises."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        logger.log_usage.side_effect = RuntimeError("DB connection lost")
        client = _mock_http_client(_make_openai_response())

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert result.text == "A game board with hexagonal tiles."


# ---------------------------------------------------------------------------
# describe_image — fallback model
# ---------------------------------------------------------------------------


class TestFallbackModel:
    """Tests for fallback model behavior when primary fails."""

    @pytest.mark.asyncio
    async def test_describe_image_falls_back_on_primary_failure(self) -> None:
        """Fallback model is used when primary model returns a server error."""
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-sonnet-4-20250514",
        )
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()

        # Primary fails, fallback succeeds
        primary_response = MagicMock(spec=httpx.Response)
        primary_response.status_code = 500
        primary_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Internal Server Error",
            request=MagicMock(spec=httpx.Request),
            response=primary_response,
        )

        fallback_response = MagicMock(spec=httpx.Response)
        fallback_response.status_code = 200
        fallback_response.json.return_value = _make_anthropic_response(text="Fallback description")
        fallback_response.raise_for_status = MagicMock()

        client = AsyncMock(spec=httpx.AsyncClient)
        # Primary retries (_MAX_RETRIES + 1 = 3 calls), then fallback
        client.post = AsyncMock(
            side_effect=[
                primary_response,
                primary_response,
                primary_response,
                fallback_response,
            ]
        )

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert result.text == "Fallback description"

    @pytest.mark.asyncio
    async def test_describe_image_raises_when_both_models_fail(self) -> None:
        """VisionProcessingError raised when both primary and fallback fail."""
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-sonnet-4-20250514",
        )
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Internal Server Error",
            request=MagicMock(spec=httpx.Request),
            response=error_response,
        )

        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=error_response)

        svc = VisionService(repo, logger, http_client=client)

        with pytest.raises(VisionProcessingError) as exc_info:
            await svc.describe_image(_SAMPLE_PNG, "png")

        assert exc_info.value.provider == "openai"
        assert exc_info.value.model_id == "gpt-4o"


# ---------------------------------------------------------------------------
# describe_image — graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for graceful degradation when vision is unavailable."""

    @pytest.mark.asyncio
    async def test_describe_image_raises_when_no_slot_configured(self) -> None:
        """VisionUnavailableError raised when no model slot exists."""
        repo = _mock_model_slot_repo(None)
        logger = _mock_token_logger()

        svc = VisionService(repo, logger)

        with pytest.raises(VisionUnavailableError):
            await svc.describe_image(_SAMPLE_PNG, "png")

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_configured(self) -> None:
        """is_available returns True when a vision slot exists."""
        repo = _mock_model_slot_repo(_make_slot())
        logger = _mock_token_logger()

        svc = VisionService(repo, logger)
        assert await svc.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_configured(self) -> None:
        """is_available returns False when no vision slot exists."""
        repo = _mock_model_slot_repo(None)
        logger = _mock_token_logger()

        svc = VisionService(repo, logger)
        assert await svc.is_available() is False


# ---------------------------------------------------------------------------
# describe_image — input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_describe_image_rejects_empty_data(self) -> None:
        """ValueError raised for empty image_data."""
        repo = _mock_model_slot_repo(_make_slot())
        logger = _mock_token_logger()

        svc = VisionService(repo, logger)

        with pytest.raises(ValueError, match="must not be empty"):
            await svc.describe_image(b"", "png")

    @pytest.mark.asyncio
    async def test_describe_image_rejects_unsupported_format(self) -> None:
        """ValueError raised for unsupported image format."""
        repo = _mock_model_slot_repo(_make_slot())
        logger = _mock_token_logger()

        svc = VisionService(repo, logger)

        with pytest.raises(ValueError, match="Unsupported image format"):
            await svc.describe_image(_SAMPLE_PNG, "bmp")

    @pytest.mark.asyncio
    async def test_describe_image_normalises_format_case(self) -> None:
        """Uppercase and dotted formats are normalised correctly."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_openai_response())

        svc = VisionService(repo, logger, http_client=client)

        # ".PNG" should be normalised to "png"
        result = await svc.describe_image(_SAMPLE_PNG, ".PNG")
        assert isinstance(result, ImageDescription)

    @pytest.mark.asyncio
    async def test_describe_image_normalises_jpg_to_jpeg(self) -> None:
        """'jpg' is normalised to 'jpeg' for MIME type consistency."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client(_make_openai_response())

        svc = VisionService(repo, logger, http_client=client)
        await svc.describe_image(_SAMPLE_JPEG, "jpg")

        call_args = client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        assert "image/jpeg" in image_url


# ---------------------------------------------------------------------------
# describe_image — retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    """Tests for retry logic on transient errors."""

    @pytest.mark.asyncio
    async def test_describe_image_retries_on_server_error(self) -> None:
        """Retries on 500 errors and succeeds on later attempt."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Server Error",
            request=MagicMock(spec=httpx.Request),
            response=error_response,
        )

        success_response = MagicMock(spec=httpx.Response)
        success_response.status_code = 200
        success_response.json.return_value = _make_openai_response()
        success_response.raise_for_status = MagicMock()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(side_effect=[error_response, success_response])

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert result.text == "A game board with hexagonal tiles."
        assert client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_describe_image_no_retry_on_client_error(self) -> None:
        """Does not retry on 4xx client errors (except 429)."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 400
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Bad Request",
            request=MagicMock(spec=httpx.Request),
            response=error_response,
        )

        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=error_response)

        svc = VisionService(repo, logger, http_client=client)

        with pytest.raises(VisionProcessingError):
            await svc.describe_image(_SAMPLE_PNG, "png")

        # Only 1 call — no retries on client errors
        assert client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_describe_image_retries_on_rate_limit(self) -> None:
        """429 rate limit errors are retried."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()

        rate_limit_response = MagicMock(spec=httpx.Response)
        rate_limit_response.status_code = 429
        rate_limit_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Rate Limited",
            request=MagicMock(spec=httpx.Request),
            response=rate_limit_response,
        )

        success_response = MagicMock(spec=httpx.Response)
        success_response.status_code = 200
        success_response.json.return_value = _make_openai_response()
        success_response.raise_for_status = MagicMock()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(side_effect=[rate_limit_response, success_response])

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert result.text == "A game board with hexagonal tiles."
        assert client.post.call_count == 2


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Tests for API response parsing."""

    @pytest.mark.asyncio
    async def test_parse_openai_malformed_response_raises(self) -> None:
        """Malformed OpenAI response raises VisionProcessingError."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client({"choices": []})  # Empty choices

        svc = VisionService(repo, logger, http_client=client)

        with pytest.raises(VisionProcessingError, match="Malformed"):
            await svc.describe_image(_SAMPLE_PNG, "png")

    @pytest.mark.asyncio
    async def test_parse_anthropic_malformed_response_raises(self) -> None:
        """Malformed Anthropic response raises VisionProcessingError."""
        slot = _make_slot(provider="anthropic", model_id="claude-sonnet-4-20250514")
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        client = _mock_http_client({})  # Missing 'content' key

        svc = VisionService(repo, logger, http_client=client)

        with pytest.raises(VisionProcessingError, match="Malformed"):
            await svc.describe_image(_SAMPLE_PNG, "png")

    @pytest.mark.asyncio
    async def test_parse_openai_missing_usage_defaults_to_zero(self) -> None:
        """Missing usage field defaults token counts to zero."""
        slot = _make_slot()
        repo = _mock_model_slot_repo(slot)
        logger = _mock_token_logger()
        response = {"choices": [{"message": {"content": "A board."}}]}
        client = _mock_http_client(response)

        svc = VisionService(repo, logger, http_client=client)
        result = await svc.describe_image(_SAMPLE_PNG, "png")

        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------


class TestConfidenceEstimation:
    """Tests for the confidence heuristic."""

    def test_estimate_confidence_high_for_long_text(self) -> None:
        """50+ word responses are high confidence."""
        text = " ".join(["word"] * 60)
        assert _estimate_confidence(text) == "high"

    def test_estimate_confidence_medium_for_moderate_text(self) -> None:
        """15-49 word responses are medium confidence."""
        text = " ".join(["word"] * 25)
        assert _estimate_confidence(text) == "medium"

    def test_estimate_confidence_low_for_short_text(self) -> None:
        """Under 15 word responses are low confidence."""
        text = " ".join(["word"] * 10)
        assert _estimate_confidence(text) == "low"

    def test_estimate_confidence_low_for_empty_text(self) -> None:
        """Empty text is low confidence."""
        assert _estimate_confidence("") == "low"


# ---------------------------------------------------------------------------
# Model data classes
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for vision data models."""

    def test_image_description_is_frozen(self) -> None:
        """ImageDescription instances are immutable."""
        desc = ImageDescription(text="test", confidence="high")
        with pytest.raises(AttributeError):
            desc.text = "modified"  # type: ignore[misc]

    def test_token_attribution_defaults_to_none(self) -> None:
        """TokenAttribution fields default to None."""
        attr = TokenAttribution()
        assert attr.user_id is None
        assert attr.document_id is None
        assert attr.session_id is None
        assert attr.message_id is None

    def test_token_attribution_accepts_uuids(self) -> None:
        """TokenAttribution accepts UUID values."""
        uid = uuid.uuid4()
        did = uuid.uuid4()
        attr = TokenAttribution(user_id=uid, document_id=did)
        assert attr.user_id == uid
        assert attr.document_id == did

    def test_vision_processing_error_stores_provider_info(self) -> None:
        """VisionProcessingError stores provider and model_id."""
        err = VisionProcessingError("failed", provider="openai", model_id="gpt-4o")
        assert err.provider == "openai"
        assert err.model_id == "gpt-4o"
        assert str(err) == "failed"

    def test_vision_unavailable_error_default_message(self) -> None:
        """VisionUnavailableError has a sensible default message."""
        err = VisionUnavailableError()
        assert "not configured" in str(err)

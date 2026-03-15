"""Unit tests for ModelClient.

Tests the provider-agnostic LLM client with mocked litellm calls.
Covers completion, streaming, retry, fallback, guardrail enforcement,
token tracking, and cancellation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import litellm
import pytest

from tabletop_oracle.errors.exceptions import (
    GuardrailExceededError,
    ModelUnavailableError,
)
from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.model.client import ModelClient
from tabletop_oracle.services.model.slot_config import ModelSlotConfig
from tabletop_oracle.services.model.token_usage import TokenAttribution
from tabletop_oracle.services.model.types import CompletionResult, GuardrailCheckResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]


def _make_slot(
    *,
    capability: ModelCapability = ModelCapability.INTENT_ANALYSIS,
    provider: str = "openai",
    model_id: str = "gpt-4o",
    max_tokens_per_call: int | None = 4096,
    temperature: float | None = 0.7,
    fallback_provider: str | None = None,
    fallback_model_id: str | None = None,
) -> ModelSlotConfig:
    """Build a ModelSlotConfig for testing."""
    return ModelSlotConfig(
        capability=capability,
        provider=provider,
        model_id=model_id,
        max_tokens_per_call=max_tokens_per_call,
        temperature=temperature,
        fallback_provider=fallback_provider,
        fallback_model_id=fallback_model_id,
    )


def _make_attribution(
    *,
    with_session: bool = True,
) -> TokenAttribution:
    """Build a TokenAttribution for testing."""
    return TokenAttribution(
        user_id=uuid4(),
        session_id=uuid4() if with_session else None,
        message_id=uuid4() if with_session else None,
    )


def _make_litellm_response(
    *,
    content: str = "Hello world",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock litellm completion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_stream_chunks(
    tokens: list[str],
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> list[MagicMock]:
    """Build mock streaming chunks with usage on the last chunk."""
    chunks = []
    for i, token in enumerate(tokens):
        delta = MagicMock()
        delta.content = token

        choice = MagicMock()
        choice.delta = delta

        chunk = MagicMock()
        chunk.choices = [choice]

        # Usage reported on last chunk
        if i == len(tokens) - 1:
            usage = MagicMock()
            usage.prompt_tokens = prompt_tokens
            usage.completion_tokens = completion_tokens
            chunk.usage = usage
        else:
            chunk.usage = None

        chunks.append(chunk)
    return chunks


async def _async_iter(items: list[Any]) -> Any:
    """Create an async iterable from a list."""
    for item in items:
        yield item


def _make_guardrail_allowed() -> AsyncMock:
    """Build a mock GuardrailService that always allows."""
    service = AsyncMock()
    service.check_pre_call = AsyncMock(return_value=GuardrailCheckResult(allowed=True))
    return service


def _make_guardrail_denied(
    message: str = "Limit exceeded",
    guardrail_type: str = "per_query_tokens",
) -> AsyncMock:
    """Build a mock GuardrailService that always denies."""
    service = AsyncMock()
    service.check_pre_call = AsyncMock(
        return_value=GuardrailCheckResult(
            allowed=False,
            message=message,
            guardrail_type=guardrail_type,
        )
    )
    return service


def _make_token_usage_service() -> AsyncMock:
    """Build a mock TokenUsageService."""
    service = AsyncMock()
    service.record = AsyncMock()
    return service


def _make_slot_service(slot: ModelSlotConfig) -> AsyncMock:
    """Build a mock ModelSlotService."""
    service = AsyncMock()
    service.get_slot = AsyncMock(return_value=slot)
    return service


def _make_client(
    *,
    slot: ModelSlotConfig | None = None,
    guardrail_service: AsyncMock | None = None,
    token_usage_service: AsyncMock | None = None,
) -> tuple[ModelClient, AsyncMock, AsyncMock, AsyncMock]:
    """Build a ModelClient with mock dependencies.

    Returns:
        Tuple of (client, slot_service, token_usage_service, guardrail_service).
    """
    if slot is None:
        slot = _make_slot()
    slot_svc = _make_slot_service(slot)
    token_svc = token_usage_service or _make_token_usage_service()
    guard_svc = guardrail_service or _make_guardrail_allowed()
    client = ModelClient(
        slot_service=slot_svc,
        token_usage_service=token_svc,
        guardrail_service=guard_svc,
    )
    return client, slot_svc, token_svc, guard_svc


# ---------------------------------------------------------------------------
# complete() — happy path
# ---------------------------------------------------------------------------


class TestCompleteHappyPath:
    """Tests that complete() returns correct CompletionResult."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_returns_completion_result(self, mock_acompletion: AsyncMock) -> None:
        """Successful complete() returns a CompletionResult with correct fields."""
        mock_acompletion.return_value = _make_litellm_response(
            content="Test answer",
            prompt_tokens=15,
            completion_tokens=8,
        )
        client, _, _, _ = _make_client()
        attribution = _make_attribution()

        result = await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            attribution,
        )

        assert isinstance(result, CompletionResult)
        assert result.content == "Test answer"
        assert result.input_tokens == 15
        assert result.output_tokens == 8
        assert result.model_used == "gpt-4o"
        assert result.is_fallback is False

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_passes_model_parameters(self, mock_acompletion: AsyncMock) -> None:
        """complete() passes model, messages, max_tokens, and temperature to litellm."""
        mock_acompletion.return_value = _make_litellm_response()
        slot = _make_slot(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens_per_call=8192,
            temperature=0.3,
        )
        client, _, _, _ = _make_client(slot=slot)

        await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(),
        )

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
        assert call_kwargs["messages"] == _MESSAGES
        assert call_kwargs["max_tokens"] == 8192
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_openai_model_id_format(self, mock_acompletion: AsyncMock) -> None:
        """OpenAI models use model_id directly (no provider/ prefix)."""
        mock_acompletion.return_value = _make_litellm_response()
        slot = _make_slot(provider="openai", model_id="gpt-4o")
        client, _, _, _ = _make_client(slot=slot)

        await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(),
        )

        assert mock_acompletion.call_args.kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# complete() — token usage tracking
# ---------------------------------------------------------------------------


class TestCompleteTokenTracking:
    """Tests that complete() records token usage after success."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_records_token_usage(self, mock_acompletion: AsyncMock) -> None:
        """Token usage is recorded after a successful completion."""
        mock_acompletion.return_value = _make_litellm_response(
            prompt_tokens=20, completion_tokens=10
        )
        client, _, token_svc, _ = _make_client()
        attribution = _make_attribution()

        await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            attribution,
        )

        token_svc.record.assert_awaited_once()
        call_kwargs = token_svc.record.call_args.kwargs
        assert call_kwargs["capability"] == ModelCapability.INTENT_ANALYSIS
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model_id"] == "gpt-4o"
        assert call_kwargs["input_tokens"] == 20
        assert call_kwargs["output_tokens"] == 10
        assert call_kwargs["attribution"] is attribution

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_token_tracking_failure_does_not_break_call(
        self, mock_acompletion: AsyncMock
    ) -> None:
        """If token tracking fails, the completion result is still returned."""
        mock_acompletion.return_value = _make_litellm_response()
        token_svc = _make_token_usage_service()
        token_svc.record.side_effect = RuntimeError("DB write failed")
        client, _, _, _ = _make_client(token_usage_service=token_svc)

        result = await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(),
        )

        assert result.content == "Hello world"


# ---------------------------------------------------------------------------
# complete() — retry on transient failures
# ---------------------------------------------------------------------------


class TestCompleteRetry:
    """Tests retry behaviour for transient litellm errors."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_retries_on_transient_error(
        self, mock_acompletion: AsyncMock, mock_sleep: AsyncMock
    ) -> None:
        """Transient failures trigger retries with backoff."""
        mock_acompletion.side_effect = [
            litellm.Timeout("timeout", "model", "provider"),
            litellm.Timeout("timeout", "model", "provider"),
            _make_litellm_response(content="Recovered"),
        ]
        client, _, _, _ = _make_client()

        result = await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(),
        )

        assert result.content == "Recovered"
        assert mock_acompletion.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_await(1.0)
        mock_sleep.assert_any_await(2.0)

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_exhausted_retries_triggers_fallback(
        self, mock_acompletion: AsyncMock, mock_sleep: AsyncMock
    ) -> None:
        """After exhausting retries, falls back to the fallback model."""
        timeout = litellm.Timeout("timeout", "model", "provider")
        mock_acompletion.side_effect = [
            timeout,
            timeout,
            timeout,
            _make_litellm_response(content="Fallback answer"),
        ]
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-haiku",
        )
        client, _, _, _ = _make_client(slot=slot)

        result = await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(),
        )

        assert result.content == "Fallback answer"
        assert result.is_fallback is True
        assert result.model_used == "anthropic/claude-haiku"


# ---------------------------------------------------------------------------
# complete() — persistent failure triggers fallback
# ---------------------------------------------------------------------------


class TestCompletePersistentFailure:
    """Tests that persistent errors skip retries and go to fallback."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_auth_error_skips_retries_uses_fallback(
        self, mock_acompletion: AsyncMock
    ) -> None:
        """Authentication errors immediately trigger fallback without retry."""
        mock_acompletion.side_effect = [
            litellm.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-4o"),
            _make_litellm_response(content="Fallback"),
        ]
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-haiku",
        )
        client, _, _, _ = _make_client(slot=slot)

        result = await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(),
        )

        assert result.content == "Fallback"
        assert result.is_fallback is True
        # Only 2 calls: 1 primary (auth error) + 1 fallback
        assert mock_acompletion.call_count == 2


# ---------------------------------------------------------------------------
# complete() — fallback failure raises ModelUnavailableError
# ---------------------------------------------------------------------------


class TestCompleteFallbackFailure:
    """Tests ModelUnavailableError when both primary and fallback fail."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_both_models_fail_raises_model_unavailable(
        self, mock_acompletion: AsyncMock, mock_sleep: AsyncMock
    ) -> None:
        """ModelUnavailableError raised when both primary and fallback fail."""
        timeout = litellm.Timeout("timeout", "model", "provider")
        mock_acompletion.side_effect = [timeout, timeout, timeout, timeout]
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-haiku",
        )
        client, _, _, _ = _make_client(slot=slot)

        with pytest.raises(ModelUnavailableError):
            await client.complete(
                ModelCapability.INTENT_ANALYSIS,
                _MESSAGES,
                _make_attribution(),
            )

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_no_fallback_configured_raises_model_unavailable(
        self, mock_acompletion: AsyncMock, mock_sleep: AsyncMock
    ) -> None:
        """ModelUnavailableError raised when primary fails and no fallback exists."""
        timeout = litellm.Timeout("timeout", "model", "provider")
        mock_acompletion.side_effect = [timeout, timeout, timeout]
        slot = _make_slot(fallback_provider=None, fallback_model_id=None)
        client, _, _, _ = _make_client(slot=slot)

        with pytest.raises(ModelUnavailableError, match="no fallback configured"):
            await client.complete(
                ModelCapability.INTENT_ANALYSIS,
                _MESSAGES,
                _make_attribution(),
            )


# ---------------------------------------------------------------------------
# complete() — guardrail enforcement
# ---------------------------------------------------------------------------


class TestCompleteGuardrails:
    """Tests guardrail checking before completion calls."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_guardrail_check_called_before_completion(
        self, mock_acompletion: AsyncMock
    ) -> None:
        """Guardrail check is performed before the litellm call."""
        mock_acompletion.return_value = _make_litellm_response()
        guard_svc = _make_guardrail_allowed()
        client, _, _, _ = _make_client(guardrail_service=guard_svc)
        attribution = _make_attribution()

        await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            attribution,
        )

        guard_svc.check_pre_call.assert_awaited_once_with(
            session_id=attribution.session_id,
            message_id=attribution.message_id,
        )

    @pytest.mark.asyncio
    async def test_guardrail_denied_raises_error(self) -> None:
        """GuardrailExceededError raised when guardrails deny the call."""
        guard_svc = _make_guardrail_denied(
            message="Token limit hit",
            guardrail_type="per_query_tokens",
        )
        client, _, _, _ = _make_client(guardrail_service=guard_svc)

        with pytest.raises(GuardrailExceededError, match="Token limit hit") as exc_info:
            await client.complete(
                ModelCapability.INTENT_ANALYSIS,
                _MESSAGES,
                _make_attribution(),
            )

        assert exc_info.value.guardrail_type == "per_query_tokens"

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_guardrail_skipped_without_session(self, mock_acompletion: AsyncMock) -> None:
        """Guardrail check is skipped when no session_id (ingestion context)."""
        mock_acompletion.return_value = _make_litellm_response()
        guard_svc = _make_guardrail_allowed()
        client, _, _, _ = _make_client(guardrail_service=guard_svc)

        await client.complete(
            ModelCapability.INTENT_ANALYSIS,
            _MESSAGES,
            _make_attribution(with_session=False),
        )

        guard_svc.check_pre_call.assert_not_awaited()


# ---------------------------------------------------------------------------
# stream() — happy path
# ---------------------------------------------------------------------------


class TestStreamHappyPath:
    """Tests that stream() yields tokens correctly."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_yields_tokens(self, mock_acompletion: AsyncMock) -> None:
        """Successful stream() yields individual tokens."""
        chunks = _make_stream_chunks(["Hello", " ", "world"])
        mock_acompletion.return_value = _async_iter(chunks)
        client, _, _, _ = _make_client()

        tokens: list[str] = []
        async for token in client.stream(
            ModelCapability.ANSWER_SYNTHESIS,
            _MESSAGES,
            _make_attribution(),
        ):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_tracks_token_usage_after_stream(self, mock_acompletion: AsyncMock) -> None:
        """Token usage is recorded after streaming completes."""
        chunks = _make_stream_chunks(["Hi"], prompt_tokens=12, completion_tokens=1)
        mock_acompletion.return_value = _async_iter(chunks)
        client, _, token_svc, _ = _make_client()
        attribution = _make_attribution()

        async for _ in client.stream(
            ModelCapability.ANSWER_SYNTHESIS,
            _MESSAGES,
            attribution,
        ):
            pass

        token_svc.record.assert_awaited_once()
        call_kwargs = token_svc.record.call_args.kwargs
        assert call_kwargs["input_tokens"] == 12
        assert call_kwargs["output_tokens"] == 1


# ---------------------------------------------------------------------------
# stream() — cancellation
# ---------------------------------------------------------------------------


class TestStreamCancellation:
    """Tests cancellation support during streaming."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_cancellation_stops_stream(self, mock_acompletion: AsyncMock) -> None:
        """Stream stops when cancel_check returns True."""
        chunks = _make_stream_chunks(["a", "b", "c", "d"])
        mock_acompletion.return_value = _async_iter(chunks)
        client, _, _, _ = _make_client()

        call_count = 0

        async def cancel_after_two() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 2

        tokens: list[str] = []
        async for token in client.stream(
            ModelCapability.ANSWER_SYNTHESIS,
            _MESSAGES,
            _make_attribution(),
            cancel_check=cancel_after_two,
        ):
            tokens.append(token)

        # Should get first 2 tokens before cancellation kicks in
        assert len(tokens) <= 2


# ---------------------------------------------------------------------------
# stream() — fallback
# ---------------------------------------------------------------------------


class TestStreamFallback:
    """Tests fallback behaviour during streaming."""

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_stream_falls_back_on_failure(
        self, mock_acompletion: AsyncMock, mock_sleep: AsyncMock
    ) -> None:
        """Stream falls back to secondary model when primary fails."""
        timeout = litellm.Timeout("timeout", "model", "provider")
        fallback_chunks = _make_stream_chunks(["fallback", " ", "answer"])
        mock_acompletion.side_effect = [
            timeout,
            timeout,
            timeout,
            _async_iter(fallback_chunks),
        ]
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-haiku",
        )
        client, _, _, _ = _make_client(slot=slot)

        tokens: list[str] = []
        async for token in client.stream(
            ModelCapability.ANSWER_SYNTHESIS,
            _MESSAGES,
            _make_attribution(),
        ):
            tokens.append(token)

        assert tokens == ["fallback", " ", "answer"]

    @pytest.mark.asyncio
    @patch("tabletop_oracle.services.model.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("tabletop_oracle.services.model.client.litellm.acompletion")
    async def test_stream_raises_when_both_fail(
        self, mock_acompletion: AsyncMock, mock_sleep: AsyncMock
    ) -> None:
        """ModelUnavailableError raised when both models fail during streaming."""
        timeout = litellm.Timeout("timeout", "model", "provider")
        mock_acompletion.side_effect = [timeout, timeout, timeout, timeout]
        slot = _make_slot(
            fallback_provider="anthropic",
            fallback_model_id="claude-haiku",
        )
        client, _, _, _ = _make_client(slot=slot)

        with pytest.raises(ModelUnavailableError):
            async for _ in client.stream(
                ModelCapability.ANSWER_SYNTHESIS,
                _MESSAGES,
                _make_attribution(),
            ):
                pass


# ---------------------------------------------------------------------------
# stream() — guardrails
# ---------------------------------------------------------------------------


class TestStreamGuardrails:
    """Tests guardrail enforcement before streaming."""

    @pytest.mark.asyncio
    async def test_guardrail_denied_before_stream(self) -> None:
        """GuardrailExceededError raised before streaming begins."""
        guard_svc = _make_guardrail_denied()
        client, _, _, _ = _make_client(guardrail_service=guard_svc)

        with pytest.raises(GuardrailExceededError):
            async for _ in client.stream(
                ModelCapability.ANSWER_SYNTHESIS,
                _MESSAGES,
                _make_attribution(),
            ):
                pass


# ---------------------------------------------------------------------------
# CompletionResult — fields
# ---------------------------------------------------------------------------


class TestCompletionResultFields:
    """Tests CompletionResult dataclass field access and immutability."""

    def test_fields_accessible(self) -> None:
        """All CompletionResult fields are accessible."""
        result = CompletionResult(
            content="answer",
            input_tokens=10,
            output_tokens=5,
            model_used="gpt-4o",
            is_fallback=False,
        )

        assert result.content == "answer"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model_used == "gpt-4o"
        assert result.is_fallback is False

    def test_frozen_immutability(self) -> None:
        """CompletionResult is frozen and cannot be modified."""
        result = CompletionResult(
            content="answer",
            input_tokens=10,
            output_tokens=5,
            model_used="gpt-4o",
            is_fallback=False,
        )

        with pytest.raises(AttributeError):
            result.content = "modified"  # type: ignore[misc]

"""Provider-agnostic LLM client using litellm.

Wraps litellm for unified access to OpenAI, Anthropic, and other
providers. Handles model slot lookup, retry with exponential backoff,
fallback model activation, token usage tracking, guardrail enforcement,
streaming, and cancellation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import litellm
import litellm.exceptions as litellm_exc

from tabletop_oracle.errors.exceptions import (
    GuardrailExceededError,
    ModelUnavailableError,
)
from tabletop_oracle.services.model.types import CompletionResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from tabletop_oracle.models.enums import ModelCapability
    from tabletop_oracle.services.model.guardrail import GuardrailServiceProtocol
    from tabletop_oracle.services.model.slot_config import ModelSlotConfig
    from tabletop_oracle.services.model.slot_service import ModelSlotService
    from tabletop_oracle.services.model.token_usage import (
        TokenAttribution,
        TokenUsageServiceProtocol,
    )

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

# litellm exception types that indicate transient failures worth retrying.
_TRANSIENT_ERRORS = (
    litellm_exc.Timeout,
    litellm_exc.RateLimitError,
    litellm_exc.ServiceUnavailableError,
    litellm_exc.InternalServerError,
    litellm_exc.APIConnectionError,
)

# litellm exception types that indicate persistent failures (skip retries).
_PERSISTENT_ERRORS = (
    litellm_exc.AuthenticationError,
    litellm_exc.NotFoundError,
    litellm_exc.BadRequestError,
)


class ModelClient:
    """Provider-agnostic LLM client.

    Wraps litellm for unified access to OpenAI, Anthropic, and other
    providers. Handles model slot lookup, fallback activation, token usage
    tracking, guardrail enforcement, retry with exponential backoff,
    streaming, and cancellation.

    Args:
        slot_service: Model slot configuration service.
        token_usage_service: Token consumption tracking service.
        guardrail_service: Pre-call guardrail enforcement service.
    """

    def __init__(
        self,
        slot_service: ModelSlotService,
        token_usage_service: TokenUsageServiceProtocol,
        guardrail_service: GuardrailServiceProtocol,
    ) -> None:
        self._slot_service = slot_service
        self._token_usage_service = token_usage_service
        self._guardrail_service = guardrail_service

    async def complete(
        self,
        capability: ModelCapability,
        messages: list[dict[str, Any]],
        attribution: TokenAttribution,
        response_format: type[Any] | None = None,
    ) -> CompletionResult:
        """Non-streaming completion.

        Looks up the model slot, checks guardrails, attempts the primary
        model with retries, falls back if needed, and tracks token usage.

        Args:
            capability: Which model slot to use.
            messages: Chat messages in OpenAI format.
            attribution: Token usage attribution context.
            response_format: Optional structured output type for the response.

        Returns:
            CompletionResult with content, token counts, and model used.

        Raises:
            GuardrailExceededError: If token/call guardrails would be exceeded.
            ModelUnavailableError: If both primary and fallback fail.
        """
        slot = await self._slot_service.get_slot(capability)
        await self._check_guardrails(attribution)

        # Try primary model with retries
        try:
            return await self._attempt_completion(
                model=self._litellm_model_id(slot.provider, slot.model_id),
                messages=messages,
                slot=slot,
                attribution=attribution,
                capability=capability,
                is_fallback=False,
                response_format=response_format,
            )
        except (ModelUnavailableError, *_PERSISTENT_ERRORS):
            pass

        # Try fallback model
        return await self._attempt_fallback_completion(
            slot=slot,
            messages=messages,
            attribution=attribution,
            capability=capability,
            response_format=response_format,
        )

    async def stream(
        self,
        capability: ModelCapability,
        messages: list[dict[str, Any]],
        attribution: TokenAttribution,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion yielding tokens.

        Looks up the model slot, checks guardrails, attempts the primary
        model with retries, falls back if needed, and tracks token usage
        after the stream completes.

        Args:
            capability: Which model slot to use.
            messages: Chat messages in OpenAI format.
            attribution: Token usage attribution context.
            cancel_check: Optional async callable returning True to abort.

        Yields:
            Individual tokens as strings.

        Raises:
            GuardrailExceededError: If guardrails exceeded before streaming starts.
            ModelUnavailableError: If both primary and fallback fail.
        """
        slot = await self._slot_service.get_slot(capability)
        await self._check_guardrails(attribution)

        # Try primary model
        primary_error: Exception | None = None
        try:
            async for token in self._attempt_stream(
                model=self._litellm_model_id(slot.provider, slot.model_id),
                messages=messages,
                slot=slot,
                attribution=attribution,
                capability=capability,
                is_fallback=False,
                cancel_check=cancel_check,
            ):
                yield token
            return
        except (ModelUnavailableError, *_TRANSIENT_ERRORS, *_PERSISTENT_ERRORS) as exc:
            primary_error = exc
            logger.warning(
                "Primary model streaming failed, attempting fallback",
                extra={
                    "provider": slot.provider,
                    "model_id": slot.model_id,
                    "error": str(exc),
                },
            )

        # Try fallback model
        if slot.fallback_provider and slot.fallback_model_id:
            fallback_model = self._litellm_model_id(slot.fallback_provider, slot.fallback_model_id)
            try:
                async for token in self._stream_single_attempt(
                    model=fallback_model,
                    messages=messages,
                    slot=slot,
                    attribution=attribution,
                    capability=capability,
                    is_fallback=True,
                    cancel_check=cancel_check,
                ):
                    yield token
                return
            except (*_TRANSIENT_ERRORS, *_PERSISTENT_ERRORS) as exc:
                logger.error(
                    "Fallback model streaming also failed",
                    extra={
                        "provider": slot.fallback_provider,
                        "model_id": slot.fallback_model_id,
                        "error": str(exc),
                    },
                )

        raise ModelUnavailableError(
            f"All models failed for capability '{capability.value}': {primary_error}",
            capability=capability.value,
        )

    async def _check_guardrails(self, attribution: TokenAttribution) -> None:
        """Check pre-call guardrails and raise if exceeded.

        Only checks when session_id and message_id are present (query
        context). Ingestion calls (no session) skip guardrail checks.

        Args:
            attribution: Token attribution with optional session/message context.

        Raises:
            GuardrailExceededError: If guardrails would be exceeded.
        """
        if attribution.session_id is None or attribution.message_id is None:
            return

        result = await self._guardrail_service.check_pre_call(
            session_id=attribution.session_id,
            message_id=attribution.message_id,
        )
        if not result.allowed:
            raise GuardrailExceededError(
                message=result.message or "Usage guardrail exceeded",
                guardrail_type=result.guardrail_type or "",
            )

    async def _attempt_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        slot: ModelSlotConfig,
        attribution: TokenAttribution,
        capability: ModelCapability,
        is_fallback: bool,
        response_format: type[Any] | None = None,
    ) -> CompletionResult:
        """Attempt a completion call with retry on transient failures.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff
        for transient errors. Persistent errors abort immediately.

        Args:
            model: litellm model identifier.
            messages: Chat messages.
            slot: Model slot configuration.
            attribution: Token attribution context.
            capability: The model capability being served.
            is_fallback: Whether this is the fallback model.
            response_format: Optional structured output type.

        Returns:
            CompletionResult on success.

        Raises:
            ModelUnavailableError: If retries are exhausted.
            litellm exceptions: Persistent errors re-raised for fallback handling.
        """
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                return await self._complete_single(
                    model=model,
                    messages=messages,
                    slot=slot,
                    attribution=attribution,
                    capability=capability,
                    is_fallback=is_fallback,
                    response_format=response_format,
                )
            except _PERSISTENT_ERRORS:
                raise
            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "Transient model error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        extra={"model": model, "error": str(exc)},
                    )
                    await asyncio.sleep(delay)

        raise ModelUnavailableError(
            f"Model '{model}' failed after {_MAX_RETRIES} attempts: {last_error}",
            capability=capability.value,
        )

    async def _attempt_fallback_completion(
        self,
        *,
        slot: ModelSlotConfig,
        messages: list[dict[str, Any]],
        attribution: TokenAttribution,
        capability: ModelCapability,
        response_format: type[Any] | None = None,
    ) -> CompletionResult:
        """Attempt the fallback model if configured.

        Args:
            slot: Model slot configuration (contains fallback config).
            messages: Chat messages.
            attribution: Token attribution context.
            capability: The model capability being served.
            response_format: Optional structured output type.

        Returns:
            CompletionResult from the fallback model.

        Raises:
            ModelUnavailableError: If no fallback configured or fallback also fails.
        """
        if not slot.fallback_provider or not slot.fallback_model_id:
            raise ModelUnavailableError(
                f"Primary model failed and no fallback configured "
                f"for capability '{capability.value}'",
                capability=capability.value,
            )

        fallback_model = self._litellm_model_id(slot.fallback_provider, slot.fallback_model_id)
        logger.info(
            "Activating fallback model",
            extra={
                "capability": capability.value,
                "fallback_model": fallback_model,
            },
        )

        try:
            return await self._complete_single(
                model=fallback_model,
                messages=messages,
                slot=slot,
                attribution=attribution,
                capability=capability,
                is_fallback=True,
                response_format=response_format,
            )
        except (*_TRANSIENT_ERRORS, *_PERSISTENT_ERRORS) as exc:
            raise ModelUnavailableError(
                f"Fallback model '{fallback_model}' also failed: {exc}",
                capability=capability.value,
            ) from exc

    async def _complete_single(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        slot: ModelSlotConfig,
        attribution: TokenAttribution,
        capability: ModelCapability,
        is_fallback: bool,
        response_format: type[Any] | None = None,
    ) -> CompletionResult:
        """Execute a single non-streaming litellm call and track usage.

        Args:
            model: litellm model identifier.
            messages: Chat messages.
            slot: Model slot configuration.
            attribution: Token attribution context.
            capability: The model capability being served.
            is_fallback: Whether this is the fallback model.
            response_format: Optional structured output type.

        Returns:
            CompletionResult with content and token counts.
        """
        kwargs = self._build_call_kwargs(model, messages, slot, response_format)
        response = await litellm.acompletion(**kwargs)

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        provider = (slot.fallback_provider or slot.provider) if is_fallback else slot.provider
        model_id = (slot.fallback_model_id or slot.model_id) if is_fallback else slot.model_id
        await self._track_usage(
            capability=capability,
            provider=provider,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attribution=attribution,
        )

        return CompletionResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_used=model,
            is_fallback=is_fallback,
        )

    async def _attempt_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        slot: ModelSlotConfig,
        attribution: TokenAttribution,
        capability: ModelCapability,
        is_fallback: bool,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Attempt a streaming call with retry on transient failures.

        Retries only apply before the first token is yielded. Once
        streaming begins, failures propagate immediately.

        Args:
            model: litellm model identifier.
            messages: Chat messages.
            slot: Model slot configuration.
            attribution: Token attribution context.
            capability: The model capability being served.
            is_fallback: Whether this is the fallback model.
            cancel_check: Optional async callable returning True to abort.

        Yields:
            Individual tokens as strings.
        """
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async for token in self._stream_single_attempt(
                    model=model,
                    messages=messages,
                    slot=slot,
                    attribution=attribution,
                    capability=capability,
                    is_fallback=is_fallback,
                    cancel_check=cancel_check,
                ):
                    yield token
                return
            except _PERSISTENT_ERRORS:
                raise
            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "Transient stream error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        extra={"model": model, "error": str(exc)},
                    )
                    await asyncio.sleep(delay)

        raise ModelUnavailableError(
            f"Model '{model}' streaming failed after {_MAX_RETRIES} attempts: {last_error}",
            capability=capability.value,
        )

    async def _stream_single_attempt(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        slot: ModelSlotConfig,
        attribution: TokenAttribution,
        capability: ModelCapability,
        is_fallback: bool,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute a single streaming litellm call and track usage.

        Yields tokens as they arrive. Checks cancellation between chunks.
        Tracks token usage after the stream completes.

        Args:
            model: litellm model identifier.
            messages: Chat messages.
            slot: Model slot configuration.
            attribution: Token attribution context.
            capability: The model capability being served.
            is_fallback: Whether this is the fallback model.
            cancel_check: Optional async callable returning True to abort.

        Yields:
            Individual tokens as strings.
        """
        kwargs = self._build_call_kwargs(model, messages, slot)
        kwargs["stream"] = True
        response = await litellm.acompletion(**kwargs)

        input_tokens = 0
        output_tokens = 0
        cancelled = False

        async for chunk in response:
            # Check for cancellation
            if cancel_check is not None and await cancel_check():
                cancelled = True
                break

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

            # Accumulate usage from chunks if available
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or input_tokens
                output_tokens = chunk.usage.completion_tokens or output_tokens

        # Track usage (best effort — some providers don't report
        # streaming usage per chunk, so counts may be zero)
        provider = (slot.fallback_provider or slot.provider) if is_fallback else slot.provider
        model_id = (slot.fallback_model_id or slot.model_id) if is_fallback else slot.model_id
        await self._track_usage(
            capability=capability,
            provider=provider,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attribution=attribution,
        )

        if cancelled:
            logger.info(
                "Streaming cancelled by client",
                extra={"model": model, "capability": capability.value},
            )

    async def _track_usage(
        self,
        *,
        capability: ModelCapability,
        provider: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        attribution: TokenAttribution,
    ) -> None:
        """Record token usage, swallowing errors to avoid breaking the call.

        Args:
            capability: Which AI capability the call served.
            provider: Model provider used.
            model_id: Model identifier used.
            input_tokens: Tokens sent to the model.
            output_tokens: Tokens generated by the model.
            attribution: Who/what triggered the call.
        """
        try:
            await self._token_usage_service.record(
                capability=capability,
                provider=provider,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attribution=attribution,
            )
        except Exception:
            logger.exception(
                "Failed to record token usage",
                extra={"provider": provider, "model_id": model_id},
            )

    @staticmethod
    def _build_call_kwargs(
        model: str,
        messages: list[dict[str, Any]],
        slot: ModelSlotConfig,
        response_format: type[Any] | None = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for a litellm.acompletion call.

        Args:
            model: litellm model identifier.
            messages: Chat messages.
            slot: Model slot configuration.
            response_format: Optional structured output type.

        Returns:
            Dict of keyword arguments for litellm.acompletion.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if slot.max_tokens_per_call is not None:
            kwargs["max_tokens"] = slot.max_tokens_per_call
        if slot.temperature is not None:
            kwargs["temperature"] = slot.temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    @staticmethod
    def _litellm_model_id(provider: str, model_id: str) -> str:
        """Build the litellm model identifier string.

        litellm uses ``provider/model_id`` format for most providers,
        but OpenAI models use just the model_id directly.

        Args:
            provider: Model provider (e.g. 'openai', 'anthropic').
            model_id: Provider-specific model identifier.

        Returns:
            litellm-compatible model identifier string.
        """
        if provider == "openai":
            return model_id
        return f"{provider}/{model_id}"

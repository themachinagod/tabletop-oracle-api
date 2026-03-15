"""Unit tests for GuardrailService.

Tests all guardrail types (per-query token, per-query model calls,
per-session query count, daily token budget, daily query budget),
the master enforcement toggle, graceful degradation messages, and
edge cases like missing config rows and NULL limits.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.services.model.guardrail import (
    _DAILY_DENIAL,
    _PER_QUERY_DENIAL,
    _PER_SESSION_DENIAL,
    GuardrailService,
    GuardrailServiceProtocol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    enforcement_enabled: bool = True,
    max_tokens_per_query: int | None = None,
    max_model_calls_per_query: int | None = None,
    max_queries_per_session: int | None = None,
    daily_token_budget: int | None = None,
    daily_query_budget: int | None = None,
) -> GuardrailConfig:
    """Build a GuardrailConfig with sensible defaults for testing."""
    config = MagicMock(spec=GuardrailConfig)
    config.enforcement_enabled = enforcement_enabled
    config.max_tokens_per_query = max_tokens_per_query
    config.max_model_calls_per_query = max_model_calls_per_query
    config.max_queries_per_session = max_queries_per_session
    config.daily_token_budget = daily_token_budget
    config.daily_query_budget = daily_query_budget
    return config


def _mock_db(config: GuardrailConfig | None = None) -> AsyncMock:
    """Build a mock AsyncSession that returns the given config from select."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = config
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _mock_token_usage(
    *,
    session_query_count: int = 0,
    daily_total: int = 0,
    daily_query_count: int = 0,
    query_total: int = 0,
    query_model_call_count: int = 0,
) -> MagicMock:
    """Build a mock TokenUsageService with configurable return values."""
    svc = MagicMock()
    svc.get_session_query_count = AsyncMock(return_value=session_query_count)
    svc.get_daily_total = AsyncMock(return_value=daily_total)
    svc.get_daily_query_count = AsyncMock(return_value=daily_query_count)
    svc.get_query_total = AsyncMock(return_value=query_total)
    svc.get_query_model_call_count = AsyncMock(return_value=query_model_call_count)
    return svc


def _make_service(
    config: GuardrailConfig | None = None,
    **usage_kwargs: int,
) -> GuardrailService:
    """Build a GuardrailService with mocked dependencies."""
    db = _mock_db(config)
    token_usage = _mock_token_usage(**usage_kwargs)
    return GuardrailService(db=db, token_usage_service=token_usage)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify GuardrailService satisfies GuardrailServiceProtocol."""

    def test_implements_protocol(self) -> None:
        """GuardrailService is a runtime instance of GuardrailServiceProtocol."""
        db = _mock_db()
        token_usage = _mock_token_usage()
        service = GuardrailService(db=db, token_usage_service=token_usage)

        assert isinstance(service, GuardrailServiceProtocol)


# ---------------------------------------------------------------------------
# Master toggle
# ---------------------------------------------------------------------------


class TestMasterToggle:
    """Tests for the enforcement_enabled master toggle."""

    async def test_check_pre_query_allowed_when_enforcement_disabled(self) -> None:
        """All pre-query checks pass when enforcement is disabled."""
        config = _make_config(
            enforcement_enabled=False,
            max_queries_per_session=1,
            daily_token_budget=1,
            daily_query_budget=1,
        )
        service = _make_service(config, session_query_count=999, daily_total=999999)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True
        assert result.message is None

    async def test_check_pre_call_allowed_when_enforcement_disabled(self) -> None:
        """All pre-call checks pass when enforcement is disabled."""
        config = _make_config(
            enforcement_enabled=False,
            max_tokens_per_query=1,
            max_model_calls_per_query=1,
        )
        service = _make_service(config, query_total=999999, query_model_call_count=999)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True
        assert result.message is None

    async def test_check_pre_query_allowed_when_no_config_row(self) -> None:
        """All checks pass when no guardrail_config row exists."""
        service = _make_service(None)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_check_pre_call_allowed_when_no_config_row(self) -> None:
        """All checks pass when no guardrail_config row exists."""
        service = _make_service(None)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# check_pre_query — per-session query count
# ---------------------------------------------------------------------------


class TestPreQuerySessionLimit:
    """Tests for per-session query count guardrail in check_pre_query."""

    async def test_allowed_when_under_limit(self) -> None:
        """Allowed when session query count is below the limit."""
        config = _make_config(max_queries_per_session=10)
        service = _make_service(config, session_query_count=5)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_denied_when_at_limit(self) -> None:
        """Denied when session query count equals the limit."""
        config = _make_config(max_queries_per_session=10)
        service = _make_service(config, session_query_count=10)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.message == _PER_SESSION_DENIAL
        assert result.guardrail_type == "per_session_query"

    async def test_denied_when_over_limit(self) -> None:
        """Denied when session query count exceeds the limit."""
        config = _make_config(max_queries_per_session=5)
        service = _make_service(config, session_query_count=7)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.guardrail_type == "per_session_query"

    async def test_allowed_when_limit_is_none(self) -> None:
        """Allowed when max_queries_per_session is NULL (no limit)."""
        config = _make_config(max_queries_per_session=None)
        service = _make_service(config, session_query_count=999)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_allowed_at_boundary_minus_one(self) -> None:
        """Allowed when session query count is one below the limit."""
        config = _make_config(max_queries_per_session=10)
        service = _make_service(config, session_query_count=9)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True


# ---------------------------------------------------------------------------
# check_pre_query — daily token budget
# ---------------------------------------------------------------------------


class TestPreQueryDailyTokenBudget:
    """Tests for daily token budget guardrail in check_pre_query."""

    async def test_allowed_when_under_budget(self) -> None:
        """Allowed when daily token total is below the budget."""
        config = _make_config(daily_token_budget=100000)
        service = _make_service(config, daily_total=50000)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_denied_when_at_budget(self) -> None:
        """Denied when daily token total equals the budget."""
        config = _make_config(daily_token_budget=100000)
        service = _make_service(config, daily_total=100000)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.message == _DAILY_DENIAL
        assert result.guardrail_type == "daily_token_budget"

    async def test_denied_when_over_budget(self) -> None:
        """Denied when daily token total exceeds the budget."""
        config = _make_config(daily_token_budget=50000)
        service = _make_service(config, daily_total=75000)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.guardrail_type == "daily_token_budget"

    async def test_allowed_when_budget_is_none(self) -> None:
        """Allowed when daily_token_budget is NULL (no limit)."""
        config = _make_config(daily_token_budget=None)
        service = _make_service(config, daily_total=999999)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True


# ---------------------------------------------------------------------------
# check_pre_query — daily query budget
# ---------------------------------------------------------------------------


class TestPreQueryDailyQueryBudget:
    """Tests for daily query budget guardrail in check_pre_query."""

    async def test_allowed_when_under_budget(self) -> None:
        """Allowed when daily query count is below the budget."""
        config = _make_config(daily_query_budget=500)
        service = _make_service(config, daily_query_count=100)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_denied_when_at_budget(self) -> None:
        """Denied when daily query count equals the budget."""
        config = _make_config(daily_query_budget=500)
        service = _make_service(config, daily_query_count=500)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.message == _DAILY_DENIAL
        assert result.guardrail_type == "daily_query_budget"

    async def test_denied_when_over_budget(self) -> None:
        """Denied when daily query count exceeds the budget."""
        config = _make_config(daily_query_budget=100)
        service = _make_service(config, daily_query_count=150)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.guardrail_type == "daily_query_budget"

    async def test_allowed_when_budget_is_none(self) -> None:
        """Allowed when daily_query_budget is NULL (no limit)."""
        config = _make_config(daily_query_budget=None)
        service = _make_service(config, daily_query_count=999)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True


# ---------------------------------------------------------------------------
# check_pre_query — short-circuit ordering
# ---------------------------------------------------------------------------


class TestPreQueryShortCircuit:
    """Tests that check_pre_query short-circuits on the first violation."""

    async def test_session_limit_checked_before_daily_token(self) -> None:
        """Per-session query limit is checked before daily token budget."""
        config = _make_config(
            max_queries_per_session=5,
            daily_token_budget=100,
        )
        service = _make_service(
            config,
            session_query_count=10,
            daily_total=200,
        )

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.guardrail_type == "per_session_query"

    async def test_daily_token_checked_before_daily_query(self) -> None:
        """Daily token budget is checked before daily query budget."""
        config = _make_config(
            daily_token_budget=100,
            daily_query_budget=10,
        )
        service = _make_service(
            config,
            daily_total=200,
            daily_query_count=20,
        )

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.guardrail_type == "daily_token_budget"


# ---------------------------------------------------------------------------
# check_pre_call — per-query token limit
# ---------------------------------------------------------------------------


class TestPreCallTokenLimit:
    """Tests for per-query token guardrail in check_pre_call."""

    async def test_allowed_when_under_limit(self) -> None:
        """Allowed when query token total is below the limit."""
        config = _make_config(max_tokens_per_query=10000)
        service = _make_service(config, query_total=5000)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True

    async def test_denied_when_at_limit(self) -> None:
        """Denied when query token total equals the limit."""
        config = _make_config(max_tokens_per_query=10000)
        service = _make_service(config, query_total=10000)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is False
        assert result.message == _PER_QUERY_DENIAL
        assert result.guardrail_type == "per_query_token"

    async def test_denied_when_over_limit(self) -> None:
        """Denied when query token total exceeds the limit."""
        config = _make_config(max_tokens_per_query=5000)
        service = _make_service(config, query_total=8000)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is False
        assert result.guardrail_type == "per_query_token"

    async def test_allowed_when_limit_is_none(self) -> None:
        """Allowed when max_tokens_per_query is NULL (no limit)."""
        config = _make_config(max_tokens_per_query=None)
        service = _make_service(config, query_total=999999)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# check_pre_call — per-query model call count
# ---------------------------------------------------------------------------


class TestPreCallModelCallLimit:
    """Tests for per-query model call count guardrail in check_pre_call."""

    async def test_allowed_when_under_limit(self) -> None:
        """Allowed when model call count is below the limit."""
        config = _make_config(max_model_calls_per_query=10)
        service = _make_service(config, query_model_call_count=5)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True

    async def test_denied_when_at_limit(self) -> None:
        """Denied when model call count equals the limit."""
        config = _make_config(max_model_calls_per_query=10)
        service = _make_service(config, query_model_call_count=10)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is False
        assert result.message == _PER_QUERY_DENIAL
        assert result.guardrail_type == "per_query_model_calls"

    async def test_denied_when_over_limit(self) -> None:
        """Denied when model call count exceeds the limit."""
        config = _make_config(max_model_calls_per_query=3)
        service = _make_service(config, query_model_call_count=5)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is False
        assert result.guardrail_type == "per_query_model_calls"

    async def test_allowed_when_limit_is_none(self) -> None:
        """Allowed when max_model_calls_per_query is NULL (no limit)."""
        config = _make_config(max_model_calls_per_query=None)
        service = _make_service(config, query_model_call_count=999)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# check_pre_call — short-circuit ordering
# ---------------------------------------------------------------------------


class TestPreCallShortCircuit:
    """Tests that check_pre_call short-circuits on the first violation."""

    async def test_token_limit_checked_before_call_count(self) -> None:
        """Per-query token limit is checked before model call count."""
        config = _make_config(
            max_tokens_per_query=100,
            max_model_calls_per_query=2,
        )
        service = _make_service(
            config,
            query_total=200,
            query_model_call_count=5,
        )

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.guardrail_type == "per_query_token"


# ---------------------------------------------------------------------------
# All limits NULL — no enforcement
# ---------------------------------------------------------------------------


class TestAllLimitsNull:
    """Tests that all checks pass when all limits are NULL."""

    async def test_pre_query_all_null(self) -> None:
        """Pre-query passes when all limits are NULL."""
        config = _make_config()
        service = _make_service(config)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_pre_call_all_null(self) -> None:
        """Pre-call passes when all limits are NULL."""
        config = _make_config()
        service = _make_service(config)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Graceful degradation messages
# ---------------------------------------------------------------------------


class TestDenialMessages:
    """Tests that denial messages are human-readable per PRD-007 Section 3.3."""

    async def test_per_query_token_message(self) -> None:
        """Per-query token denial message is human-readable."""
        config = _make_config(max_tokens_per_query=100)
        service = _make_service(config, query_total=200)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.message is not None
        assert "processing limit" in result.message
        assert "simplifying" in result.message

    async def test_per_query_model_calls_message(self) -> None:
        """Per-query model call denial uses same message as per-query token."""
        config = _make_config(max_model_calls_per_query=1)
        service = _make_service(config, query_model_call_count=1)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.message == _PER_QUERY_DENIAL

    async def test_per_session_query_message(self) -> None:
        """Per-session query denial mentions starting a new session."""
        config = _make_config(max_queries_per_session=1)
        service = _make_service(config, session_query_count=1)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.message is not None
        assert "question limit" in result.message
        assert "new session" in result.message

    async def test_daily_token_budget_message(self) -> None:
        """Daily token budget denial mentions trying again tomorrow."""
        config = _make_config(daily_token_budget=100)
        service = _make_service(config, daily_total=100)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.message is not None
        assert "daily usage limit" in result.message
        assert "tomorrow" in result.message

    async def test_daily_query_budget_message(self) -> None:
        """Daily query budget denial uses same message as daily token."""
        config = _make_config(daily_query_budget=10)
        service = _make_service(config, daily_query_count=10)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.message == _DAILY_DENIAL


# ---------------------------------------------------------------------------
# Multiple guardrails configured — all pass
# ---------------------------------------------------------------------------


class TestMultipleGuardrailsPass:
    """Tests when all guardrails are configured and usage is within limits."""

    async def test_pre_query_all_under_limits(self) -> None:
        """Pre-query passes when all configured limits are satisfied."""
        config = _make_config(
            max_queries_per_session=100,
            daily_token_budget=500000,
            daily_query_budget=1000,
        )
        service = _make_service(
            config,
            session_query_count=5,
            daily_total=1000,
            daily_query_count=10,
        )

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is True

    async def test_pre_call_all_under_limits(self) -> None:
        """Pre-call passes when all configured limits are satisfied."""
        config = _make_config(
            max_tokens_per_query=50000,
            max_model_calls_per_query=20,
        )
        service = _make_service(
            config,
            query_total=1000,
            query_model_call_count=2,
        )

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Zero limits — edge case
# ---------------------------------------------------------------------------


class TestZeroLimits:
    """Tests behavior when limits are set to zero (block everything)."""

    async def test_zero_session_query_limit_blocks_first_query(self) -> None:
        """A zero max_queries_per_session blocks even the first query."""
        config = _make_config(max_queries_per_session=0)
        service = _make_service(config, session_query_count=0)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.guardrail_type == "per_session_query"

    async def test_zero_token_limit_blocks_first_call(self) -> None:
        """A zero max_tokens_per_query blocks even with zero usage."""
        config = _make_config(max_tokens_per_query=0)
        service = _make_service(config, query_total=0)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is False
        assert result.guardrail_type == "per_query_token"

    async def test_zero_model_calls_limit_blocks_first_call(self) -> None:
        """A zero max_model_calls_per_query blocks even the first call."""
        config = _make_config(max_model_calls_per_query=0)
        service = _make_service(config, query_model_call_count=0)

        result = await service.check_pre_call(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert result.allowed is False
        assert result.guardrail_type == "per_query_model_calls"

    async def test_zero_daily_token_budget_blocks(self) -> None:
        """A zero daily_token_budget blocks even with zero usage."""
        config = _make_config(daily_token_budget=0)
        service = _make_service(config, daily_total=0)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.guardrail_type == "daily_token_budget"

    async def test_zero_daily_query_budget_blocks(self) -> None:
        """A zero daily_query_budget blocks even with zero queries."""
        config = _make_config(daily_query_budget=0)
        service = _make_service(config, daily_query_count=0)

        result = await service.check_pre_query(session_id=uuid.uuid4())

        assert result.allowed is False
        assert result.guardrail_type == "daily_query_budget"

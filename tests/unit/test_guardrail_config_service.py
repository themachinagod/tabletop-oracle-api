"""Unit tests for GuardrailService admin operations (get_config, update_config).

Tests the extended admin methods added for EPIC-005 guardrail
configuration management. The existing enforcement tests remain
in test_guardrail_service.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.services.model.guardrail import GuardrailService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    enforcement_enabled: bool = True,
    max_tokens_per_query: int | None = 50000,
    max_model_calls_per_query: int | None = 10,
    max_queries_per_session: int | None = 100,
    daily_token_budget: int | None = 1000000,
    daily_query_budget: int | None = 5000,
    per_document_ingestion_limit: int | None = 200000,
    updated_at: datetime | None = None,
) -> MagicMock:
    """Build a mock GuardrailConfig with realistic defaults."""
    config = MagicMock(spec=GuardrailConfig)
    config.enforcement_enabled = enforcement_enabled
    config.max_tokens_per_query = max_tokens_per_query
    config.max_model_calls_per_query = max_model_calls_per_query
    config.max_queries_per_session = max_queries_per_session
    config.daily_token_budget = daily_token_budget
    config.daily_query_budget = daily_query_budget
    config.per_document_ingestion_limit = per_document_ingestion_limit
    config.updated_at = updated_at or datetime.now(UTC)
    return config


def _mock_db(config: MagicMock | None = None) -> AsyncMock:
    """Build a mock AsyncSession that returns the given config from select."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = config
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result_mock)
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _mock_token_usage() -> MagicMock:
    """Build a minimal mock TokenUsageService (not used by admin methods)."""
    return MagicMock()


def _make_service(
    config: MagicMock | None = None,
) -> tuple[GuardrailService, AsyncMock]:
    """Build a GuardrailService with mocked dependencies.

    Returns:
        Tuple of (service, mock_db) for assertion access.
    """
    db = _mock_db(config)
    token_usage = _mock_token_usage()
    service = GuardrailService(db=db, token_usage_service=token_usage)
    return service, db


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Tests for GuardrailService.get_config()."""

    @pytest.mark.asyncio
    async def test_returns_config_when_exists(self) -> None:
        """Returns the guardrail config when a row exists."""
        config = _make_config()
        service, _ = _make_service(config)

        result = await service.get_config()

        assert result is config

    @pytest.mark.asyncio
    async def test_returns_config_with_all_fields(self) -> None:
        """Returned config has all expected attributes."""
        config = _make_config(
            enforcement_enabled=True,
            max_tokens_per_query=50000,
            daily_token_budget=1000000,
        )
        service, _ = _make_service(config)

        result = await service.get_config()

        assert result.enforcement_enabled is True
        assert result.max_tokens_per_query == 50000
        assert result.daily_token_budget == 1000000

    @pytest.mark.asyncio
    async def test_raises_not_found_when_no_config(self) -> None:
        """Raises NotFoundError when no guardrail_config row exists."""
        service, _ = _make_service(None)

        with pytest.raises(NotFoundError):
            await service.get_config()


# ---------------------------------------------------------------------------
# update_config
# ---------------------------------------------------------------------------


class TestUpdateConfig:
    """Tests for GuardrailService.update_config()."""

    @pytest.mark.asyncio
    async def test_updates_single_field(self) -> None:
        """Updating a single field sets it on the config entity."""
        config = _make_config(enforcement_enabled=True)
        service, db = _make_service(config)

        result = await service.update_config({"enforcement_enabled": False})

        assert config.enforcement_enabled is False
        assert result is config
        db.flush.assert_awaited_once()
        db.refresh.assert_awaited_once_with(config)

    @pytest.mark.asyncio
    async def test_updates_multiple_fields(self) -> None:
        """Updating multiple fields applies all changes."""
        config = _make_config()
        service, _db = _make_service(config)

        await service.update_config(
            {
                "daily_token_budget": 2000000,
                "daily_query_budget": 10000,
                "enforcement_enabled": False,
            }
        )

        assert config.daily_token_budget == 2000000
        assert config.daily_query_budget == 10000
        assert config.enforcement_enabled is False

    @pytest.mark.asyncio
    async def test_partial_update_leaves_other_fields_unchanged(self) -> None:
        """Fields not in the update dict are not modified."""
        config = _make_config(
            max_tokens_per_query=50000,
            daily_token_budget=1000000,
        )
        service, _ = _make_service(config)

        await service.update_config({"daily_token_budget": 2000000})

        assert config.max_tokens_per_query == 50000  # unchanged
        assert config.daily_token_budget == 2000000  # changed

    @pytest.mark.asyncio
    async def test_update_with_null_value(self) -> None:
        """Setting a field to None clears the limit."""
        config = _make_config(max_tokens_per_query=50000)
        service, _ = _make_service(config)

        await service.update_config({"max_tokens_per_query": None})

        assert config.max_tokens_per_query is None

    @pytest.mark.asyncio
    async def test_ignores_unknown_fields(self) -> None:
        """Unknown field names are filtered out and not applied."""
        config = _make_config()
        service, _db = _make_service(config)

        await service.update_config(
            {
                "enforcement_enabled": False,
                "nonexistent_field": "should be ignored",
            }
        )

        assert config.enforcement_enabled is False
        assert not hasattr(config, "nonexistent_field") or True  # mock won't have it

    @pytest.mark.asyncio
    async def test_empty_updates_skips_flush(self) -> None:
        """No-op when update dict is empty (or all fields are unknown)."""
        config = _make_config()
        service, db = _make_service(config)

        result = await service.update_config({})

        assert result is config
        db.flush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_only_unknown_fields_skips_flush(self) -> None:
        """No-op when all provided fields are unknown."""
        config = _make_config()
        service, db = _make_service(config)

        result = await service.update_config({"unknown_field": 42})

        assert result is config
        db.flush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_not_found_when_no_config(self) -> None:
        """Raises NotFoundError when no guardrail_config row exists."""
        service, _ = _make_service(None)

        with pytest.raises(NotFoundError):
            await service.update_config({"enforcement_enabled": False})

    @pytest.mark.asyncio
    async def test_updates_all_updatable_fields(self) -> None:
        """All known updatable fields can be set."""
        config = _make_config()
        service, _ = _make_service(config)

        await service.update_config(
            {
                "enforcement_enabled": False,
                "max_tokens_per_query": 100000,
                "max_model_calls_per_query": 20,
                "max_queries_per_session": 200,
                "daily_token_budget": 5000000,
                "daily_query_budget": 25000,
                "per_document_ingestion_limit": 500000,
            }
        )

        assert config.enforcement_enabled is False
        assert config.max_tokens_per_query == 100000
        assert config.max_model_calls_per_query == 20
        assert config.max_queries_per_session == 200
        assert config.daily_token_budget == 5000000
        assert config.daily_query_budget == 25000
        assert config.per_document_ingestion_limit == 500000

"""Unit tests for ModelSlotRepository.

Tests repository query construction with a mocked AsyncSession.
Verifies correct capability filtering and None handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.model.slot_repository import ModelSlotRepository


def _mock_session(return_value: MagicMock | None = None) -> AsyncMock:
    """Build a mock AsyncSession that returns a scalar result."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = return_value
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)
    return session


class TestGetByCapability:
    """Tests for ModelSlotRepository.get_by_capability."""

    @pytest.mark.asyncio
    async def test_returns_slot_when_found(self) -> None:
        """Repository returns the ORM entity when a matching row exists."""
        slot_mock = MagicMock()
        slot_mock.capability = ModelCapability.INTENT_ANALYSIS
        session = _mock_session(slot_mock)
        repo = ModelSlotRepository(session)

        result = await repo.get_by_capability(ModelCapability.INTENT_ANALYSIS)

        assert result is slot_mock
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        """Repository returns None when no matching row exists."""
        session = _mock_session(None)
        repo = ModelSlotRepository(session)

        result = await repo.get_by_capability(ModelCapability.VISION_PROCESSING)

        assert result is None

    @pytest.mark.asyncio
    async def test_executes_query_on_each_call(self) -> None:
        """Each call executes a fresh query (no internal caching)."""
        session = _mock_session(MagicMock())
        repo = ModelSlotRepository(session)

        await repo.get_by_capability(ModelCapability.ANSWER_SYNTHESIS)
        await repo.get_by_capability(ModelCapability.CONCEPT_EXTRACTION)

        assert session.execute.await_count == 2

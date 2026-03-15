"""Unit tests for ModelSlotRepository admin operations (get_all, update).

Tests repository query construction with a mocked AsyncSession for
the list-all and partial-update operations added for EPIC-005.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.model.slot_repository import ModelSlotRepository


def _mock_session(
    scalars_result: list[MagicMock] | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession that returns a scalars result."""
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = scalars_result or []

    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


class TestGetAll:
    """Tests for ModelSlotRepository.get_all."""

    @pytest.mark.asyncio
    async def test_returns_all_slots(self) -> None:
        """Repository returns all ModelSlot entities."""
        slot1 = MagicMock()
        slot1.capability = ModelCapability.INTENT_ANALYSIS
        slot2 = MagicMock()
        slot2.capability = ModelCapability.ANSWER_SYNTHESIS

        session = _mock_session([slot1, slot2])
        repo = ModelSlotRepository(session)

        result = await repo.get_all()

        assert len(result) == 2
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_slots(self) -> None:
        """Repository returns empty list when no slots exist."""
        session = _mock_session([])
        repo = ModelSlotRepository(session)

        result = await repo.get_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_executes_query(self) -> None:
        """Each get_all call executes a fresh query."""
        session = _mock_session([])
        repo = ModelSlotRepository(session)

        await repo.get_all()
        await repo.get_all()

        assert session.execute.await_count == 2


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    """Tests for ModelSlotRepository.update."""

    @pytest.mark.asyncio
    async def test_applies_fields_to_slot(self) -> None:
        """Update applies field values to the ORM entity."""
        slot = MagicMock()
        slot.provider = "openai"

        session = _mock_session()
        repo = ModelSlotRepository(session)

        await repo.update(slot, {"provider": "anthropic"})

        assert slot.provider == "anthropic"
        session.flush.assert_awaited_once()
        session.refresh.assert_awaited_once_with(slot)

    @pytest.mark.asyncio
    async def test_applies_multiple_fields(self) -> None:
        """Update applies all provided fields."""
        slot = MagicMock()

        session = _mock_session()
        repo = ModelSlotRepository(session)

        await repo.update(
            slot,
            {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-20250514",
                "temperature": 0.5,
            },
        )

        assert slot.provider == "anthropic"
        assert slot.model_id == "claude-sonnet-4-20250514"
        assert slot.temperature == 0.5

    @pytest.mark.asyncio
    async def test_returns_updated_slot(self) -> None:
        """Update returns the slot entity after flush+refresh."""
        slot = MagicMock()
        session = _mock_session()
        repo = ModelSlotRepository(session)

        result = await repo.update(slot, {"provider": "anthropic"})

        assert result is slot

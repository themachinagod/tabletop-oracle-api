"""Unit tests for ModelSlotService admin operations (list_all, update_slot).

Tests the extended service methods added for EPIC-005 model slot
admin API: listing all slots and partial updates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.model.slot_config import ModelSlotConfig
from tabletop_oracle.services.model.slot_service import ModelSlotService


def _make_slot_orm(
    capability: ModelCapability,
    *,
    provider: str = "openai",
    model_id: str = "gpt-4o",
    max_tokens_per_call: int | None = 4096,
    temperature: float | None = 0.7,
    fallback_provider: str | None = None,
    fallback_model_id: str | None = None,
) -> MagicMock:
    """Build a mock ModelSlot ORM entity."""
    mock = MagicMock()
    mock.capability = capability
    mock.provider = provider
    mock.model_id = model_id
    mock.max_tokens_per_call = max_tokens_per_call
    mock.temperature = temperature
    mock.fallback_provider = fallback_provider
    mock.fallback_model_id = fallback_model_id
    return mock


def _make_repository(
    *,
    all_slots: list[MagicMock] | None = None,
    get_slot: MagicMock | None = None,
) -> AsyncMock:
    """Build a mock ModelSlotRepository."""
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=all_slots or [])
    repo.get_by_capability = AsyncMock(return_value=get_slot)
    repo.update = AsyncMock(side_effect=lambda slot, fields: slot)
    return repo


def _apply_fields(slot: MagicMock, fields: dict) -> MagicMock:
    """Simulate applying update fields to a mock slot."""
    for k, v in fields.items():
        setattr(slot, k, v)
    return slot


class TestListAll:
    """Tests for ModelSlotService.list_all."""

    @pytest.mark.asyncio
    async def test_returns_all_slots(self) -> None:
        """list_all returns a ModelSlotConfig for each slot in the repository."""
        slots = [
            _make_slot_orm(ModelCapability.INTENT_ANALYSIS),
            _make_slot_orm(ModelCapability.ANSWER_SYNTHESIS, provider="anthropic"),
            _make_slot_orm(ModelCapability.CONCEPT_EXTRACTION),
        ]
        repo = _make_repository(all_slots=slots)
        service = ModelSlotService(repository=repo)

        result = await service.list_all()

        assert len(result) == 3
        assert all(isinstance(c, ModelSlotConfig) for c in result)
        repo.get_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_slots(self) -> None:
        """list_all returns an empty list when no slots exist."""
        repo = _make_repository(all_slots=[])
        service = ModelSlotService(repository=repo)

        result = await service.list_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_preserves_field_values(self) -> None:
        """list_all correctly maps all fields from ORM entities."""
        slot = _make_slot_orm(
            ModelCapability.RETRIEVAL_AUGMENTATION,
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens_per_call=8192,
            temperature=0.3,
            fallback_provider="openai",
            fallback_model_id="gpt-4o-mini",
        )
        repo = _make_repository(all_slots=[slot])
        service = ModelSlotService(repository=repo)

        result = await service.list_all()

        cfg = result[0]
        assert cfg.capability == ModelCapability.RETRIEVAL_AUGMENTATION
        assert cfg.provider == "anthropic"
        assert cfg.model_id == "claude-sonnet-4-20250514"
        assert cfg.max_tokens_per_call == 8192
        assert cfg.temperature == 0.3
        assert cfg.fallback_provider == "openai"
        assert cfg.fallback_model_id == "gpt-4o-mini"


class TestUpdateSlot:
    """Tests for ModelSlotService.update_slot."""

    @pytest.mark.asyncio
    async def test_updates_slot_fields(self) -> None:
        """update_slot applies provided fields and returns updated config."""
        slot = _make_slot_orm(ModelCapability.INTENT_ANALYSIS)
        repo = _make_repository(get_slot=slot)
        repo.update = AsyncMock(
            side_effect=lambda s, fields: _apply_fields(s, fields),
        )
        service = ModelSlotService(repository=repo)

        result = await service.update_slot(
            ModelCapability.INTENT_ANALYSIS,
            {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
        )

        assert isinstance(result, ModelSlotConfig)
        repo.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_not_found_for_missing_slot(self) -> None:
        """update_slot raises NotFoundError when no slot exists."""
        repo = _make_repository(get_slot=None)
        service = ModelSlotService(repository=repo)

        with pytest.raises(NotFoundError, match="intent_analysis"):
            await service.update_slot(
                ModelCapability.INTENT_ANALYSIS,
                {"provider": "anthropic"},
            )

    @pytest.mark.asyncio
    async def test_partial_update_only_valid_fields(self) -> None:
        """update_slot ignores unknown fields and only passes valid ones."""
        slot = _make_slot_orm(ModelCapability.ANSWER_SYNTHESIS)
        repo = _make_repository(get_slot=slot)
        service = ModelSlotService(repository=repo)

        await service.update_slot(
            ModelCapability.ANSWER_SYNTHESIS,
            {"provider": "anthropic", "unknown_field": "ignored"},
        )

        call_args = repo.update.call_args
        assert "unknown_field" not in call_args[0][1]
        assert "provider" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_no_op_when_no_valid_fields(self) -> None:
        """update_slot returns current config when no valid fields provided."""
        slot = _make_slot_orm(ModelCapability.ANSWER_SYNTHESIS, provider="openai")
        repo = _make_repository(get_slot=slot)
        service = ModelSlotService(repository=repo)

        result = await service.update_slot(
            ModelCapability.ANSWER_SYNTHESIS,
            {"bogus": "value"},
        )

        assert result.provider == "openai"
        repo.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nullable_fields_can_be_set_to_none(self) -> None:
        """update_slot allows setting nullable fields to None."""
        slot = _make_slot_orm(
            ModelCapability.VISION_PROCESSING,
            fallback_provider="openai",
            fallback_model_id="gpt-4o",
        )
        repo = _make_repository(get_slot=slot)
        repo.update = AsyncMock(
            side_effect=lambda s, fields: _apply_fields(s, fields),
        )
        service = ModelSlotService(repository=repo)

        result = await service.update_slot(
            ModelCapability.VISION_PROCESSING,
            {"fallback_provider": None, "fallback_model_id": None},
        )

        assert result.fallback_provider is None
        assert result.fallback_model_id is None

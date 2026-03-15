"""Unit tests for ModelSlotService.

Tests service returns correct config for valid capabilities, raises
ConfigurationError for missing slots, verifies frozen immutability,
fallback presence/absence, and no-caching behaviour (fresh DB read
per call).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import ConfigurationError
from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.model.slot_config import ModelSlotConfig
from tabletop_oracle.services.model.slot_service import ModelSlotService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_repository(return_value: MagicMock | None = None) -> AsyncMock:
    """Build a mock ModelSlotRepository."""
    repo = AsyncMock()
    repo.get_by_capability = AsyncMock(return_value=return_value)
    return repo


# ---------------------------------------------------------------------------
# get_slot — happy path
# ---------------------------------------------------------------------------


class TestGetSlotHappyPath:
    """Tests that get_slot returns correct ModelSlotConfig."""

    @pytest.mark.asyncio
    async def test_returns_config_for_valid_capability(self) -> None:
        """Service returns a ModelSlotConfig with all fields mapped."""
        slot = _make_slot_orm(
            ModelCapability.INTENT_ANALYSIS,
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens_per_call=8192,
            temperature=0.3,
        )
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        config = await service.get_slot(ModelCapability.INTENT_ANALYSIS)

        assert isinstance(config, ModelSlotConfig)
        assert config.capability == ModelCapability.INTENT_ANALYSIS
        assert config.provider == "anthropic"
        assert config.model_id == "claude-sonnet-4-20250514"
        assert config.max_tokens_per_call == 8192
        assert config.temperature == 0.3
        assert config.fallback_provider is None
        assert config.fallback_model_id is None

    @pytest.mark.asyncio
    async def test_passes_capability_to_repository(self) -> None:
        """Service delegates to repository with the correct capability."""
        slot = _make_slot_orm(ModelCapability.ANSWER_SYNTHESIS)
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        await service.get_slot(ModelCapability.ANSWER_SYNTHESIS)

        repo.get_by_capability.assert_awaited_once_with(ModelCapability.ANSWER_SYNTHESIS)


# ---------------------------------------------------------------------------
# get_slot — all capabilities
# ---------------------------------------------------------------------------


class TestGetSlotAllCapabilities:
    """Verify get_slot works for every ModelCapability value."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capability", list(ModelCapability))
    async def test_returns_config_for_each_capability(self, capability: ModelCapability) -> None:
        """Each ModelCapability value produces a valid config."""
        slot = _make_slot_orm(capability)
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        config = await service.get_slot(capability)

        assert config.capability == capability
        repo.get_by_capability.assert_awaited_once_with(capability)


# ---------------------------------------------------------------------------
# get_slot — missing slot
# ---------------------------------------------------------------------------


class TestGetSlotMissingSlot:
    """Tests that get_slot raises ConfigurationError for missing slots."""

    @pytest.mark.asyncio
    async def test_raises_configuration_error_when_slot_missing(self) -> None:
        """ConfigurationError raised when no slot exists for capability."""
        repo = _make_repository(return_value=None)
        service = ModelSlotService(repository=repo)

        with pytest.raises(ConfigurationError, match="intent_analysis"):
            await service.get_slot(ModelCapability.INTENT_ANALYSIS)

    @pytest.mark.asyncio
    async def test_error_message_includes_capability_value(self) -> None:
        """Error message includes the specific capability that is missing."""
        repo = _make_repository(return_value=None)
        service = ModelSlotService(repository=repo)

        with pytest.raises(ConfigurationError) as exc_info:
            await service.get_slot(ModelCapability.VISION_PROCESSING)

        assert "vision_processing" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_slot — fallback config
# ---------------------------------------------------------------------------


class TestGetSlotFallbackConfig:
    """Tests fallback provider/model presence and absence."""

    @pytest.mark.asyncio
    async def test_fallback_present(self) -> None:
        """Config includes fallback fields when set in the database."""
        slot = _make_slot_orm(
            ModelCapability.RETRIEVAL_AUGMENTATION,
            fallback_provider="openai",
            fallback_model_id="gpt-4o-mini",
        )
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        config = await service.get_slot(ModelCapability.RETRIEVAL_AUGMENTATION)

        assert config.fallback_provider == "openai"
        assert config.fallback_model_id == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_fallback_absent(self) -> None:
        """Config has None fallback fields when not set in the database."""
        slot = _make_slot_orm(
            ModelCapability.CONCEPT_EXTRACTION,
            fallback_provider=None,
            fallback_model_id=None,
        )
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        config = await service.get_slot(ModelCapability.CONCEPT_EXTRACTION)

        assert config.fallback_provider is None
        assert config.fallback_model_id is None


# ---------------------------------------------------------------------------
# get_slot — optional fields
# ---------------------------------------------------------------------------


class TestGetSlotOptionalFields:
    """Tests for nullable max_tokens_per_call and temperature."""

    @pytest.mark.asyncio
    async def test_none_max_tokens(self) -> None:
        """Config allows None max_tokens_per_call (unlimited)."""
        slot = _make_slot_orm(
            ModelCapability.ANSWER_SYNTHESIS,
            max_tokens_per_call=None,
        )
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        config = await service.get_slot(ModelCapability.ANSWER_SYNTHESIS)

        assert config.max_tokens_per_call is None

    @pytest.mark.asyncio
    async def test_none_temperature(self) -> None:
        """Config allows None temperature (provider default)."""
        slot = _make_slot_orm(
            ModelCapability.CLARIFICATION_GENERATION,
            temperature=None,
        )
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        config = await service.get_slot(ModelCapability.CLARIFICATION_GENERATION)

        assert config.temperature is None


# ---------------------------------------------------------------------------
# ModelSlotConfig — immutability
# ---------------------------------------------------------------------------


class TestModelSlotConfigImmutability:
    """Tests that ModelSlotConfig is frozen (immutable)."""

    def test_cannot_modify_provider(self) -> None:
        """Attempting to modify provider raises FrozenInstanceError."""
        config = ModelSlotConfig(
            capability=ModelCapability.INTENT_ANALYSIS,
            provider="openai",
            model_id="gpt-4o",
            max_tokens_per_call=4096,
            temperature=0.7,
            fallback_provider=None,
            fallback_model_id=None,
        )

        with pytest.raises(FrozenInstanceError):
            config.provider = "anthropic"  # type: ignore[misc]

    def test_cannot_modify_model_id(self) -> None:
        """Attempting to modify model_id raises FrozenInstanceError."""
        config = ModelSlotConfig(
            capability=ModelCapability.INTENT_ANALYSIS,
            provider="openai",
            model_id="gpt-4o",
            max_tokens_per_call=4096,
            temperature=0.7,
            fallback_provider=None,
            fallback_model_id=None,
        )

        with pytest.raises(FrozenInstanceError):
            config.model_id = "gpt-4o-mini"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# No caching — fresh DB read per call
# ---------------------------------------------------------------------------


class TestNoCaching:
    """Verify the service does not cache — each call hits the repository."""

    @pytest.mark.asyncio
    async def test_successive_calls_return_different_values(self) -> None:
        """Two calls return different configs when the DB changes between them."""
        slot_v1 = _make_slot_orm(
            ModelCapability.INTENT_ANALYSIS,
            provider="openai",
            model_id="gpt-4o",
        )
        slot_v2 = _make_slot_orm(
            ModelCapability.INTENT_ANALYSIS,
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
        )

        repo = AsyncMock()
        repo.get_by_capability = AsyncMock(side_effect=[slot_v1, slot_v2])
        service = ModelSlotService(repository=repo)

        config_1 = await service.get_slot(ModelCapability.INTENT_ANALYSIS)
        config_2 = await service.get_slot(ModelCapability.INTENT_ANALYSIS)

        assert config_1.provider == "openai"
        assert config_2.provider == "anthropic"
        assert repo.get_by_capability.await_count == 2

    @pytest.mark.asyncio
    async def test_each_call_queries_repository(self) -> None:
        """Every get_slot call issues a repository query."""
        slot = _make_slot_orm(ModelCapability.ANSWER_SYNTHESIS)
        repo = _make_repository(slot)
        service = ModelSlotService(repository=repo)

        await service.get_slot(ModelCapability.ANSWER_SYNTHESIS)
        await service.get_slot(ModelCapability.ANSWER_SYNTHESIS)
        await service.get_slot(ModelCapability.ANSWER_SYNTHESIS)

        assert repo.get_by_capability.await_count == 3

"""Unit tests for model slot admin Pydantic schemas.

Tests ModelSlotResponse serialization and ModelSlotUpdate partial
update semantics with UNSET sentinels.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tabletop_oracle.schemas.model_slot import ModelSlotResponse, ModelSlotUpdate
from tabletop_oracle.schemas.sentinel import UNSET

# ---------------------------------------------------------------------------
# ModelSlotResponse
# ---------------------------------------------------------------------------


class TestModelSlotResponse:
    """Tests for ModelSlotResponse schema."""

    def test_serializes_all_fields(self) -> None:
        """Response includes all fields when populated."""
        now = datetime.now(UTC)
        resp = ModelSlotResponse(
            capability="intent_analysis",
            provider="openai",
            model_id="gpt-4o",
            max_tokens_per_call=4096,
            temperature=0.7,
            fallback_provider="anthropic",
            fallback_model_id="claude-haiku-4-20250514",
            updated_at=now,
        )
        data = resp.model_dump()
        assert data["capability"] == "intent_analysis"
        assert data["provider"] == "openai"
        assert data["max_tokens_per_call"] == 4096
        assert data["fallback_provider"] == "anthropic"

    def test_nullable_fields_default_to_none(self) -> None:
        """Optional fields default to None."""
        resp = ModelSlotResponse(
            capability="answer_synthesis",
            provider="openai",
            model_id="gpt-4o",
        )
        assert resp.max_tokens_per_call is None
        assert resp.temperature is None
        assert resp.fallback_provider is None
        assert resp.fallback_model_id is None
        assert resp.updated_at is None


# ---------------------------------------------------------------------------
# ModelSlotUpdate
# ---------------------------------------------------------------------------


class TestModelSlotUpdate:
    """Tests for ModelSlotUpdate schema and UNSET semantics."""

    def test_all_fields_default_to_unset(self) -> None:
        """All fields default to UNSET when not provided."""
        update = ModelSlotUpdate()
        assert update.provider is UNSET
        assert update.model_id is UNSET
        assert update.max_tokens_per_call is UNSET
        assert update.temperature is UNSET
        assert update.fallback_provider is UNSET
        assert update.fallback_model_id is UNSET

    def test_to_update_dict_includes_only_provided_fields(self) -> None:
        """to_update_dict returns only fields explicitly set."""
        update = ModelSlotUpdate(provider="anthropic", temperature=0.5)
        result = update.to_update_dict()
        assert result == {"provider": "anthropic", "temperature": 0.5}

    def test_to_update_dict_empty_when_no_fields(self) -> None:
        """to_update_dict returns empty dict when nothing provided."""
        update = ModelSlotUpdate()
        assert update.to_update_dict() == {}

    def test_to_update_dict_includes_none_values(self) -> None:
        """Explicitly setting a field to None is included (to clear it)."""
        update = ModelSlotUpdate(fallback_provider=None, fallback_model_id=None)
        result = update.to_update_dict()
        assert result == {"fallback_provider": None, "fallback_model_id": None}

    def test_to_update_dict_all_fields(self) -> None:
        """All fields present when all are explicitly provided."""
        update = ModelSlotUpdate(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens_per_call=8192,
            temperature=0.3,
            fallback_provider="openai",
            fallback_model_id="gpt-4o-mini",
        )
        result = update.to_update_dict()
        assert len(result) == 6
        assert result["provider"] == "anthropic"
        assert result["max_tokens_per_call"] == 8192

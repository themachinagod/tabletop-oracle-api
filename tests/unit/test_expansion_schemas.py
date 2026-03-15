"""Unit tests for Expansion Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tabletop_oracle.schemas.expansion import (
    ExpansionCreate,
    ExpansionListResponse,
    ExpansionResponse,
    ExpansionUpdate,
)
from tabletop_oracle.schemas.sentinel import UNSET

# ---------------------------------------------------------------------------
# ExpansionCreate
# ---------------------------------------------------------------------------


class TestExpansionCreate:
    """Tests for ExpansionCreate request schema."""

    def test_valid_minimal(self) -> None:
        """ExpansionCreate accepts name-only input."""
        expansion = ExpansionCreate(name="Seafarers")
        assert expansion.name == "Seafarers"
        assert expansion.year_published is None
        assert expansion.description is None

    def test_valid_full(self) -> None:
        """ExpansionCreate accepts all fields."""
        expansion = ExpansionCreate(
            name="Seafarers of Catan",
            year_published=1997,
            description="Explore new islands.",
        )
        assert expansion.name == "Seafarers of Catan"
        assert expansion.year_published == 1997
        assert expansion.description == "Explore new islands."

    def test_name_required(self) -> None:
        """ExpansionCreate rejects missing name."""
        with pytest.raises(ValidationError):
            ExpansionCreate()  # type: ignore[call-arg]

    def test_name_empty_rejected(self) -> None:
        """ExpansionCreate rejects empty string name."""
        with pytest.raises(ValidationError):
            ExpansionCreate(name="")

    def test_name_max_length(self) -> None:
        """ExpansionCreate rejects names longer than 500 characters."""
        with pytest.raises(ValidationError):
            ExpansionCreate(name="x" * 501)

    def test_name_at_max_length(self) -> None:
        """ExpansionCreate accepts name at exactly 500 characters."""
        expansion = ExpansionCreate(name="x" * 500)
        assert len(expansion.name) == 500

    def test_year_must_be_positive(self) -> None:
        """ExpansionCreate rejects zero or negative year_published."""
        with pytest.raises(ValidationError):
            ExpansionCreate(name="Test", year_published=0)
        with pytest.raises(ValidationError):
            ExpansionCreate(name="Test", year_published=-1)


# ---------------------------------------------------------------------------
# ExpansionUpdate
# ---------------------------------------------------------------------------


class TestExpansionUpdate:
    """Tests for ExpansionUpdate (PATCH) request schema."""

    def test_all_fields_default_to_unset(self) -> None:
        """ExpansionUpdate defaults all fields to UNSET sentinel."""
        update = ExpansionUpdate()
        assert update.name is UNSET
        assert update.year_published is UNSET
        assert update.description is UNSET

    def test_partial_update(self) -> None:
        """ExpansionUpdate allows setting only specific fields."""
        update = ExpansionUpdate.model_validate({"description": "New description"})
        assert update.description == "New description"
        assert update.name is UNSET

    def test_null_distinguished_from_unset(self) -> None:
        """ExpansionUpdate distinguishes null from absent."""
        update = ExpansionUpdate.model_validate({"year_published": None})
        assert update.year_published is None
        assert update.description is UNSET

    def test_name_empty_rejected(self) -> None:
        """ExpansionUpdate rejects empty-string name."""
        with pytest.raises(ValidationError, match="name must not be empty"):
            ExpansionUpdate.model_validate({"name": ""})

    def test_name_whitespace_rejected(self) -> None:
        """ExpansionUpdate rejects whitespace-only name."""
        with pytest.raises(ValidationError, match="name must not be empty"):
            ExpansionUpdate.model_validate({"name": "   "})

    def test_name_over_500_rejected(self) -> None:
        """ExpansionUpdate rejects name longer than 500 characters."""
        with pytest.raises(ValidationError, match="at most 500"):
            ExpansionUpdate.model_validate({"name": "x" * 501})

    def test_year_must_be_positive(self) -> None:
        """ExpansionUpdate rejects non-positive year_published."""
        with pytest.raises(ValidationError, match="positive integer"):
            ExpansionUpdate.model_validate({"year_published": 0})

    def test_get_set_fields_excludes_unset(self) -> None:
        """get_set_fields returns only explicitly provided fields."""
        update = ExpansionUpdate.model_validate({"name": "Updated", "description": None})
        fields = update.get_set_fields()
        assert fields == {"name": "Updated", "description": None}
        assert "year_published" not in fields

    def test_get_set_fields_empty_when_no_fields(self) -> None:
        """get_set_fields returns empty dict when nothing provided."""
        update = ExpansionUpdate()
        assert update.get_set_fields() == {}


# ---------------------------------------------------------------------------
# ExpansionResponse
# ---------------------------------------------------------------------------


def _expansion_data(game_id: uuid.UUID | None = None) -> dict[str, object]:
    """Build valid ExpansionResponse input data."""
    return {
        "id": uuid.uuid4(),
        "game_id": game_id or uuid.uuid4(),
        "name": "Seafarers",
        "year_published": 1997,
        "description": "Explore new islands.",
        "archived_at": None,
        "created_at": datetime.now(tz=UTC),
        "updated_at": datetime.now(tz=UTC),
    }


class TestExpansionResponse:
    """Tests for ExpansionResponse schema."""

    def test_valid_response(self) -> None:
        """ExpansionResponse accepts valid expansion data."""
        data = _expansion_data()
        response = ExpansionResponse(**data)  # type: ignore[arg-type]
        assert response.name == "Seafarers"
        assert isinstance(response.id, uuid.UUID)
        assert isinstance(response.game_id, uuid.UUID)

    def test_nullable_fields(self) -> None:
        """ExpansionResponse allows null for optional fields."""
        data = _expansion_data()
        data["year_published"] = None
        data["description"] = None
        response = ExpansionResponse(**data)  # type: ignore[arg-type]
        assert response.year_published is None
        assert response.description is None

    def test_serialization(self) -> None:
        """ExpansionResponse serializes with expected keys."""
        data = _expansion_data()
        response = ExpansionResponse(**data)  # type: ignore[arg-type]
        dumped = response.model_dump()
        expected_keys = {
            "id",
            "game_id",
            "name",
            "year_published",
            "description",
            "archived_at",
            "created_at",
            "updated_at",
        }
        assert set(dumped.keys()) == expected_keys


# ---------------------------------------------------------------------------
# ExpansionListResponse
# ---------------------------------------------------------------------------


class TestExpansionListResponse:
    """Tests for ExpansionListResponse schema."""

    def test_includes_document_count(self) -> None:
        """ExpansionListResponse includes document_count."""
        data = _expansion_data()
        data["document_count"] = 5
        response = ExpansionListResponse(**data)  # type: ignore[arg-type]
        assert response.document_count == 5
        assert response.name == "Seafarers"

    def test_inherits_expansion_fields(self) -> None:
        """ExpansionListResponse inherits all ExpansionResponse fields."""
        data = _expansion_data()
        data["document_count"] = 0
        response = ExpansionListResponse(**data)  # type: ignore[arg-type]
        assert isinstance(response.id, uuid.UUID)
        assert isinstance(response.game_id, uuid.UUID)

"""Unit tests for Game Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tabletop_oracle.schemas.game import (
    GameComplexityFilter,
    GameCreate,
    GameDetailResponse,
    GameFilters,
    GameResponse,
    GameSummaryResponse,
    GameUpdate,
    TagCountResponse,
)
from tabletop_oracle.schemas.sentinel import UNSET

# ---------------------------------------------------------------------------
# GameComplexityFilter
# ---------------------------------------------------------------------------


class TestGameComplexityFilter:
    """Tests for GameComplexityFilter enum."""

    def test_values(self) -> None:
        """GameComplexityFilter has light, medium, heavy values."""
        assert GameComplexityFilter.LIGHT == "light"
        assert GameComplexityFilter.MEDIUM == "medium"
        assert GameComplexityFilter.HEAVY == "heavy"

    def test_is_str_enum(self) -> None:
        """GameComplexityFilter members are strings for JSON serialization."""
        assert isinstance(GameComplexityFilter.LIGHT, str)


# ---------------------------------------------------------------------------
# GameCreate
# ---------------------------------------------------------------------------


class TestGameCreate:
    """Tests for GameCreate request schema."""

    def test_valid_minimal(self) -> None:
        """GameCreate accepts name-only input with defaults."""
        game = GameCreate(name="Catan")
        assert game.name == "Catan"
        assert game.publisher is None
        assert game.year_published is None
        assert game.edition is None
        assert game.min_players is None
        assert game.max_players is None
        assert game.description is None
        assert game.cover_image_url is None
        assert game.complexity is None
        assert game.tags == []

    def test_valid_full(self) -> None:
        """GameCreate accepts all fields populated."""
        game = GameCreate(
            name="Catan",
            publisher="Catan Studio",
            year_published=1995,
            edition="5th Edition",
            min_players=3,
            max_players=4,
            description="Trade and build.",
            cover_image_url="/images/catan.png",
            complexity="medium",
            tags=["strategy", "trading"],
        )
        assert game.name == "Catan"
        assert game.publisher == "Catan Studio"
        assert game.year_published == 1995
        assert game.min_players == 3
        assert game.max_players == 4
        assert game.complexity == GameComplexityFilter.MEDIUM
        assert game.tags == ["strategy", "trading"]

    def test_name_required(self) -> None:
        """GameCreate rejects missing name."""
        with pytest.raises(ValidationError) as exc_info:
            GameCreate()  # type: ignore[call-arg]
        assert "name" in str(exc_info.value)

    def test_name_empty_string_rejected(self) -> None:
        """GameCreate rejects empty string name."""
        with pytest.raises(ValidationError):
            GameCreate(name="")

    def test_name_max_length(self) -> None:
        """GameCreate rejects names longer than 500 characters."""
        with pytest.raises(ValidationError):
            GameCreate(name="x" * 501)

    def test_name_at_max_length(self) -> None:
        """GameCreate accepts name at exactly 500 characters."""
        game = GameCreate(name="x" * 500)
        assert len(game.name) == 500

    def test_year_published_must_be_positive(self) -> None:
        """GameCreate rejects zero or negative year_published."""
        with pytest.raises(ValidationError):
            GameCreate(name="Test", year_published=0)
        with pytest.raises(ValidationError):
            GameCreate(name="Test", year_published=-1)

    def test_min_players_must_be_positive(self) -> None:
        """GameCreate rejects zero or negative min_players."""
        with pytest.raises(ValidationError):
            GameCreate(name="Test", min_players=0, max_players=4)

    def test_max_players_must_be_positive(self) -> None:
        """GameCreate rejects zero or negative max_players."""
        with pytest.raises(ValidationError):
            GameCreate(name="Test", min_players=1, max_players=0)

    def test_min_max_players_must_be_provided_together(self) -> None:
        """GameCreate rejects min_players without max_players."""
        with pytest.raises(ValidationError, match="together"):
            GameCreate(name="Test", min_players=2)
        with pytest.raises(ValidationError, match="together"):
            GameCreate(name="Test", max_players=4)

    def test_min_players_must_not_exceed_max(self) -> None:
        """GameCreate rejects min_players > max_players."""
        with pytest.raises(ValidationError, match="min_players must be <= max_players"):
            GameCreate(name="Test", min_players=5, max_players=3)

    def test_tags_default_empty(self) -> None:
        """GameCreate defaults tags to an empty list."""
        game = GameCreate(name="Test")
        assert game.tags == []

    def test_tags_max_20(self) -> None:
        """GameCreate rejects more than 20 tags."""
        tags = [f"tag-{i}" for i in range(21)]
        with pytest.raises(ValidationError):
            GameCreate(name="Test", tags=tags)

    def test_tags_normalised(self) -> None:
        """GameCreate lowercases, trims, and deduplicates tags."""
        game = GameCreate(name="Test", tags=["Strategy", " TRADING ", "strategy"])
        assert game.tags == ["strategy", "trading"]

    def test_tags_empty_strings_removed(self) -> None:
        """GameCreate removes empty strings from tags after trimming."""
        game = GameCreate(name="Test", tags=["valid", "  ", ""])
        assert game.tags == ["valid"]

    def test_invalid_complexity_rejected(self) -> None:
        """GameCreate rejects invalid complexity values."""
        with pytest.raises(ValidationError):
            GameCreate(name="Test", complexity="extreme")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GameUpdate
# ---------------------------------------------------------------------------


class TestGameUpdate:
    """Tests for GameUpdate (PATCH) request schema."""

    def test_all_fields_default_to_unset(self) -> None:
        """GameUpdate defaults all fields to UNSET sentinel."""
        update = GameUpdate()
        assert update.name is UNSET
        assert update.publisher is UNSET
        assert update.year_published is UNSET
        assert update.edition is UNSET
        assert update.min_players is UNSET
        assert update.max_players is UNSET
        assert update.description is UNSET
        assert update.cover_image_url is UNSET
        assert update.complexity is UNSET
        assert update.tags is UNSET

    def test_partial_update(self) -> None:
        """GameUpdate allows setting only specific fields."""
        update = GameUpdate.model_validate({"description": "New description"})
        assert update.description == "New description"
        assert update.name is UNSET
        assert update.publisher is UNSET

    def test_null_field_distinguished_from_unset(self) -> None:
        """GameUpdate distinguishes null (clear) from absent (no change)."""
        update = GameUpdate.model_validate({"publisher": None})
        assert update.publisher is None  # explicitly set to null
        assert update.description is UNSET  # not provided

    def test_name_empty_string_rejected(self) -> None:
        """GameUpdate rejects empty-string name."""
        with pytest.raises(ValidationError, match="name must not be empty"):
            GameUpdate.model_validate({"name": ""})

    def test_name_whitespace_only_rejected(self) -> None:
        """GameUpdate rejects whitespace-only name."""
        with pytest.raises(ValidationError, match="name must not be empty"):
            GameUpdate.model_validate({"name": "   "})

    def test_name_over_500_rejected(self) -> None:
        """GameUpdate rejects name longer than 500 characters."""
        with pytest.raises(ValidationError, match="at most 500"):
            GameUpdate.model_validate({"name": "x" * 501})

    def test_min_players_must_be_positive(self) -> None:
        """GameUpdate rejects non-positive min_players."""
        with pytest.raises(ValidationError, match="positive integer"):
            GameUpdate.model_validate({"min_players": 0})

    def test_max_players_must_be_positive(self) -> None:
        """GameUpdate rejects non-positive max_players."""
        with pytest.raises(ValidationError, match="positive integer"):
            GameUpdate.model_validate({"max_players": -1})

    def test_min_must_not_exceed_max(self) -> None:
        """GameUpdate rejects min_players > max_players."""
        with pytest.raises(ValidationError, match="min_players must be <= max_players"):
            GameUpdate.model_validate({"min_players": 5, "max_players": 3})

    def test_year_must_be_positive(self) -> None:
        """GameUpdate rejects non-positive year_published."""
        with pytest.raises(ValidationError, match="positive integer"):
            GameUpdate.model_validate({"year_published": 0})

    def test_tags_max_20(self) -> None:
        """GameUpdate rejects more than 20 tags."""
        tags = [f"tag-{i}" for i in range(21)]
        with pytest.raises(ValidationError, match="20 tags"):
            GameUpdate.model_validate({"tags": tags})

    def test_tags_normalised(self) -> None:
        """GameUpdate lowercases and deduplicates tags."""
        update = GameUpdate.model_validate({"tags": ["Strategy", "strategy"]})
        assert update.tags == ["strategy"]

    def test_tags_null_means_clear(self) -> None:
        """GameUpdate allows null tags to signal clearing all tags."""
        update = GameUpdate.model_validate({"tags": None})
        assert update.tags is None

    def test_get_set_fields_excludes_unset(self) -> None:
        """get_set_fields returns only explicitly provided fields."""
        update = GameUpdate.model_validate({"name": "Updated", "publisher": None})
        fields = update.get_set_fields()
        assert fields == {"name": "Updated", "publisher": None}
        assert "description" not in fields
        assert "tags" not in fields

    def test_get_set_fields_empty_when_no_fields(self) -> None:
        """get_set_fields returns empty dict when nothing provided."""
        update = GameUpdate()
        assert update.get_set_fields() == {}

    def test_complexity_accepts_valid_value(self) -> None:
        """GameUpdate accepts valid complexity enum value."""
        update = GameUpdate.model_validate({"complexity": "heavy"})
        assert update.complexity == GameComplexityFilter.HEAVY

    def test_complexity_null_clears(self) -> None:
        """GameUpdate allows null complexity to clear the field."""
        update = GameUpdate.model_validate({"complexity": None})
        assert update.complexity is None


# ---------------------------------------------------------------------------
# GameResponse
# ---------------------------------------------------------------------------


def _game_response_data() -> dict[str, object]:
    """Build valid GameResponse input data for testing."""
    return {
        "id": uuid.uuid4(),
        "created_by": uuid.uuid4(),
        "name": "Catan",
        "publisher": "Catan Studio",
        "year_published": 1995,
        "edition": "5th Edition",
        "min_players": 3,
        "max_players": 4,
        "description": "A game about trading.",
        "cover_image_url": None,
        "complexity": "medium",
        "tags": ["strategy"],
        "archived_at": None,
        "is_active": True,
        "created_at": datetime.now(tz=UTC),
        "updated_at": datetime.now(tz=UTC),
    }


class TestGameResponse:
    """Tests for GameResponse schema."""

    def test_valid_response(self) -> None:
        """GameResponse accepts valid game data."""
        data = _game_response_data()
        response = GameResponse(**data)  # type: ignore[arg-type]
        assert response.name == "Catan"
        assert response.is_active is True
        assert response.complexity == GameComplexityFilter.MEDIUM

    def test_nullable_fields(self) -> None:
        """GameResponse allows null for optional fields."""
        data = _game_response_data()
        data["publisher"] = None
        data["year_published"] = None
        data["complexity"] = None
        data["archived_at"] = None
        response = GameResponse(**data)  # type: ignore[arg-type]
        assert response.publisher is None
        assert response.complexity is None

    def test_serialization(self) -> None:
        """GameResponse serializes to dict with all expected keys."""
        data = _game_response_data()
        response = GameResponse(**data)  # type: ignore[arg-type]
        dumped = response.model_dump()
        expected_keys = {
            "id",
            "created_by",
            "name",
            "publisher",
            "year_published",
            "edition",
            "min_players",
            "max_players",
            "description",
            "cover_image_url",
            "complexity",
            "tags",
            "archived_at",
            "is_active",
            "created_at",
            "updated_at",
        }
        assert set(dumped.keys()) == expected_keys


# ---------------------------------------------------------------------------
# GameSummaryResponse
# ---------------------------------------------------------------------------


class TestGameSummaryResponse:
    """Tests for GameSummaryResponse schema."""

    def test_includes_counts(self) -> None:
        """GameSummaryResponse includes document_count and expansion_count."""
        data = _game_response_data()
        data["document_count"] = 3
        data["expansion_count"] = 2
        response = GameSummaryResponse(**data)  # type: ignore[arg-type]
        assert response.document_count == 3
        assert response.expansion_count == 2
        assert response.name == "Catan"

    def test_inherits_game_response_fields(self) -> None:
        """GameSummaryResponse inherits all GameResponse fields."""
        data = _game_response_data()
        data["document_count"] = 0
        data["expansion_count"] = 0
        response = GameSummaryResponse(**data)  # type: ignore[arg-type]
        assert response.is_active is True
        assert isinstance(response.id, uuid.UUID)


# ---------------------------------------------------------------------------
# GameDetailResponse
# ---------------------------------------------------------------------------


def _expansion_response_data(game_id: uuid.UUID | None = None) -> dict[str, object]:
    """Build valid ExpansionResponse input data for testing."""
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


class TestGameDetailResponse:
    """Tests for GameDetailResponse schema."""

    def test_includes_expansions_list(self) -> None:
        """GameDetailResponse includes inline expansions list."""
        game_id = uuid.uuid4()
        data = _game_response_data()
        data["id"] = game_id
        data["document_count"] = 3
        data["expansion_count"] = 1
        data["expansions"] = [_expansion_response_data(game_id)]
        response = GameDetailResponse(**data)  # type: ignore[arg-type]
        assert len(response.expansions) == 1
        assert response.expansions[0].name == "Seafarers"
        assert response.document_count == 3

    def test_empty_expansions_list(self) -> None:
        """GameDetailResponse accepts empty expansions list."""
        data = _game_response_data()
        data["document_count"] = 0
        data["expansion_count"] = 0
        data["expansions"] = []
        response = GameDetailResponse(**data)  # type: ignore[arg-type]
        assert response.expansions == []


# ---------------------------------------------------------------------------
# GameFilters
# ---------------------------------------------------------------------------


class TestGameFilters:
    """Tests for GameFilters query parameter schema."""

    def test_defaults(self) -> None:
        """GameFilters defaults: no search, no filters, sort by name."""
        filters = GameFilters()
        assert filters.search is None
        assert filters.player_count is None
        assert filters.complexity is None
        assert filters.tags is None
        assert filters.archived is False
        assert filters.sort == "name"

    def test_parses_comma_separated_complexity(self) -> None:
        """GameFilters parses comma-separated complexity string."""
        filters = GameFilters.model_validate({"complexity": "light,medium"})
        assert filters.complexity == [
            GameComplexityFilter.LIGHT,
            GameComplexityFilter.MEDIUM,
        ]

    def test_parses_comma_separated_tags(self) -> None:
        """GameFilters parses comma-separated tags string."""
        filters = GameFilters.model_validate({"tags": "strategy,cooperative"})
        assert filters.tags == ["strategy", "cooperative"]

    def test_tags_lowercased(self) -> None:
        """GameFilters lowercases tag values."""
        filters = GameFilters.model_validate({"tags": "Strategy,COOPERATIVE"})
        assert filters.tags == ["strategy", "cooperative"]

    def test_complexity_as_list_passthrough(self) -> None:
        """GameFilters accepts complexity as a pre-parsed list."""
        filters = GameFilters.model_validate({"complexity": ["light", "heavy"]})
        assert filters.complexity == [
            GameComplexityFilter.LIGHT,
            GameComplexityFilter.HEAVY,
        ]

    def test_player_count_positive(self) -> None:
        """GameFilters rejects non-positive player_count."""
        with pytest.raises(ValidationError):
            GameFilters.model_validate({"player_count": 0})

    def test_player_count_valid(self) -> None:
        """GameFilters accepts positive player_count."""
        filters = GameFilters.model_validate({"player_count": 4})
        assert filters.player_count == 4

    def test_archived_true(self) -> None:
        """GameFilters accepts archived=True."""
        filters = GameFilters.model_validate({"archived": True})
        assert filters.archived is True

    def test_sort_custom(self) -> None:
        """GameFilters accepts custom sort values."""
        filters = GameFilters.model_validate({"sort": "-updated_at"})
        assert filters.sort == "-updated_at"

    def test_empty_complexity_string_produces_none(self) -> None:
        """GameFilters handles empty complexity string gracefully."""
        filters = GameFilters.model_validate({"complexity": ""})
        # Empty string splits to empty list, which becomes []
        assert filters.complexity == []

    def test_invalid_complexity_value_rejected(self) -> None:
        """GameFilters rejects invalid complexity values."""
        with pytest.raises(ValidationError):
            GameFilters.model_validate({"complexity": "extreme"})


# ---------------------------------------------------------------------------
# TagCountResponse
# ---------------------------------------------------------------------------


class TestTagCountResponse:
    """Tests for TagCountResponse schema."""

    def test_valid(self) -> None:
        """TagCountResponse stores tag and count."""
        resp = TagCountResponse(tag="strategy", count=8)
        assert resp.tag == "strategy"
        assert resp.count == 8

    def test_serialization(self) -> None:
        """TagCountResponse serializes to expected shape."""
        resp = TagCountResponse(tag="cooperative", count=5)
        dumped = resp.model_dump()
        assert dumped == {"tag": "cooperative", "count": 5}

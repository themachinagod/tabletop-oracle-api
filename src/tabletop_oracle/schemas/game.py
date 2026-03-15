"""Pydantic schemas for Game and GameTag resources.

Covers request schemas (create/update), response schemas (single, summary,
detail), filter/query schemas, and the tag count response. All field
constraints and types match the design doc (EPIC-001) and F001 conventions.
"""

from __future__ import annotations

import uuid  # noqa: TC003
from datetime import datetime  # noqa: TC003
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tabletop_oracle.schemas.sentinel import UNSET, _UnsetType


class GameComplexityFilter(StrEnum):
    """Game complexity levels for API schema use.

    Mirrors ``models.enums.GameComplexity`` values but as a StrEnum for
    Pydantic schema serialization (string values in JSON).
    """

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class GameCreate(BaseModel):
    """Request schema for creating a game.

    Attributes:
        name: Game title. Required, non-empty, max 500 characters.
        publisher: Publisher name (optional).
        year_published: Publication year. Positive integer if provided.
        edition: Edition descriptor (optional).
        min_players: Minimum player count. Positive integer if provided.
        max_players: Maximum player count. Positive integer if provided.
        description: Free-text description (optional).
        cover_image_url: Blob store path to cover image (optional).
        complexity: Complexity classification (optional).
        tags: Tag strings. Lowercased, trimmed, deduplicated. Max 20 items.
    """

    name: str = Field(..., min_length=1, max_length=500)
    publisher: str | None = None
    year_published: int | None = Field(None, gt=0)
    edition: str | None = None
    min_players: int | None = Field(None, gt=0)
    max_players: int | None = Field(None, gt=0)
    description: str | None = None
    cover_image_url: str | None = None
    complexity: GameComplexityFilter | None = None
    tags: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_player_counts(self) -> GameCreate:
        """Ensure min_players <= max_players when both are provided.

        If only one of min_players/max_players is provided, both must be
        provided per the design spec.
        """
        if (self.min_players is None) != (self.max_players is None):
            msg = "Both min_players and max_players must be provided together."
            raise ValueError(msg)
        if (
            self.min_players is not None
            and self.max_players is not None
            and self.min_players > self.max_players
        ):
            msg = "min_players must be <= max_players."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def normalise_tags(self) -> GameCreate:
        """Lowercase, trim, and deduplicate tags while preserving order."""
        if self.tags:
            seen: set[str] = set()
            normalised: list[str] = []
            for tag in self.tags:
                clean = tag.strip().lower()
                if clean and clean not in seen:
                    seen.add(clean)
                    normalised.append(clean)
            self.tags = normalised
        return self


class GameUpdate(BaseModel):
    """Request schema for partial game update (PATCH).

    Uses the UNSET sentinel as defaults so the service layer can distinguish
    "field not sent" (UNSET) from "field explicitly set to null" (None).

    Attributes:
        name: Game title. Non-empty, max 500 chars if provided.
        publisher: Publisher name (nullable).
        year_published: Publication year (nullable).
        edition: Edition descriptor (nullable).
        min_players: Minimum player count (nullable).
        max_players: Maximum player count (nullable).
        description: Free-text description (nullable).
        cover_image_url: Cover image URL (nullable).
        complexity: Complexity classification (nullable).
        tags: Full replacement tag list (nullable — null removes all tags).
    """

    name: str | _UnsetType = Field(UNSET)
    publisher: str | None | _UnsetType = UNSET
    year_published: int | None | _UnsetType = UNSET
    edition: str | None | _UnsetType = UNSET
    min_players: int | None | _UnsetType = UNSET
    max_players: int | None | _UnsetType = UNSET
    description: str | None | _UnsetType = UNSET
    cover_image_url: str | None | _UnsetType = UNSET
    complexity: GameComplexityFilter | None | _UnsetType = UNSET
    tags: list[str] | None | _UnsetType = UNSET

    @model_validator(mode="after")
    def validate_name_not_empty(self) -> GameUpdate:
        """Reject empty-string name updates."""
        if isinstance(self.name, str) and not self.name.strip():
            msg = "name must not be empty."
            raise ValueError(msg)
        if isinstance(self.name, str) and len(self.name) > 500:
            msg = "name must be at most 500 characters."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_player_counts(self) -> GameUpdate:
        """Validate min_players <= max_players when both are provided."""
        min_p = self.min_players
        max_p = self.max_players
        if isinstance(min_p, int) and isinstance(max_p, int) and min_p > max_p:
            msg = "min_players must be <= max_players."
            raise ValueError(msg)
        if isinstance(min_p, int) and min_p <= 0:
            msg = "min_players must be a positive integer."
            raise ValueError(msg)
        if isinstance(max_p, int) and max_p <= 0:
            msg = "max_players must be a positive integer."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_year(self) -> GameUpdate:
        """Validate year_published is positive when provided."""
        if isinstance(self.year_published, int) and self.year_published <= 0:
            msg = "year_published must be a positive integer."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def normalise_tags(self) -> GameUpdate:
        """Lowercase, trim, and deduplicate tags if provided."""
        if isinstance(self.tags, list):
            if len(self.tags) > 20:
                msg = "Maximum 20 tags per game."
                raise ValueError(msg)
            seen: set[str] = set()
            normalised: list[str] = []
            for tag in self.tags:
                clean = tag.strip().lower()
                if clean and clean not in seen:
                    seen.add(clean)
                    normalised.append(clean)
            self.tags = normalised
        return self

    def get_set_fields(self) -> dict[str, Any]:
        """Return a dict of only the fields that were explicitly provided.

        Fields still set to UNSET are excluded. This enables the service
        layer to apply only the provided changes.
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
            if getattr(self, field_name) is not UNSET
        }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class GameResponse(BaseModel):
    """Response schema for a single game resource.

    Includes all game metadata, computed is_active flag, and tag list.
    Used as the base for summary and detail response variants.

    Attributes:
        id: Game UUID.
        created_by: UUID of the curator who created the game.
        name: Game title.
        publisher: Publisher name.
        year_published: Publication year.
        edition: Edition descriptor.
        min_players: Minimum player count.
        max_players: Maximum player count.
        description: Free-text description.
        cover_image_url: Cover image path.
        complexity: Complexity classification.
        tags: List of tag strings.
        archived_at: Soft-delete timestamp (null if active).
        is_active: Convenience boolean derived from archived_at.
        created_at: Record creation timestamp.
        updated_at: Record last-modified timestamp.
    """

    id: uuid.UUID
    created_by: uuid.UUID
    name: str
    publisher: str | None
    year_published: int | None
    edition: str | None
    min_players: int | None
    max_players: int | None
    description: str | None
    cover_image_url: str | None
    complexity: GameComplexityFilter | None
    tags: list[str]
    archived_at: datetime | None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class GameSummaryResponse(GameResponse):
    """Response schema for game list items with aggregated counts.

    Extends GameResponse with document_count and expansion_count for
    efficient rendering of game lists without additional API calls.

    Attributes:
        document_count: Count of processed documents for this game.
        expansion_count: Count of active (non-archived) expansions.
    """

    document_count: int
    expansion_count: int


class GameDetailResponse(GameResponse):
    """Response schema for game detail with inline expansions.

    Extends GameResponse with aggregated counts and the full list of
    expansions for rendering the game detail view.

    Attributes:
        document_count: Count of processed documents for this game.
        expansion_count: Count of active (non-archived) expansions.
        expansions: Inline list of expansion responses.
    """

    document_count: int
    expansion_count: int
    expansions: list[ExpansionResponse]


# ---------------------------------------------------------------------------
# Filter / query schemas
# ---------------------------------------------------------------------------


class GameFilters(BaseModel):
    """Parsed query parameters for the game list endpoint.

    Parses comma-separated values for complexity and tags. Designed to
    be constructed from FastAPI query parameter dependencies.

    Attributes:
        search: Full-text search on game name.
        player_count: Filter games supporting this player count.
        complexity: Complexity values to include (OR logic).
        tags: Tags the game must have (AND logic).
        archived: Whether to include archived games.
        sort: Sort field with optional - prefix for descending.
    """

    search: str | None = None
    player_count: int | None = Field(None, gt=0)
    complexity: list[GameComplexityFilter] | None = None
    tags: list[str] | None = None
    archived: bool = False
    sort: str = "name"

    @model_validator(mode="before")
    @classmethod
    def parse_comma_separated(cls, data: Any) -> Any:
        """Parse comma-separated strings for complexity and tags fields.

        Allows API callers to pass ``?complexity=light,medium`` and
        ``?tags=strategy,cooperative`` as query parameter strings.
        """
        if isinstance(data, dict):
            raw_complexity = data.get("complexity")
            if isinstance(raw_complexity, str):
                data["complexity"] = [v.strip() for v in raw_complexity.split(",") if v.strip()]
            raw_tags = data.get("tags")
            if isinstance(raw_tags, str):
                data["tags"] = [v.strip().lower() for v in raw_tags.split(",") if v.strip()]
        return data


class TagCountResponse(BaseModel):
    """Tag name with its usage count across active games.

    Attributes:
        tag: The tag string.
        count: Number of active games using this tag.
    """

    tag: str
    count: int


# ---------------------------------------------------------------------------
# Forward reference resolution — must come after ExpansionResponse import
# ---------------------------------------------------------------------------

# Import here to resolve the forward reference in GameDetailResponse.
# ExpansionResponse is defined in the expansion module but needed inline.
from tabletop_oracle.schemas.expansion import ExpansionResponse  # noqa: E402, TC001

GameDetailResponse.model_rebuild()

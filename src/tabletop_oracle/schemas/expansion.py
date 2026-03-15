"""Pydantic schemas for Expansion resources.

Covers request schemas (create/update), response schemas (single, list with
document count). All field constraints match the design doc (EPIC-001).
"""

from __future__ import annotations

import uuid  # noqa: TC003
from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tabletop_oracle.schemas.sentinel import UNSET, _UnsetType

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ExpansionCreate(BaseModel):
    """Request schema for creating an expansion under a game.

    Attributes:
        name: Expansion title. Required, non-empty, max 500 characters.
        year_published: Publication year. Positive integer if provided.
        description: Free-text description (optional).
    """

    name: str = Field(..., min_length=1, max_length=500)
    year_published: int | None = Field(None, gt=0)
    description: str | None = None


class ExpansionUpdate(BaseModel):
    """Request schema for partial expansion update (PATCH).

    Uses the UNSET sentinel as defaults so the service layer can distinguish
    "field not sent" from "field explicitly set to null".

    Attributes:
        name: Expansion title. Non-empty, max 500 chars if provided.
        year_published: Publication year (nullable).
        description: Free-text description (nullable).
    """

    name: str | _UnsetType = Field(UNSET)
    year_published: int | None | _UnsetType = UNSET
    description: str | None | _UnsetType = UNSET

    @model_validator(mode="after")
    def validate_name_not_empty(self) -> ExpansionUpdate:
        """Reject empty-string name updates."""
        if isinstance(self.name, str) and not self.name.strip():
            msg = "name must not be empty."
            raise ValueError(msg)
        if isinstance(self.name, str) and len(self.name) > 500:
            msg = "name must be at most 500 characters."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_year(self) -> ExpansionUpdate:
        """Validate year_published is positive when provided."""
        if isinstance(self.year_published, int) and self.year_published <= 0:
            msg = "year_published must be a positive integer."
            raise ValueError(msg)
        return self

    def get_set_fields(self) -> dict[str, Any]:
        """Return a dict of only the fields that were explicitly provided.

        Fields still set to UNSET are excluded.
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
            if getattr(self, field_name) is not UNSET
        }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ExpansionResponse(BaseModel):
    """Response schema for a single expansion resource.

    Attributes:
        id: Expansion UUID.
        game_id: UUID of the parent game.
        name: Expansion title.
        year_published: Publication year.
        description: Free-text description.
        archived_at: Soft-delete timestamp (null if active).
        created_at: Record creation timestamp.
        updated_at: Record last-modified timestamp.
    """

    id: uuid.UUID
    game_id: uuid.UUID
    name: str
    year_published: int | None
    description: str | None
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExpansionListResponse(ExpansionResponse):
    """Response schema for expansion list items with document count.

    Extends ExpansionResponse with the count of processed documents
    associated with this expansion.

    Attributes:
        document_count: Count of processed documents for this expansion.
    """

    document_count: int

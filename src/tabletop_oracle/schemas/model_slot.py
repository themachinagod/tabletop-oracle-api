"""Pydantic schemas for model slot admin API.

Request and response schemas for the ``/api/v1/admin/model-slots``
endpoints. The update schema uses UNSET sentinels for partial update
semantics -- only fields explicitly provided in the request body are
applied.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

from pydantic import BaseModel, Field

from tabletop_oracle.schemas.sentinel import UNSET, _UnsetType


class ModelSlotResponse(BaseModel):
    """Response schema for a single model slot.

    Attributes:
        capability: The AI capability this slot serves.
        provider: Model provider identifier (e.g. 'openai', 'anthropic').
        model_id: Provider-specific model identifier.
        max_tokens_per_call: Maximum tokens per API call, or null.
        temperature: Sampling temperature, or null.
        fallback_provider: Fallback provider, or null.
        fallback_model_id: Fallback model identifier, or null.
        updated_at: Last modification timestamp.
    """

    capability: str
    provider: str
    model_id: str
    max_tokens_per_call: int | None = None
    temperature: float | None = None
    fallback_provider: str | None = None
    fallback_model_id: str | None = None
    updated_at: datetime | None = None


class ModelSlotUpdate(BaseModel):
    """Request schema for updating a model slot (partial update).

    All fields use the UNSET sentinel as default. Only fields explicitly
    provided in the request body are applied to the database row.

    Attributes:
        provider: New model provider identifier.
        model_id: New provider-specific model identifier.
        max_tokens_per_call: New max tokens per call (null to unset).
        temperature: New sampling temperature (null to unset).
        fallback_provider: New fallback provider (null to unset).
        fallback_model_id: New fallback model identifier (null to unset).
    """

    provider: str | _UnsetType = Field(default=UNSET)
    model_id: str | _UnsetType = Field(default=UNSET)
    max_tokens_per_call: int | None | _UnsetType = Field(default=UNSET)
    temperature: float | None | _UnsetType = Field(default=UNSET)
    fallback_provider: str | None | _UnsetType = Field(default=UNSET)
    fallback_model_id: str | None | _UnsetType = Field(default=UNSET)

    def to_update_dict(self) -> dict[str, object]:
        """Extract only the fields that were explicitly provided.

        Returns:
            Dict of field names to values, excluding UNSET fields.
        """
        result: dict[str, object] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if value is not UNSET:
                result[field_name] = value
        return result

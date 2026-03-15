"""Pydantic schemas for guardrail config admin API.

Request and response schemas for the ``/api/v1/admin/guardrail-config``
endpoints. The update schema uses UNSET sentinels for partial update
semantics -- only fields explicitly provided in the request body are
applied.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

from pydantic import BaseModel, Field

from tabletop_oracle.schemas.sentinel import UNSET, _UnsetType


class GuardrailConfigResponse(BaseModel):
    """Response schema for guardrail configuration.

    Attributes:
        enforcement_enabled: Master switch for guardrail enforcement.
        max_tokens_per_query: Per-query token ceiling, or null.
        max_model_calls_per_query: Per-query model call ceiling, or null.
        max_queries_per_session: Per-session query ceiling, or null.
        daily_token_budget: Daily token budget, or null.
        daily_query_budget: Daily query budget, or null.
        per_document_ingestion_limit: Per-document ingestion token limit, or null.
        updated_at: Last modification timestamp.
    """

    enforcement_enabled: bool
    max_tokens_per_query: int | None = None
    max_model_calls_per_query: int | None = None
    max_queries_per_session: int | None = None
    daily_token_budget: int | None = None
    daily_query_budget: int | None = None
    per_document_ingestion_limit: int | None = None
    updated_at: datetime | None = None


class GuardrailConfigUpdate(BaseModel):
    """Request schema for updating guardrail configuration (partial update).

    All fields use the UNSET sentinel as default. Only fields explicitly
    provided in the request body are applied to the database row.

    Attributes:
        enforcement_enabled: New master switch value.
        max_tokens_per_query: New per-query token ceiling (null to unset).
        max_model_calls_per_query: New per-query model call ceiling (null to unset).
        max_queries_per_session: New per-session query ceiling (null to unset).
        daily_token_budget: New daily token budget (null to unset).
        daily_query_budget: New daily query budget (null to unset).
        per_document_ingestion_limit: New per-document ingestion limit (null to unset).
    """

    enforcement_enabled: bool | _UnsetType = Field(default=UNSET)
    max_tokens_per_query: int | None | _UnsetType = Field(default=UNSET)
    max_model_calls_per_query: int | None | _UnsetType = Field(default=UNSET)
    max_queries_per_session: int | None | _UnsetType = Field(default=UNSET)
    daily_token_budget: int | None | _UnsetType = Field(default=UNSET)
    daily_query_budget: int | None | _UnsetType = Field(default=UNSET)
    per_document_ingestion_limit: int | None | _UnsetType = Field(default=UNSET)

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

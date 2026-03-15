"""Pydantic schemas for usage aggregation admin API.

Response schemas for the ``/api/v1/admin/usage`` endpoints. All schemas
are read-only (no update schemas) since usage data is derived from
aggregation queries over ``token_usage_log``.
"""

from __future__ import annotations

from datetime import date  # noqa: TC003
from enum import StrEnum

from pydantic import BaseModel, Field


class GuardrailThreshold(StrEnum):
    """Traffic-light status for a guardrail metric.

    Thresholds per design doc:
    - green: usage < 70% of limit
    - amber: usage >= 70% and < 90% of limit
    - red: usage >= 90% of limit
    - disabled: limit is NULL or enforcement_enabled is False
    """

    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    DISABLED = "disabled"


class UsageSummaryResponse(BaseModel):
    """Period summary totals for usage dashboard cards.

    Attributes:
        total_tokens: Combined input + output tokens in the period.
        total_queries: Number of token usage log entries with a session_id.
        total_documents_processed: Number of entries with a document_id.
        input_tokens: Total input tokens in the period.
        output_tokens: Total output tokens in the period.
        unique_users: Distinct user count in the period.
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
    """

    total_tokens: int = Field(ge=0)
    total_queries: int = Field(ge=0)
    total_documents_processed: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    unique_users: int = Field(ge=0)
    period_start: date
    period_end: date


class DailyUsageResponse(BaseModel):
    """Single day of usage for trend charts.

    Attributes:
        date: The calendar date.
        total_tokens: Combined input + output tokens on this date.
        query_count: Number of entries with a session_id.
        document_count: Number of entries with a document_id.
    """

    date: date
    total_tokens: int = Field(ge=0)
    query_count: int = Field(ge=0)
    document_count: int = Field(ge=0)


class CapabilityUsageResponse(BaseModel):
    """Usage breakdown for a single AI capability.

    Attributes:
        capability: The model capability enum value.
        total_tokens: Combined input + output tokens.
        call_count: Number of log entries.
        input_tokens: Total input tokens.
        output_tokens: Total output tokens.
    """

    capability: str
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class GameUsageResponse(BaseModel):
    """Usage breakdown for a single game.

    Attributes:
        game_id: UUID of the game.
        game_name: Display name of the game.
        query_tokens: Tokens from session-attributed entries.
        ingestion_tokens: Tokens from document-attributed entries.
        query_count: Number of session-attributed entries.
    """

    game_id: str
    game_name: str
    query_tokens: int = Field(ge=0)
    ingestion_tokens: int = Field(ge=0)
    query_count: int = Field(ge=0)


class UserUsageResponse(BaseModel):
    """Usage breakdown for a single user.

    Attributes:
        user_id: UUID of the user.
        display_name: User's display name.
        total_tokens: Combined input + output tokens.
        query_count: Number of log entries for this user.
    """

    user_id: str
    display_name: str
    total_tokens: int = Field(ge=0)
    query_count: int = Field(ge=0)


class GuardrailIndicator(BaseModel):
    """Status indicator for a single guardrail metric.

    Attributes:
        name: Human-readable guardrail name.
        current: Current usage value.
        limit: Configured limit (None if not set).
        status: Traffic-light status.
    """

    name: str
    current: int = Field(ge=0)
    limit: int | None = None
    status: GuardrailThreshold


class GuardrailStatusResponse(BaseModel):
    """Aggregate guardrail status for the dashboard.

    Attributes:
        enforcement_enabled: Whether guardrails are actively enforced.
        indicators: Per-guardrail status indicators.
    """

    enforcement_enabled: bool
    indicators: list[GuardrailIndicator]

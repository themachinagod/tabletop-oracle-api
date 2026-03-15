"""Unit tests for guardrail config admin API schemas.

Tests GuardrailConfigResponse serialization and GuardrailConfigUpdate
partial-update semantics with UNSET sentinels.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tabletop_oracle.schemas.guardrail_config import (
    GuardrailConfigResponse,
    GuardrailConfigUpdate,
)
from tabletop_oracle.schemas.sentinel import UNSET

# ---------------------------------------------------------------------------
# GuardrailConfigResponse
# ---------------------------------------------------------------------------


class TestGuardrailConfigResponse:
    """Tests for GuardrailConfigResponse serialization."""

    def test_full_response_serialization(self) -> None:
        """All fields serialize correctly."""
        ts = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
        response = GuardrailConfigResponse(
            enforcement_enabled=True,
            max_tokens_per_query=50000,
            max_model_calls_per_query=10,
            max_queries_per_session=100,
            daily_token_budget=1000000,
            daily_query_budget=5000,
            per_document_ingestion_limit=200000,
            updated_at=ts,
        )

        data = response.model_dump()

        assert data["enforcement_enabled"] is True
        assert data["max_tokens_per_query"] == 50000
        assert data["max_model_calls_per_query"] == 10
        assert data["max_queries_per_session"] == 100
        assert data["daily_token_budget"] == 1000000
        assert data["daily_query_budget"] == 5000
        assert data["per_document_ingestion_limit"] == 200000
        assert data["updated_at"] == ts

    def test_null_limits_serialize_as_none(self) -> None:
        """Optional limit fields default to None."""
        response = GuardrailConfigResponse(
            enforcement_enabled=False,
        )

        data = response.model_dump()

        assert data["max_tokens_per_query"] is None
        assert data["max_model_calls_per_query"] is None
        assert data["max_queries_per_session"] is None
        assert data["daily_token_budget"] is None
        assert data["daily_query_budget"] is None
        assert data["per_document_ingestion_limit"] is None
        assert data["updated_at"] is None


# ---------------------------------------------------------------------------
# GuardrailConfigUpdate
# ---------------------------------------------------------------------------


class TestGuardrailConfigUpdate:
    """Tests for GuardrailConfigUpdate partial-update semantics."""

    def test_all_fields_default_to_unset(self) -> None:
        """All fields are UNSET when no values are provided."""
        update = GuardrailConfigUpdate()

        assert update.enforcement_enabled is UNSET
        assert update.max_tokens_per_query is UNSET
        assert update.max_model_calls_per_query is UNSET
        assert update.max_queries_per_session is UNSET
        assert update.daily_token_budget is UNSET
        assert update.daily_query_budget is UNSET
        assert update.per_document_ingestion_limit is UNSET

    def test_to_update_dict_empty_when_all_unset(self) -> None:
        """to_update_dict returns empty dict when all fields are UNSET."""
        update = GuardrailConfigUpdate()

        assert update.to_update_dict() == {}

    def test_to_update_dict_includes_only_set_fields(self) -> None:
        """to_update_dict includes only explicitly provided fields."""
        update = GuardrailConfigUpdate(
            enforcement_enabled=False,
            daily_token_budget=2000000,
        )

        result = update.to_update_dict()

        assert result == {
            "enforcement_enabled": False,
            "daily_token_budget": 2000000,
        }
        assert "max_tokens_per_query" not in result

    def test_to_update_dict_includes_null_values(self) -> None:
        """Explicitly setting a field to None is included in the update."""
        update = GuardrailConfigUpdate(
            max_tokens_per_query=None,
        )

        result = update.to_update_dict()

        assert result == {"max_tokens_per_query": None}

    def test_to_update_dict_all_fields_set(self) -> None:
        """All fields included when all are explicitly provided."""
        update = GuardrailConfigUpdate(
            enforcement_enabled=True,
            max_tokens_per_query=100000,
            max_model_calls_per_query=20,
            max_queries_per_session=200,
            daily_token_budget=5000000,
            daily_query_budget=25000,
            per_document_ingestion_limit=500000,
        )

        result = update.to_update_dict()

        assert len(result) == 7
        assert result["enforcement_enabled"] is True
        assert result["max_tokens_per_query"] == 100000

    def test_from_json_partial(self) -> None:
        """Parsing from JSON dict correctly sets only provided fields."""
        update = GuardrailConfigUpdate.model_validate(
            {
                "enforcement_enabled": False,
            }
        )

        result = update.to_update_dict()

        assert result == {"enforcement_enabled": False}
        assert update.max_tokens_per_query is UNSET

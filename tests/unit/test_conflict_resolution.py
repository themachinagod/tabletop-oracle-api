"""Unit tests for ConflictResolutionService.

Tests document type priority ordering, authoritative source promotion,
source retention on conflict, audit logging, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tabletop_oracle.services.kg.conflicts import (
    PRIORITY_ORDER,
    ConflictResolutionService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    *,
    source_id: None = None,
    document_type: str = "core_rules",
    document_name: str = "Core Rulebook",
    document_id: None = None,
    is_authoritative: bool = True,
) -> MagicMock:
    """Build a mock KGConceptSource."""
    source = MagicMock()
    source.id = source_id or uuid4()
    source.document_type = document_type
    source.document_name = document_name
    source.document_id = document_id or uuid4()
    source.is_authoritative = is_authoritative
    return source


def _make_service() -> tuple[ConflictResolutionService, AsyncMock, AsyncMock]:
    """Create a service with mocked repositories."""
    source_repo = AsyncMock()
    audit_repo = AsyncMock()
    service = ConflictResolutionService(
        source_repo=source_repo,
        audit_repo=audit_repo,
    )
    return service, source_repo, audit_repo


# ---------------------------------------------------------------------------
# Priority Ordering Tests
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    """Tests for document type priority constants."""

    def test_errata_highest_priority(self) -> None:
        """Errata has the highest priority."""
        assert PRIORITY_ORDER["errata"] == 60
        assert PRIORITY_ORDER["errata"] > PRIORITY_ORDER["faq"]
        assert PRIORITY_ORDER["errata"] > PRIORITY_ORDER["core_rules"]

    def test_faq_above_core_rules(self) -> None:
        """FAQ has higher priority than core rules."""
        assert PRIORITY_ORDER["faq"] > PRIORITY_ORDER["core_rules"]

    def test_core_and_expansion_equal(self) -> None:
        """Core rules and expansion rules have equal priority."""
        assert PRIORITY_ORDER["core_rules"] == PRIORITY_ORDER["expansion_rules"]

    def test_strategy_below_rules(self) -> None:
        """Strategy guides have lower priority than rules."""
        assert PRIORITY_ORDER["strategy"] < PRIORITY_ORDER["core_rules"]

    def test_other_lowest(self) -> None:
        """'other' has the lowest defined priority."""
        assert PRIORITY_ORDER["other"] == min(PRIORITY_ORDER.values())


# ---------------------------------------------------------------------------
# Conflict Resolution Tests
# ---------------------------------------------------------------------------


class TestResolveConflicts:
    """Tests for resolve_conflicts."""

    @pytest.mark.asyncio
    async def test_single_source_no_conflict(self) -> None:
        """No conflict when only one source exists."""
        service, source_repo, audit_repo = _make_service()
        source = _make_source()
        source_repo.list_by_concept.return_value = [source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=uuid4(),
            new_source=source,
        )

        source_repo.update_authoritative.assert_not_called()
        audit_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_higher_priority_becomes_authoritative(self) -> None:
        """New source with higher priority supersedes current authoritative."""
        service, source_repo, _audit_repo = _make_service()
        concept_id = uuid4()
        game_id = uuid4()

        old_source = _make_source(
            document_type="core_rules",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="errata",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=game_id,
            concept_id=concept_id,
            new_source=new_source,
        )

        source_repo.update_authoritative.assert_called_once_with(concept_id, new_source.id)

    @pytest.mark.asyncio
    async def test_lower_priority_does_not_change_authoritative(self) -> None:
        """New source with lower priority does not replace authoritative."""
        service, source_repo, _audit_repo = _make_service()

        old_source = _make_source(
            document_type="errata",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="strategy",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=uuid4(),
            new_source=new_source,
        )

        source_repo.update_authoritative.assert_not_called()

    @pytest.mark.asyncio
    async def test_equal_priority_does_not_change_authoritative(self) -> None:
        """New source with equal priority does not replace authoritative."""
        service, source_repo, _audit_repo = _make_service()

        old_source = _make_source(
            document_type="core_rules",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="expansion_rules",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=uuid4(),
            new_source=new_source,
        )

        source_repo.update_authoritative.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_sources_retained(self) -> None:
        """Both sources are retained after conflict resolution."""
        service, source_repo, _audit_repo = _make_service()
        concept_id = uuid4()

        old_source = _make_source(
            document_type="core_rules",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="errata",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=concept_id,
            new_source=new_source,
        )

        # update_authoritative is called — not delete. Both are kept.
        source_repo.update_authoritative.assert_called_once_with(concept_id, new_source.id)
        # No delete calls anywhere
        source_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_conflict_logged_to_audit(self) -> None:
        """Conflict resolution event is logged to kg_audit_log."""
        service, source_repo, audit_repo = _make_service()
        concept_id = uuid4()
        game_id = uuid4()

        old_source = _make_source(
            document_type="core_rules",
            document_name="Core Rulebook",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="errata",
            document_name="Errata v2",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=game_id,
            concept_id=concept_id,
            new_source=new_source,
        )

        audit_repo.create.assert_called_once()
        call_kwargs = audit_repo.create.call_args[1]
        assert call_kwargs["game_id"] == game_id
        assert call_kwargs["event_type"] == "conflict_resolved"
        assert call_kwargs["concept_id"] == concept_id
        details = call_kwargs["details"]
        assert details["new_document_type"] == "errata"
        assert details["previous_document_type"] == "core_rules"

    @pytest.mark.asyncio
    async def test_no_audit_when_no_conflict(self) -> None:
        """No audit log entry when lower priority source added."""
        service, source_repo, audit_repo = _make_service()

        old_source = _make_source(
            document_type="errata",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="other",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=uuid4(),
            new_source=new_source,
        )

        audit_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_current_authoritative_promotes_new(self) -> None:
        """When no source is authoritative, new source gets promoted."""
        service, source_repo, _audit_repo = _make_service()
        concept_id = uuid4()

        source_a = _make_source(is_authoritative=False)
        new_source = _make_source(is_authoritative=False)
        source_repo.list_by_concept.return_value = [source_a, new_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=concept_id,
            new_source=new_source,
        )

        source_repo.update_authoritative.assert_called_once_with(concept_id, new_source.id)

    @pytest.mark.asyncio
    async def test_new_source_already_authoritative_noop(self) -> None:
        """No change when the new source is already authoritative."""
        service, source_repo, _audit_repo = _make_service()

        new_source = _make_source(
            document_type="errata",
            is_authoritative=True,
        )
        old_source = _make_source(
            document_type="core_rules",
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [new_source, old_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=uuid4(),
            new_source=new_source,
        )

        source_repo.update_authoritative.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_document_type_uses_default_priority(self) -> None:
        """Unknown document type falls back to default (lowest) priority."""
        service, source_repo, _audit_repo = _make_service()

        old_source = _make_source(
            document_type="strategy",
            is_authoritative=True,
        )
        new_source = _make_source(
            document_type="homebrew",  # Unknown type
            is_authoritative=False,
        )
        source_repo.list_by_concept.return_value = [old_source, new_source]

        await service.resolve_conflicts(
            game_id=uuid4(),
            concept_id=uuid4(),
            new_source=new_source,
        )

        # homebrew (default 10) < strategy (20), no change
        source_repo.update_authoritative.assert_not_called()

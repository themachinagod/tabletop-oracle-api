"""Unit tests for KGAuditService.

Tests typed logging methods for all seven event types, audit history
queries, and event counting. Each method delegates to the KGAuditRepository.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tabletop_oracle.services.kg.audit import (
    EVENT_ASSOCIATIONS_DISCOVERED,
    EVENT_CONFLICT_DETECTED,
    EVENT_CONFLICT_RESOLVED,
    EVENT_DOCUMENT_REMOVED,
    EVENT_EXTRACTION_COMPLETE,
    EVENT_GRAPH_REBUILT,
    EVENT_RETRIEVAL_EXECUTED,
    KGAuditService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> tuple[KGAuditService, AsyncMock]:
    """Build a KGAuditService with a mocked repository."""
    repo = AsyncMock()
    repo.create.return_value = MagicMock()
    service = KGAuditService(audit_repo=repo)
    return service, repo


# ---------------------------------------------------------------------------
# Tests: log_extraction
# ---------------------------------------------------------------------------


class TestLogExtraction:
    """Tests for KGAuditService.log_extraction."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_extraction should call repo.create with correct event type and details."""
        service, repo = _make_service()
        game_id = uuid4()
        doc_id = uuid4()

        await service.log_extraction(game_id, doc_id, concept_count=10, chunk_count=5)

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_EXTRACTION_COMPLETE,
            document_id=doc_id,
            details={"concept_count": 10, "chunk_count": 5},
        )

    @pytest.mark.asyncio
    async def test_returns_audit_entry(self) -> None:
        """log_extraction should return the entry from the repository."""
        service, repo = _make_service()
        expected = MagicMock()
        repo.create.return_value = expected

        result = await service.log_extraction(uuid4(), uuid4(), concept_count=3, chunk_count=1)

        assert result is expected


# ---------------------------------------------------------------------------
# Tests: log_association_discovery
# ---------------------------------------------------------------------------


class TestLogAssociationDiscovery:
    """Tests for KGAuditService.log_association_discovery."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_association_discovery should pass association and cross-doc counts."""
        service, repo = _make_service()
        game_id = uuid4()
        doc_id = uuid4()

        await service.log_association_discovery(
            game_id, doc_id, association_count=8, cross_document_count=3
        )

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_ASSOCIATIONS_DISCOVERED,
            document_id=doc_id,
            details={"association_count": 8, "cross_document_count": 3},
        )


# ---------------------------------------------------------------------------
# Tests: log_conflict_detected
# ---------------------------------------------------------------------------


class TestLogConflictDetected:
    """Tests for KGAuditService.log_conflict_detected."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_conflict_detected should pass concept_id and resolution details."""
        service, repo = _make_service()
        game_id = uuid4()
        concept_id = uuid4()
        existing_doc = uuid4()
        new_doc = uuid4()

        await service.log_conflict_detected(
            game_id,
            concept_id,
            existing_source_doc=existing_doc,
            new_source_doc=new_doc,
            resolution="new_source_higher_priority",
        )

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_CONFLICT_DETECTED,
            concept_id=concept_id,
            details={
                "existing_source_doc": str(existing_doc),
                "new_source_doc": str(new_doc),
                "resolution": "new_source_higher_priority",
            },
        )


# ---------------------------------------------------------------------------
# Tests: log_conflict_resolved
# ---------------------------------------------------------------------------


class TestLogConflictResolved:
    """Tests for KGAuditService.log_conflict_resolved."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_conflict_resolved should pass document types and UUIDs."""
        service, repo = _make_service()
        game_id = uuid4()
        concept_id = uuid4()
        new_doc = uuid4()
        prev_doc = uuid4()

        await service.log_conflict_resolved(
            game_id,
            concept_id,
            new_authoritative_doc=new_doc,
            previous_authoritative_doc=prev_doc,
            new_document_type="errata",
            previous_document_type="core_rules",
        )

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_CONFLICT_RESOLVED,
            concept_id=concept_id,
            document_id=new_doc,
            details={
                "new_authoritative_doc": str(new_doc),
                "previous_authoritative_doc": str(prev_doc),
                "new_document_type": "errata",
                "previous_document_type": "core_rules",
            },
        )


# ---------------------------------------------------------------------------
# Tests: log_document_removal
# ---------------------------------------------------------------------------


class TestLogDocumentRemoval:
    """Tests for KGAuditService.log_document_removal."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_document_removal should pass removal and update counts."""
        service, repo = _make_service()
        game_id = uuid4()
        doc_id = uuid4()

        await service.log_document_removal(game_id, doc_id, concepts_removed=4, concepts_updated=2)

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_DOCUMENT_REMOVED,
            document_id=doc_id,
            details={"concepts_removed": 4, "concepts_updated": 2},
        )


# ---------------------------------------------------------------------------
# Tests: log_graph_rebuilt
# ---------------------------------------------------------------------------


class TestLogGraphRebuilt:
    """Tests for KGAuditService.log_graph_rebuilt."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_graph_rebuilt should pass rebuild summary counts."""
        service, repo = _make_service()
        game_id = uuid4()

        await service.log_graph_rebuilt(
            game_id,
            document_count=5,
            concept_count=120,
            association_count=85,
        )

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_GRAPH_REBUILT,
            details={
                "document_count": 5,
                "concept_count": 120,
                "association_count": 85,
            },
        )

    @pytest.mark.asyncio
    async def test_no_document_id(self) -> None:
        """log_graph_rebuilt should not pass document_id (game-level event)."""
        service, repo = _make_service()

        await service.log_graph_rebuilt(
            uuid4(),
            document_count=1,
            concept_count=10,
            association_count=5,
        )

        call_kwargs = repo.create.call_args[1]
        assert "document_id" not in call_kwargs


# ---------------------------------------------------------------------------
# Tests: log_retrieval
# ---------------------------------------------------------------------------


class TestLogRetrieval:
    """Tests for KGAuditService.log_retrieval."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """log_retrieval should pass query details."""
        service, repo = _make_service()
        game_id = uuid4()

        await service.log_retrieval(
            game_id,
            query_text="How does initiative work?",
            results_count=7,
            expansion_filter_active=True,
        )

        repo.create.assert_awaited_once_with(
            game_id=game_id,
            event_type=EVENT_RETRIEVAL_EXECUTED,
            details={
                "query_text": "How does initiative work?",
                "results_count": 7,
                "expansion_filter_active": True,
            },
        )

    @pytest.mark.asyncio
    async def test_no_expansion_filter(self) -> None:
        """log_retrieval with expansion_filter_active=False records correctly."""
        service, repo = _make_service()

        await service.log_retrieval(
            uuid4(),
            query_text="What are the combat rules?",
            results_count=3,
            expansion_filter_active=False,
        )

        call_kwargs = repo.create.call_args[1]
        assert call_kwargs["details"]["expansion_filter_active"] is False


# ---------------------------------------------------------------------------
# Tests: get_audit_history
# ---------------------------------------------------------------------------


class TestGetAuditHistory:
    """Tests for KGAuditService.get_audit_history."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """get_audit_history should delegate to repo.list_by_game."""
        service, repo = _make_service()
        game_id = uuid4()
        expected = [MagicMock(), MagicMock()]
        repo.list_by_game.return_value = expected

        result = await service.get_audit_history(game_id, limit=10, offset=5)

        repo.list_by_game.assert_awaited_once_with(
            game_id,
            event_type=None,
            document_id=None,
            limit=10,
            offset=5,
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_with_filters(self) -> None:
        """get_audit_history should forward event_type and document_id filters."""
        service, repo = _make_service()
        game_id = uuid4()
        doc_id = uuid4()
        repo.list_by_game.return_value = []

        await service.get_audit_history(
            game_id,
            event_type="extraction_complete",
            document_id=doc_id,
        )

        repo.list_by_game.assert_awaited_once_with(
            game_id,
            event_type="extraction_complete",
            document_id=doc_id,
            limit=50,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_default_pagination(self) -> None:
        """get_audit_history uses default limit=50, offset=0."""
        service, repo = _make_service()
        repo.list_by_game.return_value = []

        await service.get_audit_history(uuid4())

        call_kwargs = repo.list_by_game.call_args[1]
        assert call_kwargs["limit"] == 50
        assert call_kwargs["offset"] == 0


# ---------------------------------------------------------------------------
# Tests: get_event_count
# ---------------------------------------------------------------------------


class TestGetEventCount:
    """Tests for KGAuditService.get_event_count."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        """get_event_count should delegate to repo.count_by_game."""
        service, repo = _make_service()
        game_id = uuid4()
        repo.count_by_game.return_value = 15

        result = await service.get_event_count(game_id)

        repo.count_by_game.assert_awaited_once_with(
            game_id,
            event_type=None,
        )
        assert result == 15

    @pytest.mark.asyncio
    async def test_with_event_type(self) -> None:
        """get_event_count should forward event_type filter."""
        service, repo = _make_service()
        repo.count_by_game.return_value = 3

        result = await service.get_event_count(
            uuid4(),
            event_type="conflict_detected",
        )

        call_kwargs = repo.count_by_game.call_args[1]
        assert call_kwargs["event_type"] == "conflict_detected"
        assert result == 3


# ---------------------------------------------------------------------------
# Tests: event type constants
# ---------------------------------------------------------------------------


class TestEventTypeConstants:
    """Tests for event type constant values."""

    def test_all_event_types_contains_all_seven(self) -> None:
        """ALL_EVENT_TYPES should contain exactly the 7 defined event types."""
        from tabletop_oracle.services.kg.audit import ALL_EVENT_TYPES

        assert len(ALL_EVENT_TYPES) == 7
        assert EVENT_EXTRACTION_COMPLETE in ALL_EVENT_TYPES
        assert EVENT_ASSOCIATIONS_DISCOVERED in ALL_EVENT_TYPES
        assert EVENT_CONFLICT_DETECTED in ALL_EVENT_TYPES
        assert EVENT_CONFLICT_RESOLVED in ALL_EVENT_TYPES
        assert EVENT_DOCUMENT_REMOVED in ALL_EVENT_TYPES
        assert EVENT_GRAPH_REBUILT in ALL_EVENT_TYPES
        assert EVENT_RETRIEVAL_EXECUTED in ALL_EVENT_TYPES

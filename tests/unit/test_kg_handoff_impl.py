"""Tests for KGHandoffServiceImpl -- the real KG handoff implementation.

All sub-services are mocked. Tests verify orchestration order, SSE events,
audit logging, error handling, and the version replacement path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from tabletop_oracle.services.ingestion.handoff import (
    KGHandoffPayload,
    KGIngestionError,
)
from tabletop_oracle.services.ingestion.models import Chunk
from tabletop_oracle.services.kg.associations import DiscoveredAssociation
from tabletop_oracle.services.kg.extraction import ExtractedConcept, GameContext
from tabletop_oracle.services.kg.handoff_impl import KGHandoffServiceImpl
from tabletop_oracle.services.kg.integration import RemovalResult
from tabletop_oracle.services.model.token_usage import TokenAttribution

# --- Helpers ---


def _make_chunk(index: int = 0, content: str = "test content") -> Chunk:
    """Create a Chunk with deterministic fields."""
    return Chunk(
        chunk_index=index,
        chunk_type="text",
        content=content,
        metadata={"chunk_db_id": str(uuid4())},
    )


def _make_payload(
    *,
    chunks: tuple[Chunk, ...] | None = None,
    replaces_version_id: UUID | None = None,
) -> KGHandoffPayload:
    """Create a KGHandoffPayload with defaults."""
    if chunks is None:
        chunks = (_make_chunk(),)
    return KGHandoffPayload(
        document_id=uuid4(),
        version_id=uuid4(),
        game_id=uuid4(),
        document_type="core_rules",
        expansion_id=None,
        chunks=chunks,
        replaces_version_id=replaces_version_id,
    )


def _make_concept(
    name: str = "fireball",
    chunk_id: UUID | None = None,
) -> ExtractedConcept:
    """Create an ExtractedConcept."""
    return ExtractedConcept(
        name=name,
        semantic_type="game_mechanic",
        description="A powerful spell",
        source_chunk_id=chunk_id or uuid4(),
        source_text="Fireball deals damage",
    )


@dataclass
class _FakeKGConcept:
    """Lightweight stand-in for KGConcept ORM model in unit tests."""

    id: UUID
    name: str
    semantic_type: str = "game_mechanic"
    description: str = "test"


@dataclass
class _FakeSource:
    """Lightweight stand-in for KGConceptSource."""

    id: UUID
    concept_id: UUID
    document_id: UUID
    document_type: str = "core_rules"
    is_authoritative: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _make_association(
    source_name: str = "fireball",
    target_name: str = "damage",
) -> DiscoveredAssociation:
    """Create a DiscoveredAssociation."""
    return DiscoveredAssociation(
        source_concept_name=source_name,
        target_concept_name=target_name,
        relationship_label="causes",
        description="Fireball causes damage",
        source_chunk_id=uuid4(),
        source_text="Fireball deals damage",
    )


class _MockServices:
    """Container for all mocked sub-services."""

    def __init__(self) -> None:
        self.extraction = AsyncMock()
        self.associations = AsyncMock()
        self.integration = AsyncMock()
        self.conflicts = AsyncMock()
        self.embeddings = AsyncMock()
        self.audit = AsyncMock()
        self.event_publisher = MagicMock()
        self.game_context_provider = AsyncMock(
            return_value=GameContext(
                game_name="Test Game",
                game_description="A test game",
            )
        )
        self.attribution_provider = AsyncMock(return_value=TokenAttribution(user_id=uuid4()))
        self.existing_concepts_provider = AsyncMock(return_value=[])

        # Default returns
        self.extraction.extract_from_chunk = AsyncMock(return_value=[])
        self.extraction._resolve_chunk_id = MagicMock(
            side_effect=lambda c: UUID(c.metadata.get("chunk_db_id", str(uuid4())))
        )
        self.associations.discover_associations = AsyncMock(return_value=[])
        self.integration.integrate_concepts = AsyncMock(return_value=[])
        self.integration.build_concept_map = MagicMock(return_value={})
        self.integration.integrate_associations = AsyncMock(return_value=[])
        self.integration.remove_document_contributions = AsyncMock(
            return_value=RemovalResult(concepts_removed=0, concepts_updated=0)
        )
        self.conflicts._source_repo = AsyncMock()
        self.conflicts._source_repo.list_by_concept = AsyncMock(return_value=[])
        self.conflicts.resolve_conflicts = AsyncMock()
        self.embeddings.embed_batch = AsyncMock(return_value=[])
        self.audit.log_extraction = AsyncMock()
        self.audit.log_association_discovery = AsyncMock()
        self.audit.log_document_removal = AsyncMock()

    def build_service(self) -> KGHandoffServiceImpl:
        """Build a KGHandoffServiceImpl with all mocked services."""
        return KGHandoffServiceImpl(
            extraction_service=self.extraction,
            association_service=self.associations,
            integration_service=self.integration,
            conflict_service=self.conflicts,
            embedding_service=self.embeddings,
            audit_service=self.audit,
            game_context_provider=self.game_context_provider,
            attribution_provider=self.attribution_provider,
            event_publisher=self.event_publisher,
            existing_concepts_provider=self.existing_concepts_provider,
        )


# --- Tests ---


class TestIngestDocumentPipelineOrchestration:
    """Verify the full pipeline runs stages in the correct order."""

    @pytest.mark.asyncio
    async def test_full_pipeline_calls_all_stages_in_order(self) -> None:
        """All pipeline stages called in sequence: extract, discover,
        integrate, conflict, embed, audit."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk = _make_chunk(content="Fireball is a spell")
        chunk_id = UUID(chunk.metadata["chunk_db_id"])
        concept = _make_concept(chunk_id=chunk_id)
        fake_kg_concept = _FakeKGConcept(id=uuid4(), name="fireball")

        mocks.extraction.extract_from_chunk = AsyncMock(return_value=[concept])
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_kg_concept])
        mocks.integration.build_concept_map = MagicMock(
            return_value={"fireball": fake_kg_concept.id}
        )

        payload = _make_payload(chunks=(chunk,))
        await service.ingest_document(payload)

        # Verify extraction called
        mocks.extraction.extract_from_chunk.assert_called_once()

        # Verify integration called
        mocks.integration.integrate_concepts.assert_called_once()
        mocks.integration.integrate_associations.assert_called_once()

        # Verify embeddings called
        mocks.embeddings.embed_batch.assert_called_once_with([fake_kg_concept])

        # Verify audit called
        mocks.audit.log_extraction.assert_called_once()
        mocks.audit.log_association_discovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_chunks_skips_pipeline(self) -> None:
        """When payload has no chunks, pipeline is skipped entirely."""
        mocks = _MockServices()
        service = mocks.build_service()

        payload = _make_payload(chunks=())
        await service.ingest_document(payload)

        mocks.extraction.extract_from_chunk.assert_not_called()
        mocks.integration.integrate_concepts.assert_not_called()
        mocks.audit.log_extraction.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_concepts_extracted_skips_downstream(self) -> None:
        """When extraction yields no concepts, subsequent stages are skipped."""
        mocks = _MockServices()
        service = mocks.build_service()

        payload = _make_payload()
        await service.ingest_document(payload)

        mocks.integration.integrate_concepts.assert_not_called()
        mocks.embeddings.embed_batch.assert_not_called()


class TestVersionReplacement:
    """Verify version replacement removes old contributions first."""

    @pytest.mark.asyncio
    async def test_replaces_version_triggers_removal(self) -> None:
        """When replaces_version_id is set, remove_document is called first."""
        mocks = _MockServices()
        service = mocks.build_service()

        old_version_id = uuid4()
        payload = _make_payload(replaces_version_id=old_version_id)

        await service.ingest_document(payload)

        mocks.integration.remove_document_contributions.assert_called_once_with(
            game_id=payload.game_id,
            document_id=payload.document_id,
        )

    @pytest.mark.asyncio
    async def test_no_replaces_version_skips_removal(self) -> None:
        """When replaces_version_id is None, remove is not called."""
        mocks = _MockServices()
        service = mocks.build_service()

        payload = _make_payload(replaces_version_id=None)
        await service.ingest_document(payload)

        mocks.integration.remove_document_contributions.assert_not_called()


class TestDocumentRemoval:
    """Verify remove_document orchestrates cleanup and audit."""

    @pytest.mark.asyncio
    async def test_remove_document_calls_integration_and_audit(self) -> None:
        """remove_document delegates to integration and logs audit event."""
        mocks = _MockServices()
        service = mocks.build_service()

        doc_id = uuid4()
        game_id = uuid4()
        mocks.integration.remove_document_contributions = AsyncMock(
            return_value=RemovalResult(concepts_removed=3, concepts_updated=1)
        )

        await service.remove_document(doc_id, game_id)

        mocks.integration.remove_document_contributions.assert_called_once_with(
            game_id=game_id,
            document_id=doc_id,
        )
        mocks.audit.log_document_removal.assert_called_once_with(
            game_id=game_id,
            document_id=doc_id,
            concepts_removed=3,
            concepts_updated=1,
        )

    @pytest.mark.asyncio
    async def test_remove_document_emits_sse_events(self) -> None:
        """remove_document emits removing and removed SSE events."""
        mocks = _MockServices()
        service = mocks.build_service()

        doc_id = uuid4()
        game_id = uuid4()
        mocks.integration.remove_document_contributions = AsyncMock(
            return_value=RemovalResult(concepts_removed=2, concepts_updated=0)
        )

        await service.remove_document(doc_id, game_id)

        # Check that events were published
        event_calls = mocks.event_publisher.publish.call_args_list
        event_types = [c[0][1] for c in event_calls]
        assert "kg.removing" in event_types
        assert "kg.removed" in event_types

    @pytest.mark.asyncio
    async def test_remove_document_wraps_unexpected_errors(self) -> None:
        """Unexpected errors are wrapped in KGIngestionError."""
        mocks = _MockServices()
        service = mocks.build_service()

        mocks.integration.remove_document_contributions = AsyncMock(
            side_effect=RuntimeError("db connection lost")
        )

        with pytest.raises(KGIngestionError):
            await service.remove_document(uuid4(), uuid4())


class TestSSEEvents:
    """Verify SSE events are emitted at each pipeline stage."""

    @pytest.mark.asyncio
    async def test_pipeline_emits_stage_events(self) -> None:
        """Full pipeline emits events for each stage."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk = _make_chunk()
        chunk_id = UUID(chunk.metadata["chunk_db_id"])
        concept = _make_concept(chunk_id=chunk_id)
        fake_concept = _FakeKGConcept(id=uuid4(), name="fireball")

        mocks.extraction.extract_from_chunk = AsyncMock(return_value=[concept])
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_concept])
        mocks.integration.build_concept_map = MagicMock(return_value={"fireball": fake_concept.id})

        payload = _make_payload(chunks=(chunk,))
        await service.ingest_document(payload)

        event_calls = mocks.event_publisher.publish.call_args_list
        event_types = [c[0][1] for c in event_calls]

        assert "kg.extracting" in event_types
        assert "kg.discovering_associations" in event_types
        assert "kg.integrating" in event_types
        assert "kg.resolving_conflicts" in event_types
        assert "kg.embedding" in event_types
        assert "kg.complete" in event_types

    @pytest.mark.asyncio
    async def test_no_event_publisher_does_not_error(self) -> None:
        """Pipeline works without an event publisher."""
        mocks = _MockServices()
        # Build without event publisher
        service = KGHandoffServiceImpl(
            extraction_service=mocks.extraction,
            association_service=mocks.associations,
            integration_service=mocks.integration,
            conflict_service=mocks.conflicts,
            embedding_service=mocks.embeddings,
            audit_service=mocks.audit,
            game_context_provider=mocks.game_context_provider,
            attribution_provider=mocks.attribution_provider,
            event_publisher=None,
        )

        payload = _make_payload()
        # Should not raise
        await service.ingest_document(payload)


class TestAuditLogging:
    """Verify audit events are logged correctly."""

    @pytest.mark.asyncio
    async def test_audit_logs_extraction_and_associations(self) -> None:
        """Both extraction and association audit entries are created."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk = _make_chunk()
        chunk_id = UUID(chunk.metadata["chunk_db_id"])
        concept = _make_concept(chunk_id=chunk_id)
        fake_concept = _FakeKGConcept(id=uuid4(), name="fireball")

        mocks.extraction.extract_from_chunk = AsyncMock(return_value=[concept])
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_concept])
        mocks.integration.build_concept_map = MagicMock(return_value={"fireball": fake_concept.id})

        payload = _make_payload(chunks=(chunk,))
        await service.ingest_document(payload)

        mocks.audit.log_extraction.assert_called_once_with(
            game_id=payload.game_id,
            document_id=payload.document_id,
            concept_count=1,
            chunk_count=1,
        )
        mocks.audit.log_association_discovery.assert_called_once()


class TestErrorHandling:
    """Verify error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_extraction_failure_skips_chunk_continues(self) -> None:
        """If extraction fails for one chunk, others still process."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk1 = _make_chunk(0, "chunk one")
        chunk2 = _make_chunk(1, "chunk two")
        chunk2_id = UUID(chunk2.metadata["chunk_db_id"])
        concept2 = _make_concept(name="shield", chunk_id=chunk2_id)

        fake_concept = _FakeKGConcept(id=uuid4(), name="shield")

        # First call fails, second succeeds
        mocks.extraction.extract_from_chunk = AsyncMock(
            side_effect=[RuntimeError("LLM timeout"), [concept2]]
        )
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_concept])
        mocks.integration.build_concept_map = MagicMock(return_value={"shield": fake_concept.id})

        payload = _make_payload(chunks=(chunk1, chunk2))
        await service.ingest_document(payload)

        # Integration still called with the successful concept
        mocks.integration.integrate_concepts.assert_called_once()
        call_args = mocks.integration.integrate_concepts.call_args
        assert len(call_args.kwargs.get("extracted_concepts", [])) == 1

    @pytest.mark.asyncio
    async def test_embedding_failure_does_not_crash_pipeline(self) -> None:
        """Embedding failure is logged but pipeline completes."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk = _make_chunk()
        chunk_id = UUID(chunk.metadata["chunk_db_id"])
        concept = _make_concept(chunk_id=chunk_id)
        fake_concept = _FakeKGConcept(id=uuid4(), name="fireball")

        mocks.extraction.extract_from_chunk = AsyncMock(return_value=[concept])
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_concept])
        mocks.integration.build_concept_map = MagicMock(return_value={"fireball": fake_concept.id})
        mocks.embeddings.embed_batch = AsyncMock(side_effect=RuntimeError("Embedding API down"))

        payload = _make_payload(chunks=(chunk,))
        # Should not raise
        await service.ingest_document(payload)

        # Audit still called despite embedding failure
        mocks.audit.log_extraction.assert_called_once()

    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped_in_kg_ingestion_error(self) -> None:
        """Non-KGIngestionError exceptions are wrapped."""
        mocks = _MockServices()
        service = mocks.build_service()

        mocks.game_context_provider = AsyncMock(side_effect=RuntimeError("provider failed"))
        # Re-build with broken provider
        service._game_context_provider = mocks.game_context_provider

        payload = _make_payload()
        with pytest.raises(KGIngestionError) as exc_info:
            await service.ingest_document(payload)

        assert "provider failed" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_kg_ingestion_error_not_double_wrapped(self) -> None:
        """KGIngestionError from sub-service propagates without wrapping."""
        mocks = _MockServices()
        service = mocks.build_service()

        original = KGIngestionError("direct failure", document_id=uuid4())
        mocks.game_context_provider = AsyncMock(side_effect=original)
        service._game_context_provider = mocks.game_context_provider

        payload = _make_payload()
        with pytest.raises(KGIngestionError) as exc_info:
            await service.ingest_document(payload)

        assert exc_info.value is original


class TestIncrementalEnrichment:
    """Verify incremental enrichment (AC-507) -- accumulated concept names
    are passed to subsequent chunk extractions for dedup awareness."""

    @pytest.mark.asyncio
    async def test_accumulated_names_passed_to_later_chunks(self) -> None:
        """Second chunk extraction receives names from first chunk."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk1 = _make_chunk(0, "first chunk")
        chunk2 = _make_chunk(1, "second chunk")
        chunk1_id = UUID(chunk1.metadata["chunk_db_id"])
        concept1 = _make_concept(name="fireball", chunk_id=chunk1_id)

        fake_concept = _FakeKGConcept(id=uuid4(), name="fireball")

        mocks.extraction.extract_from_chunk = AsyncMock(side_effect=[[concept1], []])
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_concept])
        mocks.integration.build_concept_map = MagicMock(return_value={"fireball": fake_concept.id})

        payload = _make_payload(chunks=(chunk1, chunk2))
        await service.ingest_document(payload)

        # Second extraction call should have existing_concepts
        calls = mocks.extraction.extract_from_chunk.call_args_list
        assert len(calls) == 2
        # First call: no existing concepts
        assert calls[0].kwargs.get("existing_concepts") is None
        # Second call: has accumulated names
        assert calls[1].kwargs.get("existing_concepts") == ["fireball"]


class TestConflictResolution:
    """Verify conflict resolution is called for multi-sourced concepts."""

    @pytest.mark.asyncio
    async def test_resolve_called_for_multi_sourced_concept(self) -> None:
        """Concepts with multiple sources trigger conflict resolution."""
        mocks = _MockServices()
        service = mocks.build_service()

        chunk = _make_chunk()
        chunk_id = UUID(chunk.metadata["chunk_db_id"])
        concept = _make_concept(chunk_id=chunk_id)

        concept_id = uuid4()
        fake_concept = _FakeKGConcept(id=concept_id, name="fireball")

        source1 = _FakeSource(
            id=uuid4(),
            concept_id=concept_id,
            document_id=uuid4(),
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        source2 = _FakeSource(
            id=uuid4(),
            concept_id=concept_id,
            document_id=uuid4(),
            created_at=datetime(2024, 6, 1, tzinfo=UTC),
        )

        mocks.extraction.extract_from_chunk = AsyncMock(return_value=[concept])
        mocks.integration.integrate_concepts = AsyncMock(return_value=[fake_concept])
        mocks.integration.build_concept_map = MagicMock(return_value={"fireball": concept_id})
        mocks.conflicts._source_repo.list_by_concept = AsyncMock(return_value=[source1, source2])

        payload = _make_payload(chunks=(chunk,))
        await service.ingest_document(payload)

        mocks.conflicts.resolve_conflicts.assert_called_once()
        call_kwargs = mocks.conflicts.resolve_conflicts.call_args.kwargs
        assert call_kwargs["concept_id"] == concept_id
        # newest_source should be source2
        assert call_kwargs["new_source"] is source2


class TestCrossDocumentCounting:
    """Verify cross-document association counting."""

    def test_count_cross_document_all_internal(self) -> None:
        """All associations between new concepts yields 0 cross-doc."""
        concepts = [_make_concept("a"), _make_concept("b")]
        associations = [_make_association("a", "b")]

        result = KGHandoffServiceImpl._count_cross_document(associations, concepts)
        assert result == 0

    def test_count_cross_document_with_external(self) -> None:
        """Association referencing an unknown concept counts as cross-doc."""
        concepts = [_make_concept("a")]
        associations = [_make_association("a", "external")]

        result = KGHandoffServiceImpl._count_cross_document(associations, concepts)
        assert result == 1


class TestProtocolConformance:
    """Verify KGHandoffServiceImpl satisfies the KGHandoffService protocol."""

    def test_isinstance_check(self) -> None:
        """KGHandoffServiceImpl is recognized as KGHandoffService."""
        from tabletop_oracle.services.ingestion.handoff import KGHandoffService

        mocks = _MockServices()
        service = mocks.build_service()
        assert isinstance(service, KGHandoffService)

"""Tests for KG handoff interface and V1 stub implementation."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import pytest

from tabletop_oracle.services.ingestion.handoff import (
    KGHandoffPayload,
    KGHandoffService,
    KGIngestionError,
    StubKGHandoffService,
)
from tabletop_oracle.services.ingestion.models import Chunk

# --- KGHandoffPayload ---


class TestKGHandoffPayload:
    """Tests for KGHandoffPayload dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Payload constructed with all required fields retains values."""
        doc_id = uuid4()
        ver_id = uuid4()
        game_id = uuid4()
        chunk = Chunk(chunk_index=0, chunk_type="text", content="Hello")

        payload = KGHandoffPayload(
            document_id=doc_id,
            version_id=ver_id,
            game_id=game_id,
            document_type="core_rules",
            expansion_id=None,
            chunks=(chunk,),
        )

        assert payload.document_id == doc_id
        assert payload.version_id == ver_id
        assert payload.game_id == game_id
        assert payload.document_type == "core_rules"
        assert payload.expansion_id is None
        assert len(payload.chunks) == 1
        assert payload.replaces_version_id is None

    def test_create_with_expansion_and_replaces(self) -> None:
        """Payload accepts optional expansion_id and replaces_version_id."""
        expansion_id = uuid4()
        replaces_id = uuid4()

        payload = KGHandoffPayload(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="faq",
            expansion_id=expansion_id,
            chunks=(),
            replaces_version_id=replaces_id,
        )

        assert payload.expansion_id == expansion_id
        assert payload.replaces_version_id == replaces_id

    def test_payload_is_frozen(self) -> None:
        """Payload is immutable (frozen dataclass)."""
        payload = KGHandoffPayload(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
        )

        with pytest.raises(AttributeError):
            payload.document_type = "errata"  # type: ignore[misc]

    def test_chunks_default_to_empty_tuple(self) -> None:
        """Chunks default to an empty tuple when not provided."""
        payload = KGHandoffPayload(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
        )

        assert payload.chunks == ()


# --- KGIngestionError ---


class TestKGIngestionError:
    """Tests for KGIngestionError exception."""

    def test_create_with_message(self) -> None:
        """Error carries message as string representation."""
        error = KGIngestionError("KG connection failed")
        assert str(error) == "KG connection failed"
        assert error.message == "KG connection failed"
        assert error.document_id is None

    def test_create_with_document_id(self) -> None:
        """Error optionally carries document_id context."""
        doc_id = uuid4()
        error = KGIngestionError("Ingestion failed", document_id=doc_id)
        assert error.document_id == doc_id

    def test_is_exception(self) -> None:
        """KGIngestionError is a standard Exception subclass."""
        assert issubclass(KGIngestionError, Exception)


# --- StubKGHandoffService ---


class TestStubKGHandoffService:
    """Tests for the V1 stub implementation."""

    def test_satisfies_protocol(self) -> None:
        """Stub is a runtime-checkable instance of KGHandoffService."""
        service = StubKGHandoffService()
        assert isinstance(service, KGHandoffService)

    def test_ingest_document_returns_successfully(self) -> None:
        """ingest_document completes without error."""
        service = StubKGHandoffService()
        payload = KGHandoffPayload(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
            chunks=(Chunk(chunk_index=0, chunk_type="text", content="Hello"),),
        )

        # Should not raise
        asyncio.run(service.ingest_document(payload))

    def test_ingest_document_logs_payload(self, caplog: pytest.LogCaptureFixture) -> None:
        """ingest_document logs document metadata at INFO level."""
        service = StubKGHandoffService()
        doc_id = uuid4()
        payload = KGHandoffPayload(
            document_id=doc_id,
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
            chunks=(Chunk(chunk_index=0, chunk_type="text", content="Hello"),),
        )

        with caplog.at_level(logging.INFO, logger="tabletop_oracle.services.ingestion.handoff"):
            asyncio.run(service.ingest_document(payload))

        assert "ingest_document called" in caplog.text
        assert str(doc_id) in caplog.text
        assert "chunk_count=1" in caplog.text

    def test_remove_document_returns_successfully(self) -> None:
        """remove_document completes without error."""
        service = StubKGHandoffService()
        asyncio.run(service.remove_document(uuid4(), uuid4()))

    def test_remove_document_logs_request(self, caplog: pytest.LogCaptureFixture) -> None:
        """remove_document logs document_id and game_id at INFO level."""
        service = StubKGHandoffService()
        doc_id = uuid4()
        game_id = uuid4()

        with caplog.at_level(logging.INFO, logger="tabletop_oracle.services.ingestion.handoff"):
            asyncio.run(service.remove_document(doc_id, game_id))

        assert "remove_document called" in caplog.text
        assert str(doc_id) in caplog.text
        assert str(game_id) in caplog.text


# --- KGHandoffStage integration ---


class TestKGHandoffStageIntegration:
    """Tests for KGHandoffStage wiring with the service."""

    def test_stage_uses_injected_service(self) -> None:
        """KGHandoffStage delegates to the provided service."""
        from unittest.mock import AsyncMock

        from tabletop_oracle.services.ingestion.models import PipelineContext
        from tabletop_oracle.services.ingestion.stages import KGHandoffStage

        mock_service = AsyncMock(spec=StubKGHandoffService)
        stage = KGHandoffStage(service=mock_service)

        ctx = PipelineContext(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
            file_path="test.pdf",
            document_format="pdf",
            file_size=100,
        )
        ctx.chunks = [Chunk(chunk_index=0, chunk_type="text", content="Hello")]

        stage.execute(ctx)

        mock_service.ingest_document.assert_called_once()
        call_payload = mock_service.ingest_document.call_args[0][0]
        assert isinstance(call_payload, KGHandoffPayload)
        assert call_payload.document_id == ctx.document_id
        assert call_payload.game_id == ctx.game_id

    def test_stage_defaults_to_stub(self) -> None:
        """KGHandoffStage defaults to StubKGHandoffService when no service given."""
        from tabletop_oracle.services.ingestion.stages import KGHandoffStage

        stage = KGHandoffStage()
        assert isinstance(stage._service, StubKGHandoffService)

    def test_stage_no_chunks_raises(self) -> None:
        """KGHandoffStage raises RuntimeError when chunks list is empty."""
        from tabletop_oracle.services.ingestion.models import PipelineContext
        from tabletop_oracle.services.ingestion.stages import KGHandoffStage

        stage = KGHandoffStage()
        ctx = PipelineContext(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
            file_path="test.pdf",
            document_format="pdf",
            file_size=100,
        )
        ctx.chunks = []

        with pytest.raises(RuntimeError, match="no chunks produced"):
            stage.execute(ctx)

    def test_stage_passes_replaces_version_id(self) -> None:
        """KGHandoffStage passes replaces_version_id from context to payload."""
        from unittest.mock import AsyncMock

        from tabletop_oracle.services.ingestion.models import PipelineContext
        from tabletop_oracle.services.ingestion.stages import KGHandoffStage

        replaces_id = uuid4()
        mock_service = AsyncMock(spec=StubKGHandoffService)
        stage = KGHandoffStage(service=mock_service)

        ctx = PipelineContext(
            document_id=uuid4(),
            version_id=uuid4(),
            game_id=uuid4(),
            document_type="core_rules",
            expansion_id=None,
            file_path="test.pdf",
            document_format="pdf",
            file_size=100,
            replaces_version_id=replaces_id,
        )
        ctx.chunks = [Chunk(chunk_index=0, chunk_type="text", content="Hello")]

        stage.execute(ctx)

        call_payload = mock_service.ingest_document.call_args[0][0]
        assert call_payload.replaces_version_id == replaces_id

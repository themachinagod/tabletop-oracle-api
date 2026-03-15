"""Ingestion pipeline orchestrator.

Drives documents through the processing stages sequentially, emitting
SSE events at each transition and managing error recovery with chunk
cleanup. The orchestrator is a plain Python class — Celery task
invocation is in workers/tasks.py.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select, update

from tabletop_oracle.models.document import Document, DocumentChunk, DocumentVersion
from tabletop_oracle.models.enums import DocumentStatus
from tabletop_oracle.services.ingestion.models import PipelineContext, PipelineStage
from tabletop_oracle.services.ingestion.stages import (
    ChunkingStage,
    ExtractionStage,
    KGHandoffStage,
    StructureDetectionStage,
    ValidationStage,
)
from tabletop_oracle.streaming.event_types import ProcessingEventType

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.orm import Session

    from tabletop_oracle.services.ingestion.events import EventPublisher
    from tabletop_oracle.services.ingestion.stages import PipelineStageBase

logger = logging.getLogger(__name__)

# Map from stage class to the SSE stage name emitted in processing.stage events.
_STAGE_EVENT_MAP: dict[type[PipelineStageBase], str] = {
    ValidationStage: PipelineStage.VALIDATING,
    ExtractionStage: PipelineStage.EXTRACTING,
    StructureDetectionStage: PipelineStage.DETECTING_STRUCTURE,
    ChunkingStage: PipelineStage.CHUNKING,
    KGHandoffStage: PipelineStage.INTEGRATING,
}


class IngestionPipeline:
    """Orchestrates document processing through sequential stages.

    Each invocation processes a single document version. The pipeline:
    1. Cleans up any existing chunks (idempotent start)
    2. Runs stages in order: validation -> extraction -> structure -> chunking -> KG handoff
    3. Emits SSE events at each stage transition
    4. Updates document status in PostgreSQL at each stage
    5. On failure: cleans up partial chunks, sets error status

    Args:
        db_session: Synchronous SQLAlchemy session for database operations.
        event_publisher: Publisher for SSE events (Redis or log-based).
        stages: Ordered list of pipeline stages. Defaults to the standard
            five-stage pipeline if not provided.
    """

    def __init__(
        self,
        db_session: Session,
        event_publisher: EventPublisher,
        stages: list[PipelineStageBase] | None = None,
    ) -> None:
        self._db = db_session
        self._publisher = event_publisher
        self._stages: list[PipelineStageBase] = stages or [
            ValidationStage(),
            ExtractionStage(),
            StructureDetectionStage(),
            ChunkingStage(),
            KGHandoffStage(),
        ]

    def process(self, document_id: UUID, version_id: UUID) -> None:
        """Execute the full ingestion pipeline for a document version.

        Args:
            document_id: UUID of the document being processed.
            version_id: UUID of the specific version to process.

        Raises:
            Exception: Re-raises after cleanup if a stage fails and
                retries are exhausted.
        """
        context = self._build_context(document_id, version_id)

        # Set document status to parsing and emit started event
        self._update_document_status(document_id, DocumentStatus.PARSING)
        self._emit_started(document_id, context.file_path)

        # Idempotent start: clean up any chunks from a previous failed run
        self._cleanup_chunks(document_id, version_id)

        try:
            for stage in self._stages:
                stage_name = _STAGE_EVENT_MAP.get(type(stage), stage.name)
                self._emit_stage(document_id, stage_name)
                logger.info(
                    "Pipeline stage %s starting: document_id=%s",
                    stage_name,
                    document_id,
                )
                stage.execute(context)
                logger.info(
                    "Pipeline stage %s completed: document_id=%s",
                    stage_name,
                    document_id,
                )

            # Persist chunks
            chunk_count = self._persist_chunks(context)

            # Mark document as processed
            self._update_document_status(
                document_id,
                DocumentStatus.PROCESSED,
                processed_at=datetime.now(UTC),
            )
            self._update_version_processed(version_id)
            self._emit_completed(document_id, chunk_count)

            logger.info(
                "Pipeline completed: document_id=%s, chunks=%d",
                document_id,
                chunk_count,
            )

        except Exception:
            self._handle_failure(document_id, version_id)
            raise

    def _build_context(self, document_id: UUID, version_id: UUID) -> PipelineContext:
        """Load document and version data to build the pipeline context.

        Args:
            document_id: UUID of the document.
            version_id: UUID of the version.

        Returns:
            A populated PipelineContext.

        Raises:
            ValueError: If the document or version is not found.
        """
        doc = self._db.execute(
            select(Document).where(Document.id == document_id)
        ).scalar_one_or_none()
        if doc is None:
            msg = f"Document not found: {document_id}"
            raise ValueError(msg)

        version = self._db.execute(
            select(DocumentVersion).where(DocumentVersion.id == version_id)
        ).scalar_one_or_none()
        if version is None:
            msg = f"Document version not found: {version_id}"
            raise ValueError(msg)

        return PipelineContext(
            document_id=document_id,
            version_id=version_id,
            file_path=version.file_path,
            document_format=doc.format.value,
            file_size=version.file_size,
        )

    def _update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        *,
        processed_at: datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update document status in PostgreSQL.

        Args:
            document_id: UUID of the document.
            status: New status value.
            processed_at: Timestamp when processing completed.
            error_message: Error message if status is ERROR.
        """
        values: dict[str, Any] = {"status": status}
        if processed_at is not None:
            values["processed_at"] = processed_at
        if error_message is not None:
            values["error_message"] = error_message

        self._db.execute(update(Document).where(Document.id == document_id).values(**values))
        self._db.commit()

    def _update_version_processed(self, version_id: UUID) -> None:
        """Set processed_at on the document version.

        Args:
            version_id: UUID of the version.
        """
        self._db.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .values(processed_at=datetime.now(UTC))
        )
        self._db.commit()

    def _cleanup_chunks(self, document_id: UUID, version_id: UUID) -> None:
        """Delete existing chunks for this document version.

        Ensures idempotent processing: if a previous run partially wrote
        chunks before failing, they are cleaned up before reprocessing.

        Args:
            document_id: UUID of the document.
            version_id: UUID of the version.
        """
        self._db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.version_id == version_id,
            )
        )
        self._db.commit()

    def _persist_chunks(self, context: PipelineContext) -> int:
        """Write pipeline chunks to the database.

        Args:
            context: Pipeline context with populated chunks list.

        Returns:
            Number of chunks persisted.
        """
        for chunk in context.chunks:
            db_chunk = DocumentChunk(
                document_id=context.document_id,
                version_id=context.version_id,
                chunk_index=chunk.chunk_index,
                chunk_type=chunk.chunk_type,
                content=chunk.content,
                section_path=chunk.section_path,
                heading=chunk.heading,
                page_number=chunk.page_number,
                token_estimate=chunk.token_estimate,
                chunk_metadata=chunk.metadata,
            )
            self._db.add(db_chunk)

        self._db.commit()
        return len(context.chunks)

    def _handle_failure(self, document_id: UUID, version_id: UUID) -> None:
        """Handle pipeline failure: clean up chunks and set error status.

        Args:
            document_id: UUID of the document.
            version_id: UUID of the version.
        """
        logger.exception("Pipeline failed: document_id=%s", document_id)

        try:
            self._cleanup_chunks(document_id, version_id)
            self._update_document_status(
                document_id,
                DocumentStatus.ERROR,
                error_message="Document processing failed. Please try again.",
            )
            self._emit_error(
                document_id,
                code="PROCESSING_FAILED",
                message="Document processing failed. Please try again.",
            )
        except Exception:
            logger.exception(
                "Failed to clean up after pipeline error: document_id=%s",
                document_id,
            )

    # --- SSE event emission ---

    def _emit_started(self, document_id: UUID, filename: str) -> None:
        """Emit processing.started event."""
        self._publisher.publish(
            document_id,
            ProcessingEventType.STARTED,
            {"document_id": str(document_id), "filename": filename},
        )

    def _emit_stage(self, document_id: UUID, stage: str) -> None:
        """Emit processing.stage event."""
        self._publisher.publish(
            document_id,
            ProcessingEventType.STAGE,
            {"stage": stage, "progress": 0.0},
        )

    def _emit_completed(self, document_id: UUID, chunk_count: int) -> None:
        """Emit processing.completed event."""
        self._publisher.publish(
            document_id,
            ProcessingEventType.COMPLETED,
            {"document_id": str(document_id), "chunk_count": chunk_count},
        )

    def _emit_error(self, document_id: UUID, *, code: str, message: str) -> None:
        """Emit processing.error event."""
        self._publisher.publish(
            document_id,
            ProcessingEventType.ERROR,
            {"document_id": str(document_id), "code": code, "message": message},
        )

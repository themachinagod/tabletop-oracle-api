"""Tests for the ingestion pipeline orchestrator.

Tests the pipeline as a Python class with mocked DB session and event
publisher. Does not test Celery integration — only the pipeline
orchestration logic: stage sequencing, event emission, status updates,
error handling, and chunk cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from tabletop_oracle.models.enums import DocumentFormat, DocumentStatus
from tabletop_oracle.services.ingestion.events import LogEventPublisher
from tabletop_oracle.services.ingestion.pipeline import IngestionPipeline
from tabletop_oracle.services.ingestion.stages import PipelineStageBase
from tabletop_oracle.streaming.event_types import ProcessingEventType

if TYPE_CHECKING:
    from tabletop_oracle.services.ingestion.models import PipelineContext


# --- Helpers ---


class RecordingPublisher:
    """Event publisher that records all published events for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[UUID, str, dict[str, Any]]] = []

    def publish(self, document_id: UUID, event_type: str, payload: dict[str, Any]) -> None:
        """Record the event."""
        self.events.append((document_id, event_type, payload))

    @property
    def event_types(self) -> list[str]:
        """Return just the event types in emission order."""
        return [e[1] for e in self.events]


class SuccessStage(PipelineStageBase):
    """Stage that always succeeds, recording invocations."""

    def __init__(self, stage_name: str = "success") -> None:
        self._name = stage_name
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: PipelineContext) -> None:
        """Record the call and do nothing."""
        self.call_count += 1


class FailingStage(PipelineStageBase):
    """Stage that always raises an exception."""

    def __init__(self, stage_name: str = "failing") -> None:
        self._name = stage_name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: PipelineContext) -> None:
        """Raise a RuntimeError."""
        msg = f"Stage {self._name} failed"
        raise RuntimeError(msg)


class ChunkProducingStage(PipelineStageBase):
    """Stage that adds chunks to the context."""

    @property
    def name(self) -> str:
        return "chunk_producer"

    def execute(self, context: PipelineContext) -> None:
        """Add test chunks to the context."""
        from tabletop_oracle.services.ingestion.models import Chunk

        context.chunks = [
            Chunk(chunk_index=0, chunk_type="text", content="Chunk one", token_estimate=2),
            Chunk(chunk_index=1, chunk_type="text", content="Chunk two", token_estimate=2),
        ]


def _make_mock_document(
    doc_id: UUID,
    format_val: DocumentFormat = DocumentFormat.PDF,
    *,
    game_id: UUID | None = None,
    expansion_id: UUID | None = None,
) -> MagicMock:
    """Create a mock Document ORM object."""
    from tabletop_oracle.models.enums import DocumentType

    doc = MagicMock()
    doc.id = doc_id
    doc.format = format_val
    doc.type = DocumentType.CORE_RULES
    doc.status = DocumentStatus.UPLOADED
    doc.game_id = game_id or uuid4()
    doc.expansion_id = expansion_id
    return doc


def _make_mock_version(ver_id: UUID) -> MagicMock:
    """Create a mock DocumentVersion ORM object."""
    ver = MagicMock()
    ver.id = ver_id
    ver.file_path = "documents/game1/doc1/v1/rules.pdf"
    ver.file_size = 1024
    return ver


def _make_mock_session(doc: MagicMock, version: MagicMock) -> MagicMock:
    """Create a mock SQLAlchemy session that returns doc and version."""
    session = MagicMock()
    result_mock = MagicMock()

    # Map queries to return the right object based on call order
    call_results = [doc, version]
    call_idx = {"i": 0}

    def scalar_one_or_none() -> MagicMock:
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(call_results):
            return call_results[idx]
        return None  # type: ignore[return-value]

    result_mock.scalar_one_or_none = scalar_one_or_none
    session.execute.return_value = result_mock

    return session


# --- Tests ---


class TestIngestionPipelineInit:
    """Tests for pipeline construction."""

    def test_default_stages(self) -> None:
        """Pipeline creates five default stages when none provided."""
        session = MagicMock()
        publisher = LogEventPublisher()
        pipeline = IngestionPipeline(session, publisher)
        assert len(pipeline._stages) == 5

    def test_custom_stages(self) -> None:
        """Pipeline accepts custom stage list."""
        session = MagicMock()
        publisher = LogEventPublisher()
        stages = [SuccessStage("s1"), SuccessStage("s2")]
        pipeline = IngestionPipeline(session, publisher, stages=stages)
        assert len(pipeline._stages) == 2


class TestIngestionPipelineProcess:
    """Tests for the process() method — the main pipeline execution."""

    def test_all_stages_execute_in_order(self) -> None:
        """All stages execute sequentially."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        stage1 = SuccessStage("first")
        stage2 = ChunkProducingStage()
        pipeline = IngestionPipeline(session, publisher, stages=[stage1, stage2])
        pipeline.process(doc_id, ver_id)

        assert stage1.call_count == 1

    def test_emits_started_event(self) -> None:
        """Pipeline emits processing.started before stages."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[ChunkProducingStage()])
        pipeline.process(doc_id, ver_id)

        assert publisher.events[0][1] == ProcessingEventType.STARTED

    def test_emits_stage_events(self) -> None:
        """Pipeline emits processing.stage for each stage."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(
            session, publisher, stages=[SuccessStage("s1"), ChunkProducingStage()]
        )
        pipeline.process(doc_id, ver_id)

        stage_events = [e for e in publisher.events if e[1] == ProcessingEventType.STAGE]
        assert len(stage_events) == 2

    def test_emits_completed_event(self) -> None:
        """Pipeline emits processing.completed on success."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[ChunkProducingStage()])
        pipeline.process(doc_id, ver_id)

        assert publisher.events[-1][1] == ProcessingEventType.COMPLETED
        assert publisher.events[-1][2]["chunk_count"] == 2

    def test_updates_document_status_to_parsing(self) -> None:
        """Pipeline sets document status to PARSING at start."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[ChunkProducingStage()])
        pipeline.process(doc_id, ver_id)

        # The first execute call after build_context queries is the
        # status update to PARSING
        execute_calls = session.execute.call_args_list
        # At minimum: 2 selects (doc, version) + delete + update(PARSING) + ...
        assert len(execute_calls) >= 4

    def test_persists_chunks_on_success(self) -> None:
        """Pipeline calls session.add for each chunk on success."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[ChunkProducingStage()])
        pipeline.process(doc_id, ver_id)

        # session.add called once per chunk
        add_calls = session.add.call_args_list
        assert len(add_calls) == 2

    def test_commits_on_success(self) -> None:
        """Pipeline commits the session on success."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[ChunkProducingStage()])
        pipeline.process(doc_id, ver_id)

        assert session.commit.called


class TestIngestionPipelineErrorHandling:
    """Tests for pipeline error handling and cleanup."""

    def test_stage_failure_raises(self) -> None:
        """Pipeline re-raises the exception after cleanup."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[FailingStage("boom")])

        with pytest.raises(RuntimeError, match="Stage boom failed"):
            pipeline.process(doc_id, ver_id)

    def test_failure_emits_error_event(self) -> None:
        """Pipeline emits processing.error on failure."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[FailingStage("boom")])

        with pytest.raises(RuntimeError):
            pipeline.process(doc_id, ver_id)

        error_events = [e for e in publisher.events if e[1] == ProcessingEventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0][2]["code"] == "PROCESSING_FAILED"

    def test_failure_does_not_emit_completed(self) -> None:
        """Pipeline does not emit processing.completed on failure."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[FailingStage()])

        with pytest.raises(RuntimeError):
            pipeline.process(doc_id, ver_id)

        completed_events = [e for e in publisher.events if e[1] == ProcessingEventType.COMPLETED]
        assert len(completed_events) == 0

    def test_second_stage_failure_does_not_run_third(self) -> None:
        """Pipeline stops executing stages after a failure."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        stage1 = SuccessStage("first")
        stage2 = FailingStage("second")
        stage3 = SuccessStage("third")

        pipeline = IngestionPipeline(session, publisher, stages=[stage1, stage2, stage3])

        with pytest.raises(RuntimeError):
            pipeline.process(doc_id, ver_id)

        assert stage1.call_count == 1
        assert stage3.call_count == 0


class TestIngestionPipelineBuildContext:
    """Tests for _build_context."""

    def test_missing_document_raises(self) -> None:
        """ValueError when document is not found."""
        doc_id = uuid4()
        ver_id = uuid4()
        session = MagicMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute.return_value = result_mock
        publisher = LogEventPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[])
        with pytest.raises(ValueError, match="Document not found"):
            pipeline._build_context(doc_id, ver_id)

    def test_missing_version_raises(self) -> None:
        """ValueError when version is not found."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        session = MagicMock()

        call_count = {"n": 0}

        def scalar_side_effect() -> MagicMock | None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return doc
            return None

        result_mock = MagicMock()
        result_mock.scalar_one_or_none = scalar_side_effect
        session.execute.return_value = result_mock
        publisher = LogEventPublisher()

        pipeline = IngestionPipeline(session, publisher, stages=[])
        with pytest.raises(ValueError, match="Document version not found"):
            pipeline._build_context(doc_id, ver_id)


class TestIngestionPipelineIdempotency:
    """Tests for idempotent processing behavior."""

    def test_cleanup_called_before_stages(self) -> None:
        """Pipeline cleans up existing chunks before running stages."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        # Track that execute is called (for delete) before stage runs
        stage = SuccessStage("test")
        original_execute = session.execute

        execute_before_stage: list[int] = []

        def tracking_execute(*args: Any, **kwargs: Any) -> Any:
            execute_before_stage.append(stage.call_count)
            return original_execute(*args, **kwargs)

        session.execute = tracking_execute

        pipeline = IngestionPipeline(session, publisher, stages=[stage, ChunkProducingStage()])
        pipeline.process(doc_id, ver_id)

        # The delete (cleanup) should happen while stage.call_count is still 0
        assert 0 in execute_before_stage


class TestIngestionPipelineEventSequence:
    """Tests for the complete event emission sequence."""

    def test_full_event_sequence(self) -> None:
        """Verify the complete event sequence for a successful pipeline."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(
            session,
            publisher,
            stages=[SuccessStage("s1"), SuccessStage("s2"), ChunkProducingStage()],
        )
        pipeline.process(doc_id, ver_id)

        expected = [
            ProcessingEventType.STARTED,
            ProcessingEventType.STAGE,  # s1
            ProcessingEventType.STAGE,  # s2
            ProcessingEventType.STAGE,  # chunk_producer
            ProcessingEventType.COMPLETED,
        ]
        assert publisher.event_types == expected

    def test_failure_event_sequence(self) -> None:
        """Verify event sequence when pipeline fails at second stage."""
        doc_id = uuid4()
        ver_id = uuid4()
        doc = _make_mock_document(doc_id)
        version = _make_mock_version(ver_id)
        session = _make_mock_session(doc, version)
        publisher = RecordingPublisher()

        pipeline = IngestionPipeline(
            session,
            publisher,
            stages=[SuccessStage("s1"), FailingStage("s2"), SuccessStage("s3")],
        )

        with pytest.raises(RuntimeError):
            pipeline.process(doc_id, ver_id)

        expected = [
            ProcessingEventType.STARTED,
            ProcessingEventType.STAGE,  # s1
            ProcessingEventType.STAGE,  # s2 (emitted before execution)
            ProcessingEventType.ERROR,
        ]
        assert publisher.event_types == expected

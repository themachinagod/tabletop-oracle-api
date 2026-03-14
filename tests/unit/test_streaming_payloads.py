"""Tests for SSE payload models and serialisation."""

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from tabletop_oracle.streaming.event_types import (
    HeartbeatEventType,
    ProcessingEventType,
    StreamEventType,
)
from tabletop_oracle.streaming.events import SseEvent
from tabletop_oracle.streaming.payloads import (
    HeartbeatPayload,
    ProcessingCompletedPayload,
    ProcessingErrorPayload,
    ProcessingStagePayload,
    ProcessingStartedPayload,
    StreamCancelledPayload,
    StreamCitationPayload,
    StreamCompletedPayload,
    StreamConfidencePayload,
    StreamErrorPayload,
    StreamStartedPayload,
    StreamStepPayload,
    StreamTokenPayload,
    payload_to_sse_event,
)

# ---------------------------------------------------------------------------
# AI Streaming Payloads
# ---------------------------------------------------------------------------


class TestStreamStartedPayload:
    """stream.started payload serialisation and validation."""

    def test_serialise(self) -> None:
        mid = uuid4()
        payload = StreamStartedPayload(message_id=mid)
        data = payload.model_dump(mode="json")
        assert data == {"message_id": str(mid)}

    def test_json_roundtrip(self) -> None:
        mid = uuid4()
        payload = StreamStartedPayload(message_id=mid)
        raw = payload.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["message_id"] == str(mid)

    def test_invalid_message_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamStartedPayload(message_id="not-a-uuid")  # type: ignore[arg-type]


class TestStreamStepPayload:
    """stream.step payload serialisation and validation."""

    def test_serialise_with_detail(self) -> None:
        payload = StreamStepPayload(step="Retrieving knowledge", detail="12 sources")
        data = payload.model_dump(mode="json")
        assert data == {"step": "Retrieving knowledge", "detail": "12 sources"}

    def test_serialise_without_detail(self) -> None:
        payload = StreamStepPayload(step="Synthesising answer")
        data = payload.model_dump(mode="json")
        assert data == {"step": "Synthesising answer", "detail": ""}

    def test_empty_step_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamStepPayload(step="")


class TestStreamTokenPayload:
    """stream.token payload serialisation and validation."""

    def test_serialise(self) -> None:
        payload = StreamTokenPayload(token="Hello")
        assert payload.model_dump(mode="json") == {"token": "Hello"}

    def test_empty_token_allowed(self) -> None:
        """Tokens can be empty strings (whitespace tokens, etc.)."""
        payload = StreamTokenPayload(token="")
        assert payload.model_dump(mode="json") == {"token": ""}

    def test_whitespace_token(self) -> None:
        payload = StreamTokenPayload(token=" ")
        assert payload.model_dump(mode="json") == {"token": " "}


class TestStreamCitationPayload:
    """stream.citation payload serialisation and validation."""

    def test_serialise_full(self) -> None:
        payload = StreamCitationPayload(
            index=1,
            document_name="Core Rules v2",
            section="Chapter 3",
            excerpt="Movement phase allows...",
        )
        data = payload.model_dump(mode="json")
        assert data == {
            "index": 1,
            "document_name": "Core Rules v2",
            "section": "Chapter 3",
            "excerpt": "Movement phase allows...",
        }

    def test_serialise_minimal(self) -> None:
        payload = StreamCitationPayload(index=1, document_name="Rules")
        data = payload.model_dump(mode="json")
        assert data["index"] == 1
        assert data["document_name"] == "Rules"
        assert data["section"] == ""
        assert data["excerpt"] == ""

    def test_index_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamCitationPayload(index=0, document_name="Rules")

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamCitationPayload(index=-1, document_name="Rules")

    def test_empty_document_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamCitationPayload(index=1, document_name="")


class TestStreamConfidencePayload:
    """stream.confidence payload serialisation and validation."""

    def test_serialise(self) -> None:
        payload = StreamConfidencePayload(score=0.85, label="high")
        data = payload.model_dump(mode="json")
        assert data == {"score": 0.85, "label": "high"}

    def test_score_zero(self) -> None:
        payload = StreamConfidencePayload(score=0.0, label="none")
        assert payload.score == 0.0

    def test_score_one(self) -> None:
        payload = StreamConfidencePayload(score=1.0, label="perfect")
        assert payload.score == 1.0

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamConfidencePayload(score=1.01, label="high")

    def test_score_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamConfidencePayload(score=-0.1, label="low")

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamConfidencePayload(score=0.5, label="")


class TestStreamCompletedPayload:
    """stream.completed payload serialisation."""

    def test_serialise(self) -> None:
        mid = uuid4()
        payload = StreamCompletedPayload(message_id=mid)
        assert payload.model_dump(mode="json") == {"message_id": str(mid)}


class TestStreamErrorPayload:
    """stream.error payload serialisation and validation."""

    def test_serialise(self) -> None:
        payload = StreamErrorPayload(code="SERVICE_UNAVAILABLE", message="LLM provider down")
        data = payload.model_dump(mode="json")
        assert data == {"code": "SERVICE_UNAVAILABLE", "message": "LLM provider down"}

    def test_empty_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamErrorPayload(code="", message="Something broke")

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamErrorPayload(code="ERROR", message="")


class TestStreamCancelledPayload:
    """stream.cancelled payload serialisation."""

    def test_serialise(self) -> None:
        mid = uuid4()
        payload = StreamCancelledPayload(message_id=mid)
        assert payload.model_dump(mode="json") == {"message_id": str(mid)}


# ---------------------------------------------------------------------------
# Document Processing Payloads
# ---------------------------------------------------------------------------


class TestProcessingStartedPayload:
    """processing.started payload serialisation and validation."""

    def test_serialise(self) -> None:
        doc_id = uuid4()
        payload = ProcessingStartedPayload(document_id=doc_id, filename="rules.pdf")
        data = payload.model_dump(mode="json")
        assert data == {"document_id": str(doc_id), "filename": "rules.pdf"}

    def test_empty_filename_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingStartedPayload(document_id=uuid4(), filename="")


class TestProcessingStagePayload:
    """processing.stage payload serialisation and validation."""

    def test_serialise_with_progress(self) -> None:
        payload = ProcessingStagePayload(stage="extracting", progress=0.45)
        data = payload.model_dump(mode="json")
        assert data == {"stage": "extracting", "progress": 0.45}

    def test_serialise_without_progress(self) -> None:
        payload = ProcessingStagePayload(stage="validating")
        data = payload.model_dump(mode="json")
        assert data == {"stage": "validating", "progress": 0.0}

    def test_progress_at_boundaries(self) -> None:
        p0 = ProcessingStagePayload(stage="s", progress=0.0)
        p1 = ProcessingStagePayload(stage="s", progress=1.0)
        assert p0.progress == 0.0
        assert p1.progress == 1.0

    def test_progress_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingStagePayload(stage="s", progress=1.01)

    def test_progress_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingStagePayload(stage="s", progress=-0.1)

    def test_empty_stage_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingStagePayload(stage="")


class TestProcessingCompletedPayload:
    """processing.completed payload serialisation and validation."""

    def test_serialise(self) -> None:
        doc_id = uuid4()
        payload = ProcessingCompletedPayload(document_id=doc_id, chunk_count=42)
        data = payload.model_dump(mode="json")
        assert data == {"document_id": str(doc_id), "chunk_count": 42}

    def test_chunk_count_zero(self) -> None:
        payload = ProcessingCompletedPayload(document_id=uuid4(), chunk_count=0)
        assert payload.chunk_count == 0

    def test_negative_chunk_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingCompletedPayload(document_id=uuid4(), chunk_count=-1)


class TestProcessingErrorPayload:
    """processing.error payload serialisation and validation."""

    def test_serialise(self) -> None:
        doc_id = uuid4()
        payload = ProcessingErrorPayload(
            document_id=doc_id,
            code="PARSE_ERROR",
            message="Could not extract text from PDF",
        )
        data = payload.model_dump(mode="json")
        assert data == {
            "document_id": str(doc_id),
            "code": "PARSE_ERROR",
            "message": "Could not extract text from PDF",
        }

    def test_empty_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingErrorPayload(document_id=uuid4(), code="", message="fail")

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingErrorPayload(document_id=uuid4(), code="ERR", message="")


# ---------------------------------------------------------------------------
# Heartbeat Payload
# ---------------------------------------------------------------------------


class TestHeartbeatPayload:
    """heartbeat payload serialisation."""

    def test_serialise_empty(self) -> None:
        payload = HeartbeatPayload()
        assert payload.model_dump(mode="json") == {}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HeartbeatPayload(unexpected="field")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# payload_to_sse_event conversion
# ---------------------------------------------------------------------------


class TestPayloadToSseEvent:
    """payload_to_sse_event bridges typed payloads to SseEvent."""

    def test_stream_started(self) -> None:
        mid = uuid4()
        event = payload_to_sse_event(StreamStartedPayload(message_id=mid))
        assert isinstance(event, SseEvent)
        assert event.type == StreamEventType.STARTED
        assert event.payload == {"message_id": str(mid)}

    def test_stream_token(self) -> None:
        event = payload_to_sse_event(StreamTokenPayload(token="Hi"))
        assert event.type == StreamEventType.TOKEN
        assert event.payload == {"token": "Hi"}

    def test_stream_step(self) -> None:
        event = payload_to_sse_event(StreamStepPayload(step="Searching", detail="3 docs"))
        assert event.type == StreamEventType.STEP
        assert event.payload == {"step": "Searching", "detail": "3 docs"}

    def test_stream_citation(self) -> None:
        event = payload_to_sse_event(
            StreamCitationPayload(index=2, document_name="FAQ", section="S1", excerpt="...")
        )
        assert event.type == StreamEventType.CITATION
        assert event.payload["index"] == 2

    def test_stream_confidence(self) -> None:
        event = payload_to_sse_event(StreamConfidencePayload(score=0.9, label="high"))
        assert event.type == StreamEventType.CONFIDENCE
        assert event.payload["score"] == 0.9

    def test_stream_completed(self) -> None:
        mid = uuid4()
        event = payload_to_sse_event(StreamCompletedPayload(message_id=mid))
        assert event.type == StreamEventType.COMPLETED

    def test_stream_error(self) -> None:
        event = payload_to_sse_event(StreamErrorPayload(code="FAIL", message="Broken"))
        assert event.type == StreamEventType.ERROR

    def test_stream_cancelled(self) -> None:
        mid = uuid4()
        event = payload_to_sse_event(StreamCancelledPayload(message_id=mid))
        assert event.type == StreamEventType.CANCELLED

    def test_processing_started(self) -> None:
        doc_id = uuid4()
        event = payload_to_sse_event(
            ProcessingStartedPayload(document_id=doc_id, filename="rules.pdf")
        )
        assert event.type == ProcessingEventType.STARTED
        assert event.payload == {"document_id": str(doc_id), "filename": "rules.pdf"}

    def test_processing_stage(self) -> None:
        event = payload_to_sse_event(ProcessingStagePayload(stage="chunking", progress=0.7))
        assert event.type == ProcessingEventType.STAGE

    def test_processing_completed(self) -> None:
        doc_id = uuid4()
        event = payload_to_sse_event(
            ProcessingCompletedPayload(document_id=doc_id, chunk_count=10)
        )
        assert event.type == ProcessingEventType.COMPLETED

    def test_processing_error(self) -> None:
        doc_id = uuid4()
        event = payload_to_sse_event(
            ProcessingErrorPayload(document_id=doc_id, code="ERR", message="Failed")
        )
        assert event.type == ProcessingEventType.ERROR

    def test_heartbeat(self) -> None:
        event = payload_to_sse_event(HeartbeatPayload())
        assert event.type == HeartbeatEventType.HEARTBEAT
        assert event.payload == {}

    def test_unknown_payload_raises(self) -> None:
        """Unregistered payload types produce a clear error."""
        from pydantic import BaseModel

        class FakePayload(BaseModel):
            value: str = "x"

        with pytest.raises(ValueError, match="Unknown payload type"):
            payload_to_sse_event(FakePayload())

    def test_uuid_serialised_as_string(self) -> None:
        """UUIDs in payloads serialise as strings, not UUID objects."""
        mid = UUID("12345678-1234-5678-1234-567812345678")
        event = payload_to_sse_event(StreamStartedPayload(message_id=mid))
        assert event.payload["message_id"] == "12345678-1234-5678-1234-567812345678"
        # Verify it's JSON-serialisable without custom encoder
        json.dumps(event.payload)

    def test_event_payload_is_plain_dict(self) -> None:
        """SseEvent.payload should be a plain dict, not a Pydantic model."""
        event = payload_to_sse_event(StreamTokenPayload(token="x"))
        assert isinstance(event.payload, dict)


# ---------------------------------------------------------------------------
# JSON schema compliance (design doc)
# ---------------------------------------------------------------------------


class TestDesignDocSchemaCompliance:
    """Verify serialised JSON matches the design doc event schema structure.

    The design doc specifies the envelope: {"type", "timestamp", "payload"}.
    The sse.py module adds type and timestamp. Here we verify the payload
    portion matches per-event-type schemas.
    """

    def test_stream_started_schema(self) -> None:
        mid = uuid4()
        data = StreamStartedPayload(message_id=mid).model_dump(mode="json")
        assert set(data.keys()) == {"message_id"}

    def test_stream_step_schema(self) -> None:
        data = StreamStepPayload(step="s", detail="d").model_dump(mode="json")
        assert set(data.keys()) == {"step", "detail"}

    def test_stream_token_schema(self) -> None:
        data = StreamTokenPayload(token="t").model_dump(mode="json")
        assert set(data.keys()) == {"token"}

    def test_stream_citation_schema(self) -> None:
        data = StreamCitationPayload(
            index=1, document_name="d", section="s", excerpt="e"
        ).model_dump(mode="json")
        assert set(data.keys()) == {"index", "document_name", "section", "excerpt"}

    def test_stream_confidence_schema(self) -> None:
        data = StreamConfidencePayload(score=0.5, label="med").model_dump(mode="json")
        assert set(data.keys()) == {"score", "label"}

    def test_stream_completed_schema(self) -> None:
        data = StreamCompletedPayload(message_id=uuid4()).model_dump(mode="json")
        assert set(data.keys()) == {"message_id"}

    def test_stream_error_schema(self) -> None:
        data = StreamErrorPayload(code="c", message="m").model_dump(mode="json")
        assert set(data.keys()) == {"code", "message"}

    def test_stream_cancelled_schema(self) -> None:
        data = StreamCancelledPayload(message_id=uuid4()).model_dump(mode="json")
        assert set(data.keys()) == {"message_id"}

    def test_processing_started_schema(self) -> None:
        data = ProcessingStartedPayload(document_id=uuid4(), filename="f").model_dump(mode="json")
        assert set(data.keys()) == {"document_id", "filename"}

    def test_processing_stage_schema(self) -> None:
        data = ProcessingStagePayload(stage="s", progress=0.5).model_dump(mode="json")
        assert set(data.keys()) == {"stage", "progress"}

    def test_processing_completed_schema(self) -> None:
        data = ProcessingCompletedPayload(document_id=uuid4(), chunk_count=1).model_dump(
            mode="json"
        )
        assert set(data.keys()) == {"document_id", "chunk_count"}

    def test_processing_error_schema(self) -> None:
        data = ProcessingErrorPayload(document_id=uuid4(), code="c", message="m").model_dump(
            mode="json"
        )
        assert set(data.keys()) == {"document_id", "code", "message"}

    def test_heartbeat_schema(self) -> None:
        data = HeartbeatPayload().model_dump(mode="json")
        assert data == {}

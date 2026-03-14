"""Pydantic payload models for SSE event types.

Each model defines the ``payload`` structure for a specific event type.
Models enforce field validation and produce JSON matching the design doc
schema (EPIC-T002 design.md, Event Schema section).

Payload models are infrastructure contracts. Consuming epics (EPIC-002,
EPIC-004) construct these payloads; the SSE infrastructure serialises them.
"""

from __future__ import annotations

from uuid import UUID  # noqa: TC003  # Pydantic needs UUID at runtime for validation

from pydantic import BaseModel, Field

from tabletop_oracle.streaming.event_types import (
    HeartbeatEventType,
    ProcessingEventType,
    StreamEventType,
)
from tabletop_oracle.streaming.events import SseEvent

# ---------------------------------------------------------------------------
# AI Streaming Payloads (stream.*)
# ---------------------------------------------------------------------------


class StreamStartedPayload(BaseModel):
    """Payload for ``stream.started``: stream has begun.

    Client should prepare the response display area.
    """

    message_id: UUID


class StreamStepPayload(BaseModel):
    """Payload for ``stream.step``: reasoning/processing step indicator.

    Transient -- not persisted in message history.
    Examples: "Retrieving knowledge", "Evaluating 12 sources".
    """

    step: str = Field(min_length=1)
    detail: str = Field(default="")


class StreamTokenPayload(BaseModel):
    """Payload for ``stream.token``: a single token of answer text.

    Client appends to the accumulating response.
    """

    token: str


class StreamCitationPayload(BaseModel):
    """Payload for ``stream.citation``: a citation reference.

    Emitted as citations are assembled during response generation.
    """

    index: int = Field(ge=1)
    document_name: str = Field(min_length=1)
    section: str = Field(default="")
    excerpt: str = Field(default="")


class StreamConfidencePayload(BaseModel):
    """Payload for ``stream.confidence``: answer confidence score.

    Emitted once near the end of the stream.
    """

    score: float = Field(ge=0.0, le=1.0)
    label: str = Field(min_length=1)


class StreamCompletedPayload(BaseModel):
    """Payload for ``stream.completed``: stream completed successfully.

    Client finalises the response display.
    """

    message_id: UUID


class StreamErrorPayload(BaseModel):
    """Payload for ``stream.error``: stream terminated due to error.

    Uses F001 error codes (e.g., ``SERVICE_UNAVAILABLE``).
    """

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class StreamCancelledPayload(BaseModel):
    """Payload for ``stream.cancelled``: stream cancelled by player.

    Confirms cancellation was processed.
    """

    message_id: UUID


# ---------------------------------------------------------------------------
# Document Processing Payloads (processing.*)
# ---------------------------------------------------------------------------


class ProcessingStartedPayload(BaseModel):
    """Payload for ``processing.started``: processing has begun."""

    document_id: UUID
    filename: str = Field(min_length=1)


class ProcessingStagePayload(BaseModel):
    """Payload for ``processing.stage``: entered a new pipeline stage.

    Stages: ``validating``, ``extracting``, ``detecting_structure``,
    ``chunking``, ``integrating``. Progress is 0.0 when indeterminate.
    """

    stage: str = Field(min_length=1)
    progress: float = Field(ge=0.0, le=1.0, default=0.0)


class ProcessingCompletedPayload(BaseModel):
    """Payload for ``processing.completed``: processing finished successfully."""

    document_id: UUID
    chunk_count: int = Field(ge=0)


class ProcessingErrorPayload(BaseModel):
    """Payload for ``processing.error``: processing failed.

    Human-readable error message per PRD-004 AC-407.
    """

    document_id: UUID
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Heartbeat Payload
# ---------------------------------------------------------------------------


class HeartbeatPayload(BaseModel):
    """Payload for ``heartbeat``: empty keep-alive."""

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Payload → Event type mapping
# ---------------------------------------------------------------------------

_PAYLOAD_EVENT_TYPE: dict[type[BaseModel], str] = {
    StreamStartedPayload: StreamEventType.STARTED,
    StreamStepPayload: StreamEventType.STEP,
    StreamTokenPayload: StreamEventType.TOKEN,
    StreamCitationPayload: StreamEventType.CITATION,
    StreamConfidencePayload: StreamEventType.CONFIDENCE,
    StreamCompletedPayload: StreamEventType.COMPLETED,
    StreamErrorPayload: StreamEventType.ERROR,
    StreamCancelledPayload: StreamEventType.CANCELLED,
    ProcessingStartedPayload: ProcessingEventType.STARTED,
    ProcessingStagePayload: ProcessingEventType.STAGE,
    ProcessingCompletedPayload: ProcessingEventType.COMPLETED,
    ProcessingErrorPayload: ProcessingEventType.ERROR,
    HeartbeatPayload: HeartbeatEventType.HEARTBEAT,
}


def payload_to_sse_event(payload: BaseModel) -> SseEvent:
    """Convert a typed payload model to an SseEvent for the SSE infrastructure.

    Looks up the event type from the payload class and serialises the
    payload to a plain dict. This bridges the typed Pydantic world with
    the generic ``SseEvent`` dataclass used by ``sse_response``.

    Args:
        payload: A Pydantic payload model instance.

    Returns:
        An SseEvent with the correct type and serialised payload.

    Raises:
        ValueError: If the payload type is not registered.
    """
    event_type = _PAYLOAD_EVENT_TYPE.get(type(payload))
    if event_type is None:
        msg = f"Unknown payload type: {type(payload).__name__}"
        raise ValueError(msg)
    return SseEvent(type=event_type, payload=payload.model_dump(mode="json"))

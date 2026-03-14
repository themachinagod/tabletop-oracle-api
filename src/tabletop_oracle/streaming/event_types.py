"""SSE event type constants with dot-separated namespacing.

Event types use ``<domain>.<event>`` naming. Each domain corresponds to
an owning epic:

- ``stream.*`` -- AI response streaming (EPIC-004)
- ``processing.*`` -- Document processing status (EPIC-002)
- ``heartbeat`` -- Infrastructure keep-alive (T002)
"""

import enum


class StreamEventType(enum.StrEnum):
    """AI response streaming event types (``stream.*`` namespace).

    Emitted on the per-message SSE endpoint during AI query processing.
    Consumed by EPIC-006 (Player Frontend).
    """

    STARTED = "stream.started"
    STEP = "stream.step"
    TOKEN = "stream.token"
    CITATION = "stream.citation"
    CONFIDENCE = "stream.confidence"
    COMPLETED = "stream.completed"
    ERROR = "stream.error"
    CANCELLED = "stream.cancelled"


class ProcessingEventType(enum.StrEnum):
    """Document processing event types (``processing.*`` namespace).

    Emitted on the per-document SSE endpoint during ingestion pipeline.
    Consumed by EPIC-007 (Curator Frontend).
    """

    STARTED = "processing.started"
    STAGE = "processing.stage"
    COMPLETED = "processing.completed"
    ERROR = "processing.error"


class HeartbeatEventType(enum.StrEnum):
    """Infrastructure heartbeat event type.

    Emitted by all SSE streams at a fixed interval to keep connections
    alive through proxies and load balancers.
    """

    HEARTBEAT = "heartbeat"

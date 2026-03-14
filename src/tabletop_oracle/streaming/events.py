"""SSE event data types."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SseEvent:
    """A single SSE event ready for serialisation.

    Represents a domain event that will be formatted into SSE wire format
    by the sse_response helper. Domain async generators yield these; the
    infrastructure handles framing, IDs, and timestamps.

    Attributes:
        type: Event type identifier (e.g., 'stream.token', 'processing.stage').
              Matches the SSE ``event`` field and is duplicated in the JSON payload
              for client convenience.
        payload: Event-specific data dictionary. Structure varies by event type
                 and is owned by the consuming epic, not the infrastructure.
    """

    type: str
    payload: dict[str, Any]

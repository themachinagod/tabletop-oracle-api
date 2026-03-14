"""SSE response helper wrapping async generators as StreamingResponse.

Handles SSE wire format, heartbeat injection, monotonic event IDs,
client disconnect detection, and reconnection replay via Last-Event-ID.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from starlette.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from starlette.requests import Request

    from tabletop_oracle.streaming.events import SseEvent

HEARTBEAT_INTERVAL_SECONDS: float = 15.0
"""Heartbeat interval in seconds. Keeps connections alive through proxies."""

EVENT_BUFFER_MAX_SIZE: int = 100
"""Maximum number of events retained for reconnection replay."""

EVENT_BUFFER_MAX_AGE_SECONDS: float = 60.0
"""Maximum age of buffered events in seconds. Events older than this are pruned."""


@dataclass
class _BufferedEvent:
    """An event stored in the replay buffer with its metadata."""

    event_id: int
    raw: str
    created_at: float = field(default_factory=time.monotonic)


class _EventBuffer:
    """Bounded buffer for SSE reconnection replay.

    Retains up to EVENT_BUFFER_MAX_SIZE events and prunes entries older
    than EVENT_BUFFER_MAX_AGE_SECONDS. The two constraints apply
    independently — whichever is stricter wins.
    """

    def __init__(self) -> None:
        self._events: deque[_BufferedEvent] = deque(maxlen=EVENT_BUFFER_MAX_SIZE)

    def append(self, event_id: int, raw: str) -> None:
        """Buffer a formatted SSE event for potential replay."""
        self._events.append(_BufferedEvent(event_id=event_id, raw=raw))

    def events_after(self, last_event_id: int) -> list[str]:
        """Return formatted events with IDs greater than *last_event_id*.

        Prunes expired entries before searching.
        """
        self._prune()
        return [e.raw for e in self._events if e.event_id > last_event_id]

    def _prune(self) -> None:
        """Remove events older than EVENT_BUFFER_MAX_AGE_SECONDS."""
        cutoff = time.monotonic() - EVENT_BUFFER_MAX_AGE_SECONDS
        while self._events and self._events[0].created_at < cutoff:
            self._events.popleft()


def _format_event(event_type: str, event_id: int, data: dict[str, object]) -> str:
    """Format a single SSE event in wire format.

    Returns:
        A string with ``event:``, ``id:``, ``data:`` fields terminated by
        a blank line, per the SSE specification.
    """
    json_data = json.dumps(data, separators=(",", ":"))
    return f"event: {event_type}\nid: {event_id}\ndata: {json_data}\n\n"


def _build_heartbeat(event_id: int) -> str:
    """Build a heartbeat SSE event."""
    data: dict[str, object] = {
        "type": "heartbeat",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {},
    }
    return _format_event("heartbeat", event_id, data)


def _build_event(event: SseEvent, event_id: int) -> str:
    """Build an SSE-formatted string for a domain event."""
    data: dict[str, object] = {
        "type": event.type,
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": event.payload,
    }
    return _format_event(event.type, event_id, data)


async def _stream_events(
    event_generator: AsyncGenerator[SseEvent, None],
    request: Request,
    buffer: _EventBuffer,
    start_id: int,
) -> AsyncGenerator[str, None]:
    """Internal async generator that produces SSE-formatted strings.

    Wraps the domain generator with heartbeat injection, event ID assignment,
    disconnect detection, and replay buffering.

    Args:
        event_generator: Domain async generator yielding SseEvent instances.
        request: Starlette request for disconnect detection.
        buffer: Shared event buffer for reconnection replay.
        start_id: Starting event ID (1-based for fresh streams, may be higher
                  after replay).
    """
    event_id = start_id

    try:
        while True:
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.wait_for(
                    event_generator.__anext__(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS,
                )
            except TimeoutError:
                # No event within the heartbeat window — emit heartbeat.
                event_id += 1
                heartbeat = _build_heartbeat(event_id)
                buffer.append(event_id, heartbeat)
                yield heartbeat
                continue
            except StopAsyncIteration:
                # Domain generator is exhausted — stream complete.
                break

            event_id += 1
            formatted = _build_event(event, event_id)
            buffer.append(event_id, formatted)
            yield formatted

    except asyncio.CancelledError:
        # Client disconnected — propagate cancellation to the domain generator
        # so it can clean up resources (e.g., abort LLM calls).
        await event_generator.aclose()
        raise
    finally:
        await event_generator.aclose()


async def sse_response(
    event_generator: AsyncGenerator[SseEvent, None],
    request: Request,
    last_event_id: int | None = None,
) -> StreamingResponse:
    """Create an SSE StreamingResponse from an async event generator.

    Wraps the generator with heartbeat injection, monotonic event ID
    assignment, and client disconnect detection. If *last_event_id* is
    provided (from a reconnecting client's ``Last-Event-ID`` header),
    buffered events after that ID are replayed before resuming the
    live stream.

    Args:
        event_generator: Async generator yielding SseEvent instances.
        request: The incoming Starlette request (for disconnect detection
                 and Last-Event-ID parsing).
        last_event_id: Last event ID from a reconnecting client. When
                       provided, events after this ID that are still in
                       the buffer are replayed immediately.

    Returns:
        A StreamingResponse with ``text/event-stream`` content type,
        ``Cache-Control: no-cache``, and ``Connection: keep-alive`` headers.
    """
    buffer = _EventBuffer()
    start_id = 0

    async def _generate() -> AsyncGenerator[str, None]:
        nonlocal start_id

        # Replay buffered events if reconnecting.
        if last_event_id is not None:
            replayed = buffer.events_after(last_event_id)
            for raw in replayed:
                yield raw
            # Continue IDs from where the buffer left off.
            if replayed:
                # Extract the highest replayed event ID. The ID is on
                # the second line of each SSE block: "id: <number>\n".
                last_raw = replayed[-1]
                for line in last_raw.splitlines():
                    if line.startswith("id: "):
                        start_id = int(line[4:])
                        break

        async for chunk in _stream_events(event_generator, request, buffer, start_id):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

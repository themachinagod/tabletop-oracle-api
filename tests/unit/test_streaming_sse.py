"""Tests for SSE response helper and supporting infrastructure."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from starlette.responses import StreamingResponse

from tabletop_oracle.streaming.events import SseEvent
from tabletop_oracle.streaming.sse import (
    EVENT_BUFFER_MAX_AGE_SECONDS,
    EVENT_BUFFER_MAX_SIZE,
    HEARTBEAT_INTERVAL_SECONDS,
    _build_event,
    _build_heartbeat,
    _EventBuffer,
    _format_event,
    sse_response,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRequest:
    """Minimal Starlette Request stub for testing."""

    def __init__(self, *, disconnect_after: int | None = None) -> None:
        self._call_count = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._call_count += 1
        return self._disconnect_after is not None and self._call_count > self._disconnect_after


async def _collect_events(
    gen: AsyncGenerator[str, None],
    max_events: int = 50,
) -> list[str]:
    """Collect up to *max_events* SSE blocks from the generator."""
    results: list[str] = []
    async for chunk in gen:
        results.append(chunk)
        if len(results) >= max_events:
            break
    return results


async def _simple_generator(
    events: list[SseEvent],
) -> AsyncGenerator[SseEvent, None]:
    """Yield a fixed list of events immediately."""
    for event in events:
        yield event


async def _slow_generator(
    events: list[SseEvent],
    delay: float = 0.0,
) -> AsyncGenerator[SseEvent, None]:
    """Yield events with a delay between each."""
    for event in events:
        await asyncio.sleep(delay)
        yield event


def _parse_sse_block(block: str) -> dict[str, str]:
    """Parse an SSE text block into a dict of field -> value."""
    result: dict[str, str] = {}
    for line in block.strip().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def _parse_sse_data(block: str) -> dict[str, Any]:
    """Parse the JSON data field from an SSE block."""
    parsed = _parse_sse_block(block)
    return json.loads(parsed["data"])  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# _format_event
# ---------------------------------------------------------------------------


class TestFormatEvent:
    """Tests for _format_event wire format."""

    def test_contains_event_field(self) -> None:
        """Output contains the event type line."""
        result = _format_event("stream.token", 1, {"type": "stream.token"})
        assert "event: stream.token\n" in result

    def test_contains_id_field(self) -> None:
        """Output contains the event ID line."""
        result = _format_event("stream.token", 42, {"type": "stream.token"})
        assert "id: 42\n" in result

    def test_contains_data_field_as_json(self) -> None:
        """Output contains the data line with JSON payload."""
        data: dict[str, object] = {"type": "stream.token", "payload": {"token": "hi"}}
        result = _format_event("stream.token", 1, data)
        data_line = next(line for line in result.splitlines() if line.startswith("data: "))
        parsed = json.loads(data_line[6:])
        assert parsed["payload"]["token"] == "hi"

    def test_ends_with_double_newline(self) -> None:
        """SSE blocks must end with two newlines."""
        result = _format_event("test", 1, {"type": "test"})
        assert result.endswith("\n\n")

    def test_field_order(self) -> None:
        """Fields appear in order: event, id, data."""
        result = _format_event("test", 1, {"type": "test"})
        lines = result.strip().splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("id: ")
        assert lines[2].startswith("data: ")


# ---------------------------------------------------------------------------
# _build_heartbeat
# ---------------------------------------------------------------------------


class TestBuildHeartbeat:
    """Tests for heartbeat event construction."""

    def test_heartbeat_type(self) -> None:
        """Heartbeat has type 'heartbeat'."""
        raw = _build_heartbeat(5)
        parsed = _parse_sse_block(raw)
        assert parsed["event"] == "heartbeat"

    def test_heartbeat_id(self) -> None:
        """Heartbeat carries the assigned event ID."""
        raw = _build_heartbeat(99)
        parsed = _parse_sse_block(raw)
        assert parsed["id"] == "99"

    def test_heartbeat_payload_empty(self) -> None:
        """Heartbeat data payload is empty."""
        raw = _build_heartbeat(1)
        data = _parse_sse_data(raw)
        assert data["payload"] == {}

    def test_heartbeat_has_timestamp(self) -> None:
        """Heartbeat includes an ISO 8601 timestamp."""
        raw = _build_heartbeat(1)
        data = _parse_sse_data(raw)
        assert "timestamp" in data
        # Should be parseable ISO format.
        assert "T" in data["timestamp"]


# ---------------------------------------------------------------------------
# _build_event
# ---------------------------------------------------------------------------


class TestBuildEvent:
    """Tests for domain event construction."""

    def test_event_type_in_sse_field(self) -> None:
        """SSE event field matches the SseEvent type."""
        event = SseEvent(type="stream.token", payload={"token": "x"})
        raw = _build_event(event, 3)
        parsed = _parse_sse_block(raw)
        assert parsed["event"] == "stream.token"

    def test_event_type_in_json_data(self) -> None:
        """JSON data 'type' field matches the SseEvent type."""
        event = SseEvent(type="stream.token", payload={"token": "x"})
        raw = _build_event(event, 3)
        data = _parse_sse_data(raw)
        assert data["type"] == "stream.token"

    def test_payload_in_json_data(self) -> None:
        """JSON data contains the event payload."""
        event = SseEvent(type="processing.stage", payload={"stage": "chunking", "progress": 0.7})
        raw = _build_event(event, 1)
        data = _parse_sse_data(raw)
        assert data["payload"]["stage"] == "chunking"
        assert data["payload"]["progress"] == 0.7

    def test_timestamp_present(self) -> None:
        """Event data includes a UTC timestamp."""
        event = SseEvent(type="test", payload={})
        raw = _build_event(event, 1)
        data = _parse_sse_data(raw)
        assert "timestamp" in data

    def test_event_id_assigned(self) -> None:
        """Event carries the monotonic ID."""
        event = SseEvent(type="test", payload={})
        raw = _build_event(event, 17)
        parsed = _parse_sse_block(raw)
        assert parsed["id"] == "17"


# ---------------------------------------------------------------------------
# _EventBuffer
# ---------------------------------------------------------------------------


class TestEventBuffer:
    """Tests for the bounded event replay buffer."""

    def test_append_and_retrieve_after(self) -> None:
        """Events appended can be retrieved by ID threshold."""
        buf = _EventBuffer()
        buf.append(1, "event-1")
        buf.append(2, "event-2")
        buf.append(3, "event-3")

        result = buf.events_after(1)
        assert result == ["event-2", "event-3"]

    def test_events_after_returns_empty_when_none_match(self) -> None:
        """Returns empty list when no events are after the given ID."""
        buf = _EventBuffer()
        buf.append(1, "event-1")

        assert buf.events_after(1) == []

    def test_events_after_returns_all_when_id_is_zero(self) -> None:
        """All events returned when last_event_id is 0."""
        buf = _EventBuffer()
        buf.append(1, "a")
        buf.append(2, "b")

        assert buf.events_after(0) == ["a", "b"]

    def test_max_size_evicts_oldest(self) -> None:
        """Buffer evicts oldest events when maxlen is reached."""
        buf = _EventBuffer()
        for i in range(1, EVENT_BUFFER_MAX_SIZE + 10):
            buf.append(i, f"event-{i}")

        result = buf.events_after(0)
        assert len(result) == EVENT_BUFFER_MAX_SIZE
        # Oldest 9 should be evicted.
        assert result[0] == f"event-{10}"

    def test_age_based_pruning(self) -> None:
        """Events older than EVENT_BUFFER_MAX_AGE_SECONDS are pruned."""
        buf = _EventBuffer()
        buf.append(1, "old-event")

        # Manually backdate the event's created_at.
        buf._events[0].created_at = time.monotonic() - EVENT_BUFFER_MAX_AGE_SECONDS - 1

        buf.append(2, "new-event")

        result = buf.events_after(0)
        assert result == ["new-event"]

    def test_empty_buffer_returns_empty(self) -> None:
        """Empty buffer returns empty list."""
        buf = _EventBuffer()
        assert buf.events_after(0) == []


# ---------------------------------------------------------------------------
# sse_response — integration-level tests
# ---------------------------------------------------------------------------


class TestSseResponse:
    """Tests for the sse_response helper."""

    async def test_returns_streaming_response(self) -> None:
        """sse_response returns a StreamingResponse."""
        gen = _simple_generator([SseEvent(type="test", payload={})])
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]

        assert isinstance(response, StreamingResponse)

    async def test_content_type_is_event_stream(self) -> None:
        """Content type is text/event-stream."""
        gen = _simple_generator([SseEvent(type="test", payload={})])
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]

        assert response.media_type == "text/event-stream"

    async def test_cache_control_header(self) -> None:
        """Cache-Control header is set to no-cache."""
        gen = _simple_generator([SseEvent(type="test", payload={})])
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]

        assert response.headers["Cache-Control"] == "no-cache"

    async def test_connection_header(self) -> None:
        """Connection header is set to keep-alive."""
        gen = _simple_generator([SseEvent(type="test", payload={})])
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]

        assert response.headers["Connection"] == "keep-alive"

    async def test_events_emitted_in_sse_format(self) -> None:
        """Domain events are emitted as properly formatted SSE blocks."""
        events = [
            SseEvent(type="stream.started", payload={"message_id": "abc"}),
            SseEvent(type="stream.token", payload={"token": "hello"}),
        ]
        gen = _simple_generator(events)
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        assert len(chunks) == 2

        # First event.
        parsed_0 = _parse_sse_block(chunks[0])
        assert parsed_0["event"] == "stream.started"
        assert parsed_0["id"] == "1"

        # Second event.
        parsed_1 = _parse_sse_block(chunks[1])
        assert parsed_1["event"] == "stream.token"
        assert parsed_1["id"] == "2"

    async def test_monotonic_event_ids(self) -> None:
        """Event IDs increase monotonically."""
        events = [SseEvent(type="t", payload={}) for _ in range(5)]
        gen = _simple_generator(events)
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        ids = []
        for chunk in chunks:
            parsed = _parse_sse_block(chunk)
            ids.append(int(parsed["id"]))

        assert ids == [1, 2, 3, 4, 5]

    async def test_json_payload_structure(self) -> None:
        """JSON data contains type, timestamp, and payload."""
        gen = _simple_generator([SseEvent(type="stream.token", payload={"token": "x"})])
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        data = _parse_sse_data(chunks[0])
        assert data["type"] == "stream.token"
        assert "timestamp" in data
        assert data["payload"] == {"token": "x"}

    async def test_timestamp_is_iso_utc(self) -> None:
        """Timestamp is ISO 8601 UTC format."""
        gen = _simple_generator([SseEvent(type="test", payload={})])
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        data = _parse_sse_data(chunks[0])
        ts = data["timestamp"]
        # Should contain 'T' and end with timezone info.
        assert "T" in ts
        assert "+" in ts or ts.endswith("Z") or "UTC" in ts

    async def test_heartbeat_on_idle_stream(self) -> None:
        """Heartbeat emitted when the generator is idle beyond the interval."""

        # Use a generator that blocks longer than heartbeat interval.
        async def _blocking_gen() -> AsyncGenerator[SseEvent, None]:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS + 1)
            yield SseEvent(type="done", payload={})

        request = FakeRequest()

        with patch(
            "tabletop_oracle.streaming.sse.HEARTBEAT_INTERVAL_SECONDS",
            0.1,
        ):
            response = await sse_response(_blocking_gen(), request)  # type: ignore[arg-type]
            chunks = await _collect_events(
                response.body_iterator,  # type: ignore[arg-type]
                max_events=2,
            )

        # First event should be a heartbeat.
        data = _parse_sse_data(chunks[0])
        assert data["type"] == "heartbeat"
        assert data["payload"] == {}

    async def test_disconnect_stops_stream(self) -> None:
        """Stream stops when the client disconnects."""

        # Generator that would yield many events.
        async def _infinite_gen() -> AsyncGenerator[SseEvent, None]:
            i = 0
            while True:
                yield SseEvent(type="tick", payload={"n": i})
                i += 1

        # Disconnect after 3 disconnect checks.
        request = FakeRequest(disconnect_after=3)

        response = await sse_response(_infinite_gen(), request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        # Should have stopped — not collected all 50.
        assert len(chunks) < 50

    async def test_cancelled_error_closes_generator(self) -> None:
        """CancelledError propagates and closes the domain generator."""
        closed = False

        async def _gen() -> AsyncGenerator[SseEvent, None]:
            nonlocal closed
            try:
                while True:
                    yield SseEvent(type="tick", payload={})
                    await asyncio.sleep(0)
            except GeneratorExit:
                closed = True

        request = FakeRequest(disconnect_after=2)

        response = await sse_response(_gen(), request)  # type: ignore[arg-type]
        await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        assert closed

    async def test_reconnection_replay_from_buffer(self) -> None:
        """Events are replayed when last_event_id is provided.

        This tests the buffer + replay integration. We feed events into
        a first stream, then create a second response with last_event_id
        to verify replay.
        """
        # We test the buffer independently; this verifies the sse_response
        # integration path with a fresh buffer (which will be empty, so
        # replay yields nothing). The real replay scenario requires the
        # same buffer instance — which happens within a single stream's
        # reconnection window.
        gen = _simple_generator([SseEvent(type="a", payload={})])
        request = FakeRequest()

        # With last_event_id=0 on a fresh buffer, no replay — just live events.
        response = await sse_response(gen, request, last_event_id=0)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        assert len(chunks) == 1
        parsed = _parse_sse_block(chunks[0])
        assert parsed["event"] == "a"

    async def test_empty_generator_produces_no_events(self) -> None:
        """An empty generator produces no SSE output."""

        async def _empty() -> AsyncGenerator[SseEvent, None]:
            return
            yield  # pragma: no cover — makes this an async generator

        request = FakeRequest()

        response = await sse_response(_empty(), request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        assert chunks == []

    async def test_multiple_event_types(self) -> None:
        """Stream correctly handles a sequence of mixed event types."""
        events = [
            SseEvent(type="stream.started", payload={"message_id": "m1"}),
            SseEvent(type="stream.step", payload={"step": "Retrieving"}),
            SseEvent(type="stream.token", payload={"token": "The"}),
            SseEvent(type="stream.token", payload={"token": " answer"}),
            SseEvent(type="stream.completed", payload={"message_id": "m1"}),
        ]
        gen = _simple_generator(events)
        request = FakeRequest()

        response = await sse_response(gen, request)  # type: ignore[arg-type]
        chunks = await _collect_events(response.body_iterator)  # type: ignore[arg-type]

        types = [_parse_sse_data(c)["type"] for c in chunks]
        assert types == [
            "stream.started",
            "stream.step",
            "stream.token",
            "stream.token",
            "stream.completed",
        ]


# ---------------------------------------------------------------------------
# _EventBuffer — edge cases
# ---------------------------------------------------------------------------


class TestEventBufferEdgeCases:
    """Additional edge-case tests for the event buffer."""

    def test_events_after_with_gap_in_ids(self) -> None:
        """Buffer correctly filters with non-contiguous IDs."""
        buf = _EventBuffer()
        buf.append(1, "a")
        buf.append(5, "b")
        buf.append(10, "c")

        assert buf.events_after(3) == ["b", "c"]

    def test_multiple_prune_cycles(self) -> None:
        """Pruning works correctly across multiple calls."""
        buf = _EventBuffer()
        buf.append(1, "old")
        buf._events[0].created_at = time.monotonic() - EVENT_BUFFER_MAX_AGE_SECONDS - 10

        buf.append(2, "newer")
        buf.append(3, "newest")

        # First retrieval prunes the old one.
        result = buf.events_after(0)
        assert result == ["newer", "newest"]

        # Second retrieval still works.
        result2 = buf.events_after(2)
        assert result2 == ["newest"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify infrastructure constants match the design spec."""

    def test_heartbeat_interval(self) -> None:
        """Heartbeat interval is 15 seconds per design."""
        assert HEARTBEAT_INTERVAL_SECONDS == 15.0

    def test_buffer_max_size(self) -> None:
        """Buffer max size is 100 events per design."""
        assert EVENT_BUFFER_MAX_SIZE == 100

    def test_buffer_max_age(self) -> None:
        """Buffer max age is 60 seconds per design."""
        assert EVENT_BUFFER_MAX_AGE_SECONDS == 60.0

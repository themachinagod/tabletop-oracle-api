"""Tests for SSE event type constants."""

from tabletop_oracle.streaming.event_types import (
    HeartbeatEventType,
    ProcessingEventType,
    StreamEventType,
)


class TestStreamEventType:
    """StreamEventType enum values and namespacing."""

    def test_started_value(self) -> None:
        assert StreamEventType.STARTED == "stream.started"

    def test_step_value(self) -> None:
        assert StreamEventType.STEP == "stream.step"

    def test_token_value(self) -> None:
        assert StreamEventType.TOKEN == "stream.token"

    def test_citation_value(self) -> None:
        assert StreamEventType.CITATION == "stream.citation"

    def test_confidence_value(self) -> None:
        assert StreamEventType.CONFIDENCE == "stream.confidence"

    def test_completed_value(self) -> None:
        assert StreamEventType.COMPLETED == "stream.completed"

    def test_error_value(self) -> None:
        assert StreamEventType.ERROR == "stream.error"

    def test_cancelled_value(self) -> None:
        assert StreamEventType.CANCELLED == "stream.cancelled"

    def test_all_use_stream_namespace(self) -> None:
        for member in StreamEventType:
            assert member.value.startswith("stream."), (
                f"{member.name} does not use stream.* namespace"
            )

    def test_member_count(self) -> None:
        assert len(StreamEventType) == 8

    def test_is_str(self) -> None:
        """StrEnum values are usable as plain strings."""
        assert isinstance(StreamEventType.TOKEN, str)
        assert StreamEventType.TOKEN == "stream.token"


class TestProcessingEventType:
    """ProcessingEventType enum values and namespacing."""

    def test_started_value(self) -> None:
        assert ProcessingEventType.STARTED == "processing.started"

    def test_stage_value(self) -> None:
        assert ProcessingEventType.STAGE == "processing.stage"

    def test_completed_value(self) -> None:
        assert ProcessingEventType.COMPLETED == "processing.completed"

    def test_error_value(self) -> None:
        assert ProcessingEventType.ERROR == "processing.error"

    def test_all_use_processing_namespace(self) -> None:
        for member in ProcessingEventType:
            assert member.value.startswith("processing."), (
                f"{member.name} does not use processing.* namespace"
            )

    def test_member_count(self) -> None:
        assert len(ProcessingEventType) == 4

    def test_is_str(self) -> None:
        assert isinstance(ProcessingEventType.STAGE, str)


class TestHeartbeatEventType:
    """HeartbeatEventType enum value."""

    def test_heartbeat_value(self) -> None:
        assert HeartbeatEventType.HEARTBEAT == "heartbeat"

    def test_no_namespace(self) -> None:
        """Heartbeat is infrastructure-level, no dot namespace."""
        assert "." not in HeartbeatEventType.HEARTBEAT.value

    def test_member_count(self) -> None:
        assert len(HeartbeatEventType) == 1

    def test_is_str(self) -> None:
        assert isinstance(HeartbeatEventType.HEARTBEAT, str)

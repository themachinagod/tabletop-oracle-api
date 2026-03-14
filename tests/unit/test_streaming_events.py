"""Tests for SseEvent dataclass."""

from tabletop_oracle.streaming.events import SseEvent


class TestSseEvent:
    """Tests for the SseEvent dataclass."""

    def test_create_with_type_and_payload(self) -> None:
        """SseEvent stores type and payload."""
        event = SseEvent(type="stream.token", payload={"token": "hello"})

        assert event.type == "stream.token"
        assert event.payload == {"token": "hello"}

    def test_frozen_immutability(self) -> None:
        """SseEvent is immutable (frozen dataclass)."""
        event = SseEvent(type="heartbeat", payload={})

        try:
            event.type = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True

        assert raised, "Frozen dataclass should prevent attribute mutation"

    def test_equality_based_on_fields(self) -> None:
        """Two SseEvents with same fields are equal."""
        a = SseEvent(type="stream.started", payload={"message_id": "abc"})
        b = SseEvent(type="stream.started", payload={"message_id": "abc"})

        assert a == b

    def test_inequality_different_type(self) -> None:
        """SseEvents with different types are not equal."""
        a = SseEvent(type="stream.started", payload={})
        b = SseEvent(type="stream.completed", payload={})

        assert a != b

    def test_inequality_different_payload(self) -> None:
        """SseEvents with different payloads are not equal."""
        a = SseEvent(type="stream.token", payload={"token": "a"})
        b = SseEvent(type="stream.token", payload={"token": "b"})

        assert a != b

    def test_empty_payload(self) -> None:
        """SseEvent accepts an empty payload dict."""
        event = SseEvent(type="heartbeat", payload={})

        assert event.payload == {}

    def test_complex_payload(self) -> None:
        """SseEvent accepts nested payload structures."""
        payload = {
            "stage": "extracting",
            "progress": 0.5,
            "metadata": {"pages": 42, "format": "pdf"},
        }
        event = SseEvent(type="processing.stage", payload=payload)

        assert event.payload["metadata"]["pages"] == 42

"""Tests for ingestion pipeline event publishers."""

from uuid import uuid4

from tabletop_oracle.services.ingestion.events import LogEventPublisher


class TestLogEventPublisher:
    """Tests for LogEventPublisher."""

    def test_publish_logs_event(self) -> None:
        """LogEventPublisher logs event data without error."""
        publisher = LogEventPublisher()
        doc_id = uuid4()
        publisher.publish(
            doc_id,
            "processing.started",
            {"document_id": str(doc_id), "filename": "rules.pdf"},
        )
        # No exception = pass (log output is a side effect)

    def test_publish_multiple_events(self) -> None:
        """LogEventPublisher handles multiple events without state issues."""
        publisher = LogEventPublisher()
        doc_id = uuid4()

        publisher.publish(doc_id, "processing.started", {"document_id": str(doc_id)})
        publisher.publish(doc_id, "processing.stage", {"stage": "validating"})
        publisher.publish(doc_id, "processing.completed", {"chunk_count": 5})
        # No exception = pass

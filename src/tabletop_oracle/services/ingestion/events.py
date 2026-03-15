"""SSE event emission for the ingestion pipeline.

Publishes processing events to Redis Pub/Sub for delivery via the T002
SSE infrastructure. Events are published WITHOUT timestamps; the
sse_response helper adds timestamps at delivery time.

The Celery worker process is synchronous (prefork pool). Redis publish
calls use a synchronous Redis client to avoid asyncio complexity in
the sync task context.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)


class EventPublisher(Protocol):
    """Protocol for publishing pipeline events.

    Abstracts the Redis publish mechanism so the pipeline orchestrator
    is testable without Redis.
    """

    def publish(self, document_id: UUID, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a processing event.

        Args:
            document_id: UUID of the document being processed.
            event_type: Event type (e.g., 'processing.started').
            payload: Event-specific data.
        """
        ...


class RedisEventPublisher:
    """Publishes pipeline events to Redis Pub/Sub.

    Uses synchronous Redis client for compatibility with Celery's
    prefork pool (sync workers).

    Args:
        redis_url: Redis connection URL.
    """

    def __init__(self, redis_url: str) -> None:
        # Lazy import to avoid requiring redis at module load time
        import redis as redis_lib

        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def publish(self, document_id: UUID, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a processing event to Redis Pub/Sub.

        Events are published to channel ``document:{document_id}:processing``.

        Args:
            document_id: UUID of the document being processed.
            event_type: Event type (e.g., 'processing.started').
            payload: Event-specific data.
        """
        channel = f"document:{document_id}:processing"
        event = {"type": event_type, "payload": payload}
        self._redis.publish(channel, json.dumps(event))
        logger.debug("Published %s to %s", event_type, channel)


class LogEventPublisher:
    """Publishes pipeline events to the log.

    Fallback publisher for environments without Redis (dev, testing).
    """

    def publish(self, document_id: UUID, event_type: str, payload: dict[str, Any]) -> None:
        """Log a processing event instead of publishing to Redis.

        Args:
            document_id: UUID of the document being processed.
            event_type: Event type (e.g., 'processing.started').
            payload: Event-specific data.
        """
        logger.info(
            "Pipeline event: document_id=%s, type=%s, payload=%s",
            document_id,
            event_type,
            payload,
        )

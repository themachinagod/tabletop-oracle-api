"""Celery task definitions for background processing.

Tasks are thin wrappers that set up dependencies (DB session, event
publisher) and delegate to domain logic. The pipeline orchestrator
contains all business logic; the task provides the Celery lifecycle
(retries, error propagation).
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tabletop_oracle.config import settings
from tabletop_oracle.services.ingestion.events import RedisEventPublisher
from tabletop_oracle.services.ingestion.pipeline import IngestionPipeline
from tabletop_oracle.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Synchronous engine for Celery worker tasks (prefork pool uses sync workers).
_sync_engine = create_engine(settings.database_url, echo=False)


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="tabletop_oracle.workers.tasks.process_document",
    max_retries=2,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
)
def process_document(self: object, document_id: str, version_id: str) -> None:
    """Execute the ingestion pipeline for a document version.

    Celery task entry point. Sets up a synchronous DB session and event
    publisher, then delegates to IngestionPipeline.process().

    Uses prefork pool (sync workers). Retries up to 2 times with
    exponential backoff on any exception.

    Args:
        self: Celery task instance (bound task).
        document_id: UUID string of the document being processed.
        version_id: UUID string of the specific version to process.
    """
    from uuid import UUID

    logger.info(
        "process_document task started: document_id=%s, version_id=%s",
        document_id,
        version_id,
    )

    doc_uuid = UUID(document_id)
    ver_uuid = UUID(version_id)

    publisher = RedisEventPublisher(settings.redis_url)

    with Session(_sync_engine) as db_session:
        pipeline = IngestionPipeline(db_session, publisher)
        pipeline.process(doc_uuid, ver_uuid)

    logger.info(
        "process_document task completed: document_id=%s, version_id=%s",
        document_id,
        version_id,
    )

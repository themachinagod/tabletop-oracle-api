"""Celery application configuration.

Configures the Celery app with Redis broker, prefork pool defaults,
and task autodiscovery. Task results are persisted in PostgreSQL (not
the Celery result backend) for durability; the Redis result backend
is configured only for Celery's internal bookkeeping.
"""

from celery import Celery  # type: ignore[import-untyped]

from tabletop_oracle.config import settings

celery_app = Celery(
    "tabletop_oracle",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Worker
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    # Task defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Autodiscover tasks in the workers package
    include=["tabletop_oracle.workers.tasks"],
)

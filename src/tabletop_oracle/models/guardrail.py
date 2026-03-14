"""GuardrailConfig ORM model.

Maps to the ``guardrail_config`` table. Single-row table for global token
and query budget limits. Has ``updated_at`` (trigger-managed) but no
``created_at``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Boolean, Integer, func, text
from sqlalchemy.orm import Mapped, mapped_column

from tabletop_oracle.models.base import MappedBase


class GuardrailConfig(MappedBase):
    """Global guardrail configuration (single-row table).

    All limit fields are optional; NULL means no limit is enforced for
    that dimension. The ``enforcement_enabled`` flag is a global kill
    switch for all guardrails.

    Attributes:
        enforcement_enabled: Master switch for guardrail enforcement.
        max_tokens_per_query: Per-query token ceiling (optional).
        max_model_calls_per_query: Per-query model call ceiling (optional).
        max_queries_per_session: Per-session query ceiling (optional).
        daily_token_budget: Daily token budget across all users (optional).
        daily_query_budget: Daily query budget across all users (optional).
        per_document_ingestion_limit: Token limit per document ingestion (optional).
        updated_at: Last modification timestamp (trigger-managed).
    """

    __tablename__ = "guardrail_config"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    enforcement_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("TRUE"),
    )
    max_tokens_per_query: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_model_calls_per_query: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_queries_per_session: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_token_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_query_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    per_document_ingestion_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

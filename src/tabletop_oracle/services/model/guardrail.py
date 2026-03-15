"""Guardrail enforcement service and protocol.

Defines the ``GuardrailServiceProtocol`` interface that ``ModelClient``
depends on, and the concrete ``GuardrailService`` that reads limits from
the ``guardrail_config`` table and checks current usage via
``TokenUsageService``.

Enforcement is skipped entirely when ``guardrail_config.enforcement_enabled``
is False (master toggle per PRD-007 Section 3.3).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sqlalchemy import select

from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.services.model.types import GuardrailCheckResult

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.services.model.token_usage import TokenUsageService

logger = logging.getLogger(__name__)

# Graceful degradation messages per PRD-007 Section 3.3
_PER_QUERY_DENIAL = (
    "I've reached the processing limit for this question. "
    "Try simplifying your question or breaking it into smaller parts."
)
_PER_SESSION_DENIAL = (
    "This session has reached its question limit. Start a new session to continue."
)
_DAILY_DENIAL = (
    "The daily usage limit has been reached. Try again tomorrow or contact your administrator."
)


@runtime_checkable
class GuardrailServiceProtocol(Protocol):
    """Enforces token and query guardrails.

    Reads limits from the ``guardrail_config`` table. Checks current
    usage against limits before each model call.
    """

    async def check_pre_query(
        self,
        session_id: UUID,
    ) -> GuardrailCheckResult:
        """Check session-level and daily guardrails before starting a query.

        Args:
            session_id: The session context.

        Returns:
            GuardrailCheckResult with allowed flag and denial message.
        """
        ...

    async def check_pre_call(
        self,
        session_id: UUID,
        message_id: UUID,
    ) -> GuardrailCheckResult:
        """Check per-query guardrails before an individual model call.

        Checks per-query token total and per-query model call count.

        Args:
            session_id: The session context.
            message_id: The message context.

        Returns:
            GuardrailCheckResult with allowed flag and denial message.
        """
        ...


class GuardrailService:
    """Enforces token and query guardrails.

    Reads limits from the ``guardrail_config`` table (single row).
    Checks current usage against limits before each model call.

    Enforcement is skipped entirely when
    ``guardrail_config.enforcement_enabled`` is False (master toggle
    per PRD-007 Section 3.3).

    Args:
        db: SQLAlchemy async session for database operations.
        token_usage_service: Service providing usage aggregation queries.
    """

    def __init__(
        self,
        db: AsyncSession,
        token_usage_service: TokenUsageService,
    ) -> None:
        self._db = db
        self._token_usage_service = token_usage_service

    async def check_pre_query(
        self,
        session_id: UUID,
    ) -> GuardrailCheckResult:
        """Check session-level and daily guardrails before starting a query.

        Checks (in order, short-circuiting on first violation):
        - Per-session query count vs ``max_queries_per_session``.
        - Daily token budget vs ``daily_token_budget``.
        - Daily query budget vs ``daily_query_budget``.

        Args:
            session_id: The session context.

        Returns:
            GuardrailCheckResult with allowed flag and denial message.
        """
        config = await self._load_config()
        if config is None or not config.enforcement_enabled:
            return GuardrailCheckResult(allowed=True)

        # Per-session query count
        if config.max_queries_per_session is not None:
            count = await self._token_usage_service.get_session_query_count(
                session_id=session_id,
            )
            if count >= config.max_queries_per_session:
                logger.warning(
                    "Per-session query limit exceeded",
                    extra={
                        "guardrail_type": "per_session_query",
                        "current_value": count,
                        "limit": config.max_queries_per_session,
                        "session_id": str(session_id),
                    },
                )
                return GuardrailCheckResult(
                    allowed=False,
                    message=_PER_SESSION_DENIAL,
                    guardrail_type="per_session_query",
                )

        # Daily token budget
        if config.daily_token_budget is not None:
            daily_tokens = await self._token_usage_service.get_daily_total()
            if daily_tokens >= config.daily_token_budget:
                logger.warning(
                    "Daily token budget exceeded",
                    extra={
                        "guardrail_type": "daily_token_budget",
                        "current_value": daily_tokens,
                        "limit": config.daily_token_budget,
                    },
                )
                return GuardrailCheckResult(
                    allowed=False,
                    message=_DAILY_DENIAL,
                    guardrail_type="daily_token_budget",
                )

        # Daily query budget
        if config.daily_query_budget is not None:
            daily_queries = await self._token_usage_service.get_daily_query_count()
            if daily_queries >= config.daily_query_budget:
                logger.warning(
                    "Daily query budget exceeded",
                    extra={
                        "guardrail_type": "daily_query_budget",
                        "current_value": daily_queries,
                        "limit": config.daily_query_budget,
                    },
                )
                return GuardrailCheckResult(
                    allowed=False,
                    message=_DAILY_DENIAL,
                    guardrail_type="daily_query_budget",
                )

        logger.info(
            "Pre-query guardrails passed",
            extra={
                "guardrail_type": "pre_query",
                "session_id": str(session_id),
            },
        )
        return GuardrailCheckResult(allowed=True)

    async def check_pre_call(
        self,
        session_id: UUID,
        message_id: UUID,
    ) -> GuardrailCheckResult:
        """Check per-query guardrails before an individual model call.

        Checks (in order, short-circuiting on first violation):
        - Per-query token total vs ``max_tokens_per_query``.
        - Per-query model call count vs ``max_model_calls_per_query``.

        Args:
            session_id: The session context.
            message_id: The message context.

        Returns:
            GuardrailCheckResult with allowed flag and denial message.
        """
        config = await self._load_config()
        if config is None or not config.enforcement_enabled:
            return GuardrailCheckResult(allowed=True)

        # Per-query token total
        if config.max_tokens_per_query is not None:
            token_total = await self._token_usage_service.get_query_total(
                session_id=session_id,
                message_id=message_id,
            )
            if token_total >= config.max_tokens_per_query:
                logger.warning(
                    "Per-query token limit exceeded",
                    extra={
                        "guardrail_type": "per_query_token",
                        "current_value": token_total,
                        "limit": config.max_tokens_per_query,
                        "message_id": str(message_id),
                    },
                )
                return GuardrailCheckResult(
                    allowed=False,
                    message=_PER_QUERY_DENIAL,
                    guardrail_type="per_query_token",
                )

        # Per-query model call count
        if config.max_model_calls_per_query is not None:
            call_count = await self._token_usage_service.get_query_model_call_count(
                session_id=session_id,
                message_id=message_id,
            )
            if call_count >= config.max_model_calls_per_query:
                logger.warning(
                    "Per-query model call limit exceeded",
                    extra={
                        "guardrail_type": "per_query_model_calls",
                        "current_value": call_count,
                        "limit": config.max_model_calls_per_query,
                        "message_id": str(message_id),
                    },
                )
                return GuardrailCheckResult(
                    allowed=False,
                    message=_PER_QUERY_DENIAL,
                    guardrail_type="per_query_model_calls",
                )

        logger.info(
            "Pre-call guardrails passed",
            extra={
                "guardrail_type": "pre_call",
                "session_id": str(session_id),
                "message_id": str(message_id),
            },
        )
        return GuardrailCheckResult(allowed=True)

    async def _load_config(self) -> GuardrailConfig | None:
        """Load the single-row guardrail configuration from the database.

        Returns the first (and only) row from ``guardrail_config``, or
        None if the table is empty. No caching -- every check reads
        fresh config so admin changes take effect immediately.

        Returns:
            GuardrailConfig or None if no configuration exists.
        """
        stmt = select(GuardrailConfig).limit(1)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

"""Quality metrics service for the admin dashboard.

Aggregates quality signals from ``messages``, ``citations``,
``player_feedback``, and ``sessions`` for per-game quality reporting.
All methods are read-only aggregation queries. No data is written.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.types import Float as SAFloat

from tabletop_oracle.errors.exceptions import ValidationError
from tabletop_oracle.models.enums import FeedbackRating, MessageType
from tabletop_oracle.models.feedback import PlayerFeedback
from tabletop_oracle.models.game import Game
from tabletop_oracle.models.message import Citation, Message
from tabletop_oracle.models.session import Session
from tabletop_oracle.schemas.quality import (
    ConfidenceDistribution,
    GameQualityMetricsResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_MAX_RANGE_DAYS = 365


class QualityMetricsService:
    """Computes quality signals from existing data.

    Reads from messages (confidence_score, type), citations,
    player_feedback, and sessions (game_id). All computations
    are read-only aggregations -- no new data is written.

    Args:
        db: SQLAlchemy async session for database operations.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_metrics(
        self,
        period_start: date,
        period_end: date,
    ) -> list[GameQualityMetricsResponse]:
        """Get quality metrics per game for the given period.

        For each game with activity in the period, computes:
        - low_confidence_rate: fraction of ai_answer messages with
          confidence_score < 0.5
        - clarification_rate: fraction of user_question messages that
          triggered an ai_clarification response
        - avg_citations_per_answer: mean citation count per ai_answer
        - feedback_positive_count / feedback_negative_count: from
          player_feedback
        - confidence_distribution: histogram buckets (0-0.25, 0.25-0.5,
          0.5-0.75, 0.75-1.0)
        - avg_queries_per_session: mean user_question count per session

        Args:
            period_start: Inclusive start of the aggregation period.
            period_end: Inclusive end of the aggregation period.

        Returns:
            List of GameQualityMetricsResponse ordered by game name.

        Raises:
            ValidationError: If the date range is invalid.
        """
        self._validate_date_range(period_start, period_end)

        start_ts, end_ts = self._date_range_timestamps(period_start, period_end)

        games = await self._get_active_games(start_ts, end_ts)

        results: list[GameQualityMetricsResponse] = []
        for game_id, game_name in games:
            metrics = await self._compute_game_metrics(game_id, game_name, start_ts, end_ts)
            results.append(metrics)

        return results

    async def _get_active_games(self, start_ts: str, end_ts: str) -> list[tuple[object, str]]:
        """Find games that have message activity in the period.

        Args:
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            List of (game_id, game_name) tuples ordered by game name.
        """
        stmt = (
            select(Game.id, Game.name)
            .join(Session, Session.game_id == Game.id)
            .join(Message, Message.session_id == Session.id)
            .where(
                Message.created_at >= start_ts,
                Message.created_at < end_ts,
            )
            .group_by(Game.id, Game.name)
            .order_by(Game.name)
        )
        result = await self._db.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def _compute_game_metrics(
        self,
        game_id: object,
        game_name: str,
        start_ts: str,
        end_ts: str,
    ) -> GameQualityMetricsResponse:
        """Compute all quality metrics for a single game.

        Args:
            game_id: UUID of the game.
            game_name: Display name of the game.
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            GameQualityMetricsResponse with all computed metrics.
        """
        message_stats = await self._get_message_stats(game_id, start_ts, end_ts)
        citation_avg = await self._get_avg_citations(game_id, start_ts, end_ts)
        feedback = await self._get_feedback_counts(game_id, start_ts, end_ts)
        confidence_dist = await self._get_confidence_distribution(game_id, start_ts, end_ts)
        queries_per_session = await self._get_avg_queries_per_session(game_id, start_ts, end_ts)

        ai_answer_count = message_stats["ai_answer_count"]
        low_conf_count = message_stats["low_confidence_count"]
        user_question_count = message_stats["user_question_count"]
        clarification_count = message_stats["clarification_count"]

        low_confidence_rate = low_conf_count / ai_answer_count if ai_answer_count > 0 else 0.0
        clarification_rate = (
            clarification_count / user_question_count if user_question_count > 0 else 0.0
        )

        return GameQualityMetricsResponse(
            game_id=str(game_id),
            game_name=game_name,
            low_confidence_rate=low_confidence_rate,
            clarification_rate=clarification_rate,
            avg_citations_per_answer=citation_avg,
            feedback_positive_count=feedback["positive"],
            feedback_negative_count=feedback["negative"],
            confidence_distribution=confidence_dist,
            avg_queries_per_session=queries_per_session,
        )

    async def _get_message_stats(
        self,
        game_id: object,
        start_ts: str,
        end_ts: str,
    ) -> dict[str, int]:
        """Get message type counts and low-confidence count for a game.

        Args:
            game_id: UUID of the game.
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            Dict with ai_answer_count, low_confidence_count,
            user_question_count, and clarification_count.
        """
        stmt = (
            select(
                func.count().filter(Message.type == MessageType.AI_ANSWER),
                func.count().filter(
                    Message.type == MessageType.AI_ANSWER,
                    Message.confidence_score < 0.5,
                ),
                func.count().filter(Message.type == MessageType.USER_QUESTION),
                func.count().filter(Message.type == MessageType.AI_CLARIFICATION),
            )
            .join(Session, Message.session_id == Session.id)
            .where(
                Session.game_id == game_id,
                Message.created_at >= start_ts,
                Message.created_at < end_ts,
            )
        )
        result = await self._db.execute(stmt)
        row = result.one()

        return {
            "ai_answer_count": int(row[0]),
            "low_confidence_count": int(row[1]),
            "user_question_count": int(row[2]),
            "clarification_count": int(row[3]),
        }

    async def _get_avg_citations(
        self,
        game_id: object,
        start_ts: str,
        end_ts: str,
    ) -> float:
        """Get average citation count per ai_answer message for a game.

        Uses a subquery to count citations per message, then averages.

        Args:
            game_id: UUID of the game.
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            Average citations per answer, 0.0 if no answers exist.
        """
        citation_counts = (
            select(
                Message.id.label("message_id"),
                func.count(Citation.id).label("citation_count"),
            )
            .outerjoin(Citation, Citation.message_id == Message.id)
            .join(Session, Message.session_id == Session.id)
            .where(
                Session.game_id == game_id,
                Message.type == MessageType.AI_ANSWER,
                Message.created_at >= start_ts,
                Message.created_at < end_ts,
            )
            .group_by(Message.id)
            .subquery()
        )

        stmt = select(
            func.coalesce(func.avg(func.cast(citation_counts.c.citation_count, SAFloat)), 0.0)
        )
        result = await self._db.execute(stmt)
        return float(result.scalar_one())

    async def _get_feedback_counts(
        self,
        game_id: object,
        start_ts: str,
        end_ts: str,
    ) -> dict[str, int]:
        """Get positive and negative feedback counts for a game.

        Filters feedback by game_id (denormalised on player_feedback)
        and by the feedback creation timestamp within the period.

        Args:
            game_id: UUID of the game.
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            Dict with 'positive' and 'negative' counts.
        """
        stmt = select(
            func.count().filter(PlayerFeedback.rating == FeedbackRating.POSITIVE),
            func.count().filter(PlayerFeedback.rating == FeedbackRating.NEGATIVE),
        ).where(
            PlayerFeedback.game_id == game_id,
            PlayerFeedback.created_at >= start_ts,
            PlayerFeedback.created_at < end_ts,
        )
        result = await self._db.execute(stmt)
        row = result.one()

        return {
            "positive": int(row[0]),
            "negative": int(row[1]),
        }

    async def _get_confidence_distribution(
        self,
        game_id: object,
        start_ts: str,
        end_ts: str,
    ) -> ConfidenceDistribution:
        """Get confidence score histogram for ai_answer messages.

        Buckets: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0].
        Messages with NULL confidence_score are excluded.

        Args:
            game_id: UUID of the game.
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            ConfidenceDistribution with bucket counts.
        """
        stmt = (
            select(
                func.count().filter(
                    Message.confidence_score >= 0.0,
                    Message.confidence_score < 0.25,
                ),
                func.count().filter(
                    Message.confidence_score >= 0.25,
                    Message.confidence_score < 0.5,
                ),
                func.count().filter(
                    Message.confidence_score >= 0.5,
                    Message.confidence_score < 0.75,
                ),
                func.count().filter(
                    Message.confidence_score >= 0.75,
                ),
            )
            .join(Session, Message.session_id == Session.id)
            .where(
                Session.game_id == game_id,
                Message.type == MessageType.AI_ANSWER,
                Message.confidence_score.isnot(None),
                Message.created_at >= start_ts,
                Message.created_at < end_ts,
            )
        )
        result = await self._db.execute(stmt)
        row = result.one()

        return ConfidenceDistribution(
            low=int(row[0]),
            medium_low=int(row[1]),
            medium_high=int(row[2]),
            high=int(row[3]),
        )

    async def _get_avg_queries_per_session(
        self,
        game_id: object,
        start_ts: str,
        end_ts: str,
    ) -> float:
        """Get average user_question count per session for a game.

        Args:
            game_id: UUID of the game.
            start_ts: ISO start timestamp (inclusive).
            end_ts: ISO end timestamp (exclusive).

        Returns:
            Average queries per session, 0.0 if no sessions exist.
        """
        session_query_counts = (
            select(
                Session.id.label("session_id"),
                func.count(Message.id).label("query_count"),
            )
            .join(Message, Message.session_id == Session.id)
            .where(
                Session.game_id == game_id,
                Message.type == MessageType.USER_QUESTION,
                Message.created_at >= start_ts,
                Message.created_at < end_ts,
            )
            .group_by(Session.id)
            .subquery()
        )

        stmt = select(
            func.coalesce(func.avg(func.cast(session_query_counts.c.query_count, SAFloat)), 0.0)
        )
        result = await self._db.execute(stmt)
        return float(result.scalar_one())

    @staticmethod
    def _validate_date_range(period_start: date, period_end: date) -> None:
        """Validate the date range parameters.

        Args:
            period_start: Start of the period (inclusive).
            period_end: End of the period (inclusive).

        Raises:
            ValidationError: If start > end or range exceeds 365 days.
        """
        if period_start > period_end:
            raise ValidationError("period_start must be <= period_end")

        delta = (period_end - period_start).days
        if delta > _MAX_RANGE_DAYS:
            raise ValidationError(
                f"Date range must not exceed {_MAX_RANGE_DAYS} days (requested {delta} days)"
            )

    @staticmethod
    def _date_range_timestamps(period_start: date, period_end: date) -> tuple[str, str]:
        """Convert inclusive date range to timestamp boundaries.

        The start is midnight of period_start, the end is midnight of
        the day after period_end (exclusive upper bound).

        Args:
            period_start: Inclusive start date.
            period_end: Inclusive end date.

        Returns:
            Tuple of (start_iso, end_iso) suitable for timestamp comparison.
        """
        start = period_start.isoformat()
        end = (period_end + timedelta(days=1)).isoformat()
        return start, end

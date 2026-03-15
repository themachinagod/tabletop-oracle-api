"""Unit tests for QualityMetricsService.

Tests the get_metrics method including per-game aggregation of
low-confidence rate, clarification rate, average citations, feedback
counts, confidence distribution, and queries per session. Covers
edge cases: games with zero answers, games with no feedback, and
date range validation. Uses mock database sessions.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from tabletop_oracle.errors.exceptions import ValidationError
from tabletop_oracle.schemas.quality import (
    ConfidenceDistribution,
    GameQualityMetricsResponse,
)
from tabletop_oracle.services.admin.quality_metrics import QualityMetricsService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAME_ID = uuid.uuid4()
_GAME_NAME = "Catan"
_START = date(2026, 3, 1)
_END = date(2026, 3, 15)


def _mock_db_sequence(results: list[MagicMock]) -> AsyncMock:
    """Build a mock AsyncSession returning different results per execute call.

    Args:
        results: List of mock result objects, one per execute call.

    Returns:
        Mock AsyncSession.
    """
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=results)
    return db


def _row_result(row: tuple[object, ...]) -> MagicMock:
    """Build a result mock that returns a single row from .one()."""
    result = MagicMock()
    result.one.return_value = row
    result.scalar_one.return_value = row[0]
    result.all.return_value = [row]
    return result


def _rows_result(rows: list[tuple[object, ...]]) -> MagicMock:
    """Build a result mock that returns multiple rows from .all()."""
    result = MagicMock()
    result.all.return_value = rows
    return result


def _scalar_result(value: object) -> MagicMock:
    """Build a result mock that returns a scalar value."""
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _build_game_metrics_db(
    *,
    active_games: list[tuple[object, str]] | None = None,
    ai_answer_count: int = 10,
    low_confidence_count: int = 2,
    user_question_count: int = 15,
    clarification_count: int = 3,
    avg_citations: float = 2.5,
    positive_feedback: int = 8,
    negative_feedback: int = 2,
    conf_low: int = 1,
    conf_medium_low: int = 1,
    conf_medium_high: int = 3,
    conf_high: int = 5,
    avg_queries_per_session: float = 5.0,
) -> AsyncMock:
    """Build a mock db for a full get_metrics call.

    The service calls execute in this order per game:
    1. Get active games (once)
    2. Get message stats (per game)
    3. Get avg citations (per game)
    4. Get feedback counts (per game)
    5. Get confidence distribution (per game)
    6. Get avg queries per session (per game)
    """
    if active_games is None:
        active_games = [(_GAME_ID, _GAME_NAME)]

    results = [
        # 1. Active games
        _rows_result(active_games),
    ]

    for _ in active_games:
        results.extend(
            [
                # 2. Message stats
                _row_result(
                    (
                        ai_answer_count,
                        low_confidence_count,
                        user_question_count,
                        clarification_count,
                    )
                ),
                # 3. Avg citations
                _scalar_result(avg_citations),
                # 4. Feedback counts
                _row_result((positive_feedback, negative_feedback)),
                # 5. Confidence distribution
                _row_result((conf_low, conf_medium_low, conf_medium_high, conf_high)),
                # 6. Avg queries per session
                _scalar_result(avg_queries_per_session),
            ]
        )

    return _mock_db_sequence(results)


# ---------------------------------------------------------------------------
# Date range validation
# ---------------------------------------------------------------------------


class TestDateRangeValidation:
    """Tests for date range parameter validation."""

    @pytest.mark.asyncio
    async def test_get_metrics_rejects_start_after_end(self) -> None:
        """Rejects date range where start > end."""
        db = AsyncMock()
        service = QualityMetricsService(db)

        with pytest.raises(ValidationError, match="period_start must be <= period_end"):
            await service.get_metrics(
                period_start=date(2026, 3, 15),
                period_end=date(2026, 3, 1),
            )

    @pytest.mark.asyncio
    async def test_get_metrics_rejects_range_exceeding_365_days(self) -> None:
        """Rejects date range exceeding 365 days."""
        db = AsyncMock()
        service = QualityMetricsService(db)

        with pytest.raises(ValidationError, match="must not exceed 365 days"):
            await service.get_metrics(
                period_start=date(2025, 1, 1),
                period_end=date(2026, 3, 15),
            )

    @pytest.mark.asyncio
    async def test_get_metrics_accepts_same_start_and_end(self) -> None:
        """Accepts single-day range."""
        db = _build_game_metrics_db(active_games=[])
        service = QualityMetricsService(db)

        result = await service.get_metrics(
            period_start=date(2026, 3, 1),
            period_end=date(2026, 3, 1),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_metrics_accepts_exactly_365_days(self) -> None:
        """Accepts exactly 365 day range."""
        start = date(2025, 3, 16)
        end = start + timedelta(days=365)
        db = _build_game_metrics_db(active_games=[])
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=start, period_end=end)
        assert result == []


# ---------------------------------------------------------------------------
# No activity
# ---------------------------------------------------------------------------


class TestNoActivity:
    """Tests for periods with no game activity."""

    @pytest.mark.asyncio
    async def test_get_metrics_no_games_active_returns_empty_list(self) -> None:
        """Returns empty list when no games have activity in the period."""
        db = _build_game_metrics_db(active_games=[])
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result == []
        db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Standard metrics computation
# ---------------------------------------------------------------------------


class TestStandardMetrics:
    """Tests for standard metric computation with representative data."""

    @pytest.mark.asyncio
    async def test_get_metrics_computes_all_metrics_correctly(self) -> None:
        """Computes all metrics correctly with representative data."""
        db = _build_game_metrics_db(
            ai_answer_count=10,
            low_confidence_count=2,
            user_question_count=15,
            clarification_count=3,
            avg_citations=2.5,
            positive_feedback=8,
            negative_feedback=2,
            conf_low=1,
            conf_medium_low=1,
            conf_medium_high=3,
            conf_high=5,
            avg_queries_per_session=5.0,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert len(result) == 1
        metrics = result[0]
        assert metrics.game_id == str(_GAME_ID)
        assert metrics.game_name == _GAME_NAME
        assert metrics.low_confidence_rate == pytest.approx(0.2)
        assert metrics.clarification_rate == pytest.approx(0.2)
        assert metrics.avg_citations_per_answer == pytest.approx(2.5)
        assert metrics.feedback_positive_count == 8
        assert metrics.feedback_negative_count == 2
        assert metrics.confidence_distribution.low == 1
        assert metrics.confidence_distribution.medium_low == 1
        assert metrics.confidence_distribution.medium_high == 3
        assert metrics.confidence_distribution.high == 5
        assert metrics.avg_queries_per_session == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_get_metrics_returns_correct_response_type(self) -> None:
        """Result items are GameQualityMetricsResponse instances."""
        db = _build_game_metrics_db()
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert all(isinstance(m, GameQualityMetricsResponse) for m in result)

    @pytest.mark.asyncio
    async def test_get_metrics_confidence_distribution_type(self) -> None:
        """Confidence distribution is a ConfidenceDistribution instance."""
        db = _build_game_metrics_db()
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert isinstance(result[0].confidence_distribution, ConfidenceDistribution)


# ---------------------------------------------------------------------------
# Edge cases: zero answers
# ---------------------------------------------------------------------------


class TestZeroAnswers:
    """Tests for games with zero AI answers."""

    @pytest.mark.asyncio
    async def test_get_metrics_zero_ai_answers_low_confidence_rate_is_zero(
        self,
    ) -> None:
        """Low confidence rate is 0.0 when no ai_answer messages exist."""
        db = _build_game_metrics_db(
            ai_answer_count=0,
            low_confidence_count=0,
            user_question_count=5,
            clarification_count=0,
            avg_citations=0.0,
            conf_low=0,
            conf_medium_low=0,
            conf_medium_high=0,
            conf_high=0,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].low_confidence_rate == 0.0
        assert result[0].avg_citations_per_answer == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_zero_questions_clarification_rate_is_zero(
        self,
    ) -> None:
        """Clarification rate is 0.0 when no user_question messages exist."""
        db = _build_game_metrics_db(
            ai_answer_count=5,
            low_confidence_count=1,
            user_question_count=0,
            clarification_count=0,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].clarification_rate == 0.0


# ---------------------------------------------------------------------------
# Edge cases: no feedback
# ---------------------------------------------------------------------------


class TestNoFeedback:
    """Tests for games with no player feedback."""

    @pytest.mark.asyncio
    async def test_get_metrics_no_feedback_counts_are_zero(self) -> None:
        """Feedback counts are both zero when no feedback exists."""
        db = _build_game_metrics_db(
            positive_feedback=0,
            negative_feedback=0,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].feedback_positive_count == 0
        assert result[0].feedback_negative_count == 0


# ---------------------------------------------------------------------------
# Multiple games
# ---------------------------------------------------------------------------


class TestMultipleGames:
    """Tests for periods with multiple active games."""

    @pytest.mark.asyncio
    async def test_get_metrics_multiple_games_returns_metrics_per_game(self) -> None:
        """Returns one metrics entry per active game."""
        game_a_id = uuid.uuid4()
        game_b_id = uuid.uuid4()

        db = _build_game_metrics_db(
            active_games=[
                (game_a_id, "Azul"),
                (game_b_id, "Brass"),
            ],
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert len(result) == 2
        assert result[0].game_name == "Azul"
        assert result[1].game_name == "Brass"

    @pytest.mark.asyncio
    async def test_get_metrics_multiple_games_execute_called_correctly(self) -> None:
        """Database is queried once for active games + 5 times per game."""
        game_a_id = uuid.uuid4()
        game_b_id = uuid.uuid4()

        db = _build_game_metrics_db(
            active_games=[
                (game_a_id, "Azul"),
                (game_b_id, "Brass"),
            ],
        )
        service = QualityMetricsService(db)

        await service.get_metrics(period_start=_START, period_end=_END)

        # 1 (active games) + 2 * 5 (per-game queries) = 11
        assert db.execute.call_count == 11


# ---------------------------------------------------------------------------
# Rate computation edge cases
# ---------------------------------------------------------------------------


class TestRateComputation:
    """Tests for rate computation boundary conditions."""

    @pytest.mark.asyncio
    async def test_get_metrics_all_low_confidence_rate_is_one(self) -> None:
        """Low confidence rate is 1.0 when all answers are low confidence."""
        db = _build_game_metrics_db(
            ai_answer_count=5,
            low_confidence_count=5,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].low_confidence_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_get_metrics_no_clarifications_rate_is_zero(self) -> None:
        """Clarification rate is 0.0 when no clarifications were generated."""
        db = _build_game_metrics_db(
            user_question_count=20,
            clarification_count=0,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].clarification_rate == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_all_clarifications_rate_is_one(self) -> None:
        """Clarification rate is 1.0 when every question got a clarification."""
        db = _build_game_metrics_db(
            user_question_count=10,
            clarification_count=10,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].clarification_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Confidence distribution
# ---------------------------------------------------------------------------


class TestConfidenceDistribution:
    """Tests for confidence distribution histogram."""

    @pytest.mark.asyncio
    async def test_get_metrics_all_high_confidence_only_high_bucket(self) -> None:
        """All counts in high bucket when all scores are >= 0.75."""
        db = _build_game_metrics_db(
            conf_low=0,
            conf_medium_low=0,
            conf_medium_high=0,
            conf_high=10,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        dist = result[0].confidence_distribution
        assert dist.low == 0
        assert dist.medium_low == 0
        assert dist.medium_high == 0
        assert dist.high == 10

    @pytest.mark.asyncio
    async def test_get_metrics_empty_distribution_all_zeros(self) -> None:
        """All buckets are zero when no confidence scores exist."""
        db = _build_game_metrics_db(
            ai_answer_count=0,
            low_confidence_count=0,
            conf_low=0,
            conf_medium_low=0,
            conf_medium_high=0,
            conf_high=0,
        )
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        dist = result[0].confidence_distribution
        assert dist.low == 0
        assert dist.medium_low == 0
        assert dist.medium_high == 0
        assert dist.high == 0


# ---------------------------------------------------------------------------
# Queries per session
# ---------------------------------------------------------------------------


class TestQueriesPerSession:
    """Tests for average queries per session computation."""

    @pytest.mark.asyncio
    async def test_get_metrics_no_sessions_avg_queries_is_zero(self) -> None:
        """Avg queries per session is 0.0 when no sessions exist."""
        db = _build_game_metrics_db(avg_queries_per_session=0.0)
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].avg_queries_per_session == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_fractional_avg_preserved_correctly(self) -> None:
        """Fractional average is preserved (not rounded to int)."""
        db = _build_game_metrics_db(avg_queries_per_session=3.7)
        service = QualityMetricsService(db)

        result = await service.get_metrics(period_start=_START, period_end=_END)

        assert result[0].avg_queries_per_session == pytest.approx(3.7)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for Pydantic response schema validation."""

    def test_confidence_distribution_rejects_negative(self) -> None:
        """ConfidenceDistribution rejects negative counts."""
        with pytest.raises(PydanticValidationError):
            ConfidenceDistribution(low=-1, medium_low=0, medium_high=0, high=0)

    def test_game_quality_metrics_rejects_rate_above_one(self) -> None:
        """GameQualityMetricsResponse rejects rate > 1.0."""
        with pytest.raises(PydanticValidationError):
            GameQualityMetricsResponse(
                game_id=str(uuid.uuid4()),
                game_name="Test",
                low_confidence_rate=1.5,
                clarification_rate=0.0,
                avg_citations_per_answer=0.0,
                feedback_positive_count=0,
                feedback_negative_count=0,
                confidence_distribution=ConfidenceDistribution(
                    low=0, medium_low=0, medium_high=0, high=0
                ),
                avg_queries_per_session=0.0,
            )

    def test_game_quality_metrics_rejects_negative_feedback(self) -> None:
        """GameQualityMetricsResponse rejects negative feedback count."""
        with pytest.raises(PydanticValidationError):
            GameQualityMetricsResponse(
                game_id=str(uuid.uuid4()),
                game_name="Test",
                low_confidence_rate=0.0,
                clarification_rate=0.0,
                avg_citations_per_answer=0.0,
                feedback_positive_count=-1,
                feedback_negative_count=0,
                confidence_distribution=ConfidenceDistribution(
                    low=0, medium_low=0, medium_high=0, high=0
                ),
                avg_queries_per_session=0.0,
            )

    def test_game_quality_metrics_valid_construction(self) -> None:
        """GameQualityMetricsResponse accepts valid data."""
        metrics = GameQualityMetricsResponse(
            game_id=str(uuid.uuid4()),
            game_name="Catan",
            low_confidence_rate=0.5,
            clarification_rate=0.3,
            avg_citations_per_answer=2.5,
            feedback_positive_count=10,
            feedback_negative_count=3,
            confidence_distribution=ConfidenceDistribution(
                low=1, medium_low=2, medium_high=3, high=4
            ),
            avg_queries_per_session=4.2,
        )
        assert metrics.game_name == "Catan"
        assert metrics.confidence_distribution.high == 4

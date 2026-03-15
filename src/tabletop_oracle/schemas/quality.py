"""Pydantic schemas for quality metrics admin API.

Response schemas for the ``/api/v1/admin/quality`` endpoints. All schemas
are read-only since quality data is derived from aggregation queries
over ``messages``, ``citations``, ``player_feedback``, and ``sessions``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfidenceDistribution(BaseModel):
    """Histogram of AI answer confidence scores in four buckets.

    Each bucket counts the number of ``ai_answer`` messages whose
    ``confidence_score`` falls within the bucket range. Boundaries
    are inclusive-exclusive except the last bucket which is inclusive
    on both ends: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0].

    Attributes:
        low: Count of answers with confidence in [0, 0.25).
        medium_low: Count of answers with confidence in [0.25, 0.5).
        medium_high: Count of answers with confidence in [0.5, 0.75).
        high: Count of answers with confidence in [0.75, 1.0].
    """

    low: int = Field(ge=0)
    medium_low: int = Field(ge=0)
    medium_high: int = Field(ge=0)
    high: int = Field(ge=0)


class GameQualityMetricsResponse(BaseModel):
    """Quality metrics for a single game within a time period.

    All rates are expressed as floats in [0.0, 1.0]. When no data
    exists to compute a rate (e.g. zero ai_answer messages), the
    rate is 0.0.

    Attributes:
        game_id: UUID of the game.
        game_name: Display name of the game.
        low_confidence_rate: Fraction of ai_answer messages with
            confidence_score < 0.5.
        clarification_rate: Fraction of user_question messages that
            triggered an ai_clarification response.
        avg_citations_per_answer: Mean citation count per ai_answer
            message.
        feedback_positive_count: Number of positive feedback ratings.
        feedback_negative_count: Number of negative feedback ratings.
        confidence_distribution: Histogram of confidence score buckets.
        avg_queries_per_session: Mean user_question count per session.
    """

    game_id: str
    game_name: str
    low_confidence_rate: float = Field(ge=0.0, le=1.0)
    clarification_rate: float = Field(ge=0.0, le=1.0)
    avg_citations_per_answer: float = Field(ge=0.0)
    feedback_positive_count: int = Field(ge=0)
    feedback_negative_count: int = Field(ge=0)
    confidence_distribution: ConfidenceDistribution
    avg_queries_per_session: float = Field(ge=0.0)

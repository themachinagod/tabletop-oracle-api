"""Validates end-to-end AI quality using ground truth Q&A pairs.

Loads ground truth questions from the seed fixture, runs each through a
query pipeline protocol, evaluates answers against expected criteria
(answer_contains, answer_contains_any, must_cite, min_citations,
min_confidence, expected_document_types), and produces a structured
validation report.

Design reference: EPIC-008 design.md, GroundTruthValidator section.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tabletop_oracle.services.seed_fixture_loader import (
        GroundTruthExpected,
        GroundTruthFixture,
        GroundTruthQA,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query executor protocol — allows mocking the AI pipeline
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Result of a single query execution.

    Attributes:
        answer_text: The full answer text produced by the pipeline.
        confidence_score: Self-assessed confidence (0.0-1.0), or None.
        citations: List of citation records with at least document_name
            and document_type fields.
        error: Error message if the query failed, None on success.
    """

    answer_text: str = ""
    confidence_score: float | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


@runtime_checkable
class QueryExecutor(Protocol):
    """Protocol for executing a single query against the AI pipeline.

    Implementations must accept a question string and return a QueryResult.
    This abstraction decouples the validator from the real pipeline, enabling
    deterministic unit testing with mocked responses.
    """

    async def execute_query(self, question: str) -> QueryResult:
        """Execute a question and return the aggregated result.

        Args:
            question: The natural-language question to ask.

        Returns:
            QueryResult with answer text, confidence, and citations.
        """
        ...


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------


class CriterionStatus(Enum):
    """Outcome of a single evaluation criterion."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CriterionResult:
    """Result of evaluating one criterion for one question.

    Attributes:
        criterion: Name of the criterion (e.g. "answer_contains").
        status: Whether the criterion passed, failed, or was skipped.
        detail: Human-readable explanation of the result.
    """

    criterion: str
    status: CriterionStatus
    detail: str = ""


@dataclass
class QuestionResult:
    """Validation result for a single ground truth question.

    Attributes:
        question_id: The ground truth question ID (e.g. "GT-001").
        category: Question category (e.g. "single-doc-factual").
        question: The question text.
        passed: True if all criteria passed.
        confidence: The confidence score returned, or None.
        criteria: Per-criterion evaluation results.
        error: Error message if the query itself failed.
        duration_ms: Time taken to execute the query, in milliseconds.
    """

    question_id: str
    category: str
    question: str
    passed: bool
    confidence: float | None
    criteria: list[CriterionResult] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0


@dataclass
class ValidationResult:
    """Aggregate validation report across all ground truth questions.

    Attributes:
        game: The game name being validated.
        total: Total number of questions evaluated.
        passed: Number of questions that passed all criteria.
        failed: Number of questions that failed one or more criteria.
        errors: Number of questions where the query itself errored.
        pass_rate: Fraction of questions that passed (0.0-1.0).
        quality_bar: The target pass rate (informational, not enforced).
        questions: Per-question results.
    """

    game: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    quality_bar: float = 0.6
    questions: list[QuestionResult] = field(default_factory=list)

    def format_report(self) -> str:
        """Produce a human-readable validation report.

        Returns:
            Multi-line string matching the design doc report format.
        """
        lines = [
            "Ground Truth Validation Report",
            "=" * 30,
            f"Game: {self.game}",
            f"Questions: {self.total}",
            f"Passed: {self.passed} / {self.total} ({self._pct(self.pass_rate)})",
            f"Failed: {self.failed}",
        ]

        if self.errors > 0:
            lines.append(f"Errors: {self.errors}")

        lines.append("")
        lines.append("Results:")

        for q in self.questions:
            status = "PASS" if q.passed else ("ERROR" if q.error else "FAIL")
            conf_str = (
                f"confidence={q.confidence:.2f}" if q.confidence is not None else "confidence=N/A"
            )
            lines.append(f"  {q.question_id} [{q.category}]  {status}  {conf_str}")

            if not q.passed:
                for cr in q.criteria:
                    if cr.status == CriterionStatus.FAILED:
                        lines.append(f"    - {cr.criterion}: FAIL ({cr.detail})")
                    elif cr.status == CriterionStatus.PASSED and cr.detail:
                        lines.append(f"    - {cr.criterion}: {cr.detail}")

                if q.error:
                    lines.append(f"    - error: {q.error}")

        lines.append("")
        lines.append(
            f"Quality Bar: {self._pct(self.pass_rate)} pass rate "
            f"(target: {self._pct(self.quality_bar)})"
        )

        return "\n".join(lines)

    @staticmethod
    def _pct(value: float) -> str:
        """Format a float as a percentage string.

        Args:
            value: Float between 0.0 and 1.0.

        Returns:
            String like "80%".
        """
        return f"{value * 100:.0f}%"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class GroundTruthValidator:
    """Validates end-to-end AI quality using ground truth Q&A pairs.

    Loads a ground truth fixture, runs each question through the provided
    query executor, evaluates answers against expected criteria, and
    produces a structured validation report.

    Args:
        executor: Protocol implementation that runs queries against the
            AI pipeline (or a mock for testing).
    """

    def __init__(self, executor: QueryExecutor) -> None:
        self._executor = executor

    async def validate(self, fixture: GroundTruthFixture) -> ValidationResult:
        """Run all Q&A pairs and compare against expected answers.

        Args:
            fixture: Parsed ground truth fixture with questions and criteria.

        Returns:
            ValidationResult with per-question scores and summary metrics.
        """
        result = ValidationResult(game=fixture.game, quality_bar=0.6)

        for qa in fixture.questions:
            question_result = await self._evaluate_question(qa)
            result.questions.append(question_result)
            result.total += 1

            if question_result.error:
                result.errors += 1
            elif question_result.passed:
                result.passed += 1
            else:
                result.failed += 1

        result.pass_rate = result.passed / result.total if result.total > 0 else 0.0

        logger.info(
            "Ground truth validation complete",
            extra={
                "game": fixture.game,
                "total": result.total,
                "passed": result.passed,
                "failed": result.failed,
                "errors": result.errors,
                "pass_rate": result.pass_rate,
            },
        )

        return result

    async def _evaluate_question(self, qa: GroundTruthQA) -> QuestionResult:
        """Execute a single question and evaluate against expected criteria.

        Args:
            qa: A ground truth Q&A pair with question and expected criteria.

        Returns:
            QuestionResult with per-criterion outcomes.
        """
        start = time.monotonic()

        try:
            query_result = await self._executor.execute_query(qa.question)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "Query execution failed for %s: %s",
                qa.id,
                exc,
            )
            return QuestionResult(
                question_id=qa.id,
                category=qa.category,
                question=qa.question,
                passed=False,
                confidence=None,
                error=str(exc),
                duration_ms=elapsed_ms,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if query_result.error:
            return QuestionResult(
                question_id=qa.id,
                category=qa.category,
                question=qa.question,
                passed=False,
                confidence=query_result.confidence_score,
                error=query_result.error,
                duration_ms=elapsed_ms,
            )

        criteria = self._evaluate_criteria(qa.expected, query_result)
        all_passed = all(cr.status != CriterionStatus.FAILED for cr in criteria)

        return QuestionResult(
            question_id=qa.id,
            category=qa.category,
            question=qa.question,
            passed=all_passed,
            confidence=query_result.confidence_score,
            criteria=criteria,
            duration_ms=elapsed_ms,
        )

    def _evaluate_criteria(
        self,
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> list[CriterionResult]:
        """Evaluate all criteria for a single question's result.

        Args:
            expected: The expected criteria from the ground truth fixture.
            result: The actual query result from the pipeline.

        Returns:
            List of CriterionResult, one per applicable criterion.
        """
        criteria: list[CriterionResult] = []

        criteria.append(self._check_answer_contains(expected, result))
        criteria.append(self._check_answer_contains_any(expected, result))
        criteria.append(self._check_must_cite(expected, result))
        criteria.append(self._check_min_citations(expected, result))
        criteria.append(self._check_min_confidence(expected, result))
        criteria.append(self._check_expected_document_types(expected, result))

        return criteria

    @staticmethod
    def _check_answer_contains(
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> CriterionResult:
        """Check that all required strings appear in the answer.

        Args:
            expected: Expected criteria.
            result: Query result.

        Returns:
            CriterionResult for the answer_contains criterion.
        """
        if not expected.answer_contains:
            return CriterionResult(
                criterion="answer_contains",
                status=CriterionStatus.SKIPPED,
                detail="No required terms specified",
            )

        answer_lower = result.answer_text.lower()
        missing = [term for term in expected.answer_contains if term.lower() not in answer_lower]

        if missing:
            return CriterionResult(
                criterion="answer_contains",
                status=CriterionStatus.FAILED,
                detail=f"missing: {missing}",
            )

        return CriterionResult(
            criterion="answer_contains",
            status=CriterionStatus.PASSED,
            detail=f"all {len(expected.answer_contains)} terms found",
        )

    @staticmethod
    def _check_answer_contains_any(
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> CriterionResult:
        """Check that at least one of the expected strings appears in the answer.

        Args:
            expected: Expected criteria.
            result: Query result.

        Returns:
            CriterionResult for the answer_contains_any criterion.
        """
        if not expected.answer_contains_any:
            return CriterionResult(
                criterion="answer_contains_any",
                status=CriterionStatus.SKIPPED,
                detail="No alternative terms specified",
            )

        answer_lower = result.answer_text.lower()
        matched = [term for term in expected.answer_contains_any if term.lower() in answer_lower]

        if not matched:
            return CriterionResult(
                criterion="answer_contains_any",
                status=CriterionStatus.FAILED,
                detail=f"none of {expected.answer_contains_any} found",
            )

        return CriterionResult(
            criterion="answer_contains_any",
            status=CriterionStatus.PASSED,
            detail=f"matched: {matched}",
        )

    @staticmethod
    def _check_must_cite(
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> CriterionResult:
        """Check that at least one citation is present when required.

        Args:
            expected: Expected criteria.
            result: Query result.

        Returns:
            CriterionResult for the must_cite criterion.
        """
        if not expected.must_cite:
            return CriterionResult(
                criterion="must_cite",
                status=CriterionStatus.SKIPPED,
                detail="Citations not required",
            )

        if result.citations:
            return CriterionResult(
                criterion="must_cite",
                status=CriterionStatus.PASSED,
                detail=f"{len(result.citations)} citations present",
            )

        return CriterionResult(
            criterion="must_cite",
            status=CriterionStatus.FAILED,
            detail="no citations present",
        )

    @staticmethod
    def _check_min_citations(
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> CriterionResult:
        """Check that the minimum number of citations is met.

        Args:
            expected: Expected criteria.
            result: Query result.

        Returns:
            CriterionResult for the min_citations criterion.
        """
        if expected.min_citations is None:
            return CriterionResult(
                criterion="min_citations",
                status=CriterionStatus.SKIPPED,
                detail="No minimum specified",
            )

        count = len(result.citations)
        if count >= expected.min_citations:
            return CriterionResult(
                criterion="min_citations",
                status=CriterionStatus.PASSED,
                detail=f"{count} >= {expected.min_citations}",
            )

        return CriterionResult(
            criterion="min_citations",
            status=CriterionStatus.FAILED,
            detail=f"{count} < {expected.min_citations}",
        )

    @staticmethod
    def _check_min_confidence(
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> CriterionResult:
        """Check that the confidence score meets the threshold.

        Args:
            expected: Expected criteria.
            result: Query result.

        Returns:
            CriterionResult for the min_confidence criterion.
        """
        if expected.min_confidence is None:
            return CriterionResult(
                criterion="min_confidence",
                status=CriterionStatus.SKIPPED,
                detail="No minimum specified",
            )

        if result.confidence_score is None:
            return CriterionResult(
                criterion="min_confidence",
                status=CriterionStatus.FAILED,
                detail=f"no confidence score returned (min: {expected.min_confidence})",
            )

        if result.confidence_score >= expected.min_confidence:
            return CriterionResult(
                criterion="min_confidence",
                status=CriterionStatus.PASSED,
                detail=f"{result.confidence_score:.2f} >= {expected.min_confidence}",
            )

        return CriterionResult(
            criterion="min_confidence",
            status=CriterionStatus.FAILED,
            detail=f"{result.confidence_score:.2f} < {expected.min_confidence}",
        )

    @staticmethod
    def _check_expected_document_types(
        expected: GroundTruthExpected,
        result: QueryResult,
    ) -> CriterionResult:
        """Check that citations include expected document types.

        Args:
            expected: Expected criteria.
            result: Query result.

        Returns:
            CriterionResult for the expected_document_types criterion.
        """
        if not expected.expected_document_types:
            return CriterionResult(
                criterion="expected_document_types",
                status=CriterionStatus.SKIPPED,
                detail="No expected types specified",
            )

        cited_types = {c.get("document_type", "") for c in result.citations}
        missing_types = [dt for dt in expected.expected_document_types if dt not in cited_types]

        if missing_types:
            return CriterionResult(
                criterion="expected_document_types",
                status=CriterionStatus.FAILED,
                detail=f"missing types: {missing_types}, found: {sorted(cited_types)}",
            )

        return CriterionResult(
            criterion="expected_document_types",
            status=CriterionStatus.PASSED,
            detail=f"all expected types found in citations: {sorted(cited_types)}",
        )

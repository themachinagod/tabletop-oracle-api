"""Unit tests for GroundTruthValidator.

Tests evaluation logic for all criteria: answer_contains, answer_contains_any,
must_cite, min_citations, min_confidence, expected_document_types. Covers
edge cases: empty answer, no citations, null confidence, query errors,
and the aggregate report format.

All tests use a mocked QueryExecutor — no real AI pipeline involved.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from tabletop_oracle.services.ground_truth_validator import (
    CriterionStatus,
    GroundTruthValidator,
    QueryResult,
    ValidationResult,
)
from tabletop_oracle.services.seed_fixture_loader import (
    GroundTruthExpected,
    GroundTruthFixture,
    GroundTruthQA,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _expected(**overrides: Any) -> GroundTruthExpected:
    """Build a GroundTruthExpected with sensible defaults."""
    defaults: dict[str, Any] = {
        "answer_contains": [],
        "answer_contains_any": [],
        "must_cite": False,
        "min_citations": None,
        "min_confidence": None,
        "expected_document_types": [],
    }
    defaults.update(overrides)
    return GroundTruthExpected(**defaults)


def _qa(
    qa_id: str = "GT-001",
    category: str = "single-doc-factual",
    question: str = "What are the five resource types?",
    **expected_overrides: Any,
) -> GroundTruthQA:
    """Build a GroundTruthQA with sensible defaults."""
    return GroundTruthQA(
        id=qa_id,
        category=category,
        question=question,
        expected=_expected(**expected_overrides),
    )


def _fixture(questions: list[GroundTruthQA] | None = None) -> GroundTruthFixture:
    """Build a GroundTruthFixture wrapping the given questions."""
    return GroundTruthFixture(
        version="1.0.0",
        game="Catan",
        description="Test fixture",
        questions=questions or [],
    )


def _query_result(**overrides: Any) -> QueryResult:
    """Build a QueryResult with sensible defaults."""
    defaults: dict[str, Any] = {
        "answer_text": "The five resources are lumber, wool, grain, brick, and ore.",
        "confidence_score": 0.85,
        "citations": [
            {"document_name": "Catan Rules", "document_type": "core_rules"},
        ],
        "error": None,
    }
    defaults.update(overrides)
    return QueryResult(**defaults)


def _mock_executor(results: list[QueryResult] | QueryResult | None = None) -> AsyncMock:
    """Build a mock QueryExecutor returning the given results.

    Args:
        results: A single QueryResult (returned for all calls), a list
            (returned in sequence), or None (uses default).

    Returns:
        AsyncMock implementing the QueryExecutor protocol.
    """
    mock = AsyncMock()
    if results is None:
        mock.execute_query.return_value = _query_result()
    elif isinstance(results, list):
        mock.execute_query.side_effect = results
    else:
        mock.execute_query.return_value = results
    return mock


# ---------------------------------------------------------------------------
# answer_contains criterion
# ---------------------------------------------------------------------------


class TestAnswerContains:
    """Tests for the answer_contains evaluation criterion."""

    @pytest.mark.asyncio
    async def test_all_terms_present_passes(self) -> None:
        """All required terms found in the answer -> PASSED."""
        qa = _qa(answer_contains=["lumber", "wool", "grain"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains")
        assert ac.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_missing_term_fails(self) -> None:
        """A required term not in the answer -> FAILED."""
        qa = _qa(answer_contains=["lumber", "diamonds"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains")
        assert ac.status == CriterionStatus.FAILED
        assert "diamonds" in ac.detail

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        qa = _qa(answer_contains=["LUMBER", "Wool"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains")
        assert ac.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_empty_list_skipped(self) -> None:
        """No required terms -> SKIPPED."""
        qa = _qa(answer_contains=[])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains")
        assert ac.status == CriterionStatus.SKIPPED


# ---------------------------------------------------------------------------
# answer_contains_any criterion
# ---------------------------------------------------------------------------


class TestAnswerContainsAny:
    """Tests for the answer_contains_any evaluation criterion."""

    @pytest.mark.asyncio
    async def test_at_least_one_term_passes(self) -> None:
        """At least one alternative term found -> PASSED."""
        qa = _qa(answer_contains_any=["diamonds", "lumber", "platinum"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains_any")
        assert ac.status == CriterionStatus.PASSED
        assert "lumber" in ac.detail

    @pytest.mark.asyncio
    async def test_no_terms_found_fails(self) -> None:
        """None of the alternative terms found -> FAILED."""
        qa = _qa(answer_contains_any=["diamonds", "platinum"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains_any")
        assert ac.status == CriterionStatus.FAILED

    @pytest.mark.asyncio
    async def test_empty_list_skipped(self) -> None:
        """No alternative terms -> SKIPPED."""
        qa = _qa(answer_contains_any=[])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        ac = next(c for c in criteria if c.criterion == "answer_contains_any")
        assert ac.status == CriterionStatus.SKIPPED


# ---------------------------------------------------------------------------
# must_cite criterion
# ---------------------------------------------------------------------------


class TestMustCite:
    """Tests for the must_cite evaluation criterion."""

    @pytest.mark.asyncio
    async def test_citations_present_passes(self) -> None:
        """Citations present when required -> PASSED."""
        qa = _qa(must_cite=True)
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "must_cite")
        assert mc.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_no_citations_fails(self) -> None:
        """No citations when required -> FAILED."""
        qa = _qa(must_cite=True)
        executor = _mock_executor(_query_result(citations=[]))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "must_cite")
        assert mc.status == CriterionStatus.FAILED

    @pytest.mark.asyncio
    async def test_not_required_skipped(self) -> None:
        """Citations not required -> SKIPPED."""
        qa = _qa(must_cite=False)
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "must_cite")
        assert mc.status == CriterionStatus.SKIPPED


# ---------------------------------------------------------------------------
# min_citations criterion
# ---------------------------------------------------------------------------


class TestMinCitations:
    """Tests for the min_citations evaluation criterion."""

    @pytest.mark.asyncio
    async def test_enough_citations_passes(self) -> None:
        """Citation count meets minimum -> PASSED."""
        qa = _qa(min_citations=1)
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_citations")
        assert mc.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_too_few_citations_fails(self) -> None:
        """Citation count below minimum -> FAILED."""
        qa = _qa(min_citations=3)
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_citations")
        assert mc.status == CriterionStatus.FAILED
        assert "1 < 3" in mc.detail

    @pytest.mark.asyncio
    async def test_no_minimum_skipped(self) -> None:
        """No min_citations specified -> SKIPPED."""
        qa = _qa(min_citations=None)
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_citations")
        assert mc.status == CriterionStatus.SKIPPED


# ---------------------------------------------------------------------------
# min_confidence criterion
# ---------------------------------------------------------------------------


class TestMinConfidence:
    """Tests for the min_confidence evaluation criterion."""

    @pytest.mark.asyncio
    async def test_confidence_meets_threshold_passes(self) -> None:
        """Confidence >= threshold -> PASSED."""
        qa = _qa(min_confidence=0.7)
        executor = _mock_executor(_query_result(confidence_score=0.85))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_confidence")
        assert mc.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_fails(self) -> None:
        """Confidence < threshold -> FAILED."""
        qa = _qa(min_confidence=0.7)
        executor = _mock_executor(_query_result(confidence_score=0.35))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_confidence")
        assert mc.status == CriterionStatus.FAILED
        assert "0.35 < 0.7" in mc.detail

    @pytest.mark.asyncio
    async def test_null_confidence_fails(self) -> None:
        """Null confidence when threshold required -> FAILED."""
        qa = _qa(min_confidence=0.5)
        executor = _mock_executor(_query_result(confidence_score=None))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_confidence")
        assert mc.status == CriterionStatus.FAILED
        assert "no confidence score" in mc.detail

    @pytest.mark.asyncio
    async def test_exact_threshold_passes(self) -> None:
        """Confidence exactly at threshold -> PASSED."""
        qa = _qa(min_confidence=0.7)
        executor = _mock_executor(_query_result(confidence_score=0.7))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_confidence")
        assert mc.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_no_threshold_skipped(self) -> None:
        """No min_confidence specified -> SKIPPED."""
        qa = _qa(min_confidence=None)
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        mc = next(c for c in criteria if c.criterion == "min_confidence")
        assert mc.status == CriterionStatus.SKIPPED


# ---------------------------------------------------------------------------
# expected_document_types criterion
# ---------------------------------------------------------------------------


class TestExpectedDocumentTypes:
    """Tests for the expected_document_types evaluation criterion."""

    @pytest.mark.asyncio
    async def test_all_types_present_passes(self) -> None:
        """All expected doc types in citations -> PASSED."""
        qa = _qa(expected_document_types=["core_rules"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        edt = next(c for c in criteria if c.criterion == "expected_document_types")
        assert edt.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_missing_type_fails(self) -> None:
        """Expected doc type not in citations -> FAILED."""
        qa = _qa(expected_document_types=["core_rules", "expansion_rules"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        edt = next(c for c in criteria if c.criterion == "expected_document_types")
        assert edt.status == CriterionStatus.FAILED
        assert "expansion_rules" in edt.detail

    @pytest.mark.asyncio
    async def test_cross_doc_with_both_types_passes(self) -> None:
        """Both expected types present in multiple citations -> PASSED."""
        qa = _qa(expected_document_types=["core_rules", "expansion_rules"])
        executor = _mock_executor(
            _query_result(
                citations=[
                    {"document_name": "Rules", "document_type": "core_rules"},
                    {"document_name": "Seafarers", "document_type": "expansion_rules"},
                ]
            )
        )
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        edt = next(c for c in criteria if c.criterion == "expected_document_types")
        assert edt.status == CriterionStatus.PASSED

    @pytest.mark.asyncio
    async def test_empty_list_skipped(self) -> None:
        """No expected types -> SKIPPED."""
        qa = _qa(expected_document_types=[])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        criteria = result.questions[0].criteria
        edt = next(c for c in criteria if c.criterion == "expected_document_types")
        assert edt.status == CriterionStatus.SKIPPED


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases: empty answer, errors, timeouts."""

    @pytest.mark.asyncio
    async def test_empty_answer_text(self) -> None:
        """Empty answer text fails answer_contains checks."""
        qa = _qa(answer_contains=["lumber"])
        executor = _mock_executor(_query_result(answer_text=""))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        assert not result.questions[0].passed

    @pytest.mark.asyncio
    async def test_query_execution_error(self) -> None:
        """Query raises an exception -> counted as error."""
        qa = _qa()
        executor = _mock_executor()
        executor.execute_query.side_effect = RuntimeError("Connection timeout")
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        assert result.errors == 1
        assert result.questions[0].error == "Connection timeout"
        assert not result.questions[0].passed

    @pytest.mark.asyncio
    async def test_query_result_with_error_field(self) -> None:
        """QueryResult has error field set -> counted as error."""
        qa = _qa()
        executor = _mock_executor(_query_result(error="Model unavailable"))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        assert not result.questions[0].passed
        assert result.questions[0].error == "Model unavailable"

    @pytest.mark.asyncio
    async def test_no_citations_with_no_requirements(self) -> None:
        """No citations but none required -> question can pass."""
        qa = _qa(
            answer_contains=["lumber"],
            must_cite=False,
            min_citations=None,
            expected_document_types=[],
        )
        executor = _mock_executor(_query_result(citations=[]))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        assert result.questions[0].passed

    @pytest.mark.asyncio
    async def test_duration_ms_recorded(self) -> None:
        """Duration in milliseconds is recorded for each question."""
        qa = _qa()
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))

        assert result.questions[0].duration_ms >= 0


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for aggregate ValidationResult and report formatting."""

    @pytest.mark.asyncio
    async def test_all_pass(self) -> None:
        """All questions pass -> 100% pass rate."""
        questions = [_qa(qa_id=f"GT-{i:03d}") for i in range(1, 4)]
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture(questions))

        assert result.total == 3
        assert result.passed == 3
        assert result.failed == 0
        assert result.errors == 0
        assert result.pass_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_mixed_results(self) -> None:
        """Mix of pass and fail -> correct counts."""
        q_pass = _qa(qa_id="GT-001")
        q_fail = _qa(qa_id="GT-002", answer_contains=["nonexistent_term"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([q_pass, q_fail]))

        assert result.total == 2
        assert result.passed == 1
        assert result.failed == 1
        assert result.pass_rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_empty_fixture(self) -> None:
        """No questions -> zero totals, 0% pass rate."""
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([]))

        assert result.total == 0
        assert result.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_format_report_structure(self) -> None:
        """Report string contains expected sections."""
        qa = _qa(qa_id="GT-001", answer_contains=["lumber"])
        executor = _mock_executor()
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))
        report = result.format_report()

        assert "Ground Truth Validation Report" in report
        assert "Game: Catan" in report
        assert "Questions: 1" in report
        assert "GT-001" in report
        assert "PASS" in report
        assert "Quality Bar:" in report

    @pytest.mark.asyncio
    async def test_format_report_shows_failures(self) -> None:
        """Report shows failed criteria for failed questions."""
        qa = _qa(
            qa_id="GT-002",
            min_confidence=0.9,
        )
        executor = _mock_executor(_query_result(confidence_score=0.3))
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))
        report = result.format_report()

        assert "FAIL" in report
        assert "min_confidence" in report
        assert "0.30 < 0.9" in report

    @pytest.mark.asyncio
    async def test_format_report_shows_errors(self) -> None:
        """Report shows error count when queries fail."""
        qa = _qa(qa_id="GT-003")
        executor = _mock_executor()
        executor.execute_query.side_effect = RuntimeError("Timeout")
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture([qa]))
        report = result.format_report()

        assert "Errors: 1" in report
        assert "ERROR" in report

    def test_format_report_pct_helper(self) -> None:
        """_pct formats float as percentage string."""
        assert ValidationResult._pct(0.8) == "80%"
        assert ValidationResult._pct(1.0) == "100%"
        assert ValidationResult._pct(0.0) == "0%"
        assert ValidationResult._pct(0.333) == "33%"


# ---------------------------------------------------------------------------
# Full integration-style test with realistic Q&A
# ---------------------------------------------------------------------------


class TestRealisticScenario:
    """Realistic multi-question scenario matching the design doc example."""

    @pytest.mark.asyncio
    async def test_design_doc_example_scenario(self) -> None:
        """Simulate the 5-question scenario from the design doc report."""
        questions = [
            _qa(
                qa_id="GT-001",
                category="single-doc-factual",
                question="What are the five resource types?",
                answer_contains=["lumber", "wool", "grain", "brick", "ore"],
                must_cite=True,
                min_confidence=0.7,
                expected_document_types=["core_rules"],
            ),
            _qa(
                qa_id="GT-002",
                category="single-doc-procedural",
                question="What happens when you roll a 7?",
                answer_contains=["robber", "discard"],
                must_cite=True,
                min_confidence=0.7,
                expected_document_types=["core_rules"],
            ),
            _qa(
                qa_id="GT-003",
                category="cross-doc-association",
                question="How do ships differ from roads?",
                answer_contains=["ship", "road"],
                must_cite=True,
                min_citations=2,
                min_confidence=0.5,
                expected_document_types=["core_rules", "expansion_rules"],
            ),
            _qa(
                qa_id="GT-004",
                category="expansion-specific",
                question="What is the pirate?",
                answer_contains=["pirate"],
                must_cite=True,
                min_confidence=0.6,
                expected_document_types=["expansion_rules"],
            ),
            _qa(
                qa_id="GT-005",
                category="strategy",
                question="What is a good opening strategy?",
                answer_contains_any=["settlement", "resource"],
                must_cite=True,
                min_confidence=0.4,
                expected_document_types=["core_rules"],
            ),
        ]

        results = [
            _query_result(
                answer_text="The five resources are lumber, wool, grain, brick, and ore.",
                confidence_score=0.92,
                citations=[{"document_name": "Rules", "document_type": "core_rules"}],
            ),
            _query_result(
                answer_text=(
                    "When a 7 is rolled, the robber is activated. "
                    "Players with more than 7 cards must discard half."
                ),
                confidence_score=0.88,
                citations=[{"document_name": "Rules", "document_type": "core_rules"}],
            ),
            _query_result(
                answer_text=(
                    "Ships in Seafarers function similarly to roads but are placed on sea paths."
                ),
                confidence_score=0.71,
                citations=[
                    {"document_name": "Rules", "document_type": "core_rules"},
                    {"document_name": "Seafarers", "document_type": "expansion_rules"},
                ],
            ),
            _query_result(
                answer_text="The pirate is the sea counterpart of the robber in Seafarers.",
                confidence_score=0.85,
                citations=[{"document_name": "Seafarers", "document_type": "expansion_rules"}],
            ),
            _query_result(
                answer_text=(
                    "A good opening strategy involves placing your "
                    "first settlement near diverse resources."
                ),
                confidence_score=0.35,
                citations=[{"document_name": "Rules", "document_type": "core_rules"}],
            ),
        ]

        executor = _mock_executor(results)
        validator = GroundTruthValidator(executor)

        result = await validator.validate(_fixture(questions))

        assert result.total == 5
        assert result.passed == 4
        assert result.failed == 1
        assert result.pass_rate == pytest.approx(0.8)

        # GT-005 should fail on min_confidence
        gt005 = result.questions[4]
        assert not gt005.passed
        failed_criteria = [c for c in gt005.criteria if c.status == CriterionStatus.FAILED]
        assert any(c.criterion == "min_confidence" for c in failed_criteria)

        # Report should be well-formed
        report = result.format_report()
        assert "4 / 5" in report
        assert "80%" in report

"""Validation tests for the ground truth Q&A set (catan-qa.json).

Verifies that the Q&A pairs are structurally valid, cover all required
categories, meet the minimum count threshold, and -- critically -- that
every question's expected answer keywords reference content that actually
exists in the seed document PDFs. This ensures the Q&A set stays
consistent with the fixture documents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pdfplumber
import pytest

from tabletop_oracle.services.seed_fixture_loader import (
    GroundTruthFixture,
    SeedFixtureLoader,
)

# ---------------------------------------------------------------------------
# Shared paths and constants
# ---------------------------------------------------------------------------

_FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "tabletop_oracle" / "fixtures" / "seed"
)
_QA_PATH = _FIXTURES_DIR / "ground-truth" / "catan-qa.json"
_DOCS_DIR = _FIXTURES_DIR / "games" / "catan" / "documents"
_BASE_RULES_PDF = _DOCS_DIR / "catan-base-rules.pdf"
_SEAFARERS_PDF = _DOCS_DIR / "catan-seafarers-rules.pdf"

_VALID_CATEGORIES = {
    "single-doc-factual",
    "single-doc-procedural",
    "cross-doc-association",
    "expansion-specific",
    "strategy",
}

_VALID_DOCUMENT_TYPES = {"core_rules", "expansion_rules"}

_MIN_QUESTIONS = 15
_MAX_QUESTIONS = 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qa_data() -> dict:
    """Load the raw Q&A JSON data."""
    with _QA_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def qa_fixture() -> GroundTruthFixture:
    """Load the Q&A set via the SeedFixtureLoader for schema validation."""
    loader = SeedFixtureLoader(_FIXTURES_DIR)
    return loader.load_ground_truth()


@pytest.fixture(scope="module")
def base_rules_text() -> str:
    """Extract full text from the base rules PDF (lowercased)."""
    with pdfplumber.open(_BASE_RULES_PDF) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages).lower()


@pytest.fixture(scope="module")
def seafarers_text() -> str:
    """Extract full text from the seafarers rules PDF (lowercased)."""
    with pdfplumber.open(_SEAFARERS_PDF) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages).lower()


@pytest.fixture(scope="module")
def all_text(base_rules_text: str, seafarers_text: str) -> str:
    """Combined text from all fixture PDFs (lowercased)."""
    return base_rules_text + "\n" + seafarers_text


# ---------------------------------------------------------------------------
# Tests -- Schema and structure
# ---------------------------------------------------------------------------


class TestQAStructure:
    """Verify the Q&A file is structurally valid and meets requirements."""

    def test_qa_file_exists(self) -> None:
        """The catan-qa.json file exists on disk."""
        assert _QA_PATH.is_file(), f"Missing: {_QA_PATH}"

    def test_qa_parses_via_loader(self, qa_fixture: GroundTruthFixture) -> None:
        """The Q&A file parses through the SeedFixtureLoader."""
        assert qa_fixture.game == "Catan"
        assert qa_fixture.version

    def test_minimum_question_count(self, qa_data: dict) -> None:
        """The Q&A set has at least the minimum number of questions."""
        count = len(qa_data["questions"])
        assert count >= _MIN_QUESTIONS, f"Expected >= {_MIN_QUESTIONS} questions, got {count}"

    def test_maximum_question_count(self, qa_data: dict) -> None:
        """The Q&A set does not exceed the reasonable maximum."""
        count = len(qa_data["questions"])
        assert count <= _MAX_QUESTIONS, f"Expected <= {_MAX_QUESTIONS} questions, got {count}"

    def test_question_ids_are_unique(self, qa_data: dict) -> None:
        """All question IDs are unique."""
        ids = [q["id"] for q in qa_data["questions"]]
        dupes = [x for x in ids if ids.count(x) > 1]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {dupes}"

    def test_question_ids_follow_convention(self, qa_data: dict) -> None:
        """All question IDs follow the GT-NNN pattern."""
        for q in qa_data["questions"]:
            qid = q["id"]
            assert qid.startswith("GT-"), f"{qid} missing GT- prefix"
            assert qid[3:].isdigit(), f"{qid} non-numeric suffix"

    def test_each_question_has_required_fields(self, qa_data: dict) -> None:
        """Every question has id, category, question, and expected."""
        for q in qa_data["questions"]:
            qid = q.get("id", "?")
            assert "id" in q, f"Missing 'id': {q}"
            assert "category" in q, f"{qid} missing 'category'"
            assert "question" in q, f"{qid} missing 'question'"
            assert "expected" in q, f"{qid} missing 'expected'"

    def test_each_expected_has_citation_requirement(self, qa_data: dict) -> None:
        """Every expected block declares must_cite."""
        for q in qa_data["questions"]:
            assert "must_cite" in q["expected"], f"{q['id']} missing 'must_cite'"

    def test_each_expected_has_answer_criteria(self, qa_data: dict) -> None:
        """Every question has at least one answer matching criterion."""
        for q in qa_data["questions"]:
            expected = q["expected"]
            has = bool(expected.get("answer_contains")) or bool(
                expected.get("answer_contains_any")
            )
            assert has, f"{q['id']} has no answer criteria"


# ---------------------------------------------------------------------------
# Tests -- Category coverage
# ---------------------------------------------------------------------------


class TestCategoryCoverage:
    """Verify all required categories are represented."""

    def test_all_categories_present(self, qa_data: dict) -> None:
        """Every required category has at least one question."""
        present = {q["category"] for q in qa_data["questions"]}
        missing = _VALID_CATEGORIES - present
        assert not missing, f"Missing categories: {missing}"

    def test_no_invalid_categories(self, qa_data: dict) -> None:
        """No question uses a category outside the valid set."""
        for q in qa_data["questions"]:
            assert q["category"] in _VALID_CATEGORIES, (
                f"{q['id']} invalid category '{q['category']}'"
            )

    def test_factual_questions_exist(self, qa_data: dict) -> None:
        """At least 2 single-doc-factual questions exist."""
        n = sum(1 for q in qa_data["questions"] if q["category"] == "single-doc-factual")
        assert n >= 2, f"Expected >= 2 factual, got {n}"

    def test_procedural_questions_exist(self, qa_data: dict) -> None:
        """At least 2 single-doc-procedural questions exist."""
        n = sum(1 for q in qa_data["questions"] if q["category"] == "single-doc-procedural")
        assert n >= 2, f"Expected >= 2 procedural, got {n}"

    def test_cross_doc_questions_exist(self, qa_data: dict) -> None:
        """At least 2 cross-doc-association questions exist."""
        n = sum(1 for q in qa_data["questions"] if q["category"] == "cross-doc-association")
        assert n >= 2, f"Expected >= 2 cross-doc, got {n}"

    def test_expansion_questions_exist(self, qa_data: dict) -> None:
        """At least 2 expansion-specific questions exist."""
        n = sum(1 for q in qa_data["questions"] if q["category"] == "expansion-specific")
        assert n >= 2, f"Expected >= 2 expansion, got {n}"

    def test_strategy_questions_exist(self, qa_data: dict) -> None:
        """At least 1 strategy question exists."""
        n = sum(1 for q in qa_data["questions"] if q["category"] == "strategy")
        assert n >= 1, f"Expected >= 1 strategy, got {n}"


# ---------------------------------------------------------------------------
# Tests -- Document type references
# ---------------------------------------------------------------------------


class TestDocumentTypeReferences:
    """Verify expected_document_types reference valid types."""

    def test_all_document_types_are_valid(self, qa_data: dict) -> None:
        """Every expected_document_types entry is a known type."""
        for q in qa_data["questions"]:
            for dt in q["expected"].get("expected_document_types", []):
                assert dt in _VALID_DOCUMENT_TYPES, f"{q['id']} references unknown type '{dt}'"

    def test_cross_doc_questions_reference_multiple_types(self, qa_data: dict) -> None:
        """Cross-doc questions reference at least 2 document types."""
        for q in qa_data["questions"]:
            if q["category"] == "cross-doc-association":
                dt = q["expected"].get("expected_document_types", [])
                assert len(dt) >= 2, f"{q['id']} cross-doc but only {len(dt)} type(s)"

    def test_expansion_questions_reference_expansion_rules(self, qa_data: dict) -> None:
        """Expansion-specific questions reference expansion_rules."""
        for q in qa_data["questions"]:
            if q["category"] == "expansion-specific":
                dt = q["expected"].get("expected_document_types", [])
                assert "expansion_rules" in dt, f"{q['id']} expansion but no expansion_rules ref"


# ---------------------------------------------------------------------------
# Tests -- Answer keywords match fixture content
# ---------------------------------------------------------------------------


class TestAnswerKeywordsMatchContent:
    """Verify expected answer keywords exist in fixture PDF text."""

    def _get_text(self, question: dict, base: str, seafarers: str, combined: str) -> str:
        """Return the text corpus relevant to a question."""
        dt = question["expected"].get("expected_document_types", [])
        if not dt:
            return combined
        if set(dt) == {"core_rules"}:
            return base
        if set(dt) == {"expansion_rules"}:
            return seafarers
        return combined

    def test_answer_contains_keywords_exist(
        self,
        qa_data: dict,
        base_rules_text: str,
        seafarers_text: str,
        all_text: str,
    ) -> None:
        """Every answer_contains keyword appears in relevant docs."""
        for q in qa_data["questions"]:
            kws = q["expected"].get("answer_contains", [])
            if not kws:
                continue
            text = self._get_text(q, base_rules_text, seafarers_text, all_text)
            for kw in kws:
                dt = q["expected"].get("expected_document_types", [])
                assert kw.lower() in text, f"{q['id']}: '{kw}' not in docs ({dt})"

    def test_answer_contains_any_keywords_exist(
        self,
        qa_data: dict,
        base_rules_text: str,
        seafarers_text: str,
        all_text: str,
    ) -> None:
        """At least one answer_contains_any keyword appears in docs."""
        for q in qa_data["questions"]:
            kws = q["expected"].get("answer_contains_any", [])
            if not kws:
                continue
            text = self._get_text(q, base_rules_text, seafarers_text, all_text)
            found = [kw for kw in kws if kw.lower() in text]
            dt = q["expected"].get("expected_document_types", [])
            assert found, f"{q['id']}: none of {kws} in docs ({dt})"

    def test_cross_doc_keywords_span_both_documents(
        self,
        qa_data: dict,
        base_rules_text: str,
        seafarers_text: str,
    ) -> None:
        """Cross-doc questions have keywords in both documents."""
        for q in qa_data["questions"]:
            if q["category"] != "cross-doc-association":
                continue
            kws = q["expected"].get("answer_contains", []) + q["expected"].get(
                "answer_contains_any", []
            )
            in_base = any(kw.lower() in base_rules_text for kw in kws)
            in_sea = any(kw.lower() in seafarers_text for kw in kws)
            assert in_base, f"{q['id']}: no keywords in base rules"
            assert in_sea, f"{q['id']}: no keywords in seafarers"


# ---------------------------------------------------------------------------
# Tests -- Confidence thresholds
# ---------------------------------------------------------------------------


class TestConfidenceThresholds:
    """Verify confidence thresholds follow design spec conventions."""

    def test_factual_confidence_at_least_0_7(self, qa_data: dict) -> None:
        """Factual questions have min_confidence >= 0.7."""
        for q in qa_data["questions"]:
            if q["category"] == "single-doc-factual":
                c = q["expected"].get("min_confidence")
                if c is not None:
                    assert c >= 0.7, f"{q['id']}: conf {c} < 0.7"

    def test_procedural_confidence_at_least_0_7(self, qa_data: dict) -> None:
        """Procedural questions have min_confidence >= 0.7."""
        for q in qa_data["questions"]:
            if q["category"] == "single-doc-procedural":
                c = q["expected"].get("min_confidence")
                if c is not None:
                    assert c >= 0.7, f"{q['id']}: conf {c} < 0.7"

    def test_cross_doc_confidence_at_least_0_5(self, qa_data: dict) -> None:
        """Cross-doc questions have min_confidence >= 0.5."""
        for q in qa_data["questions"]:
            if q["category"] == "cross-doc-association":
                c = q["expected"].get("min_confidence")
                if c is not None:
                    assert c >= 0.5, f"{q['id']}: conf {c} < 0.5"

    def test_strategy_confidence_lower_bar(self, qa_data: dict) -> None:
        """Strategy questions have a lower confidence bar (<= 0.5)."""
        for q in qa_data["questions"]:
            if q["category"] == "strategy":
                c = q["expected"].get("min_confidence")
                if c is not None:
                    assert c <= 0.5, f"{q['id']}: conf {c} > 0.5"

    def test_all_questions_have_min_confidence(self, qa_data: dict) -> None:
        """Every question declares a min_confidence value."""
        for q in qa_data["questions"]:
            assert q["expected"].get("min_confidence") is not None, (
                f"{q['id']} missing min_confidence"
            )

    def test_all_questions_require_citations(self, qa_data: dict) -> None:
        """Every question requires citations (must_cite is true)."""
        for q in qa_data["questions"]:
            assert q["expected"]["must_cite"] is True, f"{q['id']} must_cite is not True"

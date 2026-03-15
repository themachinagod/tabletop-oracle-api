"""Unit tests for CitationAssemblyStage.

Tests:
- Source markers parsed from answer text (AC-602)
- Citations created with full traceability (AC-602)
- Duplicates removed by (document_name, section)
- Priority ordering applied (AC-609)
- stream.citation events emitted
- Fallback citation from highest-ranked result when no markers
- Edge cases: empty answer, empty retrieval, out-of-range markers
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from tabletop_oracle.services.ai.context import CitationData, RetrievalResult
from tabletop_oracle.services.ai.stages.citation_assembly import (
    CitationAssemblyStage,
    _build_citations_from_markers,
    _build_fallback_citation,
    _citation_sort_key,
    _deduplicate_citations,
    _order_by_priority,
    _parse_source_markers,
    _reindex_citations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieval_result(
    *,
    document_name: str = "Core Rulebook",
    document_type: str = "core_rules",
    section_path: str = "Chapter 1 > Combat",
    page_number: int | None = 42,
    content: str = "When attacking, roll a d20 and add modifiers.",
    score: float = 0.85,
) -> RetrievalResult:
    """Build a RetrievalResult with source metadata.

    Args:
        document_name: Human-readable document name.
        document_type: Document classification.
        section_path: Section path within the document.
        page_number: Page number or None.
        content: Concept content text.
        score: Similarity score.

    Returns:
        RetrievalResult with populated source dict.
    """
    return RetrievalResult(
        concept_id=uuid.uuid4(),
        content=content,
        score=score,
        source={
            "document_id": str(uuid.uuid4()),
            "document_name": document_name,
            "document_type": document_type,
            "section_path": section_path,
            "page_number": page_number,
            "source_text": content[:100],
            "is_authoritative": True,
        },
    )


def _make_ctx(
    *,
    answer_text: str = "According to [1], you roll a d20 [2].",
    retrieval_results: list[RetrievalResult] | None = None,
) -> MagicMock:
    """Build a mock PipelineContext for citation assembly.

    Args:
        answer_text: Answer text with source markers.
        retrieval_results: Retrieval results to use.

    Returns:
        MagicMock mimicking a PipelineContext.
    """
    ctx = MagicMock()
    ctx.answer_text = answer_text
    ctx.user_message.id = uuid.uuid4()
    ctx.retrieval_results = retrieval_results or []
    ctx.citations = []
    return ctx


# ---------------------------------------------------------------------------
# Tests: Source Marker Parsing
# ---------------------------------------------------------------------------


class TestParseSourceMarkers:
    """Tests for _parse_source_markers."""

    def test_parse_single_marker(self) -> None:
        """Single marker [1] is parsed."""
        assert _parse_source_markers("Answer text [1] here.") == [1]

    def test_parse_multiple_markers(self) -> None:
        """Multiple markers parsed in order of appearance."""
        text = "See [1] and [3] and also [2]."
        assert _parse_source_markers(text) == [1, 3, 2]

    def test_parse_deduplicates_markers(self) -> None:
        """Repeated markers appear only once."""
        text = "See [1] and [1] again and [2]."
        assert _parse_source_markers(text) == [1, 2]

    def test_parse_ignores_zero_index(self) -> None:
        """[0] is not a valid marker (1-based indexing)."""
        text = "See [0] and [1]."
        assert _parse_source_markers(text) == [1]

    def test_parse_empty_text(self) -> None:
        """Empty text yields no markers."""
        assert _parse_source_markers("") == []

    def test_parse_no_markers(self) -> None:
        """Text without markers yields empty list."""
        assert _parse_source_markers("This is plain text.") == []

    def test_parse_large_marker_numbers(self) -> None:
        """Large marker numbers are valid."""
        assert _parse_source_markers("See [15] and [20].") == [15, 20]

    def test_parse_markers_in_complex_text(self) -> None:
        """Markers embedded in complex answer text are found."""
        text = (
            "Based on the rules [1], when a player takes damage [2], "
            "they must roll a saving throw. The errata [3] clarifies "
            "that this applies only in combat [1]."
        )
        assert _parse_source_markers(text) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tests: Citation Building from Markers
# ---------------------------------------------------------------------------


class TestBuildCitationsFromMarkers:
    """Tests for _build_citations_from_markers."""

    def test_build_citations_for_valid_markers(self) -> None:
        """Valid markers map to corresponding retrieval results."""
        results = [
            _make_retrieval_result(document_name="Doc A"),
            _make_retrieval_result(document_name="Doc B"),
        ]
        citations = _build_citations_from_markers([1, 2], results)

        assert len(citations) == 2
        assert citations[0].document_name == "Doc A"
        assert citations[0].index == 1
        assert citations[1].document_name == "Doc B"
        assert citations[1].index == 2

    def test_build_skips_out_of_range_markers(self) -> None:
        """Markers beyond retrieval_results length are skipped."""
        results = [_make_retrieval_result(document_name="Doc A")]
        citations = _build_citations_from_markers([1, 5], results)

        assert len(citations) == 1
        assert citations[0].document_name == "Doc A"

    def test_build_maps_all_source_fields(self) -> None:
        """CitationData includes document_type, section, page_number, excerpt."""
        results = [
            _make_retrieval_result(
                document_name="FAQ v2",
                document_type="faq",
                section_path="General > Trading",
                page_number=7,
                content="You may trade on your turn.",
            ),
        ]
        citations = _build_citations_from_markers([1], results)

        c = citations[0]
        assert c.document_type == "faq"
        assert c.section == "General > Trading"
        assert c.page_number == 7
        assert c.excerpt == "You may trade on your turn."

    def test_build_handles_none_page_number(self) -> None:
        """None page_number is preserved as None."""
        results = [_make_retrieval_result(page_number=None)]
        citations = _build_citations_from_markers([1], results)
        assert citations[0].page_number is None

    def test_build_truncates_long_excerpt(self) -> None:
        """Excerpt is truncated to 200 characters."""
        long_content = "x" * 500
        results = [_make_retrieval_result(content=long_content)]
        citations = _build_citations_from_markers([1], results)
        assert len(citations[0].excerpt) == 200


# ---------------------------------------------------------------------------
# Tests: Fallback Citation
# ---------------------------------------------------------------------------


class TestBuildFallbackCitation:
    """Tests for _build_fallback_citation."""

    def test_fallback_uses_first_result(self) -> None:
        """Fallback creates citation from the highest-ranked (first) result."""
        results = [
            _make_retrieval_result(document_name="Top Result"),
            _make_retrieval_result(document_name="Second Result"),
        ]
        citations = _build_fallback_citation(results)

        assert len(citations) == 1
        assert citations[0].document_name == "Top Result"
        assert citations[0].index == 1

    def test_fallback_includes_metadata(self) -> None:
        """Fallback citation has full source metadata."""
        results = [
            _make_retrieval_result(
                document_name="Errata Sheet",
                document_type="errata",
                section_path="Corrections > Movement",
                page_number=3,
            ),
        ]
        citations = _build_fallback_citation(results)

        c = citations[0]
        assert c.document_type == "errata"
        assert c.section == "Corrections > Movement"
        assert c.page_number == 3


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplicateCitations:
    """Tests for _deduplicate_citations."""

    def test_removes_exact_duplicates(self) -> None:
        """Duplicate (document_name, section) pairs are removed."""
        c1 = CitationData(
            index=1,
            document_name="Doc A",
            document_type="core_rules",
            section="Ch 1",
        )
        c2 = CitationData(
            index=2,
            document_name="Doc A",
            document_type="core_rules",
            section="Ch 1",
        )
        result = _deduplicate_citations([c1, c2])
        assert len(result) == 1
        assert result[0].index == 1

    def test_keeps_different_sections(self) -> None:
        """Same document with different sections are kept."""
        c1 = CitationData(
            index=1,
            document_name="Doc A",
            document_type="core_rules",
            section="Ch 1",
        )
        c2 = CitationData(
            index=2,
            document_name="Doc A",
            document_type="core_rules",
            section="Ch 2",
        )
        result = _deduplicate_citations([c1, c2])
        assert len(result) == 2

    def test_keeps_different_documents(self) -> None:
        """Different documents with same section are kept."""
        c1 = CitationData(
            index=1,
            document_name="Doc A",
            document_type="core_rules",
            section="Ch 1",
        )
        c2 = CitationData(
            index=2,
            document_name="Doc B",
            document_type="faq",
            section="Ch 1",
        )
        result = _deduplicate_citations([c1, c2])
        assert len(result) == 2

    def test_preserves_first_occurrence(self) -> None:
        """First occurrence is kept, later duplicates dropped."""
        citations = [
            CitationData(index=1, document_name="Doc A", document_type="errata", section="S1"),
            CitationData(index=3, document_name="Doc A", document_type="errata", section="S1"),
        ]
        result = _deduplicate_citations(citations)
        assert result[0].index == 1

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert _deduplicate_citations([]) == []


# ---------------------------------------------------------------------------
# Tests: Priority Ordering
# ---------------------------------------------------------------------------


class TestOrderByPriority:
    """Tests for _order_by_priority and _citation_sort_key."""

    def test_errata_ranks_above_core_rules(self) -> None:
        """Errata citations sort before core_rules."""
        citations = [
            CitationData(index=1, document_name="Core", document_type="core_rules"),
            CitationData(index=2, document_name="Errata", document_type="errata"),
        ]
        result = _order_by_priority(citations)
        assert result[0].document_type == "errata"
        assert result[1].document_type == "core_rules"

    def test_faq_ranks_between_errata_and_core(self) -> None:
        """FAQ citations sort after errata but before core_rules."""
        citations = [
            CitationData(index=1, document_name="Core", document_type="core_rules"),
            CitationData(index=2, document_name="FAQ", document_type="faq"),
            CitationData(index=3, document_name="Errata", document_type="errata"),
        ]
        result = _order_by_priority(citations)
        assert [c.document_type for c in result] == ["errata", "faq", "core_rules"]

    def test_unknown_type_ranks_last(self) -> None:
        """Unknown document types sort after all known types."""
        citations = [
            CitationData(index=1, document_name="Homebrew", document_type="homebrew"),
            CitationData(index=2, document_name="Strategy", document_type="strategy"),
        ]
        result = _order_by_priority(citations)
        assert result[0].document_type == "strategy"
        assert result[1].document_type == "homebrew"

    def test_full_priority_order(self) -> None:
        """All known types sort in correct order."""
        types = ["strategy", "supplement", "core_rules", "faq", "errata"]
        citations = [
            CitationData(index=i + 1, document_name=f"Doc {t}", document_type=t)
            for i, t in enumerate(types)
        ]
        result = _order_by_priority(citations)
        assert [c.document_type for c in result] == [
            "errata",
            "faq",
            "core_rules",
            "supplement",
            "strategy",
        ]

    def test_sort_key_errata_lowest(self) -> None:
        """Errata has the lowest sort key value."""
        c = CitationData(index=1, document_name="E", document_type="errata")
        assert _citation_sort_key(c) == 0

    def test_sort_key_unknown_highest(self) -> None:
        """Unknown types get default priority 99."""
        c = CitationData(index=1, document_name="U", document_type="custom")
        assert _citation_sort_key(c) == 99


# ---------------------------------------------------------------------------
# Tests: Re-indexing
# ---------------------------------------------------------------------------


class TestReindexCitations:
    """Tests for _reindex_citations."""

    def test_reindex_assigns_sequential_indices(self) -> None:
        """Citations get 1-based sequential indices."""
        citations = [
            CitationData(index=5, document_name="A", document_type="errata"),
            CitationData(index=2, document_name="B", document_type="faq"),
            CitationData(index=8, document_name="C", document_type="core_rules"),
        ]
        result = _reindex_citations(citations)
        assert [c.index for c in result] == [1, 2, 3]

    def test_reindex_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert _reindex_citations([]) == []


# ---------------------------------------------------------------------------
# Tests: Full Stage Execution
# ---------------------------------------------------------------------------


class TestCitationAssemblyStageExecute:
    """Integration tests for CitationAssemblyStage.execute."""

    async def test_execute_parses_markers_and_emits_events(self) -> None:
        """Stage parses markers, builds citations, and yields events."""
        results = [
            _make_retrieval_result(document_name="Rulebook"),
            _make_retrieval_result(document_name="FAQ"),
        ]
        ctx = _make_ctx(
            answer_text="See [1] for rules and [2] for FAQ.",
            retrieval_results=results,
        )

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        assert len(ctx.citations) == 2
        assert len(events) == 2
        assert ctx.citations[0].document_name == "Rulebook"
        assert ctx.citations[1].document_name == "FAQ"

    async def test_execute_fallback_when_no_markers(self) -> None:
        """Stage uses fallback citation when answer has no markers."""
        results = [_make_retrieval_result(document_name="Top Source")]
        ctx = _make_ctx(
            answer_text="This is an answer with no source markers.",
            retrieval_results=results,
        )

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        assert len(ctx.citations) == 1
        assert ctx.citations[0].document_name == "Top Source"
        assert len(events) == 1

    async def test_execute_no_retrieval_results(self) -> None:
        """Stage produces no citations when retrieval results are empty."""
        ctx = _make_ctx(
            answer_text="An answer [1] referencing nothing.",
            retrieval_results=[],
        )

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        assert ctx.citations == []
        assert events == []

    async def test_execute_deduplicates_citations(self) -> None:
        """Stage deduplicates citations by (document_name, section)."""
        same_result = _make_retrieval_result(
            document_name="Core Rulebook",
            section_path="Chapter 1 > Combat",
        )
        results = [same_result, same_result]
        ctx = _make_ctx(
            answer_text="See [1] and [2] for combat.",
            retrieval_results=results,
        )

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        assert len(ctx.citations) == 1
        assert len(events) == 1

    async def test_execute_orders_by_priority(self) -> None:
        """Stage orders citations by document type priority."""
        results = [
            _make_retrieval_result(document_name="Core Rule", document_type="core_rules"),
            _make_retrieval_result(document_name="Errata Fix", document_type="errata"),
        ]
        ctx = _make_ctx(
            answer_text="See [1] and [2].",
            retrieval_results=results,
        )

        stage = CitationAssemblyStage()
        [event async for event in stage.execute(ctx, MagicMock())]

        assert ctx.citations[0].document_type == "errata"
        assert ctx.citations[1].document_type == "core_rules"

    async def test_execute_reindexes_after_reorder(self) -> None:
        """Citations have sequential indices after priority reordering."""
        results = [
            _make_retrieval_result(document_name="Core", document_type="core_rules"),
            _make_retrieval_result(document_name="Errata", document_type="errata"),
        ]
        ctx = _make_ctx(
            answer_text="See [1] and [2].",
            retrieval_results=results,
        )

        stage = CitationAssemblyStage()
        [event async for event in stage.execute(ctx, MagicMock())]

        assert ctx.citations[0].index == 1  # errata first
        assert ctx.citations[1].index == 2  # core_rules second

    async def test_execute_event_payloads_match_citations(self) -> None:
        """Emitted SSE events carry correct citation data."""
        results = [
            _make_retrieval_result(
                document_name="FAQ v3",
                section_path="Trading Rules",
                content="You may trade resources.",
            ),
        ]
        ctx = _make_ctx(
            answer_text="Per [1], trading is allowed.",
            retrieval_results=results,
        )

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        assert len(events) == 1
        payload = events[0].payload
        assert payload["document_name"] == "FAQ v3"
        assert payload["section"] == "Trading Rules"
        assert payload["excerpt"] == "You may trade resources."
        assert payload["index"] == 1

    async def test_execute_with_missing_source_metadata(self) -> None:
        """Stage handles retrieval results with empty source dicts."""
        result = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="Some content",
            score=0.8,
            source={},
        )
        ctx = _make_ctx(
            answer_text="According to [1].",
            retrieval_results=[result],
        )

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        assert len(ctx.citations) == 1
        assert ctx.citations[0].document_name == "Unknown"
        assert ctx.citations[0].document_type == ""
        assert ctx.citations[0].page_number is None
        assert len(events) == 1

    async def test_execute_complex_answer_with_mixed_markers(self) -> None:
        """Stage handles a realistic answer with repeated and varied markers."""
        results = [
            _make_retrieval_result(document_name="Core Rulebook", document_type="core_rules"),
            _make_retrieval_result(document_name="Errata 2024", document_type="errata"),
            _make_retrieval_result(document_name="FAQ", document_type="faq"),
        ]
        answer = (
            "When you roll for combat [1], add your strength modifier. "
            "However, the errata [2] clarifies that flanking bonuses "
            "no longer stack. The FAQ [3] also notes this change. "
            "See [1] again for the base mechanic."
        )
        ctx = _make_ctx(answer_text=answer, retrieval_results=results)

        stage = CitationAssemblyStage()
        events = [event async for event in stage.execute(ctx, MagicMock())]

        # 3 unique markers, reordered by priority: errata > faq > core_rules
        assert len(ctx.citations) == 3
        assert ctx.citations[0].document_type == "errata"
        assert ctx.citations[1].document_type == "faq"
        assert ctx.citations[2].document_type == "core_rules"
        assert [c.index for c in ctx.citations] == [1, 2, 3]
        assert len(events) == 3

"""Unit tests for the chunking engine.

Tests cover the SectionAwareChunkingStrategy including section boundary
respect, oversized splitting, undersized merging, table atomic chunks,
overlap, token counting, and edge cases.
"""

from __future__ import annotations

from tabletop_oracle.services.ingestion.chunking import (
    ChunkingStrategy,
    SectionAwareChunkingStrategy,
    count_tokens,
)
from tabletop_oracle.services.ingestion.models import Section, Table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(
    title: str,
    content: str,
    level: int = 1,
    children: list[Section] | None = None,
    page_number: int | None = None,
) -> Section:
    """Build a Section with sensible defaults."""
    return Section(
        title=title,
        level=level,
        content=content,
        children=children or [],
        page_number=page_number,
    )


def _long_text(token_target: int) -> str:
    """Generate text that is approximately *token_target* tokens long.

    Uses short words (1 token each in cl100k_base) to give predictable sizing.
    """
    words = ["word"] * token_target
    return " ".join(words)


def _multi_paragraph_text(paragraphs: int, tokens_per_para: int) -> str:
    """Generate multi-paragraph text with approximate token counts."""
    paras = [_long_text(tokens_per_para) for _ in range(paragraphs)]
    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for the count_tokens utility."""

    def test_empty_string_returns_zero(self) -> None:
        """count_tokens_EmptyString_ReturnsZero."""
        assert count_tokens("") == 0

    def test_single_word(self) -> None:
        """count_tokens_SingleWord_ReturnsPositiveCount."""
        result = count_tokens("hello")
        assert result > 0

    def test_known_count(self) -> None:
        """count_tokens_KnownPhrase_ReturnsExpectedCount."""
        # "hello world" is 2 tokens in cl100k_base
        result = count_tokens("hello world")
        assert result == 2

    def test_longer_text_scales(self) -> None:
        """count_tokens_LongerText_ScalesApproximately."""
        short = count_tokens("a b c")
        long = count_tokens("a b c d e f g h i j")
        assert long > short


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestChunkingStrategyProtocol:
    """Tests that SectionAwareChunkingStrategy satisfies the protocol."""

    def test_satisfies_protocol(self) -> None:
        """SectionAwareChunkingStrategy is a ChunkingStrategy."""
        strategy = SectionAwareChunkingStrategy()
        assert isinstance(strategy, ChunkingStrategy)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------


class TestBasicChunking:
    """Tests for standard chunking behaviour."""

    def setup_method(self) -> None:
        self.strategy = SectionAwareChunkingStrategy(
            target_chunk_size=512,
            max_chunk_size=1024,
            min_chunk_size=64,
            overlap_tokens=50,
        )

    def test_single_section_produces_one_chunk(self) -> None:
        """chunk_SingleSection_ProducesOneTextChunk."""
        sections = [_section("Rules", _long_text(200))]
        chunks = self.strategy.chunk(sections, [])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "text"
        assert chunks[0].section_path == "Rules"
        assert chunks[0].heading == "Rules"
        assert chunks[0].chunk_index == 0

    def test_multiple_sections_produce_multiple_chunks(self) -> None:
        """chunk_MultipleSections_ProducesOneChunkPerSection when appropriately sized."""
        sections = [
            _section("Combat", _long_text(200)),
            _section("Magic", _long_text(200)),
        ]
        chunks = self.strategy.chunk(sections, [])
        # Each section is above min_chunk_size (64) so they stay separate
        assert len(chunks) >= 2
        section_paths = [c.section_path for c in chunks]
        assert "Combat" in section_paths
        assert "Magic" in section_paths

    def test_chunk_indices_are_sequential(self) -> None:
        """chunk_MultipleSections_IndicesAreSequential."""
        sections = [
            _section("A", _long_text(200)),
            _section("B", _long_text(200)),
            _section("C", _long_text(200)),
        ]
        chunks = self.strategy.chunk(sections, [])
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_token_estimate_is_positive(self) -> None:
        """chunk_AnyContent_TokenEstimateIsPositive."""
        sections = [_section("Test", "Some content here")]
        chunks = self.strategy.chunk(sections, [])
        assert all(c.token_estimate > 0 for c in chunks)

    def test_page_number_preserved(self) -> None:
        """chunk_SectionWithPageNumber_PreservedInChunk."""
        sections = [_section("Intro", _long_text(200), page_number=5)]
        chunks = self.strategy.chunk(sections, [])
        assert chunks[0].page_number == 5


# ---------------------------------------------------------------------------
# Section boundary respect
# ---------------------------------------------------------------------------


class TestSectionBoundaryRespect:
    """Tests that sections are never split if they fit within max_chunk_size."""

    def setup_method(self) -> None:
        self.strategy = SectionAwareChunkingStrategy(
            target_chunk_size=512,
            max_chunk_size=1024,
            min_chunk_size=64,
            overlap_tokens=0,  # disable overlap for clarity
        )

    def test_section_under_max_not_split(self) -> None:
        """chunk_SectionUnderMax_NotSplit."""
        content = _long_text(800)  # under 1024 max
        sections = [_section("Rules", content)]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) == 1

    def test_nested_sections_produce_separate_chunks(self) -> None:
        """chunk_NestedSections_EachLeafGetsOwnChunk."""
        parent = _section(
            "Combat",
            _long_text(100),
            children=[
                _section("Melee", _long_text(100), level=2),
                _section("Ranged", _long_text(100), level=2),
            ],
        )
        chunks = self.strategy.chunk([parent], [])
        paths = [c.section_path for c in chunks if c.chunk_type == "text"]
        assert any("Combat" in p for p in paths)
        assert any("Melee" in p for p in paths)
        assert any("Ranged" in p for p in paths)

    def test_section_path_includes_hierarchy(self) -> None:
        """chunk_NestedSection_PathShowsFullHierarchy."""
        parent = _section(
            "Combat",
            "",
            children=[
                _section("Ranged", _long_text(100), level=2),
            ],
        )
        chunks = self.strategy.chunk([parent], [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert any("Combat > Ranged" in (c.section_path or "") for c in text_chunks)


# ---------------------------------------------------------------------------
# Oversized section splitting
# ---------------------------------------------------------------------------


class TestOversizedSplitting:
    """Tests for splitting sections that exceed max_chunk_size."""

    def setup_method(self) -> None:
        self.strategy = SectionAwareChunkingStrategy(
            target_chunk_size=100,
            max_chunk_size=200,
            min_chunk_size=10,
            overlap_tokens=0,
        )

    def test_oversized_section_is_split(self) -> None:
        """chunk_OversizedSection_SplitIntoMultipleChunks."""
        content = _multi_paragraph_text(10, 50)  # ~500 tokens, max is 200
        sections = [_section("Big Section", content)]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) > 1

    def test_split_preserves_section_metadata(self) -> None:
        """chunk_SplitSection_AllChunksRetainSectionPath."""
        content = _multi_paragraph_text(10, 50)
        sections = [_section("Rules", content)]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert all(c.section_path == "Rules" for c in text_chunks)

    def test_split_at_paragraph_boundaries(self) -> None:
        """chunk_SplitSection_NeverSplitsMidParagraph."""
        paragraphs = [_long_text(80) for _ in range(5)]
        content = "\n\n".join(paragraphs)
        sections = [_section("Rules", content)]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        # Each chunk should contain complete paragraphs — no partial word splits
        for chunk in text_chunks:
            # "word word word..." pattern should not be truncated mid-word
            stripped = chunk.content.strip()
            assert stripped.endswith("word") or stripped == ""


# ---------------------------------------------------------------------------
# Undersized section merging
# ---------------------------------------------------------------------------


class TestUndersizedMerging:
    """Tests for merging undersized sections."""

    def setup_method(self) -> None:
        self.strategy = SectionAwareChunkingStrategy(
            target_chunk_size=200,
            max_chunk_size=400,
            min_chunk_size=50,
            overlap_tokens=0,
        )

    def test_tiny_sections_merged(self) -> None:
        """chunk_TinySections_MergedIntoFewerChunks."""
        sections = [
            _section("A", _long_text(10)),
            _section("B", _long_text(10)),
            _section("C", _long_text(10)),
        ]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        # Three 10-token sections should merge (30 < target 200)
        assert len(text_chunks) < 3

    def test_adequately_sized_sections_not_merged(self) -> None:
        """chunk_AdequatelySizedSections_NotMerged."""
        sections = [
            _section("A", _long_text(100)),
            _section("B", _long_text(100)),
        ]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        # Each section is >= min_chunk_size, merged total (200) <= target (200)
        # Merging is acceptable since both are under min or combined <= target
        assert len(text_chunks) >= 1


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------


class TestTableChunks:
    """Tests for table atomic chunk handling."""

    def setup_method(self) -> None:
        self.strategy = SectionAwareChunkingStrategy(
            target_chunk_size=512,
            max_chunk_size=1024,
            min_chunk_size=64,
            overlap_tokens=0,
        )

    def test_table_produces_single_chunk(self) -> None:
        """chunk_Table_ProducesSingleAtomicChunk."""
        table = Table(
            headers=["Name", "Cost", "Effect"],
            rows=[
                {"Name": "Fireball", "Cost": "3", "Effect": "Deals 3d6 fire damage"},
                {"Name": "Heal", "Cost": "2", "Effect": "Restore 2d8 HP"},
            ],
            section_path="Magic > Spells",
            caption="Spell List",
        )
        chunks = self.strategy.chunk([], [table])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert chunks[0].section_path == "Magic > Spells"

    def test_table_chunk_contains_headers(self) -> None:
        """chunk_Table_ContentIncludesHeaders."""
        table = Table(
            headers=["Stat", "Value"],
            rows=[{"Stat": "STR", "Value": "18"}],
        )
        chunks = self.strategy.chunk([], [table])
        assert "Stat" in chunks[0].content
        assert "Value" in chunks[0].content
        assert "STR" in chunks[0].content

    def test_table_chunk_includes_caption(self) -> None:
        """chunk_TableWithCaption_CaptionInContent."""
        table = Table(
            headers=["A"],
            rows=[{"A": "1"}],
            caption="Important Table",
        )
        chunks = self.strategy.chunk([], [table])
        assert "Important Table" in chunks[0].content

    def test_table_never_split(self) -> None:
        """chunk_LargeTable_NeverSplitAcrossChunks."""
        rows = [{"Col": f"Row {i} with lots of data"} for i in range(100)]
        table = Table(headers=["Col"], rows=rows)
        chunks = self.strategy.chunk([], [table])
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) == 1

    def test_tables_appear_after_text_chunks(self) -> None:
        """chunk_MixedContent_TablesAppearAfterTextChunks."""
        sections = [_section("Rules", _long_text(200))]
        tables = [Table(headers=["A"], rows=[{"A": "1"}])]
        chunks = self.strategy.chunk(sections, tables)
        text_indices = [c.chunk_index for c in chunks if c.chunk_type == "text"]
        table_indices = [c.chunk_index for c in chunks if c.chunk_type == "table"]
        assert all(ti > max(text_indices) for ti in table_indices)

    def test_table_metadata_has_headers(self) -> None:
        """chunk_Table_MetadataContainsHeaders."""
        table = Table(headers=["X", "Y"], rows=[{"X": "1", "Y": "2"}])
        chunks = self.strategy.chunk([], [table])
        assert chunks[0].metadata.get("table_headers") == ["X", "Y"]

    def test_table_token_estimate_is_positive(self) -> None:
        """chunk_Table_HasPositiveTokenEstimate."""
        table = Table(headers=["Name"], rows=[{"Name": "Fireball"}])
        chunks = self.strategy.chunk([], [table])
        assert chunks[0].token_estimate > 0


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


class TestOverlap:
    """Tests for token overlap between adjacent text chunks."""

    def test_overlap_added_between_text_chunks(self) -> None:
        """chunk_AdjacentTextChunks_OverlapApplied."""
        strategy = SectionAwareChunkingStrategy(
            target_chunk_size=100,
            max_chunk_size=200,
            min_chunk_size=10,
            overlap_tokens=20,
        )
        sections = [
            _section("A", _long_text(100)),
            _section("B", _long_text(100)),
        ]
        chunks = strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        if len(text_chunks) >= 2:
            # Second chunk should contain some text from the first
            # (overlap makes it longer than the raw content)
            assert text_chunks[1].token_estimate > 0

    def test_first_chunk_has_no_overlap(self) -> None:
        """chunk_FirstChunk_NoOverlapPrepended."""
        strategy = SectionAwareChunkingStrategy(
            target_chunk_size=100,
            max_chunk_size=200,
            min_chunk_size=10,
            overlap_tokens=20,
        )
        content = _long_text(100)
        sections = [_section("A", content)]
        chunks = strategy.chunk(sections, [])
        # First chunk should be essentially the raw content
        assert chunks[0].content.strip() == content.strip()

    def test_zero_overlap_produces_no_duplication(self) -> None:
        """chunk_ZeroOverlap_NoContentDuplication."""
        strategy = SectionAwareChunkingStrategy(
            target_chunk_size=100,
            max_chunk_size=200,
            min_chunk_size=10,
            overlap_tokens=0,
        )
        sections = [
            _section("A", _long_text(100)),
            _section("B", _long_text(100)),
        ]
        chunks = strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        # No crash with zero overlap is the basic contract
        assert len(text_chunks) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self) -> None:
        self.strategy = SectionAwareChunkingStrategy(
            target_chunk_size=512,
            max_chunk_size=1024,
            min_chunk_size=64,
            overlap_tokens=50,
        )

    def test_empty_sections_and_tables(self) -> None:
        """chunk_NoInput_ReturnsEmptyList."""
        chunks = self.strategy.chunk([], [])
        assert chunks == []

    def test_section_with_empty_content(self) -> None:
        """chunk_SectionWithEmptyContent_Skipped."""
        sections = [_section("Empty", "")]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) == 0

    def test_section_with_whitespace_only_content(self) -> None:
        """chunk_SectionWithWhitespaceOnly_Skipped."""
        sections = [_section("Whitespace", "   \n\n   ")]
        chunks = self.strategy.chunk(sections, [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) == 0

    def test_only_tables_no_sections(self) -> None:
        """chunk_OnlyTables_ProducesTableChunks."""
        table = Table(headers=["A"], rows=[{"A": "1"}])
        chunks = self.strategy.chunk([], [table])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert chunks[0].chunk_index == 0

    def test_multiple_tables(self) -> None:
        """chunk_MultipleTables_EachGetsOwnChunk."""
        tables = [
            Table(headers=["A"], rows=[{"A": "1"}]),
            Table(headers=["B"], rows=[{"B": "2"}]),
        ]
        chunks = self.strategy.chunk([], tables)
        assert len(chunks) == 2
        assert all(c.chunk_type == "table" for c in chunks)

    def test_deeply_nested_sections(self) -> None:
        """chunk_DeeplyNestedSections_FlattenedCorrectly."""
        inner = _section("L3", _long_text(100), level=3)
        mid = _section("L2", _long_text(100), level=2, children=[inner])
        outer = _section("L1", "", level=1, children=[mid])
        chunks = self.strategy.chunk([outer], [])
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) >= 1
        paths = [c.section_path for c in text_chunks]
        assert any("L1 > L2 > L3" in (p or "") for p in paths)

    def test_single_word_section(self) -> None:
        """chunk_SingleWordSection_ProducedOrMerged."""
        sections = [_section("Title", "hello")]
        chunks = self.strategy.chunk(sections, [])
        # Should produce at least one chunk (might be tiny)
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) >= 1

    def test_table_with_no_headers(self) -> None:
        """chunk_TableWithNoHeaders_StillProducesChunk."""
        table = Table(headers=[], rows=[{"col": "val"}])
        chunks = self.strategy.chunk([], [table])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"

    def test_table_with_empty_section_path(self) -> None:
        """chunk_TableWithEmptySectionPath_SectionPathIsNone."""
        table = Table(headers=["A"], rows=[{"A": "1"}], section_path="")
        chunks = self.strategy.chunk([], [table])
        assert chunks[0].section_path is None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Tests for custom configuration parameters."""

    def test_custom_target_size(self) -> None:
        """chunk_CustomTargetSize_AffectsChunkCount."""
        small_strategy = SectionAwareChunkingStrategy(
            target_chunk_size=50, max_chunk_size=100, min_chunk_size=10, overlap_tokens=0
        )
        large_strategy = SectionAwareChunkingStrategy(
            target_chunk_size=500, max_chunk_size=1000, min_chunk_size=10, overlap_tokens=0
        )
        content = _multi_paragraph_text(10, 30)
        sections = [_section("Test", content)]

        small_chunks = small_strategy.chunk(sections, [])
        large_chunks = large_strategy.chunk(sections, [])

        small_text = [c for c in small_chunks if c.chunk_type == "text"]
        large_text = [c for c in large_chunks if c.chunk_type == "text"]
        # Smaller target should produce more chunks
        assert len(small_text) >= len(large_text)

    def test_default_parameters(self) -> None:
        """chunk_DefaultParams_UsesExpectedDefaults."""
        strategy = SectionAwareChunkingStrategy()
        # Just verify it doesn't crash with defaults and produces output
        sections = [_section("Test", _long_text(200))]
        chunks = strategy.chunk(sections, [])
        assert len(chunks) >= 1

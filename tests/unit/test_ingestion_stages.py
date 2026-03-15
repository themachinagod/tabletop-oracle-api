"""Tests for ingestion pipeline stage implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tabletop_oracle.services.ingestion.models import (
    ParseResult,
    PipelineContext,
    Section,
    StructureResult,
)
from tabletop_oracle.services.ingestion.stages import (
    ChunkingStage,
    ExtractionStage,
    KGHandoffStage,
    StructureDetectionStage,
    ValidationStage,
)


def _make_context(**overrides: object) -> PipelineContext:
    """Build a PipelineContext with sensible defaults."""
    defaults: dict[str, object] = {
        "document_id": uuid4(),
        "version_id": uuid4(),
        "game_id": uuid4(),
        "document_type": "core_rules",
        "expansion_id": None,
        "file_path": "documents/game1/doc1/v1/rules.pdf",
        "document_format": "pdf",
        "file_size": 1024,
    }
    defaults.update(overrides)
    return PipelineContext(**defaults)  # type: ignore[arg-type]


class TestValidationStage:
    """Tests for ValidationStage."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert ValidationStage().name == "validation"

    def test_execute_valid_context(self) -> None:
        """Validation passes with valid context."""
        ctx = _make_context()
        ValidationStage().execute(ctx)
        # No exception = pass

    def test_execute_empty_file_path_raises(self) -> None:
        """Validation fails when file_path is empty."""
        ctx = _make_context(file_path="")
        with pytest.raises(ValueError, match="File path is empty"):
            ValidationStage().execute(ctx)

    def test_execute_empty_format_raises(self) -> None:
        """Validation fails when document_format is empty."""
        ctx = _make_context(document_format="")
        with pytest.raises(ValueError, match="Document format is not set"):
            ValidationStage().execute(ctx)

    def test_execute_zero_file_size_raises(self) -> None:
        """Validation fails when file_size is zero."""
        ctx = _make_context(file_size=0)
        with pytest.raises(ValueError, match="Invalid file size"):
            ValidationStage().execute(ctx)

    def test_execute_negative_file_size_raises(self) -> None:
        """Validation fails when file_size is negative."""
        ctx = _make_context(file_size=-1)
        with pytest.raises(ValueError, match="Invalid file size"):
            ValidationStage().execute(ctx)


class TestExtractionStage:
    """Tests for ExtractionStage — dispatches to format-specific parsers."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert ExtractionStage().name == "extraction"

    def test_execute_sets_parse_result(self, tmp_path: Path) -> None:
        """Extraction populates context.parse_result with real file."""
        text_file = tmp_path / "rules.txt"
        text_file.write_text("Some rules content here.")
        ctx = _make_context(file_path=str(text_file), document_format="text")
        ExtractionStage().execute(ctx)
        assert ctx.parse_result is not None
        assert isinstance(ctx.parse_result, ParseResult)
        assert ctx.parse_result.raw_text != ""
        assert len(ctx.parse_result.sections) > 0

    def test_execute_preserves_format_in_metadata(self, tmp_path: Path) -> None:
        """Extraction includes document format in metadata."""
        md_file = tmp_path / "rules.md"
        md_file.write_text("# Rules\n\nContent.\n")
        ctx = _make_context(file_path=str(md_file), document_format="markdown")
        ExtractionStage().execute(ctx)
        assert ctx.parse_result is not None
        assert ctx.parse_result.metadata.get("format") == "markdown"

    def test_execute_unknown_format_raises(self) -> None:
        """Extraction raises RuntimeError for unsupported formats."""
        ctx = _make_context(document_format="xyz_unknown")
        with pytest.raises(RuntimeError, match="No parser available"):
            ExtractionStage().execute(ctx)


class TestStructureDetectionStage:
    """Tests for StructureDetectionStage (stub)."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert StructureDetectionStage().name == "structure_detection"

    def test_execute_sets_structure_result(self) -> None:
        """Structure detection populates context.structure_result."""
        ctx = _make_context()
        ctx.parse_result = ParseResult(
            raw_text="Test",
            sections=[Section(title="Intro", level=1, content="Hello")],
        )
        StructureDetectionStage().execute(ctx)
        assert ctx.structure_result is not None
        assert isinstance(ctx.structure_result, StructureResult)
        assert len(ctx.structure_result.sections) == 1

    def test_execute_no_parse_result_raises(self) -> None:
        """Structure detection fails when parse_result is not set."""
        ctx = _make_context()
        with pytest.raises(RuntimeError, match="parse_result is not set"):
            StructureDetectionStage().execute(ctx)


class TestChunkingStage:
    """Tests for ChunkingStage with pluggable strategy."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert ChunkingStage().name == "chunking"

    def test_execute_produces_chunks(self) -> None:
        """Chunking produces chunks from sections via strategy."""
        ctx = _make_context()
        ctx.structure_result = StructureResult(
            sections=[
                Section(title="Combat", level=1, content="Combat rules here"),
                Section(title="Magic", level=1, content="Magic rules here"),
            ],
        )
        ChunkingStage().execute(ctx)
        assert len(ctx.chunks) >= 1
        assert ctx.chunks[0].chunk_index == 0
        # Sections may be merged if undersized — verify at least one has content
        assert any("Combat" in c.section_path for c in ctx.chunks if c.section_path)

    def test_execute_chunk_has_token_estimate(self) -> None:
        """Chunking calculates token estimate via tiktoken."""
        ctx = _make_context()
        ctx.structure_result = StructureResult(
            sections=[Section(title="Test", level=1, content="one two three")],
        )
        ChunkingStage().execute(ctx)
        assert ctx.chunks[0].token_estimate > 0

    def test_execute_no_structure_result_raises(self) -> None:
        """Chunking fails when structure_result is not set."""
        ctx = _make_context()
        with pytest.raises(RuntimeError, match="structure_result is not set"):
            ChunkingStage().execute(ctx)

    def test_execute_uses_custom_strategy(self) -> None:
        """Chunking delegates to the injected strategy."""
        from tabletop_oracle.services.ingestion.models import Chunk, Table

        class FixedStrategy:
            """Always returns one fixed chunk."""

            def chunk(self, sections: list[Section], tables: list[Table]) -> list[Chunk]:
                return [Chunk(chunk_index=0, chunk_type="text", content="fixed")]

        ctx = _make_context()
        ctx.structure_result = StructureResult(
            sections=[Section(title="X", level=1, content="ignored")],
        )
        ChunkingStage(strategy=FixedStrategy()).execute(ctx)
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].content == "fixed"


class TestKGHandoffStage:
    """Tests for KGHandoffStage (stub)."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert KGHandoffStage().name == "kg_handoff"

    def test_execute_with_chunks_succeeds(self) -> None:
        """KG handoff succeeds when chunks are present."""
        from tabletop_oracle.services.ingestion.models import Chunk

        ctx = _make_context()
        ctx.chunks = [Chunk(chunk_index=0, chunk_type="text", content="Hello")]
        KGHandoffStage().execute(ctx)
        # No exception = pass

    def test_execute_no_chunks_raises(self) -> None:
        """KG handoff fails when no chunks are produced."""
        ctx = _make_context()
        ctx.chunks = []
        with pytest.raises(RuntimeError, match="no chunks produced"):
            KGHandoffStage().execute(ctx)

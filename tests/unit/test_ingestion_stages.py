"""Tests for ingestion pipeline stage implementations."""

from uuid import uuid4

import pytest

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
    defaults = {
        "document_id": uuid4(),
        "version_id": uuid4(),
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
    """Tests for ExtractionStage (stub)."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert ExtractionStage().name == "extraction"

    def test_execute_sets_parse_result(self) -> None:
        """Extraction stub populates context.parse_result."""
        ctx = _make_context()
        ExtractionStage().execute(ctx)
        assert ctx.parse_result is not None
        assert isinstance(ctx.parse_result, ParseResult)
        assert ctx.parse_result.raw_text != ""
        assert len(ctx.parse_result.sections) > 0

    def test_execute_preserves_format_in_metadata(self) -> None:
        """Extraction stub includes document format in metadata."""
        ctx = _make_context(document_format="markdown")
        ExtractionStage().execute(ctx)
        assert ctx.parse_result is not None
        assert ctx.parse_result.metadata.get("format") == "markdown"


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
    """Tests for ChunkingStage (stub)."""

    def test_name(self) -> None:
        """Stage has the expected name."""
        assert ChunkingStage().name == "chunking"

    def test_execute_produces_chunks(self) -> None:
        """Chunking stub produces one chunk per section."""
        ctx = _make_context()
        ctx.structure_result = StructureResult(
            sections=[
                Section(title="Combat", level=1, content="Combat rules here"),
                Section(title="Magic", level=1, content="Magic rules here"),
            ],
        )
        ChunkingStage().execute(ctx)
        assert len(ctx.chunks) == 2
        assert ctx.chunks[0].chunk_index == 0
        assert ctx.chunks[1].chunk_index == 1
        assert ctx.chunks[0].section_path == "Combat"
        assert ctx.chunks[1].section_path == "Magic"

    def test_execute_chunk_has_token_estimate(self) -> None:
        """Chunking stub calculates token estimate based on word count."""
        ctx = _make_context()
        ctx.structure_result = StructureResult(
            sections=[Section(title="Test", level=1, content="one two three")],
        )
        ChunkingStage().execute(ctx)
        assert ctx.chunks[0].token_estimate == 3

    def test_execute_no_structure_result_raises(self) -> None:
        """Chunking fails when structure_result is not set."""
        ctx = _make_context()
        with pytest.raises(RuntimeError, match="structure_result is not set"):
            ChunkingStage().execute(ctx)


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

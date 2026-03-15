"""Tests for ingestion pipeline data models."""

from uuid import uuid4

from tabletop_oracle.services.ingestion.models import (
    Chunk,
    ExtractedImage,
    ParseResult,
    PipelineContext,
    PipelineStage,
    Section,
    StructureResult,
    Table,
)


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_stage_values(self) -> None:
        """All expected stages are defined with correct string values."""
        assert PipelineStage.VALIDATING == "validating"
        assert PipelineStage.EXTRACTING == "extracting"
        assert PipelineStage.DETECTING_STRUCTURE == "detecting_structure"
        assert PipelineStage.CHUNKING == "chunking"
        assert PipelineStage.INTEGRATING == "integrating"

    def test_stage_count(self) -> None:
        """Exactly five stages are defined."""
        assert len(PipelineStage) == 5


class TestSection:
    """Tests for Section dataclass."""

    def test_create_minimal(self) -> None:
        """Section with required fields only."""
        section = Section(title="Intro", level=1, content="Hello")
        assert section.title == "Intro"
        assert section.level == 1
        assert section.content == "Hello"
        assert section.children == []
        assert section.page_number is None
        assert section.source_location == ""

    def test_create_with_children(self) -> None:
        """Section with nested children."""
        child = Section(title="Sub", level=2, content="World")
        parent = Section(title="Parent", level=1, content="Hello", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].title == "Sub"


class TestTable:
    """Tests for Table dataclass."""

    def test_create_table(self) -> None:
        """Table with headers and rows."""
        table = Table(
            headers=["Name", "HP"],
            rows=[{"Name": "Goblin", "HP": "7"}],
            section_path="Monsters",
            caption="Monster Stats",
        )
        assert table.headers == ["Name", "HP"]
        assert len(table.rows) == 1
        assert table.caption == "Monster Stats"


class TestExtractedImage:
    """Tests for ExtractedImage dataclass."""

    def test_create_image(self) -> None:
        """ExtractedImage with required fields."""
        img = ExtractedImage(
            image_data=b"\x89PNG",
            format="png",
            location="page 3",
        )
        assert img.image_data == b"\x89PNG"
        assert img.format == "png"
        assert img.alt_text is None


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_create_empty(self) -> None:
        """ParseResult with only raw_text."""
        result = ParseResult(raw_text="Hello")
        assert result.raw_text == "Hello"
        assert result.sections == []
        assert result.tables == []
        assert result.images == []
        assert result.metadata == {}


class TestStructureResult:
    """Tests for StructureResult dataclass."""

    def test_create_empty(self) -> None:
        """StructureResult with defaults."""
        result = StructureResult()
        assert result.sections == []
        assert result.tables == []
        assert result.metadata == {}


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_minimal(self) -> None:
        """Chunk with required fields."""
        chunk = Chunk(chunk_index=0, chunk_type="text", content="Hello world")
        assert chunk.chunk_index == 0
        assert chunk.chunk_type == "text"
        assert chunk.content == "Hello world"
        assert chunk.section_path is None
        assert chunk.heading is None
        assert chunk.page_number is None
        assert chunk.token_estimate == 0
        assert chunk.metadata == {}


class TestPipelineContext:
    """Tests for PipelineContext dataclass."""

    def test_create_context(self) -> None:
        """PipelineContext with required fields."""
        doc_id = uuid4()
        ver_id = uuid4()
        game_id = uuid4()
        ctx = PipelineContext(
            document_id=doc_id,
            version_id=ver_id,
            game_id=game_id,
            document_type="core_rules",
            expansion_id=None,
            file_path="documents/game1/doc1/v1/rules.pdf",
            document_format="pdf",
            file_size=1024,
        )
        assert ctx.document_id == doc_id
        assert ctx.version_id == ver_id
        assert ctx.game_id == game_id
        assert ctx.document_type == "core_rules"
        assert ctx.expansion_id is None
        assert ctx.replaces_version_id is None
        assert ctx.parse_result is None
        assert ctx.structure_result is None
        assert ctx.chunks == []

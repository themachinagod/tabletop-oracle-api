"""Internal data models for the ingestion pipeline.

These models carry data between pipeline stages. They are internal to
the pipeline — not persisted directly. Final output is written to the
DocumentChunk ORM model.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uuid import UUID


class PipelineStage(enum.StrEnum):
    """Pipeline stage identifiers matching SSE event stage values."""

    VALIDATING = "validating"
    EXTRACTING = "extracting"
    DETECTING_STRUCTURE = "detecting_structure"
    CHUNKING = "chunking"
    INTEGRATING = "integrating"


@dataclass
class Section:
    """A document section with hierarchy.

    Attributes:
        title: Section heading text. Empty string for untitled sections.
        level: Heading level (1 = top-level, 2 = sub-section, etc.).
        content: Plain text content of this section (excluding children).
        children: Nested sub-sections.
        page_number: Page where this section starts (PDF only).
        source_location: Format-specific location identifier.
    """

    title: str
    level: int
    content: str
    children: list[Section] = field(default_factory=list)
    page_number: int | None = None
    source_location: str = ""


@dataclass
class Table:
    """An extracted table.

    Attributes:
        headers: Column header labels.
        rows: Table data as list of row dictionaries.
        section_path: Section context where this table appears.
        caption: Table caption if present.
    """

    headers: list[str]
    rows: list[dict[str, str]]
    section_path: str = ""
    caption: str | None = None


@dataclass
class ExtractedImage:
    """An image extracted from a document.

    Attributes:
        image_data: Raw image bytes.
        format: Image format (png, jpeg, etc.).
        location: Where in the document the image appears.
        section_path: Section context for traceability.
        alt_text: Alt text if provided in the source document.
    """

    image_data: bytes
    format: str
    location: str
    section_path: str = ""
    alt_text: str | None = None


@dataclass
class ParseResult:
    """Output of a format-specific parser.

    Attributes:
        raw_text: Full extracted text content.
        sections: Hierarchical section structure.
        tables: Extracted tables as structured data.
        images: Extracted image references with location context.
        metadata: Format-specific metadata (page count, word count, etc.).
    """

    raw_text: str
    sections: list[Section] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureResult:
    """Output of structure detection.

    Attributes:
        sections: Normalised section hierarchy.
        tables: Tables with enriched section context.
        metadata: Additional structure metadata.
    """

    sections: list[Section] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A content chunk produced by the chunking stage.

    Attributes:
        chunk_index: Sequential position within the document version.
        chunk_type: Content type (text, table, image_description).
        content: The chunk text content.
        section_path: Hierarchical section context.
        heading: Nearest heading this chunk falls under.
        page_number: Source page number (PDF only).
        token_estimate: Approximate token count.
        metadata: Format-specific metadata.
    """

    chunk_index: int
    chunk_type: str
    content: str
    section_path: str | None = None
    heading: str | None = None
    page_number: int | None = None
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable context passed through pipeline stages.

    Each stage reads from and writes to this context. It accumulates
    intermediate results as the document progresses through the pipeline.

    Attributes:
        document_id: UUID of the document being processed.
        version_id: UUID of the specific version to process.
        game_id: UUID of the owning game (for KG scoping).
        document_type: Document classification (e.g. core_rules, faq).
        expansion_id: Expansion association (optional, for KG filtering).
        file_path: Blob store path to the document file.
        document_format: File format identifier.
        file_size: File size in bytes.
        replaces_version_id: Previous version ID if this is a replacement.
        parse_result: Output from the extraction stage.
        structure_result: Output from the structure detection stage.
        chunks: Output from the chunking stage.
    """

    document_id: UUID
    version_id: UUID
    game_id: UUID
    document_type: str
    expansion_id: UUID | None
    file_path: str
    document_format: str
    file_size: int
    replaces_version_id: UUID | None = None
    parse_result: ParseResult | None = None
    structure_result: StructureResult | None = None
    chunks: list[Chunk] = field(default_factory=list)

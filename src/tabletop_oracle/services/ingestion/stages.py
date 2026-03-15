"""Pipeline stage interfaces and stub implementations.

Each stage implements a common protocol: receive a PipelineContext, perform
work, mutate the context with results. Stub implementations provide the
interface contract for real implementations in downstream tasks (#32, #33, #34).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tabletop_oracle.services.ingestion.models import (
    Chunk,
    ParseResult,
    Section,
    StructureResult,
)

if TYPE_CHECKING:
    from tabletop_oracle.services.ingestion.models import PipelineContext

logger = logging.getLogger(__name__)


class PipelineStageBase(ABC):
    """Abstract base for pipeline stages.

    Each stage receives a PipelineContext, performs its work, and mutates
    the context with its output. Stages raise exceptions on failure; the
    pipeline orchestrator handles error recovery.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name for logging and events."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> None:
        """Execute this pipeline stage.

        Args:
            context: Mutable pipeline context with accumulated results.

        Raises:
            Exception: On processing failure. The pipeline orchestrator
                handles cleanup.
        """


class ValidationStage(PipelineStageBase):
    """Validates file integrity and confirms format detection.

    Stub implementation — verifies basic preconditions. Real validation
    (magic byte checks, format-specific integrity) is task #32.
    """

    @property
    def name(self) -> str:
        return "validation"

    def execute(self, context: PipelineContext) -> None:
        """Validate the document file.

        Args:
            context: Pipeline context with file_path and document_format.

        Raises:
            ValueError: If basic preconditions fail.
        """
        if not context.file_path:
            msg = "File path is empty"
            raise ValueError(msg)
        if not context.document_format:
            msg = "Document format is not set"
            raise ValueError(msg)
        if context.file_size <= 0:
            msg = f"Invalid file size: {context.file_size}"
            raise ValueError(msg)

        logger.info(
            "Validation passed: format=%s, size=%d, path=%s",
            context.document_format,
            context.file_size,
            context.file_path,
        )


class ExtractionStage(PipelineStageBase):
    """Extracts text and structure from the document file.

    Stub implementation — produces a minimal ParseResult. Real format-specific
    parsers (PDF, Markdown, HTML, DOCX, text) are task #32.
    """

    @property
    def name(self) -> str:
        return "extraction"

    def execute(self, context: PipelineContext) -> None:
        """Extract content from the document.

        Args:
            context: Pipeline context with file_path and document_format.
                Sets context.parse_result on success.
        """
        logger.info(
            "Extraction stub: format=%s, path=%s",
            context.document_format,
            context.file_path,
        )
        # Stub: produce a placeholder parse result.
        # Real implementations (#32) will dispatch to format-specific parsers.
        context.parse_result = ParseResult(
            raw_text="[stub] Extracted text placeholder",
            sections=[
                Section(
                    title="Document Content",
                    level=1,
                    content="[stub] Content placeholder for extraction stage",
                ),
            ],
            metadata={"stub": True, "format": context.document_format},
        )


class StructureDetectionStage(PipelineStageBase):
    """Detects and normalises document structure.

    Stub implementation — passes through extraction sections. Real structure
    detection (heading hierarchy, table identification) is task #33.
    """

    @property
    def name(self) -> str:
        return "structure_detection"

    def execute(self, context: PipelineContext) -> None:
        """Detect document structure from parsed content.

        Args:
            context: Pipeline context with parse_result set.
                Sets context.structure_result on success.

        Raises:
            RuntimeError: If parse_result is not set (extraction skipped).
        """
        if context.parse_result is None:
            msg = "Cannot detect structure: parse_result is not set"
            raise RuntimeError(msg)

        logger.info("Structure detection stub: sections=%d", len(context.parse_result.sections))

        # Stub: pass through sections from extraction
        context.structure_result = StructureResult(
            sections=context.parse_result.sections,
            tables=context.parse_result.tables,
            metadata={"stub": True},
        )


class ChunkingStage(PipelineStageBase):
    """Breaks structured content into traceable chunks.

    Stub implementation — produces one chunk per section. Real chunking
    strategies (semantic, size-based, overlap) are task #33.
    """

    @property
    def name(self) -> str:
        return "chunking"

    def execute(self, context: PipelineContext) -> None:
        """Chunk the structured content.

        Args:
            context: Pipeline context with structure_result set.
                Sets context.chunks on success.

        Raises:
            RuntimeError: If structure_result is not set.
        """
        if context.structure_result is None:
            msg = "Cannot chunk: structure_result is not set"
            raise RuntimeError(msg)

        logger.info(
            "Chunking stub: sections=%d",
            len(context.structure_result.sections),
        )

        # Stub: produce one chunk per section
        chunks: list[Chunk] = []
        for idx, section in enumerate(context.structure_result.sections):
            content = section.content or section.title
            chunks.append(
                Chunk(
                    chunk_index=idx,
                    chunk_type="text",
                    content=content,
                    section_path=section.title,
                    heading=section.title,
                    page_number=section.page_number,
                    token_estimate=len(content.split()),
                    metadata={"stub": True},
                )
            )

        context.chunks = chunks


class KGHandoffStage(PipelineStageBase):
    """Hands off processed chunks to the Knowledge Graph engine.

    Stub implementation — logs the handoff. Real KG integration is task #34.
    """

    @property
    def name(self) -> str:
        return "kg_handoff"

    def execute(self, context: PipelineContext) -> None:
        """Hand off chunks to the KG engine.

        Args:
            context: Pipeline context with chunks set.

        Raises:
            RuntimeError: If chunks list is empty.
        """
        if not context.chunks:
            msg = "Cannot hand off to KG: no chunks produced"
            raise RuntimeError(msg)

        logger.info(
            "KG handoff stub: document_id=%s, chunk_count=%d",
            context.document_id,
            len(context.chunks),
        )

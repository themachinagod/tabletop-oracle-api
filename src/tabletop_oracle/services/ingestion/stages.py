"""Pipeline stage interfaces and implementations.

Each stage implements a common protocol: receive a PipelineContext, perform
work, mutate the context with results. The ExtractionStage dispatches to
format-specific parsers via the ParserRegistry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from tabletop_oracle.services.ingestion.chunking import (
    ChunkingStrategy,
    SectionAwareChunkingStrategy,
)
from tabletop_oracle.services.ingestion.handoff import (
    KGHandoffPayload,
    KGHandoffService,
    StubKGHandoffService,
)
from tabletop_oracle.services.ingestion.models import (
    StructureResult,
)
from tabletop_oracle.services.ingestion.parsers.registry import (
    ParserRegistry,
    create_default_registry,
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

    Dispatches to format-specific parsers via the ParserRegistry.
    Each parser produces a standardised ParseResult.

    Args:
        registry: Parser registry to use for format lookup. Defaults to
            the built-in registry with all five parsers.
    """

    def __init__(self, registry: ParserRegistry | None = None) -> None:
        self._registry = registry or create_default_registry()

    @property
    def name(self) -> str:
        return "extraction"

    def execute(self, context: PipelineContext) -> None:
        """Extract content from the document using a format-specific parser.

        Looks up the parser for ``context.document_format`` and delegates
        parsing to it. Raises RuntimeError if no parser supports the format.

        Args:
            context: Pipeline context with file_path and document_format.
                Sets context.parse_result on success.

        Raises:
            RuntimeError: If no parser supports the document format.
        """
        parser = self._registry.get_parser(context.document_format)
        if parser is None:
            msg = f"No parser available for format: {context.document_format}"
            raise RuntimeError(msg)

        logger.info(
            "Extracting document: format=%s, parser=%s, path=%s",
            context.document_format,
            type(parser).__name__,
            context.file_path,
        )

        file_path = Path(context.file_path)
        context.parse_result = parser.parse(
            file_path,
            metadata={"format": context.document_format},
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

    Delegates to a pluggable ``ChunkingStrategy``. Defaults to
    ``SectionAwareChunkingStrategy`` with standard size parameters.

    Args:
        strategy: Chunking strategy to use. Defaults to
            ``SectionAwareChunkingStrategy()``.
    """

    def __init__(self, strategy: ChunkingStrategy | None = None) -> None:
        self._strategy: ChunkingStrategy = strategy or SectionAwareChunkingStrategy()

    @property
    def name(self) -> str:
        return "chunking"

    def execute(self, context: PipelineContext) -> None:
        """Chunk the structured content using the configured strategy.

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
            "Chunking: sections=%d, tables=%d, strategy=%s",
            len(context.structure_result.sections),
            len(context.structure_result.tables),
            type(self._strategy).__name__,
        )

        context.chunks = self._strategy.chunk(
            context.structure_result.sections,
            context.structure_result.tables,
        )


class KGHandoffStage(PipelineStageBase):
    """Hands off processed chunks to the Knowledge Graph engine.

    Delegates to a pluggable ``KGHandoffService``. Defaults to the V1
    stub that logs and returns. Replaced via DI when EPIC-003 delivers
    the real KG integration.

    Args:
        service: KG handoff service implementation. Defaults to
            ``StubKGHandoffService()``.
    """

    def __init__(self, service: KGHandoffService | None = None) -> None:
        self._service: KGHandoffService = service or StubKGHandoffService()

    @property
    def name(self) -> str:
        return "kg_handoff"

    def execute(self, context: PipelineContext) -> None:
        """Build a KGHandoffPayload and delegate to the KG service.

        Runs the async service method synchronously via asyncio.run()
        because the pipeline executes in Celery's sync prefork pool.

        Args:
            context: Pipeline context with chunks set.

        Raises:
            RuntimeError: If chunks list is empty.
            KGIngestionError: If the KG service fails.
        """
        import asyncio

        if not context.chunks:
            msg = "Cannot hand off to KG: no chunks produced"
            raise RuntimeError(msg)

        payload = KGHandoffPayload(
            document_id=context.document_id,
            version_id=context.version_id,
            game_id=context.game_id,
            document_type=context.document_type,
            expansion_id=context.expansion_id,
            chunks=tuple(context.chunks),
            replaces_version_id=context.replaces_version_id,
        )

        logger.info(
            "KG handoff: document_id=%s, chunk_count=%d",
            context.document_id,
            len(context.chunks),
        )

        asyncio.run(self._service.ingest_document(payload))

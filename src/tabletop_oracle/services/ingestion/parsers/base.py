"""Abstract base class for format-specific document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tabletop_oracle.services.ingestion.models import ParseResult


class DocumentParser(ABC):
    """Interface for format-specific document parsers.

    Each parser extracts text, structure, tables, and images from a specific
    document format, producing a standardised ParseResult.
    """

    @abstractmethod
    def parse(self, file_path: Path, metadata: dict[str, object]) -> ParseResult:
        """Parse a document file and return structured content.

        Args:
            file_path: Path to the document file on disk.
            metadata: Additional context (e.g., original filename, content type).

        Returns:
            ParseResult with extracted text, sections, tables, and images.

        Raises:
            ExtractionError: If parsing fails.
        """

    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """Check whether this parser handles the given content type.

        Args:
            content_type: MIME type or format identifier (e.g., 'pdf', 'text/html').

        Returns:
            True if this parser can handle the content type.
        """

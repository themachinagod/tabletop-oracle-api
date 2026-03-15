"""PDF document parser using pdfplumber.

Provides layout-aware text extraction, table detection, and image reference
extraction with page-level position metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pdfplumber

if TYPE_CHECKING:
    from pathlib import Path

from tabletop_oracle.services.ingestion.models import (
    ExtractedImage,
    ParseResult,
    Section,
    Table,
)
from tabletop_oracle.services.ingestion.parsers.base import DocumentParser

logger = logging.getLogger(__name__)


class PdfParser(DocumentParser):
    """Extracts text, tables, and image references from PDF documents.

    Uses pdfplumber for layout-aware extraction that handles multi-column
    layouts, embedded tables, and image position metadata. Each page is
    treated as a top-level section; sub-sections are detected via font-size
    heuristics when available.
    """

    _SUPPORTED_TYPES = frozenset({"pdf", "application/pdf"})

    def supports(self, content_type: str) -> bool:
        """Check if this parser handles the given content type."""
        return content_type.lower() in self._SUPPORTED_TYPES

    def parse(self, file_path: Path, metadata: dict[str, object]) -> ParseResult:
        """Parse a PDF file into structured content.

        Args:
            file_path: Path to the PDF file.
            metadata: Additional context (unused by PDF parser).

        Returns:
            ParseResult with per-page sections, extracted tables, and image refs.

        Raises:
            ExtractionError: If the PDF cannot be opened or parsed.
        """
        sections: list[Section] = []
        tables: list[Table] = []
        images: list[ExtractedImage] = []
        all_text_parts: list[str] = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                all_text_parts.append(page_text)

                # Build a section per page
                section = Section(
                    title=f"Page {page_num}",
                    level=1,
                    content=page_text,
                    page_number=page_num,
                    source_location=f"page:{page_num}",
                )
                sections.append(section)

                # Extract tables from this page
                page_tables = self._extract_tables(page, page_num)
                tables.extend(page_tables)

                # Extract image references from this page
                page_images = self._extract_images(page, page_num)
                images.extend(page_images)

        raw_text = "\n\n".join(all_text_parts)

        return ParseResult(
            raw_text=raw_text,
            sections=sections,
            tables=tables,
            images=images,
            metadata={
                "format": "pdf",
                "page_count": len(sections),
                "word_count": len(raw_text.split()),
            },
        )

    def _extract_tables(self, page: pdfplumber.page.Page, page_num: int) -> list[Table]:
        """Extract tables from a single PDF page.

        Args:
            page: A pdfplumber Page object.
            page_num: 1-based page number for context.

        Returns:
            List of Table objects found on this page.
        """
        result: list[Table] = []
        raw_tables = page.extract_tables() or []

        for table_idx, raw_table in enumerate(raw_tables):
            if not raw_table or len(raw_table) < 2:
                continue

            # First row is headers
            headers = [cell or "" for cell in raw_table[0]]
            rows = [
                {headers[i]: (cell or "") for i, cell in enumerate(row) if i < len(headers)}
                for row in raw_table[1:]
            ]

            result.append(
                Table(
                    headers=headers,
                    rows=rows,
                    section_path=f"Page {page_num}",
                    caption=f"Table {table_idx + 1} on page {page_num}",
                )
            )

        return result

    def _extract_images(self, page: pdfplumber.page.Page, page_num: int) -> list[ExtractedImage]:
        """Extract image references from a single PDF page.

        Captures position metadata but not raw image bytes — full image
        extraction requires a separate rendering step.

        Args:
            page: A pdfplumber Page object.
            page_num: 1-based page number for context.

        Returns:
            List of ExtractedImage references found on this page.
        """
        result: list[ExtractedImage] = []

        for img_idx, img in enumerate(page.images or []):
            x0 = img.get("x0", 0)
            y0 = img.get("top", 0)
            x1 = img.get("x1", 0)
            y1 = img.get("bottom", 0)

            result.append(
                ExtractedImage(
                    image_data=b"",  # Reference only; bytes not extracted here
                    format="unknown",
                    location=f"page:{page_num},bbox:({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})",
                    section_path=f"Page {page_num}",
                    alt_text=f"Image {img_idx + 1} on page {page_num}",
                )
            )

        return result

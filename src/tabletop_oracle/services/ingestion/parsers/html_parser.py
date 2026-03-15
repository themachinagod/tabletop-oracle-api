"""HTML document parser using BeautifulSoup4.

Strips navigation and chrome elements, preserves semantic structure
(headings, lists, tables), and produces a clean section hierarchy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

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

# HTML elements considered navigation/chrome — stripped before extraction
_CHROME_TAGS = frozenset(
    {
        "nav",
        "header",
        "footer",
        "aside",
        "script",
        "style",
        "noscript",
        "iframe",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "meta",
        "link",
    }
)

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


class HtmlParser(DocumentParser):
    """Extracts structured content from HTML documents.

    Strips navigation elements, scripts, and chrome. Preserves semantic
    structure: headings become sections, tables are extracted as structured
    data, and images are captured as references.
    """

    _SUPPORTED_TYPES = frozenset({"html", "text/html", "application/xhtml+xml"})

    def supports(self, content_type: str) -> bool:
        """Check if this parser handles the given content type."""
        return content_type.lower() in self._SUPPORTED_TYPES

    def parse(self, file_path: Path, metadata: dict[str, object]) -> ParseResult:
        """Parse an HTML file into structured content.

        Args:
            file_path: Path to the HTML file.
            metadata: Additional context (unused by HTML parser).

        Returns:
            ParseResult with heading-based sections, tables, and image refs.
        """
        raw_html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")

        self._strip_chrome(soup)

        body = soup.body or soup
        raw_text = body.get_text(separator="\n", strip=True)

        sections = self._build_sections(body)
        tables = self._extract_tables(body)
        images = self._extract_images(body)

        return ParseResult(
            raw_text=raw_text,
            sections=sections,
            tables=tables,
            images=images,
            metadata={
                "format": "html",
                "word_count": len(raw_text.split()),
            },
        )

    def _strip_chrome(self, soup: BeautifulSoup) -> None:
        """Remove navigation, scripts, and other non-content elements.

        Args:
            soup: BeautifulSoup document tree (modified in place).
        """
        for tag_name in _CHROME_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

    def _build_sections(self, root: Tag | BeautifulSoup) -> list[Section]:
        """Build sections from heading elements in the HTML.

        Walks the document tree collecting content between headings.
        Each heading starts a new section at the heading's level.

        Args:
            root: The root element to extract from (typically <body>).

        Returns:
            Flat list of Section objects with level reflecting heading depth.
        """
        sections: list[Section] = []
        current_title = ""
        current_level = 0
        content_parts: list[str] = []

        for element in root.descendants:
            if isinstance(element, Tag) and element.name in _HEADING_TAGS:
                # Flush previous section
                if current_title or content_parts:
                    sections.append(
                        Section(
                            title=current_title,
                            level=current_level or 1,
                            content="\n".join(content_parts).strip(),
                            source_location=f"html:{element.name}",
                        )
                    )
                    content_parts = []

                current_title = element.get_text(strip=True)
                current_level = int(element.name[1])

            elif isinstance(element, NavigableString):
                text = str(element).strip()
                # Skip text inside headings (already captured as title)
                parent = element.parent
                if parent and isinstance(parent, Tag) and parent.name in _HEADING_TAGS:
                    continue
                # Skip text inside tables (extracted separately)
                if parent and isinstance(parent, Tag) and _is_inside_table(parent):
                    continue
                if text:
                    content_parts.append(text)

        # Flush final section
        if current_title or content_parts:
            sections.append(
                Section(
                    title=current_title,
                    level=current_level or 1,
                    content="\n".join(content_parts).strip(),
                    source_location="html:body",
                )
            )

        # If no headings found, create a single root section
        if not sections:
            text = root.get_text(separator="\n", strip=True) if isinstance(root, Tag) else ""
            sections.append(
                Section(
                    title="",
                    level=1,
                    content=text,
                    source_location="html:body",
                )
            )

        return sections

    def _extract_tables(self, root: Tag | BeautifulSoup) -> list[Table]:
        """Extract HTML tables as structured data.

        Args:
            root: The root element to search for tables.

        Returns:
            List of Table objects.
        """
        tables: list[Table] = []

        for table_el in root.find_all("table"):
            if not isinstance(table_el, Tag):
                continue

            headers: list[str] = []
            rows: list[dict[str, str]] = []

            # Extract headers from <thead> or first <tr>
            thead = table_el.find("thead")
            if thead and isinstance(thead, Tag):
                header_cells = thead.find_all("th")
                headers = [
                    cell.get_text(strip=True) for cell in header_cells if isinstance(cell, Tag)
                ]

            # Extract body rows
            tbody = table_el.find("tbody")
            body_rows_source = tbody if tbody and isinstance(tbody, Tag) else table_el
            tr_elements = body_rows_source.find_all("tr")

            for tr in tr_elements:
                if not isinstance(tr, Tag):
                    continue
                cells = tr.find_all(["td", "th"])
                cell_texts = [c.get_text(strip=True) for c in cells if isinstance(c, Tag)]

                # If no headers yet, use first row as headers
                if not headers and cell_texts:
                    headers = cell_texts
                    continue

                if headers and cell_texts:
                    row_dict = {
                        headers[i]: (cell_texts[i] if i < len(cell_texts) else "")
                        for i in range(len(headers))
                    }
                    rows.append(row_dict)

            if headers:
                # Look for a <caption>
                caption_el = table_el.find("caption")
                caption = (
                    caption_el.get_text(strip=True)
                    if caption_el and isinstance(caption_el, Tag)
                    else None
                )
                tables.append(Table(headers=headers, rows=rows, caption=caption))

        return tables

    def _extract_images(self, root: Tag | BeautifulSoup) -> list[ExtractedImage]:
        """Extract image references from HTML.

        Args:
            root: The root element to search for images.

        Returns:
            List of ExtractedImage references (no raw bytes).
        """
        images: list[ExtractedImage] = []

        for img in root.find_all("img"):
            if not isinstance(img, Tag):
                continue
            src = img.get("src", "") or ""
            alt = img.get("alt", "") or ""

            images.append(
                ExtractedImage(
                    image_data=b"",
                    format=_guess_image_format(str(src)),
                    location=f"html:img[src={src}]",
                    alt_text=str(alt) if alt else None,
                )
            )

        return images


def _is_inside_table(element: Tag) -> bool:
    """Check if an element is nested inside a <table>.

    Args:
        element: A BeautifulSoup Tag.

    Returns:
        True if any ancestor is a <table> element.
    """
    parent = element.parent
    while parent:
        if isinstance(parent, Tag) and parent.name == "table":
            return True
        parent = parent.parent
    return False


def _guess_image_format(src: str) -> str:
    """Guess image format from a src attribute.

    Args:
        src: The image source URL or path.

    Returns:
        Format string (e.g., 'png', 'jpeg') or 'unknown'.
    """
    lower = src.lower()
    for fmt in ("png", "jpg", "jpeg", "gif", "svg", "webp"):
        if lower.endswith(f".{fmt}"):
            return "jpeg" if fmt == "jpg" else fmt
    return "unknown"

"""Markdown document parser using markdown-it-py.

AST-based parsing that preserves heading hierarchy, code blocks, and tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from tabletop_oracle.services.ingestion.models import ParseResult, Section, Table
from tabletop_oracle.services.ingestion.parsers.base import DocumentParser

logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """Extracts structured content from Markdown documents.

    Uses markdown-it-py to produce an AST, then walks the token stream
    to build a section hierarchy from headings, extract tables, and
    collect code blocks as content.
    """

    _SUPPORTED_TYPES = frozenset({"markdown", "text/markdown", "text/x-markdown"})

    def supports(self, content_type: str) -> bool:
        """Check if this parser handles the given content type."""
        return content_type.lower() in self._SUPPORTED_TYPES

    def parse(self, file_path: Path, metadata: dict[str, object]) -> ParseResult:
        """Parse a Markdown file into structured content.

        Args:
            file_path: Path to the Markdown file.
            metadata: Additional context (unused by Markdown parser).

        Returns:
            ParseResult with heading-based sections and extracted tables.
        """
        raw_text = file_path.read_text(encoding="utf-8")
        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(raw_text)

        sections = self._build_sections(tokens)
        tables = self._extract_tables(tokens)

        return ParseResult(
            raw_text=raw_text,
            sections=sections,
            tables=tables,
            metadata={
                "format": "markdown",
                "word_count": len(raw_text.split()),
            },
        )

    def _build_sections(self, tokens: Sequence[Any]) -> list[Section]:
        """Build a flat list of sections from the markdown token stream.

        Each heading creates a new section. Content between headings is
        accumulated into the most recent section.

        Args:
            tokens: markdown-it-py token list.

        Returns:
            Flat list of Section objects (hierarchy via level field).
        """
        sections: list[Section] = []
        current_title = ""
        current_level = 0
        content_parts: list[str] = []
        in_heading = False
        heading_level = 0

        for token in tokens:
            token_type = getattr(token, "type", "")
            tag = getattr(token, "tag", "")
            content = getattr(token, "content", "")
            children = getattr(token, "children", None)

            if token_type == "heading_open":
                # Flush previous section
                if current_title or content_parts:
                    sections.append(
                        Section(
                            title=current_title,
                            level=current_level or 1,
                            content="\n".join(content_parts).strip(),
                            source_location=f"heading:L{current_level}",
                        )
                    )
                    content_parts = []

                in_heading = True
                heading_level = int(tag[1]) if tag and len(tag) > 1 and tag[1:].isdigit() else 1

            elif token_type == "heading_close":
                in_heading = False
                current_level = heading_level

            elif token_type == "inline" and in_heading:
                current_title = self._extract_inline_text(children, content)

            elif token_type == "inline" and not in_heading:
                text = self._extract_inline_text(children, content)
                if text:
                    content_parts.append(text)

            elif token_type == "fence":
                # Code blocks
                lang = getattr(token, "info", "") or ""
                code_text = content
                label = f"```{lang}" if lang else "```"
                content_parts.append(f"{label}\n{code_text}```")

        # Flush final section
        if current_title or content_parts:
            sections.append(
                Section(
                    title=current_title,
                    level=current_level or 1,
                    content="\n".join(content_parts).strip(),
                    source_location=f"heading:L{current_level}",
                )
            )

        # If no headings found, create a single root section
        if not sections:
            sections.append(
                Section(
                    title="",
                    level=1,
                    content="",
                    source_location="root",
                )
            )

        return sections

    def _extract_inline_text(self, children: Sequence[Any] | None, fallback: str) -> str:
        """Extract plain text from inline token children.

        Args:
            children: Child tokens of an inline token.
            fallback: Fallback text if no children.

        Returns:
            Concatenated plain text content.
        """
        if not children:
            return fallback
        parts: list[str] = []
        for child in children:
            child_content = getattr(child, "content", "")
            if child_content:
                parts.append(child_content)
        return "".join(parts) if parts else fallback

    def _extract_tables(self, tokens: Sequence[Any]) -> list[Table]:
        """Extract tables from the markdown token stream.

        Walks through table_open/table_close token groups and builds
        Table objects from header and body rows.

        Args:
            tokens: markdown-it-py token list.

        Returns:
            List of Table objects.
        """
        tables: list[Table] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if getattr(token, "type", "") == "table_open":
                table, i = self._parse_table_tokens(tokens, i + 1)
                if table is not None:
                    tables.append(table)
            else:
                i += 1
        return tables

    def _parse_table_tokens(self, tokens: Sequence[Any], start: int) -> tuple[Table | None, int]:
        """Parse tokens between table_open and table_close into a Table.

        Args:
            tokens: Full token list.
            start: Index after table_open.

        Returns:
            Tuple of (Table or None, next index after table_close).
        """
        headers: list[str] = []
        rows: list[dict[str, str]] = []
        in_thead = False
        in_tbody = False
        current_row: list[str] = []
        i = start

        while i < len(tokens):
            token = tokens[i]
            token_type = getattr(token, "type", "")

            if token_type == "table_close":
                return (
                    Table(headers=headers, rows=rows) if headers else None,
                    i + 1,
                )
            elif token_type == "thead_open":
                in_thead = True
            elif token_type == "thead_close":
                in_thead = False
            elif token_type == "tbody_open":
                in_tbody = True
            elif token_type == "tbody_close":
                in_tbody = False
            elif token_type == "tr_close":
                if in_thead:
                    headers = current_row
                elif in_tbody and headers:
                    row_dict = {
                        headers[j]: (current_row[j] if j < len(current_row) else "")
                        for j in range(len(headers))
                    }
                    rows.append(row_dict)
                current_row = []
            elif token_type == "inline":
                content = getattr(token, "content", "")
                children = getattr(token, "children", None)
                current_row.append(self._extract_inline_text(children, content))

            i += 1

        return None, i

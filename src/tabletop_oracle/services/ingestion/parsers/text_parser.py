"""Plain text document parser with heuristic section detection.

Detects section boundaries using blank line gaps, capitalisation patterns,
and numbering conventions common in tabletop game rulebooks.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from tabletop_oracle.services.ingestion.models import ParseResult, Section

if TYPE_CHECKING:
    from pathlib import Path
from tabletop_oracle.services.ingestion.parsers.base import DocumentParser

logger = logging.getLogger(__name__)

# Patterns that indicate a section heading in plain text
_HEADING_PATTERNS = [
    # ALL CAPS line (at least 3 chars, not just numbers/punctuation)
    re.compile(r"^[A-Z][A-Z\s\d\-:]{2,}$"),
    # Numbered heading: "1. Title", "1.2 Title", "1.2.3 Title"
    re.compile(r"^\d+(?:\.\d+)*\.?\s+\S"),
    # Roman numeral heading: "I. Title", "IV. Title"
    re.compile(r"^[IVXLC]+\.\s+\S"),
    # Dashed or equals underlined heading (detected via following line)
    # Handled separately in _is_underline_heading
]


class TextParser(DocumentParser):
    """Extracts structured content from plain text documents.

    Uses heuristic patterns to detect section boundaries: blank line gaps,
    ALL CAPS headings, numbered headings, and underline-style headings.
    No external dependencies — uses only the standard library.
    """

    _SUPPORTED_TYPES = frozenset({"text", "text/plain"})

    def supports(self, content_type: str) -> bool:
        """Check if this parser handles the given content type."""
        return content_type.lower() in self._SUPPORTED_TYPES

    def parse(self, file_path: Path, metadata: dict[str, object]) -> ParseResult:
        """Parse a plain text file into structured content.

        Args:
            file_path: Path to the text file.
            metadata: Additional context (unused by text parser).

        Returns:
            ParseResult with heuristically detected sections.
        """
        raw_text = file_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
        sections = self._detect_sections(lines)

        return ParseResult(
            raw_text=raw_text,
            sections=sections,
            metadata={
                "format": "text",
                "word_count": len(raw_text.split()),
                "line_count": len(lines),
            },
        )

    def _detect_sections(self, lines: list[str]) -> list[Section]:
        """Detect sections using heading heuristics and blank line gaps.

        Args:
            lines: Document split into lines.

        Returns:
            List of Section objects.
        """
        sections: list[Section] = []
        current_title = ""
        current_level = 1
        content_parts: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for underline-style heading (next line is ===== or -----)
            if i + 1 < len(lines) and stripped and self._is_underline(lines[i + 1].strip()):
                # Flush previous section
                if current_title or content_parts:
                    section = self._make_section(current_title, current_level, content_parts)
                    sections.append(section)
                    content_parts = []

                current_title = stripped
                underline_char = lines[i + 1].strip()[0]
                current_level = 1 if underline_char == "=" else 2
                i += 2  # Skip the underline
                continue

            # Check for heading patterns
            heading_level = self._detect_heading(stripped)
            if heading_level is not None:
                # Flush previous section
                if current_title or content_parts:
                    section = self._make_section(current_title, current_level, content_parts)
                    sections.append(section)
                    content_parts = []

                current_title = stripped
                current_level = heading_level
                i += 1
                continue

            # Regular content line
            if stripped:
                content_parts.append(stripped)
            elif content_parts:
                # Blank line — keep as paragraph separator
                content_parts.append("")

            i += 1

        # Flush final section
        if current_title or content_parts:
            sections.append(self._make_section(current_title, current_level, content_parts))

        # If no sections detected, create a single root section
        if not sections:
            full_text = "\n".join(lines).strip()
            sections.append(
                Section(title="", level=1, content=full_text, source_location="text:root")
            )

        return sections

    def _detect_heading(self, line: str) -> int | None:
        """Check if a line matches a heading pattern.

        Args:
            line: Stripped line of text.

        Returns:
            Heading level (1 or 2) if detected, None otherwise.
        """
        if not line or len(line) > 120:
            return None

        for pattern in _HEADING_PATTERNS:
            if pattern.match(line):
                # Numbered headings: level from dot count
                if line[0].isdigit():
                    dot_count = line.split()[0].count(".")
                    return min(dot_count + 1, 4)
                # ALL CAPS or Roman numerals: level 1
                return 1

        return None

    def _is_underline(self, line: str) -> bool:
        """Check if a line is a heading underline (===== or -----).

        Args:
            line: Stripped line.

        Returns:
            True if the line consists of at least 3 repeated = or - chars.
        """
        if len(line) < 3:
            return False
        return all(c == "=" for c in line) or all(c == "-" for c in line)

    def _make_section(self, title: str, level: int, content_parts: list[str]) -> Section:
        """Create a Section from accumulated content parts.

        Args:
            title: Section heading text.
            level: Heading level.
            content_parts: Lines of content for this section.

        Returns:
            A Section dataclass instance.
        """
        # Strip trailing blank lines
        while content_parts and not content_parts[-1]:
            content_parts.pop()

        return Section(
            title=title,
            level=level,
            content="\n".join(content_parts).strip(),
            source_location="text:line",
        )

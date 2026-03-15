"""Parser registry for selecting format-specific document parsers.

Provides a factory that maps content types to parser instances, with
default registration of all built-in parsers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabletop_oracle.services.ingestion.parsers.docx_parser import DocxParser
from tabletop_oracle.services.ingestion.parsers.html_parser import HtmlParser
from tabletop_oracle.services.ingestion.parsers.markdown_parser import MarkdownParser
from tabletop_oracle.services.ingestion.parsers.pdf_parser import PdfParser
from tabletop_oracle.services.ingestion.parsers.text_parser import TextParser

if TYPE_CHECKING:
    from tabletop_oracle.services.ingestion.parsers.base import DocumentParser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry mapping content types to document parsers.

    Maintains an ordered list of parsers. On lookup, returns the first
    parser whose ``supports()`` method returns True for the given content type.
    """

    def __init__(self) -> None:
        self._parsers: list[DocumentParser] = []

    def register(self, parser: DocumentParser) -> None:
        """Register a parser instance.

        Args:
            parser: A DocumentParser implementation.
        """
        self._parsers.append(parser)

    def get_parser(self, content_type: str) -> DocumentParser | None:
        """Find a parser that supports the given content type.

        Args:
            content_type: MIME type or format identifier.

        Returns:
            The first matching parser, or None if no parser supports the type.
        """
        for parser in self._parsers:
            if parser.supports(content_type):
                return parser
        return None

    @property
    def supported_types(self) -> list[str]:
        """List all content types that have a registered parser.

        Returns a representative type string for each registered parser.
        """
        types: list[str] = []
        for parser in self._parsers:
            cls_name = type(parser).__name__
            types.append(cls_name)
        return types


def create_default_registry() -> ParserRegistry:
    """Create a ParserRegistry with all built-in parsers registered.

    Returns:
        A fully populated ParserRegistry.
    """
    registry = ParserRegistry()
    registry.register(PdfParser())
    registry.register(MarkdownParser())
    registry.register(HtmlParser())
    registry.register(TextParser())
    registry.register(DocxParser())
    return registry

"""Format-specific document parsers.

Each parser implements the DocumentParser interface and produces a standardised
ParseResult. The ParserRegistry selects the appropriate parser by content type.
"""

from tabletop_oracle.services.ingestion.parsers.base import DocumentParser
from tabletop_oracle.services.ingestion.parsers.registry import ParserRegistry

__all__ = ["DocumentParser", "ParserRegistry"]

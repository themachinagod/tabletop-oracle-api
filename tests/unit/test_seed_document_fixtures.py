"""Unit tests for seed document fixtures (Catan PDFs).

Verifies that the seed document PDF fixtures exist, are well-formed,
contain expected content for KG construction and AI query validation,
and are consistent with their .meta.json sidecar files.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from tabletop_oracle.services.ingestion.parsers.pdf_parser import PdfParser
from tabletop_oracle.services.seed_fixture_loader import SeedFixtureLoader

# ---------------------------------------------------------------------------
# Shared paths and constants
# ---------------------------------------------------------------------------

_FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "tabletop_oracle" / "fixtures" / "seed"
)
_DOCS_DIR = _FIXTURES_DIR / "games" / "catan" / "documents"

_BASE_RULES_PDF = _DOCS_DIR / "catan-base-rules.pdf"
_SEAFARERS_PDF = _DOCS_DIR / "catan-seafarers-rules.pdf"


# ---------------------------------------------------------------------------
# Tests -- File existence and basic integrity
# ---------------------------------------------------------------------------


class TestDocumentFixtureFiles:
    """Verify that seed document PDFs exist and are well-formed."""

    def test_base_rules_pdf_exists(self) -> None:
        """The base rules PDF fixture file exists on disk."""
        assert _BASE_RULES_PDF.is_file(), f"Missing: {_BASE_RULES_PDF}"

    def test_seafarers_rules_pdf_exists(self) -> None:
        """The seafarers rules PDF fixture file exists on disk."""
        assert _SEAFARERS_PDF.is_file(), f"Missing: {_SEAFARERS_PDF}"

    def test_base_rules_pdf_not_empty(self) -> None:
        """The base rules PDF has meaningful content (not a zero-byte file)."""
        assert _BASE_RULES_PDF.stat().st_size > 1000

    def test_seafarers_rules_pdf_not_empty(self) -> None:
        """The seafarers rules PDF has meaningful content."""
        assert _SEAFARERS_PDF.stat().st_size > 1000

    def test_base_rules_pdf_is_valid_pdf(self) -> None:
        """The base rules file is a valid PDF that pdfplumber can open."""
        with pdfplumber.open(_BASE_RULES_PDF) as pdf:
            assert len(pdf.pages) >= 1

    def test_seafarers_rules_pdf_is_valid_pdf(self) -> None:
        """The seafarers rules file is a valid PDF that pdfplumber can open."""
        with pdfplumber.open(_SEAFARERS_PDF) as pdf:
            assert len(pdf.pages) >= 1


# ---------------------------------------------------------------------------
# Tests -- Meta.json consistency
# ---------------------------------------------------------------------------


class TestMetaJsonConsistency:
    """Verify that PDF files match their .meta.json sidecar declarations."""

    def test_base_rules_meta_references_existing_pdf(self) -> None:
        """The base rules meta.json 'file' field points to an existing PDF."""
        loader = SeedFixtureLoader(_FIXTURES_DIR)
        metas = loader.load_document_metas()
        base_meta = next(m for m in metas if m.type == "core_rules")
        pdf_path = _DOCS_DIR / base_meta.file
        assert pdf_path.is_file(), f"Meta references {base_meta.file} but file is missing"

    def test_seafarers_meta_references_existing_pdf(self) -> None:
        """The seafarers meta.json 'file' field points to an existing PDF."""
        loader = SeedFixtureLoader(_FIXTURES_DIR)
        metas = loader.load_document_metas()
        seafarers_meta = next(m for m in metas if m.type == "expansion_rules")
        pdf_path = _DOCS_DIR / seafarers_meta.file
        assert pdf_path.is_file(), f"Meta references {seafarers_meta.file} but file is missing"

    def test_base_rules_meta_format_matches(self) -> None:
        """The base rules meta declares format 'pdf' matching the file extension."""
        loader = SeedFixtureLoader(_FIXTURES_DIR)
        metas = loader.load_document_metas()
        base_meta = next(m for m in metas if m.type == "core_rules")
        assert base_meta.format == "pdf"
        assert base_meta.file.endswith(".pdf")

    def test_seafarers_meta_format_matches(self) -> None:
        """The seafarers meta declares format 'pdf' matching the file extension."""
        loader = SeedFixtureLoader(_FIXTURES_DIR)
        metas = loader.load_document_metas()
        seafarers_meta = next(m for m in metas if m.type == "expansion_rules")
        assert seafarers_meta.format == "pdf"
        assert seafarers_meta.file.endswith(".pdf")


# ---------------------------------------------------------------------------
# Tests -- Base rules content coverage
# ---------------------------------------------------------------------------


class TestBaseRulesContent:
    """Verify that base rules PDF contains content for KG construction.

    The acceptance criteria require coverage of: setup, gameplay, building,
    trading, development cards, robber, and victory conditions.
    """

    @pytest.fixture()
    def base_text(self) -> str:
        """Extract full text from the base rules PDF."""
        with pdfplumber.open(_BASE_RULES_PDF) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).lower()

    def test_contains_setup_content(self, base_text: str) -> None:
        """Base rules contain setup instructions."""
        assert "setup" in base_text
        assert "settlement" in base_text
        assert "terrain" in base_text

    def test_contains_gameplay_content(self, base_text: str) -> None:
        """Base rules contain gameplay/turn structure information."""
        assert "turn" in base_text
        assert "dice" in base_text or "roll" in base_text

    def test_contains_building_content(self, base_text: str) -> None:
        """Base rules contain building costs and rules."""
        assert "road" in base_text
        assert "settlement" in base_text
        assert "city" in base_text
        assert "brick" in base_text
        assert "lumber" in base_text

    def test_contains_trading_content(self, base_text: str) -> None:
        """Base rules contain trading mechanics."""
        assert "trade" in base_text or "trading" in base_text
        assert "harbour" in base_text or "harbor" in base_text or "maritime" in base_text

    def test_contains_development_cards_content(self, base_text: str) -> None:
        """Base rules contain development card descriptions."""
        assert "development card" in base_text
        assert "knight" in base_text
        assert "monopoly" in base_text
        assert "year of plenty" in base_text

    def test_contains_robber_content(self, base_text: str) -> None:
        """Base rules contain robber mechanics."""
        assert "robber" in base_text
        assert "7" in base_text or "seven" in base_text

    def test_contains_victory_conditions(self, base_text: str) -> None:
        """Base rules contain victory point conditions."""
        assert "victory point" in base_text
        assert "10" in base_text
        assert "longest road" in base_text
        assert "largest army" in base_text

    def test_contains_resource_types(self, base_text: str) -> None:
        """Base rules mention all five resource types."""
        assert "lumber" in base_text
        assert "wool" in base_text
        assert "grain" in base_text
        assert "brick" in base_text
        assert "ore" in base_text


# ---------------------------------------------------------------------------
# Tests -- Seafarers rules content coverage
# ---------------------------------------------------------------------------


class TestSeafarersRulesContent:
    """Verify that Seafarers rules PDF contains expansion-specific content.

    The acceptance criteria require coverage of: ships, gold rivers, pirate,
    and island scenarios.
    """

    @pytest.fixture()
    def seafarers_text(self) -> str:
        """Extract full text from the seafarers rules PDF."""
        with pdfplumber.open(_SEAFARERS_PDF) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).lower()

    def test_contains_ships_content(self, seafarers_text: str) -> None:
        """Seafarers rules contain ship building and movement rules."""
        assert "ship" in seafarers_text
        assert "wool" in seafarers_text
        assert "lumber" in seafarers_text
        assert "shipping route" in seafarers_text

    def test_contains_gold_river_content(self, seafarers_text: str) -> None:
        """Seafarers rules contain gold river hex mechanics."""
        assert "gold river" in seafarers_text

    def test_contains_pirate_content(self, seafarers_text: str) -> None:
        """Seafarers rules contain pirate mechanics."""
        assert "pirate" in seafarers_text

    def test_contains_island_scenarios(self, seafarers_text: str) -> None:
        """Seafarers rules describe island scenarios."""
        assert "scenario" in seafarers_text
        assert "island" in seafarers_text

    def test_contains_specific_scenarios(self, seafarers_text: str) -> None:
        """Seafarers rules mention specific scenario names."""
        assert "heading to new shores" in seafarers_text or "new shores" in seafarers_text
        assert "four islands" in seafarers_text or "fog island" in seafarers_text

    def test_contains_building_costs(self, seafarers_text: str) -> None:
        """Seafarers rules include updated building costs with ships."""
        assert "ship" in seafarers_text
        assert "1 wool" in seafarers_text or "wool" in seafarers_text


# ---------------------------------------------------------------------------
# Tests -- PDF parser integration
# ---------------------------------------------------------------------------


class TestPdfParserIntegration:
    """Verify that the ingestion pipeline's PdfParser handles the fixtures."""

    def test_base_rules_parseable_by_pdf_parser(self) -> None:
        """The base rules PDF is parseable by the ingestion PdfParser."""
        parser = PdfParser()
        result = parser.parse(_BASE_RULES_PDF, {})
        assert result.raw_text
        assert len(result.sections) >= 1
        assert result.metadata["format"] == "pdf"
        assert result.metadata["word_count"] > 100

    def test_seafarers_rules_parseable_by_pdf_parser(self) -> None:
        """The seafarers rules PDF is parseable by the ingestion PdfParser."""
        parser = PdfParser()
        result = parser.parse(_SEAFARERS_PDF, {})
        assert result.raw_text
        assert len(result.sections) >= 1
        assert result.metadata["format"] == "pdf"
        assert result.metadata["word_count"] > 100

    def test_base_rules_word_count_sufficient(self) -> None:
        """Base rules contain enough text for meaningful KG construction."""
        parser = PdfParser()
        result = parser.parse(_BASE_RULES_PDF, {})
        # Should have substantial content -- at least 500 words
        assert result.metadata["word_count"] >= 500

    def test_seafarers_rules_word_count_sufficient(self) -> None:
        """Seafarers rules contain enough text for meaningful KG construction."""
        parser = PdfParser()
        result = parser.parse(_SEAFARERS_PDF, {})
        # Should have substantial content -- at least 400 words
        assert result.metadata["word_count"] >= 400

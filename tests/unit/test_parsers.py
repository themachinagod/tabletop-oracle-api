"""Unit tests for format-specific document parsers.

Each parser is tested with programmatically generated content — no external
test files. Tests verify section extraction, table parsing, image references,
and metadata production.
"""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, tostring

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from tabletop_oracle.services.ingestion.models import ParseResult
from tabletop_oracle.services.ingestion.parsers.html_parser import HtmlParser
from tabletop_oracle.services.ingestion.parsers.markdown_parser import MarkdownParser
from tabletop_oracle.services.ingestion.parsers.registry import (
    ParserRegistry,
    create_default_registry,
)
from tabletop_oracle.services.ingestion.parsers.text_parser import TextParser

# ---------------------------------------------------------------------------
# Markdown Parser Tests
# ---------------------------------------------------------------------------


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    def setup_method(self) -> None:
        self.parser = MarkdownParser()

    def test_supports_markdown_types(self) -> None:
        """supports_Markdown_ReturnsTrue for all markdown content types."""
        assert self.parser.supports("markdown")
        assert self.parser.supports("text/markdown")
        assert self.parser.supports("text/x-markdown")

    def test_supports_rejects_non_markdown(self) -> None:
        """supports_NonMarkdown_ReturnsFalse."""
        assert not self.parser.supports("pdf")
        assert not self.parser.supports("text/plain")

    def test_parse_simple_headings(self, tmp_path: Path) -> None:
        """parse_HeadingsAndContent_ExtractsSectionsWithLevels."""
        md_content = "# Title\n\nSome intro text.\n\n## Section One\n\nContent here.\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        result = self.parser.parse(md_file, {})

        assert isinstance(result, ParseResult)
        assert result.raw_text == md_content
        assert len(result.sections) >= 2
        assert result.sections[0].title == "Title"
        assert result.sections[0].level == 1
        assert result.sections[1].title == "Section One"
        assert result.sections[1].level == 2

    def test_parse_code_blocks(self, tmp_path: Path) -> None:
        """parse_CodeBlocks_IncludedInSectionContent."""
        md_content = "# Code Example\n\n```python\nprint('hello')\n```\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        result = self.parser.parse(md_file, {})

        assert len(result.sections) >= 1
        assert "print('hello')" in result.sections[0].content

    def test_parse_table(self, tmp_path: Path) -> None:
        """parse_MarkdownTable_ExtractsHeadersAndRows."""
        md_content = (
            "# Data\n\n| Name | Value |\n|------|-------|\n| Alpha | 10 |\n| Beta | 20 |\n"
        )
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        result = self.parser.parse(md_file, {})

        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.headers == ["Name", "Value"]
        assert len(table.rows) == 2
        assert table.rows[0]["Name"] == "Alpha"
        assert table.rows[1]["Value"] == "20"

    def test_parse_no_headings(self, tmp_path: Path) -> None:
        """parse_NoHeadings_CreatesSingleRootSection."""
        md_content = "Just some plain text without headings.\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        result = self.parser.parse(md_file, {})

        assert len(result.sections) >= 1

    def test_parse_metadata(self, tmp_path: Path) -> None:
        """parse_AnyContent_IncludesFormatAndWordCount."""
        md_content = "# Hello\n\nWorld of testing.\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        result = self.parser.parse(md_file, {})

        assert result.metadata["format"] == "markdown"
        assert result.metadata["word_count"] > 0


# ---------------------------------------------------------------------------
# HTML Parser Tests
# ---------------------------------------------------------------------------


class TestHtmlParser:
    """Tests for HtmlParser."""

    def setup_method(self) -> None:
        self.parser = HtmlParser()

    def test_supports_html_types(self) -> None:
        """supports_HTML_ReturnsTrue for all HTML content types."""
        assert self.parser.supports("html")
        assert self.parser.supports("text/html")
        assert self.parser.supports("application/xhtml+xml")

    def test_supports_rejects_non_html(self) -> None:
        """supports_NonHTML_ReturnsFalse."""
        assert not self.parser.supports("pdf")
        assert not self.parser.supports("text/plain")

    def test_parse_headings_and_content(self, tmp_path: Path) -> None:
        """parse_HTMLWithHeadings_ExtractsSections."""
        html = (
            "<html><body>"
            "<h1>Main Title</h1>"
            "<p>Intro paragraph.</p>"
            "<h2>Sub Section</h2>"
            "<p>More content here.</p>"
            "</body></html>"
        )
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        result = self.parser.parse(html_file, {})

        assert isinstance(result, ParseResult)
        assert len(result.sections) >= 2
        assert result.sections[0].title == "Main Title"
        assert result.sections[0].level == 1
        assert result.sections[1].title == "Sub Section"
        assert result.sections[1].level == 2

    def test_parse_strips_chrome(self, tmp_path: Path) -> None:
        """parse_HTMLWithNavAndScript_StripsNonContent."""
        html = (
            "<html><body>"
            "<nav><a href='/'>Home</a></nav>"
            "<script>alert('hi');</script>"
            "<h1>Real Content</h1>"
            "<p>Body text.</p>"
            "<footer>Copyright 2024</footer>"
            "</body></html>"
        )
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        result = self.parser.parse(html_file, {})

        assert "Home" not in result.raw_text
        assert "alert" not in result.raw_text
        assert "Copyright" not in result.raw_text
        assert "Real Content" in result.raw_text
        assert "Body text." in result.raw_text

    def test_parse_table(self, tmp_path: Path) -> None:
        """parse_HTMLTable_ExtractsHeadersAndRows."""
        html = (
            "<html><body>"
            "<table>"
            "<thead><tr><th>Name</th><th>HP</th></tr></thead>"
            "<tbody>"
            "<tr><td>Warrior</td><td>100</td></tr>"
            "<tr><td>Mage</td><td>60</td></tr>"
            "</tbody>"
            "</table>"
            "</body></html>"
        )
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        result = self.parser.parse(html_file, {})

        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.headers == ["Name", "HP"]
        assert len(table.rows) == 2
        assert table.rows[0]["Name"] == "Warrior"
        assert table.rows[1]["HP"] == "60"

    def test_parse_images(self, tmp_path: Path) -> None:
        """parse_HTMLWithImages_ExtractsImageReferences."""
        html = (
            "<html><body>"
            '<img src="hero.png" alt="Hero portrait" />'
            '<img src="map.jpg" />'
            "</body></html>"
        )
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        result = self.parser.parse(html_file, {})

        assert len(result.images) == 2
        assert result.images[0].alt_text == "Hero portrait"
        assert result.images[0].format == "png"
        assert result.images[1].format == "jpeg"

    def test_parse_no_headings(self, tmp_path: Path) -> None:
        """parse_HTMLNoHeadings_CreatesSingleSection."""
        html = "<html><body><p>Just text.</p></body></html>"
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        result = self.parser.parse(html_file, {})

        assert len(result.sections) == 1
        assert "Just text." in result.sections[0].content

    def test_parse_table_with_caption(self, tmp_path: Path) -> None:
        """parse_HTMLTableWithCaption_ExtractsCaption."""
        html = (
            "<html><body>"
            "<table>"
            "<caption>Stats Table</caption>"
            "<thead><tr><th>Stat</th><th>Value</th></tr></thead>"
            "<tbody><tr><td>STR</td><td>18</td></tr></tbody>"
            "</table>"
            "</body></html>"
        )
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        result = self.parser.parse(html_file, {})

        assert len(result.tables) == 1
        assert result.tables[0].caption == "Stats Table"


# ---------------------------------------------------------------------------
# Plain Text Parser Tests
# ---------------------------------------------------------------------------


class TestTextParser:
    """Tests for TextParser."""

    def setup_method(self) -> None:
        self.parser = TextParser()

    def test_supports_text_types(self) -> None:
        """supports_Text_ReturnsTrue for text content types."""
        assert self.parser.supports("text")
        assert self.parser.supports("text/plain")

    def test_supports_rejects_non_text(self) -> None:
        """supports_NonText_ReturnsFalse."""
        assert not self.parser.supports("pdf")
        assert not self.parser.supports("html")

    def test_parse_allcaps_headings(self, tmp_path: Path) -> None:
        """parse_AllCapsLines_DetectsAsSectionHeadings."""
        content = (
            "COMBAT RULES\n\nWhen engaging in combat, roll dice.\n\n"
            "MOVEMENT\n\nMove up to speed.\n"
        )
        text_file = tmp_path / "rules.txt"
        text_file.write_text(content)

        result = self.parser.parse(text_file, {})

        assert len(result.sections) >= 2
        assert result.sections[0].title == "COMBAT RULES"
        assert "roll dice" in result.sections[0].content
        assert result.sections[1].title == "MOVEMENT"

    def test_parse_numbered_headings(self, tmp_path: Path) -> None:
        """parse_NumberedLines_DetectsAsSectionHeadings."""
        content = "1. Introduction\n\nWelcome.\n\n2. Setup\n\nPlace the board.\n"
        text_file = tmp_path / "rules.txt"
        text_file.write_text(content)

        result = self.parser.parse(text_file, {})

        assert len(result.sections) >= 2
        assert result.sections[0].title == "1. Introduction"
        assert result.sections[1].title == "2. Setup"

    def test_parse_underline_heading(self, tmp_path: Path) -> None:
        """parse_UnderlinedHeadings_DetectsAsLevel1And2."""
        content = "Main Title\n==========\n\nContent.\n\nSub Title\n---------\n\nMore.\n"
        text_file = tmp_path / "rules.txt"
        text_file.write_text(content)

        result = self.parser.parse(text_file, {})

        assert len(result.sections) >= 2
        assert result.sections[0].title == "Main Title"
        assert result.sections[0].level == 1
        assert result.sections[1].title == "Sub Title"
        assert result.sections[1].level == 2

    def test_parse_plain_body(self, tmp_path: Path) -> None:
        """parse_NoHeadings_CreatesSingleSection."""
        content = "Just a simple paragraph of text.\nAnother line.\n"
        text_file = tmp_path / "simple.txt"
        text_file.write_text(content)

        result = self.parser.parse(text_file, {})

        assert len(result.sections) >= 1
        assert "simple paragraph" in result.sections[0].content

    def test_parse_metadata(self, tmp_path: Path) -> None:
        """parse_AnyContent_IncludesFormatLineAndWordCount."""
        content = "Hello world\nSecond line\n"
        text_file = tmp_path / "test.txt"
        text_file.write_text(content)

        result = self.parser.parse(text_file, {})

        assert result.metadata["format"] == "text"
        assert result.metadata["word_count"] > 0
        assert result.metadata["line_count"] == 2

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """parse_EmptyFile_CreatesSingleEmptySection."""
        text_file = tmp_path / "empty.txt"
        text_file.write_text("")

        result = self.parser.parse(text_file, {})

        assert len(result.sections) >= 1
        assert result.raw_text == ""


# ---------------------------------------------------------------------------
# DOCX Parser Tests
# ---------------------------------------------------------------------------


def _build_minimal_docx(paragraphs: list[tuple[str, str | None]]) -> bytes:
    """Build a minimal DOCX file in memory.

    Creates a valid DOCX (ZIP with required XML) containing paragraphs with
    optional heading styles.

    Args:
        paragraphs: List of (text, style_id) tuples. style_id is None for
            normal paragraphs, or "Heading1"/"Heading2"/etc for headings.

    Returns:
        Raw bytes of a valid DOCX file.
    """
    # DOCX namespace
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    pkg_r_ns = "http://schemas.openxmlformats.org/package/2006/relationships"

    # Build document.xml
    doc_el = Element(f"{{{w_ns}}}document")
    body = SubElement(doc_el, f"{{{w_ns}}}body")
    for text, style_id in paragraphs:
        p = SubElement(body, f"{{{w_ns}}}p")
        if style_id:
            ppr = SubElement(p, f"{{{w_ns}}}pPr")
            pstyle = SubElement(ppr, f"{{{w_ns}}}pStyle")
            pstyle.set(f"{{{w_ns}}}val", style_id)
        r = SubElement(p, f"{{{w_ns}}}r")
        t = SubElement(r, f"{{{w_ns}}}t")
        t.text = text

    doc_xml = tostring(doc_el, xml_declaration=True, encoding="UTF-8")

    # Build styles.xml with heading style definitions
    styles_el = Element(f"{{{w_ns}}}styles")
    for i in range(1, 5):
        style = SubElement(styles_el, f"{{{w_ns}}}style")
        style.set(f"{{{w_ns}}}type", "paragraph")
        style.set(f"{{{w_ns}}}styleId", f"Heading{i}")
        name = SubElement(style, f"{{{w_ns}}}name")
        name.set(f"{{{w_ns}}}val", f"heading {i}")
    styles_xml = tostring(styles_el, xml_declaration=True, encoding="UTF-8")

    # Content types
    rels_ct = "application/vnd.openxmlformats-package.relationships+xml"
    doc_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    styles_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"
    types_el = Element("Types", xmlns=ct_ns)
    SubElement(types_el, "Default", Extension="rels", ContentType=rels_ct)
    SubElement(types_el, "Default", Extension="xml", ContentType="application/xml")
    SubElement(types_el, "Override", PartName="/word/document.xml", ContentType=doc_ct)
    SubElement(types_el, "Override", PartName="/word/styles.xml", ContentType=styles_ct)
    ct_xml = tostring(types_el, xml_declaration=True, encoding="UTF-8")

    # Relationships
    rels_el = Element("Relationships", xmlns=pkg_r_ns)
    SubElement(
        rels_el,
        "Relationship",
        Id="rId1",
        Type=f"{r_ns}/officeDocument",
        Target="word/document.xml",
    )
    rels_xml = tostring(rels_el, xml_declaration=True, encoding="UTF-8")

    word_rels_el = Element("Relationships", xmlns=pkg_r_ns)
    SubElement(word_rels_el, "Relationship", Id="rId1", Type=f"{r_ns}/styles", Target="styles.xml")
    word_rels_xml = tostring(word_rels_el, xml_declaration=True, encoding="UTF-8")

    # Assemble ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", word_rels_xml)

    return buf.getvalue()


class TestDocxParser:
    """Tests for DocxParser."""

    def setup_method(self) -> None:
        from tabletop_oracle.services.ingestion.parsers.docx_parser import DocxParser

        self.parser = DocxParser()

    def test_supports_docx_types(self) -> None:
        """supports_DOCX_ReturnsTrue for DOCX content types."""
        assert self.parser.supports("docx")
        assert self.parser.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_supports_rejects_non_docx(self) -> None:
        """supports_NonDOCX_ReturnsFalse."""
        assert not self.parser.supports("pdf")
        assert not self.parser.supports("text/plain")

    def test_parse_headings(self, tmp_path: Path) -> None:
        """parse_DocxWithHeadings_ExtractsSectionsWithLevels."""
        docx_bytes = _build_minimal_docx(
            [
                ("Chapter One", "Heading1"),
                ("Content of chapter one.", None),
                ("Section 1.1", "Heading2"),
                ("More detail.", None),
            ]
        )
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(docx_bytes)

        result = self.parser.parse(docx_file, {})

        assert isinstance(result, ParseResult)
        assert len(result.sections) >= 2
        assert result.sections[0].title == "Chapter One"
        assert result.sections[0].level == 1
        assert "Content of chapter one." in result.sections[0].content
        assert result.sections[1].title == "Section 1.1"
        assert result.sections[1].level == 2

    def test_parse_no_headings(self, tmp_path: Path) -> None:
        """parse_DocxNoHeadings_CreatesSingleSection."""
        docx_bytes = _build_minimal_docx(
            [
                ("Just a paragraph.", None),
                ("Another paragraph.", None),
            ]
        )
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(docx_bytes)

        result = self.parser.parse(docx_file, {})

        assert len(result.sections) >= 1
        assert "Just a paragraph" in result.raw_text

    def test_parse_metadata(self, tmp_path: Path) -> None:
        """parse_AnyDocx_IncludesFormatAndWordCount."""
        docx_bytes = _build_minimal_docx(
            [
                ("Test Title", "Heading1"),
                ("Test body content.", None),
            ]
        )
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(docx_bytes)

        result = self.parser.parse(docx_file, {})

        assert result.metadata["format"] == "docx"
        assert result.metadata["word_count"] > 0
        assert result.metadata["paragraph_count"] > 0


# ---------------------------------------------------------------------------
# PDF Parser Tests
# ---------------------------------------------------------------------------


class TestPdfParser:
    """Tests for PdfParser."""

    def setup_method(self) -> None:
        from tabletop_oracle.services.ingestion.parsers.pdf_parser import PdfParser

        self.parser = PdfParser()

    def test_supports_pdf_types(self) -> None:
        """supports_PDF_ReturnsTrue for PDF content types."""
        assert self.parser.supports("pdf")
        assert self.parser.supports("application/pdf")

    def test_supports_rejects_non_pdf(self) -> None:
        """supports_NonPDF_ReturnsFalse."""
        assert not self.parser.supports("html")
        assert not self.parser.supports("docx")

    def test_parse_simple_pdf(self, tmp_path: Path) -> None:
        """parse_ValidPDF_ExtractsTextAndSections."""
        # Create a minimal valid PDF using pypdfium2 (available via pdfplumber dep)
        pdf_bytes = _build_minimal_pdf("Hello World\nThis is a test document.")
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(pdf_bytes)

        result = self.parser.parse(pdf_file, {})

        assert isinstance(result, ParseResult)
        assert result.metadata["format"] == "pdf"
        assert result.metadata["page_count"] >= 1
        assert len(result.sections) >= 1
        assert result.sections[0].page_number == 1

    def test_parse_empty_pdf(self, tmp_path: Path) -> None:
        """parse_EmptyPDF_ProducesEmptyResult."""
        pdf_bytes = _build_minimal_pdf("")
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(pdf_bytes)

        result = self.parser.parse(pdf_file, {})

        assert result.metadata["page_count"] >= 1


def _build_minimal_pdf(text: str) -> bytes:
    """Build a minimal valid single-page PDF with the given text.

    Uses raw PDF syntax to avoid heavy test dependencies.
    """
    # Escape special PDF characters in text
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # Split text into lines for positioning
    lines = escaped.split("\\n") if "\\n" in escaped else [escaped]
    text_ops: list[str] = []
    y_pos = 750
    for line in lines:
        text_ops.append(f"BT /F1 12 Tf {72} {y_pos} Td ({line}) Tj ET")
        y_pos -= 20

    stream_content = "\n".join(text_ops)
    stream_bytes = stream_content.encode("latin-1")

    objects: list[str] = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")

    # Object 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")

    # Object 3: Page
    objects.append(
        "3 0 obj\n"
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        "endobj"
    )

    # Object 4: Content stream
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n{stream_content}\nendstream\nendobj"
    )

    # Object 5: Font
    objects.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj")

    # Build PDF
    pdf_parts: list[str] = ["%PDF-1.4\n"]
    offsets: list[int] = []

    for obj in objects:
        offsets.append(len("".join(pdf_parts).encode("latin-1")))
        pdf_parts.append(obj + "\n")

    xref_offset = len("".join(pdf_parts).encode("latin-1"))
    pdf_parts.append("xref\n")
    pdf_parts.append(f"0 {len(objects) + 1}\n")
    pdf_parts.append("0000000000 65535 f \n")
    for offset in offsets:
        pdf_parts.append(f"{offset:010d} 00000 n \n")

    pdf_parts.append("trailer\n")
    pdf_parts.append(f"<< /Root 1 0 R /Size {len(objects) + 1} >>\n")
    pdf_parts.append("startxref\n")
    pdf_parts.append(f"{xref_offset}\n")
    pdf_parts.append("%%EOF\n")

    return "".join(pdf_parts).encode("latin-1")


# ---------------------------------------------------------------------------
# Parser Registry Tests
# ---------------------------------------------------------------------------


class TestParserRegistry:
    """Tests for ParserRegistry."""

    def test_default_registry_has_all_parsers(self) -> None:
        """create_default_registry_AllFormats_HasFiveParsers."""
        registry = create_default_registry()

        assert registry.get_parser("pdf") is not None
        assert registry.get_parser("markdown") is not None
        assert registry.get_parser("html") is not None
        assert registry.get_parser("text") is not None
        assert registry.get_parser("docx") is not None

    def test_get_parser_returns_none_for_unknown(self) -> None:
        """get_parser_UnknownType_ReturnsNone."""
        registry = create_default_registry()

        assert registry.get_parser("xml") is None
        assert registry.get_parser("csv") is None

    def test_get_parser_returns_correct_type(self) -> None:
        """get_parser_KnownTypes_ReturnsCorrectParserClass."""
        registry = create_default_registry()

        assert type(registry.get_parser("markdown")).__name__ == "MarkdownParser"
        assert type(registry.get_parser("html")).__name__ == "HtmlParser"
        assert type(registry.get_parser("text")).__name__ == "TextParser"

    def test_register_custom_parser(self) -> None:
        """register_CustomParser_FoundByGetParser."""
        registry = ParserRegistry()

        parser = MarkdownParser()
        registry.register(parser)

        assert registry.get_parser("markdown") is parser

    def test_get_parser_mime_types(self) -> None:
        """get_parser_MIMETypes_ResolvesCorrectly."""
        registry = create_default_registry()

        assert registry.get_parser("application/pdf") is not None
        assert registry.get_parser("text/html") is not None
        assert registry.get_parser("text/markdown") is not None
        assert registry.get_parser("text/plain") is not None

    def test_supported_types_lists_registered(self) -> None:
        """supported_types_DefaultRegistry_ListsAllParsers."""
        registry = create_default_registry()

        types = registry.supported_types
        assert len(types) == 5


# ---------------------------------------------------------------------------
# ExtractionStage Integration Tests
# ---------------------------------------------------------------------------


class TestExtractionStageIntegration:
    """Tests that ExtractionStage dispatches to parsers correctly."""

    def test_extraction_dispatches_to_text_parser(self, tmp_path: Path) -> None:
        """execute_TextFormat_DispatchesToTextParser."""
        from uuid import uuid4

        from tabletop_oracle.services.ingestion.models import PipelineContext
        from tabletop_oracle.services.ingestion.stages import ExtractionStage

        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello world content.")

        context = PipelineContext(
            document_id=uuid4(),
            version_id=uuid4(),
            file_path=str(text_file),
            document_format="text",
            file_size=20,
        )

        stage = ExtractionStage()
        stage.execute(context)

        assert context.parse_result is not None
        assert "Hello world content." in context.parse_result.raw_text

    def test_extraction_dispatches_to_markdown_parser(self, tmp_path: Path) -> None:
        """execute_MarkdownFormat_DispatchesToMarkdownParser."""
        from uuid import uuid4

        from tabletop_oracle.services.ingestion.models import PipelineContext
        from tabletop_oracle.services.ingestion.stages import ExtractionStage

        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nBody text.\n")

        context = PipelineContext(
            document_id=uuid4(),
            version_id=uuid4(),
            file_path=str(md_file),
            document_format="markdown",
            file_size=25,
        )

        stage = ExtractionStage()
        stage.execute(context)

        assert context.parse_result is not None
        assert context.parse_result.metadata["format"] == "markdown"

    def test_extraction_raises_for_unknown_format(self, tmp_path: Path) -> None:
        """execute_UnknownFormat_RaisesRuntimeError."""
        from uuid import uuid4

        from tabletop_oracle.services.ingestion.models import PipelineContext
        from tabletop_oracle.services.ingestion.stages import ExtractionStage

        context = PipelineContext(
            document_id=uuid4(),
            version_id=uuid4(),
            file_path=str(tmp_path / "test.xyz"),
            document_format="xyz",
            file_size=10,
        )

        stage = ExtractionStage()
        with pytest.raises(RuntimeError, match="No parser available"):
            stage.execute(context)

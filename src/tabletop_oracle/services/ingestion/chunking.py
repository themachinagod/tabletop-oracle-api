"""Chunking engine with pluggable strategy interface.

Breaks structured document content into traceable chunks suitable for
KG processing and LLM context windows. The default strategy respects
section boundaries, keeps tables atomic, and applies configurable size
limits with overlap.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import tiktoken

from tabletop_oracle.services.ingestion.models import Chunk, Section, Table

logger = logging.getLogger(__name__)

# Lazily cached tokenizer instance (module-level singleton).
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """Return the cl100k_base tokenizer, cached after first call."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in *text* using the cl100k_base encoding.

    Args:
        text: The string to tokenize.

    Returns:
        Number of tokens.
    """
    return len(_get_encoding().encode(text))


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Interface for document chunking strategies.

    Implementations receive flattened sections and tables from the
    structure detection stage and return a list of ``Chunk`` objects
    with full traceability metadata.
    """

    def chunk(self, sections: list[Section], tables: list[Table]) -> list[Chunk]:
        """Break structured content into chunks.

        Args:
            sections: Hierarchical sections from structure detection.
            tables: Extracted tables.

        Returns:
            Ordered list of chunks with metadata.
        """
        ...


class SectionAwareChunkingStrategy:
    """Default V1 chunking strategy: section-aware with size limits.

    Chunks are created at natural section boundaries. Sections smaller
    than the minimum size are merged with their siblings. Sections larger
    than the maximum size are split at paragraph boundaries. Tables are
    always kept as single chunks (never split across chunks).

    Args:
        target_chunk_size: Target chunk size in tokens (default 512).
        max_chunk_size: Maximum chunk size in tokens (default 1024).
        min_chunk_size: Minimum chunk size in tokens (default 64).
        overlap_tokens: Token overlap between adjacent text chunks (default 50).
    """

    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 64,
        overlap_tokens: int = 50,
    ) -> None:
        self._target = target_chunk_size
        self._max = max_chunk_size
        self._min = min_chunk_size
        self._overlap = overlap_tokens

    def chunk(self, sections: list[Section], tables: list[Table]) -> list[Chunk]:
        """Chunk sections and tables into traceable content pieces.

        Algorithm:
        1. Flatten the section hierarchy into leaf text blocks.
        2. Walk blocks in order, merging undersized neighbours.
        3. Split oversized blocks at paragraph boundaries.
        4. Add overlap between sequential text chunks.
        5. Append tables as atomic chunks.

        Args:
            sections: Hierarchical sections from structure detection.
            tables: Extracted tables.

        Returns:
            Ordered list of ``Chunk`` objects.
        """
        flat_blocks = self._flatten_sections(sections)
        sized_blocks = self._apply_size_limits(flat_blocks)
        overlapped = self._add_overlap(sized_blocks)
        chunks = self._to_chunks(overlapped)

        # Append table chunks after text chunks
        for table in tables:
            chunks.append(self._table_to_chunk(table, len(chunks)))

        # Assign final sequential indices
        for idx, ch in enumerate(chunks):
            ch.chunk_index = idx

        logger.info(
            "Chunked document: text_blocks=%d, tables=%d, total_chunks=%d",
            len(overlapped),
            len(tables),
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_sections(self, sections: list[Section], path: str = "") -> list[_TextBlock]:
        """Recursively flatten a section tree into leaf text blocks.

        Args:
            sections: Sections at the current level.
            path: Accumulated section path from ancestors.

        Returns:
            Ordered list of ``_TextBlock`` with section metadata.
        """
        blocks: list[_TextBlock] = []
        for section in sections:
            section_path = f"{path} > {section.title}" if path else section.title
            if section.content.strip():
                blocks.append(
                    _TextBlock(
                        text=section.content.strip(),
                        section_path=section_path,
                        heading=section.title,
                        page_number=section.page_number,
                    )
                )
            if section.children:
                blocks.extend(self._flatten_sections(section.children, section_path))
        return blocks

    def _apply_size_limits(self, blocks: list[_TextBlock]) -> list[_TextBlock]:
        """Merge undersized blocks and split oversized ones.

        Args:
            blocks: Flattened text blocks.

        Returns:
            Size-adjusted blocks.
        """
        if not blocks:
            return []

        # First pass: split oversized blocks at paragraph boundaries
        split_blocks: list[_TextBlock] = []
        for block in blocks:
            tokens = count_tokens(block.text)
            if tokens > self._max:
                split_blocks.extend(self._split_block(block))
            else:
                split_blocks.append(block)

        # Second pass: merge undersized neighbours
        return self._merge_undersized(split_blocks)

    def _split_block(self, block: _TextBlock) -> list[_TextBlock]:
        """Split an oversized block at paragraph boundaries.

        Paragraphs are delimited by double newlines. If a single paragraph
        exceeds max size, it is kept as-is (never split mid-paragraph).

        Args:
            block: A text block exceeding ``max_chunk_size``.

        Returns:
            List of smaller blocks.
        """
        paragraphs = block.text.split("\n\n")
        result: list[_TextBlock] = []
        current_text = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = f"{current_text}\n\n{para}".strip() if current_text else para
            if count_tokens(candidate) <= self._target and current_text:
                current_text = candidate
            elif current_text:
                # Flush the accumulated text
                result.append(block.with_text(current_text))
                current_text = para
            else:
                current_text = para

        if current_text:
            result.append(block.with_text(current_text))

        return result if result else [block]

    def _merge_undersized(self, blocks: list[_TextBlock]) -> list[_TextBlock]:
        """Merge consecutive undersized blocks that share a parent section.

        Args:
            blocks: Blocks after splitting.

        Returns:
            Blocks with undersized neighbours merged.
        """
        if not blocks:
            return []

        result: list[_TextBlock] = [blocks[0]]
        for block in blocks[1:]:
            prev = result[-1]
            prev_tokens = count_tokens(prev.text)
            block_tokens = count_tokens(block.text)

            # Merge if both are under min OR combined stays under target
            can_merge = (prev_tokens < self._min or block_tokens < self._min) and (
                prev_tokens + block_tokens
            ) <= self._target

            if can_merge:
                merged_text = f"{prev.text}\n\n{block.text}"
                result[-1] = prev.with_text(merged_text)
            else:
                result.append(block)

        return result

    def _add_overlap(self, blocks: list[_TextBlock]) -> list[_TextBlock]:
        """Prepend overlap text from the previous block to each block.

        The overlap ensures concepts spanning a chunk boundary appear in
        at least one complete chunk. Only text chunks receive overlap;
        the first chunk has no predecessor so gets none.

        Args:
            blocks: Size-adjusted text blocks.

        Returns:
            Blocks with overlap text prepended where applicable.
        """
        if len(blocks) <= 1 or self._overlap <= 0:
            return blocks

        enc = _get_encoding()
        result: list[_TextBlock] = [blocks[0]]

        for block in blocks[1:]:
            prev_tokens = enc.encode(result[-1].text)
            overlap_tokens = (
                prev_tokens[-self._overlap :] if len(prev_tokens) > self._overlap else prev_tokens
            )
            overlap_text = enc.decode(overlap_tokens).strip()

            if overlap_text:
                new_text = f"{overlap_text}\n\n{block.text}"
                result.append(block.with_text(new_text))
            else:
                result.append(block)

        return result

    def _to_chunks(self, blocks: list[_TextBlock]) -> list[Chunk]:
        """Convert text blocks to Chunk dataclass instances.

        Args:
            blocks: Final text blocks.

        Returns:
            List of Chunk objects (chunk_index placeholder; caller re-indexes).
        """
        return [
            Chunk(
                chunk_index=0,
                chunk_type="text",
                content=block.text,
                section_path=block.section_path,
                heading=block.heading,
                page_number=block.page_number,
                token_estimate=count_tokens(block.text),
                metadata={},
            )
            for block in blocks
        ]

    def _table_to_chunk(self, table: Table, index: int) -> Chunk:
        """Convert an extracted table to a single atomic chunk.

        The table is serialised as a pipe-delimited markdown table for
        LLM readability.

        Args:
            table: Extracted table data.
            index: Provisional chunk index.

        Returns:
            A single Chunk of type ``table``.
        """
        lines: list[str] = []
        if table.caption:
            lines.append(table.caption)
            lines.append("")
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
        for row in table.rows:
            cells = (
                [row.get(h, "") for h in table.headers] if table.headers else list(row.values())
            )
            lines.append("| " + " | ".join(cells) + " |")

        content = "\n".join(lines)
        return Chunk(
            chunk_index=index,
            chunk_type="table",
            content=content,
            section_path=table.section_path or None,
            heading=table.caption,
            page_number=None,
            token_estimate=count_tokens(content),
            metadata={"table_headers": table.headers},
        )


class _TextBlock:
    """Internal intermediate representation for a text segment.

    Carries section metadata alongside the text content. Immutable-style:
    ``with_text`` returns a copy with different content.

    Attributes:
        text: The text content.
        section_path: Hierarchical section context.
        heading: Nearest heading.
        page_number: Source page (PDF only).
    """

    __slots__ = ("heading", "page_number", "section_path", "text")

    def __init__(
        self,
        text: str,
        section_path: str,
        heading: str,
        page_number: int | None,
    ) -> None:
        self.text = text
        self.section_path = section_path
        self.heading = heading
        self.page_number = page_number

    def with_text(self, new_text: str) -> _TextBlock:
        """Return a copy with replaced text, preserving metadata.

        Args:
            new_text: Replacement text content.

        Returns:
            New ``_TextBlock`` with the same metadata.
        """
        return _TextBlock(
            text=new_text,
            section_path=self.section_path,
            heading=self.heading,
            page_number=self.page_number,
        )

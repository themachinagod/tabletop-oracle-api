"""Stage 6: Citation assembly from KG retrieval sources.

Maps source markers in the answer text to structured citation records,
deduplicates by document + section, orders by document type priority,
and yields stream.citation SSE events. Falls back to the highest-ranked
retrieval result when no source markers are found in the answer.

Design reference: EPIC-004 design.md, Stage 6: Citation Assembly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from tabletop_oracle.services.ai.context import CitationData
from tabletop_oracle.streaming.payloads import (
    StreamCitationPayload,
    payload_to_sse_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tabletop_oracle.services.ai.context import PipelineContext, RetrievalResult
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

# Regex for numbered source markers in the answer text: [1], [2], etc.
_SOURCE_MARKER_PATTERN = re.compile(r"\[(\d+)\]")

# Document type priority for ordering (lower index = higher priority).
_DOCUMENT_TYPE_PRIORITY: dict[str, int] = {
    "errata": 0,
    "faq": 1,
    "core_rules": 2,
    "supplement": 3,
    "strategy": 4,
}

_DEFAULT_PRIORITY = 99


class CitationAssemblyStage:
    """Maps KG retrieval sources to citation records.

    Builds CitationData records from the source references in the
    retrieval results that were actually used in the answer.
    Deduplicates by document + section. Orders by document type
    priority (Errata > FAQ > Core Rules > Other).

    Populates on ``PipelineContext``:
    - ``citations``: Ordered list of CitationData records.

    Yields:
    - ``stream.citation`` SSE events for each citation.
    """

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Assemble citations from retrieval sources referenced in the answer.

        1. Parse source markers from answer text (e.g., [1], [2]).
        2. Map each marker to its corresponding retrieval result.
        3. Build CitationData records with full traceability.
        4. Deduplicate by (document_name, section).
        5. Order by document type priority.
        6. Yield stream.citation events for each citation.
        7. Fallback: if no markers found, create citation from top result.

        Populates ``ctx.citations``.

        Args:
            ctx: Pipeline context with retrieval_results and answer_text.
            model_client: Model client (unused -- no LLM call in this stage).

        Yields:
            SSE stream.citation events for each assembled citation.
        """
        referenced_indices = _parse_source_markers(ctx.answer_text)

        if referenced_indices and ctx.retrieval_results:
            raw_citations = _build_citations_from_markers(
                referenced_indices, ctx.retrieval_results
            )
        elif ctx.retrieval_results:
            logger.warning(
                "No source markers in answer, using fallback citation",
                extra={"message_id": str(ctx.user_message.id)},
            )
            raw_citations = _build_fallback_citation(ctx.retrieval_results)
        else:
            logger.warning(
                "No retrieval results available for citation assembly",
                extra={"message_id": str(ctx.user_message.id)},
            )
            raw_citations = []

        deduplicated = _deduplicate_citations(raw_citations)
        ordered = _order_by_priority(deduplicated)

        # Re-index citations sequentially after dedup + ordering
        final_citations = _reindex_citations(ordered)

        ctx.citations = final_citations

        logger.info(
            "Citation assembly complete",
            extra={
                "citation_count": len(final_citations),
                "markers_found": len(referenced_indices),
                "used_fallback": not referenced_indices and bool(ctx.retrieval_results),
            },
        )

        for citation in final_citations:
            yield payload_to_sse_event(
                StreamCitationPayload(
                    index=citation.index,
                    document_name=citation.document_name,
                    section=citation.section,
                    excerpt=citation.excerpt,
                ),
            )


def _parse_source_markers(answer_text: str) -> list[int]:
    """Parse numbered source markers from the answer text.

    Extracts unique 1-based indices from markers like [1], [2], [3].
    Returns them in the order they first appear.

    Args:
        answer_text: The generated answer text with source markers.

    Returns:
        List of unique 1-based marker indices in appearance order.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _SOURCE_MARKER_PATTERN.finditer(answer_text):
        index = int(match.group(1))
        if index > 0 and index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def _build_citations_from_markers(
    marker_indices: list[int],
    retrieval_results: list[RetrievalResult],
) -> list[CitationData]:
    """Build citation records from parsed source markers.

    Each marker index corresponds to a 1-based position in the
    retrieval_results list. Out-of-range markers are skipped.

    Args:
        marker_indices: 1-based source marker indices from the answer.
        retrieval_results: Ordered retrieval results from the KG stage.

    Returns:
        List of CitationData records for valid markers.
    """
    citations: list[CitationData] = []
    for marker_index in marker_indices:
        result_index = marker_index - 1  # Convert to 0-based
        if 0 <= result_index < len(retrieval_results):
            result = retrieval_results[result_index]
            citations.append(_citation_from_result(marker_index, result))
    return citations


def _build_fallback_citation(
    retrieval_results: list[RetrievalResult],
) -> list[CitationData]:
    """Build a fallback citation from the highest-ranked retrieval result.

    Used when the answer contains no source markers.

    Args:
        retrieval_results: Non-empty retrieval results list.

    Returns:
        Single-element list with a citation from the top result.
    """
    top = retrieval_results[0]
    return [_citation_from_result(1, top)]


def _citation_from_result(
    index: int,
    result: RetrievalResult,
) -> CitationData:
    """Build a CitationData from a retrieval result.

    Args:
        index: 1-based citation index.
        result: The retrieval result with source metadata.

    Returns:
        A CitationData record with full traceability.
    """
    source = result.source
    page_number_raw = source.get("page_number")
    page_number = int(page_number_raw) if page_number_raw is not None else None

    return CitationData(
        index=index,
        document_name=str(source.get("document_name", "Unknown")),
        document_type=str(source.get("document_type", "")),
        section=str(source.get("section_path", "")),
        page_number=page_number,
        excerpt=result.content[:200],
    )


def _deduplicate_citations(citations: list[CitationData]) -> list[CitationData]:
    """Remove duplicate citations by (document_name, section).

    Keeps the first occurrence (preserves the answer's reference order).

    Args:
        citations: Raw citation list potentially with duplicates.

    Returns:
        Deduplicated citation list.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[CitationData] = []
    for citation in citations:
        key = (citation.document_name, citation.section)
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique


def _order_by_priority(citations: list[CitationData]) -> list[CitationData]:
    """Order citations by document type priority.

    Errata > FAQ > Core Rules > Supplement > Strategy > Other.

    Args:
        citations: Deduplicated citation list.

    Returns:
        Citations ordered by document type priority.
    """
    return sorted(citations, key=_citation_sort_key)


def _citation_sort_key(citation: CitationData) -> int:
    """Sort key for citation ordering by document type priority.

    Args:
        citation: A citation record.

    Returns:
        Integer priority value (lower = higher priority).
    """
    return _DOCUMENT_TYPE_PRIORITY.get(citation.document_type, _DEFAULT_PRIORITY)


def _reindex_citations(citations: list[CitationData]) -> list[CitationData]:
    """Re-assign sequential 1-based indices after dedup and reordering.

    After deduplication and priority reordering, the original marker
    indices may have gaps or be out of sequence. This assigns clean
    1-based indices for the final citation list.

    Args:
        citations: Ordered, deduplicated citation list.

    Returns:
        Citations with sequential 1-based indices.
    """
    for i, citation in enumerate(citations, start=1):
        citation.index = i
    return citations

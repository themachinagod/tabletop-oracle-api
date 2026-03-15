"""Stage 4: Knowledge retrieval from the Knowledge Graph.

Queries the KG via KGRetrievalService for concepts relevant to the
player's question, traverses associations for top results, deduplicates,
and ranks by relevance and document type priority. No LLM call in this
stage -- pure KG interaction.

Design reference: EPIC-004 design.md, Stage 4: Knowledge Retrieval.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabletop_oracle.services.ai.context import RetrievalResult, TraversalResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.kg.retrieval import (
        KGRetrievalService,
        TraversalNode,
    )
    from tabletop_oracle.services.kg.retrieval import (
        RetrievalResult as KGRetrievalResult,
    )
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

# Number of top semantic search results to traverse associations for.
_TOP_RESULTS_FOR_TRAVERSAL = 5

# Maximum association traversal depth.
_TRAVERSAL_DEPTH = 2

# Semantic search result limit.
_SEARCH_LIMIT = 20

# Document type priority for ranking (lower index = higher priority).
_DOCUMENT_TYPE_PRIORITY: dict[str, int] = {
    "errata": 0,
    "faq": 1,
    "core_rules": 2,
    "supplement": 3,
    "strategy": 4,
}

_DEFAULT_PRIORITY = 99


class KnowledgeRetrievalStage:
    """Retrieves relevant knowledge from the KG.

    Uses KGRetrievalService.semantic_search() with the analysed intent
    and original question. Follows up with traverse_associations() for
    the top-ranked concepts to pull in related rules, exceptions, and
    conditions.

    The retrieval is scoped to the session's game_id and filtered by
    active_expansion_ids.

    Args:
        kg_retrieval_service: Service for KG semantic search and traversal.
    """

    def __init__(
        self,
        kg_retrieval_service: KGRetrievalService,
    ) -> None:
        self._kg = kg_retrieval_service

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Retrieve knowledge from the KG.

        1. Build query text from the user's question augmented with intent.
        2. Semantic search using KGRetrievalService (limit=20).
        3. Traverse associations for top 5 results (depth=2).
        4. Deduplicate and rank results by relevance + document type priority.
        5. Flag empty results for synthesis acknowledgement.

        Populates ``ctx.retrieval_results``, ``ctx.traversal_results``,
        and ``ctx.knowledge_insufficient``.

        Args:
            ctx: Pipeline context with game_id, intent, and user message.
            model_client: Model client (unused -- no LLM call in this stage).

        Yields:
            No SSE events. This stage is silent.
        """
        query_text = self._build_query_text(ctx)

        kg_results = await self._kg.semantic_search(
            game_id=ctx.game_id,
            query_text=query_text,
            active_expansion_ids=ctx.active_expansion_ids,
            limit=_SEARCH_LIMIT,
        )

        logger.info(
            "KG semantic search completed",
            extra={
                "game_id": str(ctx.game_id),
                "result_count": len(kg_results),
                "query_length": len(query_text),
            },
        )

        # Traverse associations for top results
        traversal_nodes = await self._traverse_top_results(
            kg_results[:_TOP_RESULTS_FOR_TRAVERSAL],
            ctx.game_id,
            ctx.active_expansion_ids,
        )

        # Map KG types to pipeline context types, deduplicate, and rank
        retrieval_results = _map_retrieval_results(kg_results)
        traversal_results = _map_traversal_nodes(traversal_nodes)

        # Deduplicate traversal results against retrieval results
        seen_concept_ids: set[UUID] = {r.concept_id for r in retrieval_results}
        unique_traversal = [t for t in traversal_results if t.concept_id not in seen_concept_ids]

        # Rank retrieval results by score and document type priority
        retrieval_results.sort(key=_retrieval_sort_key)

        ctx.retrieval_results = retrieval_results
        ctx.traversal_results = unique_traversal

        if not retrieval_results:
            ctx.knowledge_insufficient = True
            logger.warning(
                "KG retrieval returned zero results",
                extra={
                    "game_id": str(ctx.game_id),
                    "query_text": query_text,
                },
            )

        logger.info(
            "Knowledge retrieval stage complete",
            extra={
                "retrieval_count": len(retrieval_results),
                "traversal_count": len(unique_traversal),
                "knowledge_insufficient": ctx.knowledge_insufficient,
            },
        )

        # Async generator protocol: must yield to satisfy AsyncGenerator return type
        return
        yield  # pragma: no cover

    async def _traverse_top_results(
        self,
        top_results: list[KGRetrievalResult],
        game_id: UUID,
        active_expansion_ids: list[UUID],
    ) -> list[tuple[TraversalNode, UUID]]:
        """Traverse associations for the top semantic search results.

        Args:
            top_results: Top KG retrieval results to traverse from.
            game_id: Game scope for traversal.
            active_expansion_ids: Expansion filter for traversal.

        Returns:
            List of (TraversalNode, root_concept_id) tuples.
        """
        all_nodes: list[tuple[TraversalNode, UUID]] = []

        for result in top_results:
            traversal = await self._kg.traverse_associations(
                game_id=game_id,
                concept_id=result.concept_id,
                max_depth=_TRAVERSAL_DEPTH,
                active_expansion_ids=active_expansion_ids,
            )
            for node in traversal.nodes:
                all_nodes.append((node, result.concept_id))

        logger.debug(
            "Association traversal completed",
            extra={
                "seed_count": len(top_results),
                "total_nodes": len(all_nodes),
            },
        )

        return all_nodes

    @staticmethod
    def _build_query_text(ctx: PipelineContext) -> str:
        """Build the search query from question and analysed intent.

        Augments the raw question with the intent analysis when available,
        giving the embedding model richer semantic signal.

        Args:
            ctx: Pipeline context with user_message and intent.

        Returns:
            Combined query string for semantic search.
        """
        question = ctx.user_message.content or ""
        if ctx.intent:
            return f"{question}\n\nIntent: {ctx.intent}"
        return question


def _map_retrieval_results(
    kg_results: list[KGRetrievalResult],
) -> list[RetrievalResult]:
    """Map KG RetrievalResult objects to pipeline context RetrievalResult.

    Args:
        kg_results: Results from KGRetrievalService.semantic_search().

    Returns:
        List of pipeline RetrievalResult with source metadata.
    """
    results: list[RetrievalResult] = []
    for kg_r in kg_results:
        # Build source dict from the best (authoritative) source reference
        source: dict[str, object] = {}
        if kg_r.sources:
            best = kg_r.sources[0]  # Already sorted by is_authoritative DESC
            source = {
                "document_id": str(best.document_id),
                "document_name": best.document_name,
                "document_type": best.document_type,
                "section_path": best.section_path,
                "page_number": best.page_number,
                "source_text": best.source_text,
                "is_authoritative": best.is_authoritative,
            }

        results.append(
            RetrievalResult(
                concept_id=kg_r.concept_id,
                content=kg_r.description,
                score=kg_r.similarity_score,
                source=source,
            )
        )
    return results


def _map_traversal_nodes(
    nodes_with_roots: list[tuple[TraversalNode, UUID]],
) -> list[TraversalResult]:
    """Map KG TraversalNode objects to pipeline context TraversalResult.

    Deduplicates by concept_id, keeping the shallowest traversal path.

    Args:
        nodes_with_roots: List of (TraversalNode, root_concept_id) tuples.

    Returns:
        Deduplicated list of pipeline TraversalResult.
    """
    seen: dict[UUID, TraversalResult] = {}
    for node, _root_id in nodes_with_roots:
        if node.concept_id in seen:
            # Keep the shallowest path (lower depth = closer to seed)
            if node.depth < seen[node.concept_id].depth:
                seen[node.concept_id] = TraversalResult(
                    concept_id=node.concept_id,
                    content=node.concept_name,
                    relationship=node.relationship_label,
                    depth=node.depth,
                    source={"concept_type": node.concept_type},
                )
        else:
            seen[node.concept_id] = TraversalResult(
                concept_id=node.concept_id,
                content=node.concept_name,
                relationship=node.relationship_label,
                depth=node.depth,
                source={"concept_type": node.concept_type},
            )
    return list(seen.values())


def _retrieval_sort_key(result: RetrievalResult) -> tuple[int, float]:
    """Sort key for ranking retrieval results.

    Primary: document type priority (lower = better).
    Secondary: similarity score (negated so higher score sorts first).

    Args:
        result: A pipeline RetrievalResult.

    Returns:
        Tuple of (document_type_priority, -score) for sorting.
    """
    doc_type = result.source.get("document_type", "")
    priority = _DOCUMENT_TYPE_PRIORITY.get(str(doc_type), _DEFAULT_PRIORITY)
    return (priority, -result.score)

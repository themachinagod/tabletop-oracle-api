"""Unit tests for KnowledgeRetrievalStage.

Tests:
- Semantic search called with game_id and expansion filtering (AC-601)
- Query text built from question + intent
- Associations traversed for top 5 results
- Results deduplicated and ranked by priority + score
- Empty results flagged as knowledge_insufficient
- No SSE events yielded (silent stage)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from tabletop_oracle.services.ai.context import RetrievalResult
from tabletop_oracle.services.ai.stages.knowledge_retrieval import (
    _SEARCH_LIMIT,
    _TOP_RESULTS_FOR_TRAVERSAL,
    _TRAVERSAL_DEPTH,
    KnowledgeRetrievalStage,
    _retrieval_sort_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAME_ID = uuid.uuid4()
_EXPANSION_IDS = [uuid.uuid4(), uuid.uuid4()]


def _make_kg_source(
    *,
    document_type: str = "core_rules",
    document_name: str = "Core Rulebook",
    is_authoritative: bool = True,
) -> MagicMock:
    """Build a mock SourceReference from KGRetrievalService.

    Args:
        document_type: Document type classification.
        document_name: Human-readable document name.
        is_authoritative: Whether this is the authoritative source.

    Returns:
        MagicMock mimicking a SourceReference.
    """
    src = MagicMock()
    src.document_id = uuid.uuid4()
    src.document_name = document_name
    src.document_type = document_type
    src.section_path = "Chapter 1 > Combat"
    src.page_number = 42
    src.source_text = "When attacking, roll a d20."
    src.is_authoritative = is_authoritative
    src.expansion_id = None
    return src


def _make_kg_result(
    *,
    concept_id: uuid.UUID | None = None,
    name: str = "Attack Roll",
    similarity_score: float = 0.85,
    document_type: str = "core_rules",
    sources: list[Any] | None = None,
) -> MagicMock:
    """Build a mock KGRetrievalService RetrievalResult.

    Args:
        concept_id: UUID for the concept.
        name: Concept name.
        similarity_score: Cosine similarity score.
        document_type: Type of the primary source document.
        sources: Override source list.

    Returns:
        MagicMock mimicking a KG RetrievalResult.
    """
    result = MagicMock()
    result.concept_id = concept_id or uuid.uuid4()
    result.name = name
    result.semantic_type = "rule"
    result.description = f"Description of {name}"
    result.similarity_score = similarity_score
    result.sources = (
        sources
        if sources is not None
        else [
            _make_kg_source(document_type=document_type),
        ]
    )
    result.associations = []
    return result


def _make_traversal_result(
    *,
    root_concept_id: uuid.UUID | None = None,
    nodes: list[Any] | None = None,
) -> MagicMock:
    """Build a mock KGRetrievalService TraversalResult.

    Args:
        root_concept_id: Root concept UUID.
        nodes: List of TraversalNode mocks.

    Returns:
        MagicMock mimicking a KG TraversalResult.
    """
    result = MagicMock()
    result.root_concept_id = root_concept_id or uuid.uuid4()
    result.root_concept_name = "Root Concept"
    result.nodes = nodes or []
    return result


def _make_traversal_node(
    *,
    concept_id: uuid.UUID | None = None,
    concept_name: str = "Related Concept",
    relationship_label: str = "modifies",
    depth: int = 1,
) -> MagicMock:
    """Build a mock TraversalNode.

    Args:
        concept_id: UUID for the node concept.
        concept_name: Name of the related concept.
        relationship_label: Relationship type.
        depth: Traversal depth.

    Returns:
        MagicMock mimicking a TraversalNode.
    """
    node = MagicMock()
    node.concept_id = concept_id or uuid.uuid4()
    node.concept_name = concept_name
    node.concept_type = "rule"
    node.association_id = uuid.uuid4()
    node.relationship_label = relationship_label
    node.direction = "outgoing"
    node.depth = depth
    return node


def _make_ctx(
    *,
    question: str = "How does combat work?",
    intent: str | None = "rules_query: combat mechanics",
    game_id: uuid.UUID | None = None,
    expansion_ids: list[uuid.UUID] | None = None,
) -> MagicMock:
    """Build a mock PipelineContext.

    Args:
        question: User question content.
        intent: Analysed intent string.
        game_id: Game UUID.
        expansion_ids: Active expansion IDs.

    Returns:
        MagicMock mimicking a PipelineContext.
    """
    ctx = MagicMock()
    ctx.user_message.content = question
    ctx.intent = intent
    ctx.game_id = game_id or _GAME_ID
    ctx.active_expansion_ids = expansion_ids if expansion_ids is not None else _EXPANSION_IDS
    ctx.retrieval_results = []
    ctx.traversal_results = []
    ctx.knowledge_insufficient = False
    return ctx


def _make_stage(
    *,
    search_results: list[Any] | None = None,
    traversal_results: list[Any] | None = None,
) -> tuple[KnowledgeRetrievalStage, AsyncMock]:
    """Build a KnowledgeRetrievalStage with mocked KGRetrievalService.

    Args:
        search_results: Results returned by semantic_search.
        traversal_results: Results returned by traverse_associations.

    Returns:
        Tuple of (stage, kg_service_mock).
    """
    kg_service = AsyncMock()
    kg_service.semantic_search.return_value = search_results or []

    if traversal_results is not None:
        kg_service.traverse_associations.side_effect = traversal_results
    else:
        kg_service.traverse_associations.return_value = _make_traversal_result(nodes=[])

    stage = KnowledgeRetrievalStage(kg_retrieval_service=kg_service)
    return stage, kg_service


# ---------------------------------------------------------------------------
# Tests: Semantic Search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    """Tests for KG semantic search invocation."""

    async def test_search_called_with_game_id_and_expansions(self) -> None:
        """Semantic search is called with game_id and expansion filter (AC-601)."""
        stage, kg_svc = _make_stage()
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        kg_svc.semantic_search.assert_awaited_once_with(
            game_id=ctx.game_id,
            query_text="How does combat work?\n\nIntent: rules_query: combat mechanics",
            active_expansion_ids=_EXPANSION_IDS,
            limit=_SEARCH_LIMIT,
        )

    async def test_search_query_without_intent(self) -> None:
        """Without intent, search uses raw question only."""
        stage, kg_svc = _make_stage()
        ctx = _make_ctx(intent=None)

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        call_kwargs = kg_svc.semantic_search.call_args
        assert call_kwargs.kwargs["query_text"] == "How does combat work?"

    async def test_search_query_with_intent_augmented(self) -> None:
        """With intent, search query includes question + intent."""
        stage, kg_svc = _make_stage()
        ctx = _make_ctx(question="What is flanking?", intent="tactical_query: positioning")

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        call_kwargs = kg_svc.semantic_search.call_args
        assert call_kwargs.kwargs["query_text"] == (
            "What is flanking?\n\nIntent: tactical_query: positioning"
        )


# ---------------------------------------------------------------------------
# Tests: Association Traversal
# ---------------------------------------------------------------------------


class TestAssociationTraversal:
    """Tests for association traversal of top results."""

    async def test_traversal_called_for_top_5_results(self) -> None:
        """traverse_associations is called for top 5 search results."""
        results = [_make_kg_result() for _ in range(8)]
        traversal = _make_traversal_result(nodes=[])
        stage, kg_svc = _make_stage(
            search_results=results,
            traversal_results=[traversal] * _TOP_RESULTS_FOR_TRAVERSAL,
        )
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert kg_svc.traverse_associations.await_count == _TOP_RESULTS_FOR_TRAVERSAL
        for i in range(_TOP_RESULTS_FOR_TRAVERSAL):
            call = kg_svc.traverse_associations.call_args_list[i]
            assert call.kwargs["concept_id"] == results[i].concept_id
            assert call.kwargs["max_depth"] == _TRAVERSAL_DEPTH
            assert call.kwargs["game_id"] == _GAME_ID

    async def test_traversal_not_called_when_no_results(self) -> None:
        """No traversal calls when semantic search returns empty."""
        stage, kg_svc = _make_stage(search_results=[])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        kg_svc.traverse_associations.assert_not_awaited()

    async def test_traversal_called_for_fewer_than_5_results(self) -> None:
        """When fewer than 5 results, traversal is called for all of them."""
        results = [_make_kg_result() for _ in range(3)]
        traversal = _make_traversal_result(nodes=[])
        stage, kg_svc = _make_stage(
            search_results=results,
            traversal_results=[traversal] * 3,
        )
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert kg_svc.traverse_associations.await_count == 3


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for result deduplication."""

    async def test_traversal_results_deduplicated_against_retrieval(self) -> None:
        """Traversal results sharing concept_id with retrieval are excluded."""
        shared_id = uuid.uuid4()
        kg_result = _make_kg_result(concept_id=shared_id)
        node = _make_traversal_node(concept_id=shared_id)
        traversal = _make_traversal_result(nodes=[node])

        stage, _ = _make_stage(
            search_results=[kg_result],
            traversal_results=[traversal],
        )
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert len(ctx.retrieval_results) == 1
        assert len(ctx.traversal_results) == 0

    async def test_unique_traversal_results_kept(self) -> None:
        """Traversal results with unique concept_ids are kept."""
        kg_result = _make_kg_result()
        unique_node = _make_traversal_node()
        traversal = _make_traversal_result(nodes=[unique_node])

        stage, _ = _make_stage(
            search_results=[kg_result],
            traversal_results=[traversal],
        )
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert len(ctx.traversal_results) == 1
        assert ctx.traversal_results[0].concept_id == unique_node.concept_id

    async def test_traversal_nodes_deduplicated_by_concept_id_shallowest_wins(
        self,
    ) -> None:
        """When same concept appears at multiple depths, shallowest is kept."""
        shared_concept_id = uuid.uuid4()
        node_deep = _make_traversal_node(
            concept_id=shared_concept_id,
            depth=2,
            relationship_label="deep_rel",
        )
        node_shallow = _make_traversal_node(
            concept_id=shared_concept_id,
            depth=1,
            relationship_label="shallow_rel",
        )
        traversal = _make_traversal_result(nodes=[node_deep, node_shallow])

        stage, _ = _make_stage(
            search_results=[_make_kg_result()],
            traversal_results=[traversal],
        )
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        matching = [t for t in ctx.traversal_results if t.concept_id == shared_concept_id]
        assert len(matching) == 1
        assert matching[0].depth == 1
        assert matching[0].relationship == "shallow_rel"


# ---------------------------------------------------------------------------
# Tests: Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    """Tests for result ranking by document type priority and score."""

    async def test_results_ranked_by_document_type_priority(self) -> None:
        """Errata results rank above core_rules, even with lower score."""
        errata_result = _make_kg_result(
            name="Errata Fix",
            similarity_score=0.70,
            document_type="errata",
        )
        core_result = _make_kg_result(
            name="Core Rule",
            similarity_score=0.90,
            document_type="core_rules",
        )
        stage, _ = _make_stage(search_results=[core_result, errata_result])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.retrieval_results[0].source["document_type"] == "errata"
        assert ctx.retrieval_results[1].source["document_type"] == "core_rules"

    async def test_same_type_ranked_by_score_descending(self) -> None:
        """Within same document type, higher score ranks first."""
        high = _make_kg_result(
            name="High Score",
            similarity_score=0.95,
            document_type="core_rules",
        )
        low = _make_kg_result(
            name="Low Score",
            similarity_score=0.60,
            document_type="core_rules",
        )
        stage, _ = _make_stage(search_results=[low, high])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.retrieval_results[0].score == 0.95
        assert ctx.retrieval_results[1].score == 0.60

    async def test_unknown_document_type_ranks_last(self) -> None:
        """Results with unknown document type rank after known types."""
        unknown = _make_kg_result(
            name="Unknown Type",
            similarity_score=0.99,
            document_type="homebrew",
        )
        faq = _make_kg_result(
            name="FAQ Entry",
            similarity_score=0.50,
            document_type="faq",
        )
        stage, _ = _make_stage(search_results=[unknown, faq])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.retrieval_results[0].source["document_type"] == "faq"
        assert ctx.retrieval_results[1].source["document_type"] == "homebrew"


# ---------------------------------------------------------------------------
# Tests: Empty Results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    """Tests for empty result handling."""

    async def test_empty_results_flag_knowledge_insufficient(self) -> None:
        """Empty semantic search sets knowledge_insufficient to True."""
        stage, _ = _make_stage(search_results=[])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.knowledge_insufficient is True
        assert ctx.retrieval_results == []
        assert ctx.traversal_results == []

    async def test_non_empty_results_leave_flag_false(self) -> None:
        """Non-empty results leave knowledge_insufficient as False."""
        stage, _ = _make_stage(search_results=[_make_kg_result()])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.knowledge_insufficient is False


# ---------------------------------------------------------------------------
# Tests: No Events Yielded
# ---------------------------------------------------------------------------


class TestNoEventsYielded:
    """Tests that the stage yields no SSE events."""

    async def test_stage_yields_no_events(self) -> None:
        """Stage produces no SSE events (silent stage)."""
        stage, _ = _make_stage(search_results=[_make_kg_result()])
        ctx = _make_ctx()
        events = [event async for event in stage.execute(ctx, MagicMock())]
        assert events == []


# ---------------------------------------------------------------------------
# Tests: Result Mapping
# ---------------------------------------------------------------------------


class TestResultMapping:
    """Tests for mapping KG types to pipeline context types."""

    async def test_retrieval_result_maps_source_metadata(self) -> None:
        """KG result sources are mapped to pipeline source dict."""
        source = _make_kg_source(
            document_type="faq",
            document_name="FAQ v2",
        )
        kg_result = _make_kg_result(sources=[source])
        stage, _ = _make_stage(search_results=[kg_result])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        result = ctx.retrieval_results[0]
        assert result.source["document_type"] == "faq"
        assert result.source["document_name"] == "FAQ v2"
        assert result.source["is_authoritative"] is True

    async def test_retrieval_result_without_sources_has_empty_source_dict(
        self,
    ) -> None:
        """KG result with no sources maps to empty source dict."""
        kg_result = _make_kg_result(sources=[])
        stage, _ = _make_stage(search_results=[kg_result])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.retrieval_results[0].source == {}

    async def test_traversal_node_maps_to_traversal_result(self) -> None:
        """KG TraversalNode maps to pipeline TraversalResult."""
        node = _make_traversal_node(
            concept_name="Shield Wall",
            relationship_label="enables",
            depth=1,
        )
        traversal = _make_traversal_result(nodes=[node])
        stage, _ = _make_stage(
            search_results=[_make_kg_result()],
            traversal_results=[traversal],
        )
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        matching = [t for t in ctx.traversal_results if t.concept_id == node.concept_id]
        assert len(matching) == 1
        assert matching[0].content == "Shield Wall"
        assert matching[0].relationship == "enables"
        assert matching[0].depth == 1


# ---------------------------------------------------------------------------
# Tests: Helper Functions
# ---------------------------------------------------------------------------


class TestRetrievalSortKey:
    """Tests for the _retrieval_sort_key helper."""

    def test_errata_ranks_above_core_rules(self) -> None:
        errata = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="errata fix",
            score=0.5,
            source={"document_type": "errata"},
        )
        core = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="core rule",
            score=0.9,
            source={"document_type": "core_rules"},
        )
        assert _retrieval_sort_key(errata) < _retrieval_sort_key(core)

    def test_higher_score_ranks_first_within_same_type(self) -> None:
        high = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="high",
            score=0.95,
            source={"document_type": "faq"},
        )
        low = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="low",
            score=0.60,
            source={"document_type": "faq"},
        )
        assert _retrieval_sort_key(high) < _retrieval_sort_key(low)

    def test_unknown_type_gets_default_priority(self) -> None:
        unknown = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="unknown",
            score=0.99,
            source={"document_type": "homebrew"},
        )
        core = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="core",
            score=0.01,
            source={"document_type": "core_rules"},
        )
        assert _retrieval_sort_key(unknown) > _retrieval_sort_key(core)

    def test_empty_source_gets_default_priority(self) -> None:
        no_source = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="no source",
            score=0.5,
            source={},
        )
        key = _retrieval_sort_key(no_source)
        assert key[0] == 99  # _DEFAULT_PRIORITY

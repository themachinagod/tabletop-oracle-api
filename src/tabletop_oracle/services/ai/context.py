"""Mutable pipeline context accumulated across query pipeline stages.

Each stage reads upstream data and writes its own outputs. The pipeline
orchestrator passes this context to every stage. Designed per the EPIC-004
design doc ``PipelineContext`` section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.models.message import Message
    from tabletop_oracle.models.session import Session


@dataclass
class CitationData:
    """Citation reference assembled from KG retrieval sources.

    Attributes:
        index: 1-based citation index as referenced in the answer text.
        document_name: Human-readable document title.
        document_type: Document classification (core_rules, faq, errata, etc.).
        section: Section path within the document.
        page_number: Page number if available.
        excerpt: Relevant excerpt from the source.
    """

    index: int
    document_name: str
    document_type: str
    section: str = ""
    page_number: int | None = None
    excerpt: str = ""


@dataclass
class RetrievalResult:
    """A single result from KG semantic search.

    Attributes:
        concept_id: The KG concept identifier.
        content: The concept text content.
        score: Similarity score from the search.
        source: Source attribution metadata.
    """

    concept_id: UUID
    content: str
    score: float
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraversalResult:
    """A single result from KG association traversal.

    Attributes:
        concept_id: The KG concept identifier.
        content: The concept text content.
        relationship: The relationship type to the parent concept.
        depth: Traversal depth from the seed concept.
        source: Source attribution metadata.
    """

    concept_id: UUID
    content: str
    relationship: str
    depth: int
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable state accumulated across pipeline stages.

    Each stage reads upstream data and writes its own outputs. The
    pipeline orchestrator passes this to every stage.

    Attributes:
        session: The game session.
        user_message: The player's question message.
        game_id: Game ID from session (convenience).
        active_expansion_ids: Active expansion IDs from session config.
        player_count: Player count from session (may be None).
        conversation_history: Relevant prior messages for context.
        ad_hoc_text_context: Text context from player.
        ad_hoc_image_descriptions: Processed image descriptions.
        intent: Parsed intent from analysis.
        is_ambiguous: Whether the question requires clarification.
        ambiguity_reason: Why the question is ambiguous (if applicable).
        suggested_clarification: Draft clarification question from intent analysis.
        clarification_round: Current clarification round (0, 1, or 2).
        retrieval_results: Knowledge retrieved from KG.
        traversal_results: Association traversal results.
        knowledge_insufficient: True when KG retrieval returned zero results.
        answer_text: Generated answer text.
        citations: Assembled citation records.
        confidence_score: Self-assessed confidence value.
        token_usage: Running total of tokens consumed in this query.
        model_call_count: Number of model calls made in this query.
    """

    # Immutable inputs (set at pipeline start)
    session: Session
    user_message: Message
    game_id: UUID
    active_expansion_ids: list[UUID]
    player_count: int | None

    # Populated by ContextPreparationStage
    conversation_history: list[Message] = field(default_factory=list)
    ad_hoc_text_context: str | None = None
    ad_hoc_image_descriptions: list[str] = field(default_factory=list)

    # Populated by IntentAnalysisStage
    intent: str | None = None
    is_ambiguous: bool = False
    ambiguity_reason: str | None = None
    suggested_clarification: str | None = None
    clarification_round: int = 0

    # Populated by KnowledgeRetrievalStage
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    traversal_results: list[TraversalResult] = field(default_factory=list)
    knowledge_insufficient: bool = False

    # Populated by AnswerSynthesisStage
    answer_text: str = ""

    # Populated by CitationAssemblyStage
    citations: list[CitationData] = field(default_factory=list)

    # Populated by ConfidenceScoreStage
    confidence_score: float | None = None

    # Running counters (checked by guardrails)
    token_usage: int = 0
    model_call_count: int = 0

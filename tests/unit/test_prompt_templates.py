"""Unit tests for prompt template content and quality.

Tests that prompt templates enforce the oracle persona, grounding
constraints, strategy framing, confidence calibration, and source
handling requirements from the design doc and PRD-006.

These tests verify the prompt text itself, not the stage logic.
Stage-level tests are in test_intent_analysis.py, test_answer_synthesis.py,
test_clarification_stage.py, and test_confidence_score.py.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

from tabletop_oracle.services.ai.prompts.answer_synthesis import (
    build_answer_synthesis_messages,
)
from tabletop_oracle.services.ai.prompts.clarification import (
    build_clarification_messages,
)
from tabletop_oracle.services.ai.prompts.confidence import (
    build_confidence_messages,
)
from tabletop_oracle.services.ai.prompts.intent import (
    build_intent_messages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()
_GAME_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()


def _make_base_ctx(
    *,
    question: str = "How does trading work?",
    game_name: str = "Catan",
    player_count: int | None = 4,
) -> MagicMock:
    """Build a minimal mock PipelineContext.

    Args:
        question: The player's question text.
        game_name: Name of the game.
        player_count: Number of players.

    Returns:
        MagicMock mimicking a PipelineContext.
    """
    ctx = MagicMock()
    ctx.session.id = _SESSION_ID
    ctx.session.user_id = _USER_ID
    ctx.session.game.name = game_name
    ctx.user_message.id = _MESSAGE_ID
    ctx.user_message.content = question
    ctx.game_id = _GAME_ID
    ctx.player_count = player_count
    ctx.conversation_history = []
    ctx.ad_hoc_text_context = None
    ctx.ad_hoc_image_descriptions = []
    return ctx


def _make_intent_ctx(**kwargs: Any) -> MagicMock:
    """Build a context for intent analysis prompts."""
    return _make_base_ctx(**kwargs)


def _make_clarification_ctx(**kwargs: Any) -> MagicMock:
    """Build a context for clarification prompts."""
    ctx = _make_base_ctx(**kwargs)
    ctx.ambiguity_reason = "Could refer to domestic or maritime trade"
    ctx.suggested_clarification = (
        "Are you asking about trading with other players or with the bank?"
    )
    return ctx


def _make_retrieval_result(
    *,
    content: str = "Players may trade resources on their turn.",
    score: float = 0.92,
    document_name: str = "Catan Rulebook",
    document_type: str = "core_rules",
    section_path: str = "Trading",
    page_number: int | None = 8,
    is_authoritative: bool = True,
) -> MagicMock:
    """Build a mock RetrievalResult.

    Args:
        content: The content text.
        score: Similarity score.
        document_name: Source document name.
        document_type: Source document type.
        section_path: Section path.
        page_number: Page number.
        is_authoritative: Whether the source is authoritative.

    Returns:
        MagicMock mimicking a RetrievalResult.
    """
    result = MagicMock()
    result.content = content
    result.score = score
    result.source = {
        "document_name": document_name,
        "document_type": document_type,
        "section_path": section_path,
        "page_number": page_number,
        "is_authoritative": is_authoritative,
    }
    return result


def _make_synthesis_ctx(**kwargs: Any) -> MagicMock:
    """Build a context for answer synthesis prompts."""
    ctx = _make_base_ctx(**kwargs)
    ctx.retrieval_results = [_make_retrieval_result()]
    ctx.traversal_results = []
    ctx.knowledge_insufficient = False
    ctx.intent = "Rules question about trading mechanics"
    return ctx


def _make_confidence_ctx(**kwargs: Any) -> MagicMock:
    """Build a context for confidence scoring prompts."""
    ctx = _make_base_ctx(**kwargs)
    ctx.answer_text = "You can trade resources with other players on your turn [1]."
    ctx.retrieval_results = [_make_retrieval_result()]
    ctx.traversal_results = []
    return ctx


# ---------------------------------------------------------------------------
# Tests: Oracle Persona Consistency
# ---------------------------------------------------------------------------


class TestOraclePersona:
    """Tests that all prompts use the Tabletop Oracle persona consistently."""

    def test_intent_prompt_has_oracle_persona(self) -> None:
        """Intent analysis system prompt identifies as Tabletop Oracle."""
        ctx = _make_intent_ctx()
        messages = build_intent_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Tabletop Oracle" in system_msg

    def test_clarification_prompt_has_oracle_persona(self) -> None:
        """Clarification system prompt identifies as Tabletop Oracle."""
        ctx = _make_clarification_ctx()
        messages = build_clarification_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Tabletop Oracle" in system_msg

    def test_synthesis_prompt_has_oracle_persona(self) -> None:
        """Answer synthesis system prompt identifies as Tabletop Oracle."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Tabletop Oracle" in system_msg

    def test_confidence_prompt_has_oracle_persona(self) -> None:
        """Confidence scoring system prompt references Tabletop Oracle."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Tabletop Oracle" in system_msg


# ---------------------------------------------------------------------------
# Tests: Intent Analysis Prompt Quality
# ---------------------------------------------------------------------------


class TestIntentPromptQuality:
    """Tests for intent analysis prompt refinements."""

    def test_ambiguity_criteria_include_game_specific_example(self) -> None:
        """Ambiguity criteria include a concrete game example."""
        ctx = _make_intent_ctx()
        messages = build_intent_messages(ctx)
        system_msg = messages[0]["content"]

        assert "e.g." in system_msg or "for example" in system_msg.lower()

    def test_do_not_flag_guidance_present(self) -> None:
        """Prompt includes guidance on when NOT to flag as ambiguous."""
        ctx = _make_intent_ctx()
        messages = build_intent_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Do NOT flag as ambiguous" in system_msg

    def test_conversation_context_reduces_ambiguity(self) -> None:
        """Prompt instructs that conversation history can resolve ambiguity."""
        ctx = _make_intent_ctx()
        messages = build_intent_messages(ctx)
        system_msg = messages[0]["content"]

        assert "conversation" in system_msg.lower()
        assert "context" in system_msg.lower()

    def test_materiality_criterion_present(self) -> None:
        """Prompt requires ambiguity to be material (different answer)."""
        ctx = _make_intent_ctx()
        messages = build_intent_messages(ctx)
        system_msg = messages[0]["content"]

        assert "materially different" in system_msg.lower()


# ---------------------------------------------------------------------------
# Tests: Clarification Prompt Quality
# ---------------------------------------------------------------------------


class TestClarificationPromptQuality:
    """Tests for clarification prompt refinements."""

    def test_concrete_options_instruction(self) -> None:
        """Prompt instructs offering concrete options as closed questions."""
        ctx = _make_clarification_ctx()
        messages = build_clarification_messages(ctx)
        system_msg = messages[0]["content"]

        assert "concrete options" in system_msg.lower()

    def test_many_options_handling(self) -> None:
        """Prompt handles cases with many possible interpretations."""
        ctx = _make_clarification_ctx()
        messages = build_clarification_messages(ctx)
        system_msg = messages[0]["content"]

        assert "something else" in system_msg.lower()


# ---------------------------------------------------------------------------
# Tests: Answer Synthesis Prompt Quality
# ---------------------------------------------------------------------------


class TestSynthesisPromptQuality:
    """Tests for answer synthesis prompt refinements."""

    def test_no_training_data_instruction(self) -> None:
        """Prompt explicitly forbids using training data for facts."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "training data" in system_msg.lower()

    def test_interpretation_transparency(self) -> None:
        """Prompt requires transparency when interpreting ambiguous sources."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "interpret" in system_msg.lower()

    def test_strategy_insufficient_context_guidance(self) -> None:
        """Prompt addresses what to do when strategy knowledge is lacking."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        # AC-607: acknowledge when insufficient context for strategy
        assert (
            "lacks strategy content" in system_msg.lower()
            or "knowledge base lacks" in system_msg.lower()
        )

    def test_conflict_resolution_explicit(self) -> None:
        """Prompt requires explicit conflict noting with both sources cited."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "cite both" in system_msg.lower()

    def test_response_style_section_present(self) -> None:
        """Prompt includes response style guidance."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Response Style" in system_msg

    def test_authoritative_flag_shown_in_knowledge(self) -> None:
        """Authoritative sources are labelled in the knowledge section."""
        result = _make_retrieval_result(is_authoritative=True)
        ctx = _make_synthesis_ctx()
        ctx.retrieval_results = [result]
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "authoritative" in user_msg.lower()

    def test_non_authoritative_source_no_label(self) -> None:
        """Non-authoritative sources do not get the authoritative label."""
        result = _make_retrieval_result(is_authoritative=False)
        ctx = _make_synthesis_ctx()
        ctx.retrieval_results = [result]
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "authoritative" not in user_msg.lower()

    def test_insufficient_knowledge_instructs_no_general_knowledge(
        self,
    ) -> None:
        """Insufficient knowledge message forbids general knowledge answers."""
        ctx = _make_synthesis_ctx()
        ctx.retrieval_results = []
        ctx.knowledge_insufficient = True
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "general knowledge" in user_msg.lower()

    def test_source_marker_instruction_forbids_invented_markers(self) -> None:
        """Source marker instructions forbid inventing source numbers."""
        ctx = _make_synthesis_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Do not invent source numbers" in system_msg


# ---------------------------------------------------------------------------
# Tests: Confidence Scoring Prompt Quality
# ---------------------------------------------------------------------------


class TestConfidencePromptQuality:
    """Tests for confidence scoring prompt calibration."""

    def test_scoring_has_five_bands(self) -> None:
        """Confidence scoring uses 5 calibration bands for granularity."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        system_msg = messages[0]["content"]

        assert "0.9-1.0" in system_msg
        assert "0.7-0.89" in system_msg
        assert "0.5-0.69" in system_msg
        assert "0.3-0.49" in system_msg
        assert "0.0-0.29" in system_msg

    def test_criteria_ordered_by_importance(self) -> None:
        """Evaluation criteria are presented in order of importance."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        system_msg = messages[0]["content"]

        assert "order of importance" in system_msg.lower()

    def test_source_authority_criterion_present(self) -> None:
        """Source authority is an explicit evaluation criterion."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        system_msg = messages[0]["content"]

        assert "authoritative" in system_msg.lower()

    def test_priority_hierarchy_referenced(self) -> None:
        """Confidence prompt references the source priority hierarchy."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Errata" in system_msg
        assert "FAQ" in system_msg
        assert "Core Rules" in system_msg

    def test_source_type_included_in_source_section(self) -> None:
        """Source entries in the confidence prompt include document type."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        user_msg = messages[1]["content"]

        assert "type=core_rules" in user_msg

    def test_authoritative_flag_in_source_section(self) -> None:
        """Authoritative sources are flagged in the confidence source list."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        user_msg = messages[1]["content"]

        assert "authoritative" in user_msg.lower()

    def test_justification_kept_brief(self) -> None:
        """Justification instruction specifies 1-2 sentences."""
        ctx = _make_confidence_ctx()
        messages = build_confidence_messages(ctx)
        system_msg = messages[0]["content"]

        assert "1-2 sentence" in system_msg

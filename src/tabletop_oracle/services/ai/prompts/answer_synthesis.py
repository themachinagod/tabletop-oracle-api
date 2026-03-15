"""Prompt templates for the answer synthesis pipeline stage.

Builds the system and user messages for the answer synthesis LLM call.
The system prompt establishes grounding constraints, document type
priority, strategy framing rules, and insufficient knowledge behaviour.
The user prompt formats the retrieved knowledge, conversation history,
ad-hoc context, and the player's question.

Design reference: EPIC-004 design.md, Stage 5: Answer Synthesis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabletop_oracle.services.ai.context import (
        PipelineContext,
        RetrievalResult,
        TraversalResult,
    )

_SYSTEM_PROMPT = """\
You are a tabletop game rules oracle. Your role is to answer rules questions \
accurately and helpfully, grounded in the provided knowledge base.

## Grounding Rules
- Base your answer ONLY on the knowledge provided below. Do not fabricate rules, \
mechanics, or interactions.
- When the source material is ambiguous, make your best judgement and clearly \
communicate your level of confidence.
- If the provided knowledge is insufficient to answer the question, say so honestly. \
Do not guess or make up rules.

## Source Priority (highest to lowest)
When sources conflict, apply this priority:
1. Errata — official corrections that supersede all other sources.
2. FAQ — official clarifications that supersede core rules.
3. Core Rules — the baseline rules of the game.
4. Other sources — supplements, expansions, and other materials.

When you identify a conflict between sources, note the conflict and explain which \
source takes priority and why.

## Source References
Include numbered source markers [1], [2], etc. in your answer to indicate which \
source supports each claim. Place markers at the end of the relevant sentence or \
clause. Every factual claim must have at least one source marker.

## Strategy Content
Content from strategy guides or strategy-type sources should be framed as \
guidance, suggestions, or recommendations — NOT as rules. Use language like \
"strategy guides suggest", "a common approach is", or "experienced players \
recommend" rather than definitive statements.

## Player Context
{player_count_line}\
Answer with appropriate specificity for the game context provided."""

_USER_TEMPLATE = """\
{conversation_section}\
{knowledge_section}\
{ad_hoc_section}\
Player's question: {question}"""


def build_answer_synthesis_messages(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Build the chat messages for the answer synthesis LLM call.

    Assembles the system prompt with grounding constraints and a user
    message containing retrieved knowledge, conversation history,
    ad-hoc context, and the player's question.

    Args:
        ctx: Pipeline context with retrieval results and session data.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    player_count_line = (
        f"There are {ctx.player_count} players in this session.\n"
        if ctx.player_count is not None
        else ""
    )

    system_content = _SYSTEM_PROMPT.format(player_count_line=player_count_line)

    knowledge_section = _build_knowledge_section(
        ctx.retrieval_results, ctx.traversal_results, ctx.knowledge_insufficient
    )
    conversation_section = _build_conversation_section(ctx)
    ad_hoc_section = _build_ad_hoc_section(ctx)
    question = ctx.user_message.content or ""

    user_content = _USER_TEMPLATE.format(
        conversation_section=conversation_section,
        knowledge_section=knowledge_section,
        ad_hoc_section=ad_hoc_section,
        question=question,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _build_knowledge_section(
    retrieval_results: list[RetrievalResult],
    traversal_results: list[TraversalResult],
    knowledge_insufficient: bool,
) -> str:
    """Build the knowledge context section with numbered source references.

    Each retrieval result is formatted with a numbered marker that the
    model references in its answer. Traversal results are included as
    supplementary related concepts.

    Args:
        retrieval_results: Ranked retrieval results from KG search.
        traversal_results: Association traversal results.
        knowledge_insufficient: Whether the retrieval returned no results.

    Returns:
        Formatted knowledge section or insufficient knowledge notice.
    """
    if knowledge_insufficient or not retrieval_results:
        return (
            "Knowledge base:\n"
            "No relevant knowledge was found for this question. "
            "Acknowledge this honestly in your response.\n\n"
        )

    lines = ["Knowledge base:"]

    for i, result in enumerate(retrieval_results, start=1):
        source_label = _format_source_label(result.source)
        doc_type = result.source.get("document_type", "unknown")
        lines.append(f"[{i}] ({doc_type}) {source_label}: {result.content}")

    if traversal_results:
        lines.append("")
        lines.append("Related concepts:")
        for tr in traversal_results:
            lines.append(f"  - {tr.content} ({tr.relationship})")

    lines.append("")
    return "\n".join(lines) + "\n"


def _format_source_label(source: dict[str, Any]) -> str:
    """Format a human-readable source label from source metadata.

    Args:
        source: Source attribution dict from a RetrievalResult.

    Returns:
        Formatted label like "Core Rulebook, Chapter 3, p.42".
    """
    parts: list[str] = []

    doc_name = source.get("document_name")
    if doc_name:
        parts.append(str(doc_name))

    section = source.get("section_path")
    if section:
        parts.append(str(section))

    page = source.get("page_number")
    if page is not None:
        parts.append(f"p.{page}")

    return ", ".join(parts) if parts else "Unknown source"


def _build_conversation_section(ctx: PipelineContext) -> str:
    """Build the conversation history section for the synthesis prompt.

    Args:
        ctx: Pipeline context with conversation_history populated.

    Returns:
        Formatted conversation section or empty string.
    """
    if not ctx.conversation_history:
        return ""

    lines = ["Conversation history:"]
    for msg in ctx.conversation_history:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        lines.append(f"  [{role}]: {content}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_ad_hoc_section(ctx: PipelineContext) -> str:
    """Build the ad-hoc context section for the synthesis prompt.

    Args:
        ctx: Pipeline context with ad-hoc context data.

    Returns:
        Formatted ad-hoc context section or empty string.
    """
    parts: list[str] = []

    if ctx.ad_hoc_text_context:
        parts.append(f"Player-provided context:\n{ctx.ad_hoc_text_context}")

    if ctx.ad_hoc_image_descriptions:
        descriptions = "\n".join(f"  - {desc}" for desc in ctx.ad_hoc_image_descriptions)
        parts.append(f"Image context:\n{descriptions}")

    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"

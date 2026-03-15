"""Prompt templates for the clarification pipeline stage.

Builds the system and user messages for generating a clarification
question when the intent analysis stage detects ambiguity. Uses the
``clarification_generation`` model slot.

Design reference: EPIC-004 design.md, Stage 3: Clarification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabletop_oracle.services.ai.context import PipelineContext

_SYSTEM_PROMPT = """\
You are a tabletop game rules assistant. The player asked a question \
that is ambiguous and needs clarification before you can answer accurately.

Your task is to write a single, clear clarification question that will \
help you understand exactly what the player is asking. The question should:
- Be concise (1-2 sentences).
- Reference the specific ambiguity.
- Offer concrete options when possible (e.g., "Are you asking about X or Y?").
- Be friendly and helpful in tone.

Respond with the clarification question only. No preamble, no JSON, no markdown."""

_USER_TEMPLATE = """\
Game: {game_name}
Player's question: {question}
Ambiguity reason: {ambiguity_reason}
{suggestion_line}\
{conversation_section}"""


def build_clarification_messages(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Build chat messages for the clarification generation LLM call.

    Assembles the system prompt and a user message including the
    player's question, ambiguity reason, any draft suggestion from
    intent analysis, and conversation history.

    Args:
        ctx: Pipeline context with intent analysis results populated.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    game_name = _resolve_game_name(ctx)
    ambiguity_reason = ctx.ambiguity_reason or "Question is ambiguous"
    suggestion_line = (
        f"Draft suggestion: {ctx.suggested_clarification}\n"
        if ctx.suggested_clarification
        else ""
    )
    conversation_section = _build_conversation_section(ctx)

    user_content = _USER_TEMPLATE.format(
        game_name=game_name,
        question=ctx.user_message.content,
        ambiguity_reason=ambiguity_reason,
        suggestion_line=suggestion_line,
        conversation_section=conversation_section,
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _resolve_game_name(ctx: PipelineContext) -> str:
    """Resolve the game name from the session.

    Args:
        ctx: Pipeline context with session data.

    Returns:
        The game name string.
    """
    try:
        if hasattr(ctx.session, "game") and ctx.session.game is not None:
            return ctx.session.game.name
    except Exception:
        pass
    return "Unknown game"


def _build_conversation_section(ctx: PipelineContext) -> str:
    """Build conversation history section for the clarification prompt.

    Args:
        ctx: Pipeline context with conversation_history populated.

    Returns:
        Formatted conversation section or empty string.
    """
    if not ctx.conversation_history:
        return ""

    lines = ["\nConversation history:"]
    for msg in ctx.conversation_history:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        lines.append(f"  [{role}]: {content}")
    lines.append("")
    return "\n".join(lines) + "\n"

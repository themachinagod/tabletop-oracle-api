"""Prompt templates for the intent analysis pipeline stage.

Builds the system and user messages for the intent analysis LLM call.
The prompt instructs the model to determine the player's intent and
flag ambiguity when the question is unclear.

Design reference: EPIC-004 design.md, Stage 2: Intent Analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabletop_oracle.services.ai.context import PipelineContext

_SYSTEM_PROMPT = """\
You are a tabletop game rules assistant. Your task is to analyse a player's \
question to determine their intent and detect ambiguity.

Respond with a JSON object containing exactly these fields:
- "intent": A concise free-text summary of what the player is asking about.
- "is_ambiguous": true if the question needs clarification before answering, false otherwise.
- "ambiguity_reason": If ambiguous, explain why. If not ambiguous, set to null.
- "suggested_clarification": If ambiguous, provide a clarification question to ask the player. \
If not ambiguous, set to null.

Flag a question as ambiguous when:
- It could apply to multiple game entities or mechanics.
- It spans multiple rule systems, phases, or game modes.
- It references undefined or unclear terms.
- It is too broad to answer concisely.

Always respond with valid JSON only. No additional text or markdown."""

_USER_TEMPLATE = """\
Game: {game_name}
{player_count_line}\
{conversation_section}\
{ad_hoc_section}\

Player's question: {question}"""


def build_intent_messages(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Build the chat messages for the intent analysis LLM call.

    Assembles the system prompt and a user message that includes the
    player's question, conversation history, session config, and any
    ad-hoc context.

    Args:
        ctx: Pipeline context with session, message, and context data.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    game_name = _resolve_game_name(ctx)
    player_count_line = (
        f"Player count: {ctx.player_count}\n" if ctx.player_count is not None else ""
    )
    conversation_section = _build_conversation_section(ctx)
    ad_hoc_section = _build_ad_hoc_section(ctx)
    question = ctx.user_message.content

    user_content = _USER_TEMPLATE.format(
        game_name=game_name,
        player_count_line=player_count_line,
        conversation_section=conversation_section,
        ad_hoc_section=ad_hoc_section,
        question=question,
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _resolve_game_name(ctx: PipelineContext) -> str:
    """Resolve the game name from the session.

    Attempts to read from the session's game relationship. Falls back
    to a generic label if the relationship is not loaded.

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
    """Build the conversation history section for the prompt.

    Formats prior messages as a compact exchange log. Returns an
    empty string if there is no history.

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


def _build_ad_hoc_section(ctx: PipelineContext) -> str:
    """Build the ad-hoc context section for the prompt.

    Includes text context and image descriptions if present.

    Args:
        ctx: Pipeline context with ad-hoc context data.

    Returns:
        Formatted ad-hoc context section or empty string.
    """
    parts: list[str] = []

    if ctx.ad_hoc_text_context:
        parts.append(f"\nPlayer-provided context:\n{ctx.ad_hoc_text_context}")

    if ctx.ad_hoc_image_descriptions:
        descriptions = "\n".join(f"  - {desc}" for desc in ctx.ad_hoc_image_descriptions)
        parts.append(f"\nImage context:\n{descriptions}")

    if not parts:
        return ""
    return "\n".join(parts) + "\n"

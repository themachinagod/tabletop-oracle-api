"""Prompt templates for the confidence scoring pipeline stage.

Builds the system and user messages for the confidence assessment LLM call.
The prompt instructs the model to evaluate how well the generated answer is
supported by the provided knowledge graph sources.

Design reference: EPIC-004 design.md, Stage 7: Confidence Score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabletop_oracle.services.ai.context import PipelineContext

_SYSTEM_PROMPT = """\
You are a confidence assessment engine for a tabletop game rules assistant. \
Your task is to evaluate how confident the system should be in an answer \
it generated, based on the supporting knowledge sources.

Evaluate confidence based on:
- Number and quality of knowledge sources supporting the answer.
- Whether sources agree or conflict with each other.
- Completeness of the answer relative to the question asked.
- Whether the answer required inference beyond what the sources explicitly state.

Respond with a JSON object containing exactly these fields:
- "score": A float between 0.0 and 1.0 representing confidence.
- "justification": A brief explanation of why this score was assigned.

Scoring guidelines:
- 0.8-1.0: Answer is directly and fully supported by multiple agreeing sources.
- 0.5-0.79: Answer is partially supported or required some inference.
- 0.0-0.49: Answer has weak support, conflicting sources, or significant inference.

Always respond with valid JSON only. No additional text or markdown."""

_USER_TEMPLATE = """\
Question: {question}

Answer: {answer}

Sources ({source_count} total):
{sources_section}"""


def build_confidence_messages(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Build the chat messages for the confidence scoring LLM call.

    Assembles the system prompt and a user message containing the player's
    question, the generated answer, and a summary of the knowledge sources
    used.

    Args:
        ctx: Pipeline context with answer_text and retrieval/traversal results.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    question = ctx.user_message.content
    answer = ctx.answer_text
    sources_section = _build_sources_section(ctx)
    source_count = len(ctx.retrieval_results) + len(ctx.traversal_results)

    user_content = _USER_TEMPLATE.format(
        question=question,
        answer=answer,
        source_count=source_count,
        sources_section=sources_section,
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_sources_section(ctx: PipelineContext) -> str:
    """Build a compact summary of knowledge sources for the prompt.

    Combines retrieval results and traversal results into a numbered
    list showing content excerpts and similarity scores.

    Args:
        ctx: Pipeline context with retrieval and traversal results.

    Returns:
        Formatted sources section or 'No sources available'.
    """
    lines: list[str] = []

    for i, result in enumerate(ctx.retrieval_results, start=1):
        excerpt = result.content[:200]
        lines.append(f"{i}. [score={result.score:.2f}] {excerpt}")

    offset = len(ctx.retrieval_results)
    for i, traversal in enumerate(ctx.traversal_results, start=offset + 1):
        excerpt = traversal.content[:200]
        lines.append(f"{i}. [traversal, depth={traversal.depth}] {excerpt}")

    if not lines:
        return "No sources available."

    return "\n".join(lines)

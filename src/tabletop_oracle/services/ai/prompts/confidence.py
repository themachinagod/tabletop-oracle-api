"""Prompt templates for the confidence scoring pipeline stage.

Builds the system and user messages for the confidence assessment LLM call.
The prompt instructs the model to evaluate how well the generated answer is
supported by the provided knowledge graph sources, using calibrated scoring
bands and structured evaluation criteria.

Design reference: EPIC-004 design.md, Stage 7: Confidence Score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabletop_oracle.services.ai.context import PipelineContext

_SYSTEM_PROMPT = """\
You are a confidence assessor for the Tabletop Oracle, a rules advisory \
system for tabletop games. Your task is to evaluate how confident the \
system should be in the answer it generated, based on how well the \
answer is supported by the provided knowledge sources.

Evaluate confidence using these criteria (in order of importance):
1. **Source coverage:** Does the answer address the question using \
content from the provided sources? Are the key claims backed by source \
references?
2. **Source agreement:** Do the sources agree with each other, or are \
there conflicts? If conflicts exist, were they acknowledged and resolved \
using the correct priority hierarchy (Errata > FAQ > Core Rules > Other)?
3. **Directness of support:** Is the answer directly stated in the \
sources, or did it require inference, interpretation, or synthesis \
across multiple sources?
4. **Completeness:** Does the answer fully address the question, or \
does it only cover part of what was asked?
5. **Source authority:** Are the supporting sources authoritative \
(official rules, errata, FAQ) or lower-authority (community guides, \
strategy content)?

Respond with a JSON object containing exactly these fields:
- "score": A float between 0.0 and 1.0 representing confidence.
- "justification": A 1-2 sentence explanation of the primary factor \
driving the score.

Scoring calibration:
- 0.9-1.0: Answer is directly and explicitly stated in one or more \
authoritative sources (core rules, errata, FAQ). No inference needed.
- 0.7-0.89: Answer is well-supported but required minor interpretation \
or synthesis across sources. Or: strongly supported by a single source.
- 0.5-0.69: Answer required meaningful inference, or sources partially \
conflict (but conflicts were resolved), or only non-authoritative \
sources support the answer.
- 0.3-0.49: Answer has weak source support, or significant inference \
was required, or the answer acknowledges gaps in available knowledge.
- 0.0-0.29: Answer is largely unsupported by the provided sources, or \
the system acknowledged it could not answer the question.

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
    list showing content excerpts, similarity scores, document type,
    and authority status.

    Args:
        ctx: Pipeline context with retrieval and traversal results.

    Returns:
        Formatted sources section or 'No sources available'.
    """
    lines: list[str] = []

    for i, result in enumerate(ctx.retrieval_results, start=1):
        excerpt = result.content[:200]
        doc_type = getattr(result, "source", {}).get("document_type", "unknown")
        authority = (
            ", authoritative" if getattr(result, "source", {}).get("is_authoritative") else ""
        )
        lines.append(f"{i}. [score={result.score:.2f}, type={doc_type}{authority}] {excerpt}")

    offset = len(ctx.retrieval_results)
    for i, traversal in enumerate(ctx.traversal_results, start=offset + 1):
        excerpt = traversal.content[:200]
        lines.append(f"{i}. [traversal, depth={traversal.depth}] {excerpt}")

    if not lines:
        return "No sources available."

    return "\n".join(lines)

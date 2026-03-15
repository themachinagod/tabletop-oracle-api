"""LLM-driven concept extraction from document chunks.

Extracts meaningful concepts (game mechanics, components, terms, etc.)
from individual document chunks using an LLM. Concepts emerge from
content — semantic types are not predefined.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tabletop_oracle.models.enums import ModelCapability

if TYPE_CHECKING:
    from tabletop_oracle.services.ingestion.models import Chunk
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.model.token_usage import TokenAttribution

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = """\
You are a knowledge extraction specialist for tabletop board games.

## Game Context
- **Game:** {game_name}
- **Description:** {game_description}

## Task
Analyse the following text chunk from a game document. Identify all \
meaningful concepts — game mechanics, components, rules, conditions, \
phases, player actions, victory conditions, thematic elements, or any \
other significant term or idea.

For each concept:
1. **name**: The canonical name (lowercase, singular form preferred).
2. **semantic_type**: A descriptive category that emerges from the \
content (e.g., "game_mechanic", "component", "phase", "condition", \
"player_action", "resource", "victory_condition"). Do NOT force \
predefined categories — let the type describe what the concept \
actually is.
3. **description**: A one-sentence description of the concept in the \
context of this game.
4. **source_text**: The exact sentence or phrase from the chunk that \
defines or mentions this concept. Copy verbatim.
5. **aliases**: Alternative names, abbreviations, or phrasings that \
refer to the same concept. Empty list if none.

{existing_concepts_section}

## Chunk Text
```
{chunk_text}
```

## Output Format
Return a JSON array of objects. Each object must have exactly these \
fields: "name", "semantic_type", "description", "source_text", "aliases".

If no meaningful concepts are found, return an empty array: []

Return ONLY the JSON array — no markdown fencing, no explanation."""

_EXISTING_CONCEPTS_SECTION = """\
## Existing Concepts (for deduplication)
The following concepts have already been extracted. Avoid creating \
duplicates. If you find a concept that matches an existing one \
(same meaning, different phrasing), list the new phrasing as an \
alias instead of creating a new concept.

Existing: {existing_concepts}"""


@dataclass(frozen=True)
class GameContext:
    """Game metadata provided to the extraction prompt.

    Attributes:
        game_name: Display name of the game.
        game_description: Brief description of the game for LLM context.
    """

    game_name: str
    game_description: str


@dataclass(frozen=True)
class ExtractedConcept:
    """A concept extracted from a single document chunk.

    Immutable value object carrying extraction results before
    persistence. The ``source_chunk_id`` and ``source_text`` fields
    provide full traceability back to the originating chunk.

    Attributes:
        name: Canonical concept name.
        semantic_type: Content-derived type (not from a fixed enum).
        description: Brief description in game context.
        source_chunk_id: UUID of the chunk this concept was extracted from.
        source_text: Verbatim text excerpt from the chunk.
        aliases: Alternative names or phrasings for this concept.
    """

    name: str
    semantic_type: str
    description: str
    source_chunk_id: UUID
    source_text: str
    aliases: list[str] = field(default_factory=list)


class ConceptExtractionService:
    """Extracts concepts from document chunks using an LLM.

    Processes chunks individually for precise source traceability.
    Uses the ``concept_extraction`` model capability slot and delegates
    retry/fallback to ``ModelClient``.

    Args:
        model_client: Provider-agnostic LLM client.
        prompt_template: Optional custom prompt template. Must contain
            ``{game_name}``, ``{game_description}``,
            ``{existing_concepts_section}``, and ``{chunk_text}``
            placeholders.
    """

    def __init__(
        self,
        model_client: ModelClient,
        prompt_template: str | None = None,
    ) -> None:
        self._model_client = model_client
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE

    async def extract_from_chunk(
        self,
        chunk: Chunk,
        game_context: GameContext,
        attribution: TokenAttribution,
        existing_concepts: list[str] | None = None,
    ) -> list[ExtractedConcept]:
        """Extract concepts from a single document chunk.

        Builds an extraction prompt with game context and existing
        concept names, sends it to the LLM, and parses the structured
        JSON response into ``ExtractedConcept`` instances.

        Args:
            chunk: The document chunk to extract concepts from.
            game_context: Game name and description for prompt context.
            attribution: Token usage attribution for this call.
            existing_concepts: Names of already-extracted concepts for
                deduplication awareness. The LLM uses these to avoid
                creating duplicates.

        Returns:
            List of extracted concepts. Empty list if the chunk contains
            no meaningful concepts or if the LLM response is malformed.
        """
        if not chunk.content or not chunk.content.strip():
            return []

        prompt = self._build_prompt(chunk, game_context, existing_concepts)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        result = await self._model_client.complete(
            capability=ModelCapability.CONCEPT_EXTRACTION,
            messages=messages,
            attribution=attribution,
        )

        return self._parse_response(result.content, chunk)

    def _build_prompt(
        self,
        chunk: Chunk,
        game_context: GameContext,
        existing_concepts: list[str] | None,
    ) -> str:
        """Build the extraction prompt from template and context.

        Args:
            chunk: Source chunk providing the text to analyse.
            game_context: Game metadata for the prompt.
            existing_concepts: Known concept names for dedup awareness.

        Returns:
            Fully rendered prompt string.
        """
        if existing_concepts:
            existing_section = _EXISTING_CONCEPTS_SECTION.format(
                existing_concepts=", ".join(existing_concepts),
            )
        else:
            existing_section = ""

        return self._prompt_template.format(
            game_name=game_context.game_name,
            game_description=game_context.game_description,
            existing_concepts_section=existing_section,
            chunk_text=chunk.content,
        )

    def _parse_response(
        self,
        content: str,
        chunk: Chunk,
    ) -> list[ExtractedConcept]:
        """Parse LLM JSON response into ExtractedConcept instances.

        Handles common LLM output quirks: markdown code fencing,
        trailing commas, and partial/malformed entries. Individual
        concept entries that fail validation are skipped rather than
        failing the entire extraction.

        Args:
            content: Raw LLM response text.
            chunk: Source chunk for traceability metadata.

        Returns:
            List of validated ExtractedConcept instances.
        """
        cleaned = self._strip_markdown_fencing(content.strip())

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM extraction response as JSON",
                extra={"content_preview": cleaned[:200]},
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "LLM extraction response is not a JSON array",
                extra={"type": type(parsed).__name__},
            )
            return []

        # Need a stable chunk_id — use a deterministic UUID from chunk
        # metadata if available, otherwise generate from content hash.
        chunk_id = self._resolve_chunk_id(chunk)

        concepts: list[ExtractedConcept] = []
        for idx, entry in enumerate(parsed):
            concept = self._validate_entry(entry, chunk_id, idx)
            if concept is not None:
                concepts.append(concept)

        logger.info(
            "Extracted %d concepts from chunk (index=%d)",
            len(concepts),
            chunk.chunk_index,
        )
        return concepts

    def _validate_entry(
        self,
        entry: Any,
        chunk_id: UUID,
        index: int,
    ) -> ExtractedConcept | None:
        """Validate and convert a single JSON entry to ExtractedConcept.

        Args:
            entry: A single entry from the parsed JSON array.
            chunk_id: UUID of the source chunk.
            index: Position in the array (for logging).

        Returns:
            An ExtractedConcept if valid, None otherwise.
        """
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry at index %d", index)
            return None

        required_fields = ("name", "semantic_type", "description", "source_text")
        for field_name in required_fields:
            if field_name not in entry or not isinstance(entry[field_name], str):
                logger.warning(
                    "Skipping entry at index %d: missing or invalid '%s'",
                    index,
                    field_name,
                )
                return None
            if not entry[field_name].strip():
                logger.warning(
                    "Skipping entry at index %d: empty '%s'",
                    index,
                    field_name,
                )
                return None

        # Aliases are optional — default to empty list
        raw_aliases = entry.get("aliases", [])
        if not isinstance(raw_aliases, list):
            raw_aliases = []
        aliases = [str(a).strip() for a in raw_aliases if isinstance(a, str) and a.strip()]

        return ExtractedConcept(
            name=entry["name"].strip(),
            semantic_type=entry["semantic_type"].strip(),
            description=entry["description"].strip(),
            source_chunk_id=chunk_id,
            source_text=entry["source_text"].strip(),
            aliases=aliases,
        )

    @staticmethod
    def _strip_markdown_fencing(text: str) -> str:
        """Remove markdown code fences if present.

        LLMs sometimes wrap JSON in ```json ... ``` despite instructions
        not to. This strips that wrapping.

        Args:
            text: Raw LLM output.

        Returns:
            Text with markdown fences removed.
        """
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()
        return text

    @staticmethod
    def _resolve_chunk_id(chunk: Chunk) -> UUID:
        """Resolve a UUID for the chunk.

        Uses the chunk's metadata ``chunk_db_id`` if available (set
        after persistence), otherwise generates a deterministic UUID
        from chunk index and content hash.

        Args:
            chunk: The source chunk.

        Returns:
            A UUID identifying this chunk.
        """
        from uuid import NAMESPACE_DNS, uuid5

        db_id = chunk.metadata.get("chunk_db_id")
        if db_id is not None:
            if isinstance(db_id, UUID):
                return db_id
            return UUID(str(db_id))

        # Deterministic fallback from content
        seed = f"chunk:{chunk.chunk_index}:{hash(chunk.content)}"
        return uuid5(NAMESPACE_DNS, seed)

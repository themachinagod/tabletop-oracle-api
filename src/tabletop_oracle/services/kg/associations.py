"""LLM-driven association discovery between concepts.

Discovers relationships between concepts extracted from document chunks
using an LLM. Supports both intra-document (between concepts from the
same document) and cross-document (between new concepts and existing
KG concepts) association discovery.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tabletop_oracle.models.enums import ModelCapability

if TYPE_CHECKING:
    from tabletop_oracle.services.ingestion.models import Chunk
    from tabletop_oracle.services.kg.extraction import ExtractedConcept
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.model.token_usage import TokenAttribution

logger = logging.getLogger(__name__)

# Maximum number of existing concepts to include in a single LLM prompt.
# Prevents context window overflow for games with large knowledge graphs.
_MAX_EXISTING_CONCEPTS_PER_PROMPT = 200

_DEFAULT_PROMPT_TEMPLATE = """\
You are a knowledge graph specialist for tabletop board games.

## Task
Analyse the following text chunk and the concepts extracted from it. \
Identify all meaningful relationships (associations) between the concepts.

Consider two kinds of relationships:
1. **Intra-document**: Relationships between the newly extracted concepts.
2. **Cross-document**: Relationships between newly extracted concepts and \
existing concepts already in the knowledge graph.

For each relationship:
1. **source_concept_name**: The name of the source concept.
2. **target_concept_name**: The name of the target concept.
3. **relationship_label**: A semantic label describing the relationship \
(e.g., "modifies", "requires", "part_of", "overrides", "supersedes", \
"clarifies", "contains", "enables", "restricts"). Do NOT use a fixed \
set — let the label emerge from the content.
4. **description**: A one-sentence description of the relationship.
5. **source_text**: The exact sentence or phrase from the chunk text \
that evidences this relationship. Copy verbatim.

## Newly Extracted Concepts
{new_concepts_section}

{existing_concepts_section}

## Chunk Text
```
{chunk_text}
```

## Output Format
Return a JSON array of objects. Each object must have exactly these \
fields: "source_concept_name", "target_concept_name", \
"relationship_label", "description", "source_text".

If no meaningful relationships are found, return an empty array: []

Return ONLY the JSON array — no markdown fencing, no explanation."""

_NEW_CONCEPTS_TEMPLATE = """\
{concepts_list}"""

_EXISTING_CONCEPTS_SECTION = """\
## Existing Knowledge Graph Concepts
The following concepts already exist in the game's knowledge graph. \
Look for relationships between the newly extracted concepts and these \
existing concepts.

{existing_concepts_list}"""


@dataclass(frozen=True)
class ConceptSummary:
    """Summary of an existing concept in the knowledge graph.

    Lightweight representation used to provide context to the LLM
    without including full concept details.

    Attributes:
        name: Canonical concept name.
        semantic_type: Content-derived type (e.g., "game_mechanic").
        description: Brief description in game context.
    """

    name: str
    semantic_type: str
    description: str


@dataclass(frozen=True)
class DiscoveredAssociation:
    """An association discovered between two concepts.

    Immutable value object carrying discovery results before persistence.
    The ``source_chunk_id`` and ``source_text`` fields provide full
    traceability back to the originating chunk.

    Attributes:
        source_concept_name: Name of the source concept.
        target_concept_name: Name of the target concept.
        relationship_label: Semantic label for the relationship
            (e.g., "modifies", "requires", "overrides", "contains").
        description: Brief description of the relationship.
        source_chunk_id: UUID of the chunk where this association was discovered.
        source_text: Text excerpt evidencing the association.
    """

    source_concept_name: str
    target_concept_name: str
    relationship_label: str
    description: str
    source_chunk_id: UUID
    source_text: str


class AssociationDiscoveryService:
    """Discovers relationships between concepts using an LLM.

    Operates in two modes:
    1. Intra-document: Associations between concepts extracted from the
       same document. Run with an empty ``existing_concepts`` list.
    2. Cross-document: Associations between newly extracted concepts and
       existing concepts in the game's KG. Provides existing concept
       summaries (paginated if large) to the LLM for cross-linking.

    Args:
        model_client: Provider-agnostic LLM client.
        prompt_template: Optional custom prompt template. Must contain
            ``{new_concepts_section}``, ``{existing_concepts_section}``,
            and ``{chunk_text}`` placeholders.
        max_existing_concepts: Maximum existing concepts per prompt.
            Defaults to ``_MAX_EXISTING_CONCEPTS_PER_PROMPT``.
    """

    def __init__(
        self,
        model_client: ModelClient,
        prompt_template: str | None = None,
        max_existing_concepts: int = _MAX_EXISTING_CONCEPTS_PER_PROMPT,
    ) -> None:
        self._model_client = model_client
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        self._max_existing_concepts = max_existing_concepts

    async def discover_associations(
        self,
        new_concepts: list[ExtractedConcept],
        existing_concepts: list[ConceptSummary],
        chunk: Chunk,
        attribution: TokenAttribution,
    ) -> list[DiscoveredAssociation]:
        """Discover associations between new and existing concepts.

        The LLM receives the chunk text, newly extracted concepts, and
        summaries of existing KG concepts. It identifies relationships
        with semantic labels.

        When ``existing_concepts`` is empty, only intra-document
        associations (between new concepts) are discovered.

        Large existing concept lists are truncated to
        ``max_existing_concepts`` to fit within LLM context windows.

        Args:
            new_concepts: Concepts just extracted from this chunk.
            existing_concepts: Existing KG concepts for cross-document
                association discovery. Empty list for intra-document only.
            chunk: The source document chunk.
            attribution: Token usage attribution for this call.

        Returns:
            List of discovered associations. Empty list if no concepts
            provided, no relationships found, or the LLM response is
            malformed.
        """
        if not new_concepts:
            return []

        if not chunk.content or not chunk.content.strip():
            return []

        prompt = self._build_prompt(new_concepts, existing_concepts, chunk)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        result = await self._model_client.complete(
            capability=ModelCapability.CONCEPT_EXTRACTION,
            messages=messages,
            attribution=attribution,
        )

        chunk_id = self._resolve_chunk_id(chunk)
        return self._parse_response(result.content, chunk_id)

    def _build_prompt(
        self,
        new_concepts: list[ExtractedConcept],
        existing_concepts: list[ConceptSummary],
        chunk: Chunk,
    ) -> str:
        """Build the association discovery prompt from template and context.

        Args:
            new_concepts: Newly extracted concepts to include.
            existing_concepts: Existing KG concepts (will be truncated
                if exceeding ``max_existing_concepts``).
            chunk: Source chunk providing the text to analyse.

        Returns:
            Fully rendered prompt string.
        """
        new_section = self._format_new_concepts(new_concepts)

        if existing_concepts:
            truncated = existing_concepts[: self._max_existing_concepts]
            existing_section = _EXISTING_CONCEPTS_SECTION.format(
                existing_concepts_list=self._format_existing_concepts(truncated),
            )
            if len(existing_concepts) > self._max_existing_concepts:
                existing_section += (
                    f"\n\n(Showing {self._max_existing_concepts} of "
                    f"{len(existing_concepts)} existing concepts)"
                )
        else:
            existing_section = ""

        return self._prompt_template.format(
            new_concepts_section=new_section,
            existing_concepts_section=existing_section,
            chunk_text=chunk.content,
        )

    @staticmethod
    def _format_new_concepts(concepts: list[ExtractedConcept]) -> str:
        """Format newly extracted concepts for the prompt.

        Args:
            concepts: List of extracted concepts.

        Returns:
            Formatted string listing each concept with its type and
            description.
        """
        lines: list[str] = []
        for c in concepts:
            lines.append(f"- **{c.name}** ({c.semantic_type}): {c.description}")
        return "\n".join(lines)

    @staticmethod
    def _format_existing_concepts(concepts: list[ConceptSummary]) -> str:
        """Format existing KG concepts for the prompt.

        Args:
            concepts: List of concept summaries.

        Returns:
            Formatted string listing each concept with its type and
            description.
        """
        lines: list[str] = []
        for c in concepts:
            lines.append(f"- **{c.name}** ({c.semantic_type}): {c.description}")
        return "\n".join(lines)

    def _parse_response(
        self,
        content: str,
        chunk_id: UUID,
    ) -> list[DiscoveredAssociation]:
        """Parse LLM JSON response into DiscoveredAssociation instances.

        Handles common LLM output quirks: markdown code fencing and
        partial/malformed entries. Individual entries that fail validation
        are skipped rather than failing the entire discovery.

        Args:
            content: Raw LLM response text.
            chunk_id: UUID of the source chunk for traceability.

        Returns:
            List of validated DiscoveredAssociation instances.
        """
        cleaned = self._strip_markdown_fencing(content.strip())

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM association response as JSON",
                extra={"content_preview": cleaned[:200]},
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "LLM association response is not a JSON array",
                extra={"type": type(parsed).__name__},
            )
            return []

        associations: list[DiscoveredAssociation] = []
        for idx, entry in enumerate(parsed):
            association = self._validate_entry(entry, chunk_id, idx)
            if association is not None:
                associations.append(association)

        logger.info(
            "Discovered %d associations from chunk",
            len(associations),
        )
        return associations

    @staticmethod
    def _validate_entry(
        entry: Any,
        chunk_id: UUID,
        index: int,
    ) -> DiscoveredAssociation | None:
        """Validate and convert a single JSON entry to DiscoveredAssociation.

        Args:
            entry: A single entry from the parsed JSON array.
            chunk_id: UUID of the source chunk.
            index: Position in the array (for logging).

        Returns:
            A DiscoveredAssociation if valid, None otherwise.
        """
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry at index %d", index)
            return None

        required_fields = (
            "source_concept_name",
            "target_concept_name",
            "relationship_label",
            "description",
            "source_text",
        )
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

        return DiscoveredAssociation(
            source_concept_name=entry["source_concept_name"].strip(),
            target_concept_name=entry["target_concept_name"].strip(),
            relationship_label=entry["relationship_label"].strip(),
            description=entry["description"].strip(),
            source_chunk_id=chunk_id,
            source_text=entry["source_text"].strip(),
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
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
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

        seed = f"chunk:{chunk.chunk_index}:{hash(chunk.content)}"
        return uuid5(NAMESPACE_DNS, seed)

"""Unit tests for AssociationDiscoveryService.

Tests LLM-driven association discovery with a mocked ModelClient.
Covers successful discovery, intra-document and cross-document modes,
field validation, malformed responses, context truncation, prompt
construction, and source traceability.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ingestion.models import Chunk
from tabletop_oracle.services.kg.associations import (
    AssociationDiscoveryService,
    ConceptSummary,
    DiscoveredAssociation,
)
from tabletop_oracle.services.kg.extraction import ExtractedConcept
from tabletop_oracle.services.model.token_usage import TokenAttribution
from tabletop_oracle.services.model.types import CompletionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    content: str = "The robber blocks resource production on the hex.",
    chunk_index: int = 0,
    chunk_db_id: UUID | None = None,
) -> Chunk:
    """Build a Chunk for testing."""
    metadata: dict = {}
    if chunk_db_id is not None:
        metadata["chunk_db_id"] = chunk_db_id
    return Chunk(
        chunk_index=chunk_index,
        chunk_type="text",
        content=content,
        section_path="Rules > Robber",
        heading="Robber",
        page_number=5,
        token_estimate=12,
        metadata=metadata,
    )


def _make_attribution() -> TokenAttribution:
    """Build a TokenAttribution for testing."""
    return TokenAttribution(
        user_id=uuid4(),
        document_id=uuid4(),
    )


def _make_completion(content: str) -> CompletionResult:
    """Build a CompletionResult with the given content."""
    return CompletionResult(
        content=content,
        input_tokens=150,
        output_tokens=80,
        model_used="openai/gpt-4o",
        is_fallback=False,
    )


def _make_extracted_concept(
    *,
    name: str = "robber",
    semantic_type: str = "game_mechanic",
    description: str = "Token that blocks resource production.",
    source_text: str = "The robber blocks resource production on the hex.",
    chunk_id: UUID | None = None,
) -> ExtractedConcept:
    """Build an ExtractedConcept for testing."""
    return ExtractedConcept(
        name=name,
        semantic_type=semantic_type,
        description=description,
        source_chunk_id=chunk_id or uuid4(),
        source_text=source_text,
        aliases=[],
    )


def _make_concept_summary(
    *,
    name: str = "settlement",
    semantic_type: str = "component",
    description: str = "Building placed at intersections.",
) -> ConceptSummary:
    """Build a ConceptSummary for testing."""
    return ConceptSummary(
        name=name,
        semantic_type=semantic_type,
        description=description,
    )


def _make_service(
    complete_return: CompletionResult | None = None,
    prompt_template: str | None = None,
    max_existing_concepts: int = 200,
) -> tuple[AssociationDiscoveryService, AsyncMock]:
    """Build an AssociationDiscoveryService with a mocked ModelClient."""
    mock_client = AsyncMock()
    if complete_return is not None:
        mock_client.complete.return_value = complete_return
    service = AssociationDiscoveryService(
        model_client=mock_client,
        prompt_template=prompt_template,
        max_existing_concepts=max_existing_concepts,
    )
    return service, mock_client


def _valid_associations_json(
    associations: list[dict] | None = None,
) -> str:
    """Return a valid JSON string of discovered associations."""
    if associations is None:
        associations = [
            {
                "source_concept_name": "robber",
                "target_concept_name": "hex",
                "relationship_label": "blocks",
                "description": "The robber blocks resource production on a hex.",
                "source_text": "The robber blocks resource production on the hex.",
            },
            {
                "source_concept_name": "robber",
                "target_concept_name": "resource",
                "relationship_label": "prevents_collection",
                "description": "The robber prevents collection of resources.",
                "source_text": "The robber blocks resource production on the hex.",
            },
        ]
    return json.dumps(associations)


# ---------------------------------------------------------------------------
# Tests — Successful discovery
# ---------------------------------------------------------------------------


class TestDiscoverAssociationsSuccess:
    """Tests for successful association discovery."""

    @pytest.mark.asyncio
    async def test_returns_list_of_discovered_associations(self) -> None:
        """discover_associations returns a list of DiscoveredAssociation."""
        service, _ = _make_service(_make_completion(_valid_associations_json()))
        chunk = _make_chunk()
        new_concepts = [
            _make_extracted_concept(name="robber"),
            _make_extracted_concept(name="hex", semantic_type="component"),
        ]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(a, DiscoveredAssociation) for a in result)

    @pytest.mark.asyncio
    async def test_association_fields_set_correctly(self) -> None:
        """Discovered associations have correct field values."""
        chunk_id = uuid4()
        service, _ = _make_service(_make_completion(_valid_associations_json()))
        chunk = _make_chunk(chunk_db_id=chunk_id)
        new_concepts = [_make_extracted_concept(name="robber")]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assoc = result[0]
        assert assoc.source_concept_name == "robber"
        assert assoc.target_concept_name == "hex"
        assert assoc.relationship_label == "blocks"
        assert assoc.description == "The robber blocks resource production on a hex."
        assert assoc.source_text == "The robber blocks resource production on the hex."
        assert assoc.source_chunk_id == chunk_id

    @pytest.mark.asyncio
    async def test_relationship_labels_discovered_from_content(self) -> None:
        """Relationship labels come from the LLM, not a fixed set."""
        custom_associations = [
            {
                "source_concept_name": "knight",
                "target_concept_name": "robber",
                "relationship_label": "displaces",
                "description": "Playing a knight card displaces the robber.",
                "source_text": "Play a knight to move the robber.",
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(custom_associations)))
        chunk = _make_chunk(content="Play a knight to move the robber.")
        new_concepts = [_make_extracted_concept(name="knight")]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result[0].relationship_label == "displaces"


# ---------------------------------------------------------------------------
# Tests — Intra-document associations
# ---------------------------------------------------------------------------


class TestIntraDocumentAssociations:
    """Tests for intra-document mode (no existing concepts)."""

    @pytest.mark.asyncio
    async def test_intra_document_with_empty_existing(self) -> None:
        """Empty existing_concepts list discovers intra-document associations only."""
        service, mock_client = _make_service(_make_completion(_valid_associations_json()))
        chunk = _make_chunk()
        new_concepts = [
            _make_extracted_concept(name="robber"),
            _make_extracted_concept(name="hex"),
        ]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert len(result) == 2

        # Verify no existing concepts section in prompt
        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "Existing Knowledge Graph" not in prompt_text


# ---------------------------------------------------------------------------
# Tests — Cross-document associations
# ---------------------------------------------------------------------------


class TestCrossDocumentAssociations:
    """Tests for cross-document mode (new + existing concepts)."""

    @pytest.mark.asyncio
    async def test_cross_document_with_existing_concepts(self) -> None:
        """Existing concepts are included in the prompt for cross-document discovery."""
        service, mock_client = _make_service(_make_completion(_valid_associations_json()))
        chunk = _make_chunk()
        new_concepts = [_make_extracted_concept(name="robber")]
        existing = [
            _make_concept_summary(name="settlement"),
            _make_concept_summary(name="road", description="Path between nodes."),
        ]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=existing,
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert len(result) == 2

        # Verify existing concepts appear in prompt
        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "Existing Knowledge Graph Concepts" in prompt_text
        assert "settlement" in prompt_text
        assert "road" in prompt_text


# ---------------------------------------------------------------------------
# Tests — Empty / no-concept cases
# ---------------------------------------------------------------------------


class TestDiscoverAssociationsEmpty:
    """Tests for empty or trivial input."""

    @pytest.mark.asyncio
    async def test_empty_new_concepts_returns_empty_list(self) -> None:
        """Empty new_concepts list returns empty without calling LLM."""
        service, mock_client = _make_service()
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_chunk_content_returns_empty_list(self) -> None:
        """Empty chunk content returns empty without calling LLM."""
        service, mock_client = _make_service()
        chunk = _make_chunk(content="")
        new_concepts = [_make_extracted_concept()]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_chunk_returns_empty_list(self) -> None:
        """Whitespace-only chunk returns empty without calling LLM."""
        service, mock_client = _make_service()
        chunk = _make_chunk(content="   \n\t  ")
        new_concepts = [_make_extracted_concept()]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_returns_empty_array(self) -> None:
        """LLM returning [] produces an empty result list."""
        service, _ = _make_service(_make_completion("[]"))
        chunk = _make_chunk()
        new_concepts = [_make_extracted_concept()]

        result = await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []


# ---------------------------------------------------------------------------
# Tests — Malformed LLM responses
# ---------------------------------------------------------------------------


class TestDiscoverAssociationsMalformedResponse:
    """Tests for graceful handling of malformed LLM output."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty_list(self) -> None:
        """Completely invalid JSON returns an empty list."""
        service, _ = _make_service(_make_completion("not json at all"))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_json_object_instead_of_array_returns_empty(self) -> None:
        """A JSON object (not array) returns an empty list."""
        service, _ = _make_service(_make_completion('{"source_concept_name": "robber"}'))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_entry_missing_required_field_skipped(self) -> None:
        """Entries missing required fields are skipped; valid ones kept."""
        associations = [
            {
                "source_concept_name": "robber",
                "target_concept_name": "hex",
                "relationship_label": "blocks",
                "description": "Blocks production.",
                "source_text": "The robber blocks.",
            },
            {
                "source_concept_name": "broken",
                # missing target_concept_name
                "relationship_label": "unknown",
                "description": "Incomplete.",
                "source_text": "Something.",
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(associations)))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert len(result) == 1
        assert result[0].source_concept_name == "robber"

    @pytest.mark.asyncio
    async def test_entry_with_empty_required_field_skipped(self) -> None:
        """Entries with empty string required fields are skipped."""
        associations = [
            {
                "source_concept_name": "",
                "target_concept_name": "hex",
                "relationship_label": "blocks",
                "description": "Desc.",
                "source_text": "Text.",
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(associations)))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_non_dict_entries_skipped(self) -> None:
        """Non-dict entries in the array are skipped."""
        payload = json.dumps(["not a dict", 42, None])
        service, _ = _make_service(_make_completion(payload))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_parsed(self) -> None:
        """JSON wrapped in markdown code fences is parsed correctly."""
        raw = "```json\n" + _valid_associations_json() + "\n```"
        service, _ = _make_service(_make_completion(raw))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests — Large concept list truncation
# ---------------------------------------------------------------------------


class TestLargeConceptListTruncation:
    """Tests for pagination/truncation of large existing concept lists."""

    @pytest.mark.asyncio
    async def test_existing_concepts_truncated_to_max(self) -> None:
        """Large existing concept lists are truncated to max_existing_concepts."""
        max_concepts = 5
        service, mock_client = _make_service(
            _make_completion("[]"),
            max_existing_concepts=max_concepts,
        )
        chunk = _make_chunk()
        new_concepts = [_make_extracted_concept()]

        # Create more concepts than the limit
        existing = [
            _make_concept_summary(
                name=f"concept_{i}",
                description=f"Description {i}.",
            )
            for i in range(20)
        ]

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=existing,
            chunk=chunk,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]

        # First 5 concepts should be present
        for i in range(max_concepts):
            assert f"concept_{i}" in prompt_text

        # Concepts beyond the limit should NOT be present
        assert "concept_10" not in prompt_text
        assert "concept_19" not in prompt_text

        # Truncation notice should be present
        assert "Showing 5 of 20" in prompt_text

    @pytest.mark.asyncio
    async def test_no_truncation_notice_when_within_limit(self) -> None:
        """No truncation notice when existing concepts fit within the limit."""
        service, mock_client = _make_service(
            _make_completion("[]"),
            max_existing_concepts=200,
        )
        chunk = _make_chunk()
        new_concepts = [_make_extracted_concept()]
        existing = [_make_concept_summary(name="settlement")]

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=existing,
            chunk=chunk,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "Showing" not in prompt_text


# ---------------------------------------------------------------------------
# Tests — Source traceability
# ---------------------------------------------------------------------------


class TestSourceTraceability:
    """Tests for source traceability fields on discovered associations."""

    @pytest.mark.asyncio
    async def test_source_chunk_id_from_metadata(self) -> None:
        """source_chunk_id uses chunk_db_id from metadata when available."""
        expected_id = uuid4()
        service, _ = _make_service(_make_completion(_valid_associations_json()))
        chunk = _make_chunk(chunk_db_id=expected_id)

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        for assoc in result:
            assert assoc.source_chunk_id == expected_id

    @pytest.mark.asyncio
    async def test_source_chunk_id_deterministic_fallback(self) -> None:
        """source_chunk_id is deterministic when no chunk_db_id in metadata."""
        service, _ = _make_service(_make_completion(_valid_associations_json()))
        chunk = _make_chunk(content="Some text")

        result1 = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )
        result2 = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result1[0].source_chunk_id == result2[0].source_chunk_id

    @pytest.mark.asyncio
    async def test_source_text_preserved_from_llm_response(self) -> None:
        """source_text is taken from the LLM response for traceability."""
        associations = [
            {
                "source_concept_name": "robber",
                "target_concept_name": "hex",
                "relationship_label": "occupies",
                "description": "The robber occupies a hex tile.",
                "source_text": "Place the robber on any hex tile.",
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(associations)))
        chunk = _make_chunk()

        result = await service.discover_associations(
            new_concepts=[_make_extracted_concept()],
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        assert result[0].source_text == "Place the robber on any hex tile."


# ---------------------------------------------------------------------------
# Tests — Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Tests for prompt building and context inclusion."""

    @pytest.mark.asyncio
    async def test_new_concepts_included_in_prompt(self) -> None:
        """New concepts appear in the LLM prompt."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()
        new_concepts = [
            _make_extracted_concept(name="robber", semantic_type="game_mechanic"),
            _make_extracted_concept(name="hex", semantic_type="component"),
        ]

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "robber" in prompt_text
        assert "hex" in prompt_text
        assert "game_mechanic" in prompt_text

    @pytest.mark.asyncio
    async def test_chunk_text_included_in_prompt(self) -> None:
        """Chunk content appears in the prompt."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk(content="Knights can move the robber.")
        new_concepts = [_make_extracted_concept()]

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "Knights can move the robber." in prompt_text

    @pytest.mark.asyncio
    async def test_uses_concept_extraction_capability(self) -> None:
        """The service uses the CONCEPT_EXTRACTION model capability."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()
        new_concepts = [_make_extracted_concept()]

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["capability"] == ModelCapability.CONCEPT_EXTRACTION

    @pytest.mark.asyncio
    async def test_attribution_passed_to_model_client(self) -> None:
        """The attribution is forwarded to the ModelClient."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()
        new_concepts = [_make_extracted_concept()]
        attribution = _make_attribution()

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=attribution,
        )

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["attribution"] is attribution

    @pytest.mark.asyncio
    async def test_custom_prompt_template_used(self) -> None:
        """A custom prompt template replaces the default."""
        custom = "New: {new_concepts_section}. {existing_concepts_section} Text: {chunk_text}"
        service, mock_client = _make_service(
            _make_completion("[]"),
            prompt_template=custom,
        )
        chunk = _make_chunk(content="Test content")
        new_concepts = [_make_extracted_concept(name="robber")]

        await service.discover_associations(
            new_concepts=new_concepts,
            existing_concepts=[],
            chunk=chunk,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert prompt_text.startswith("New: - **robber**")
        assert "Test content" in prompt_text

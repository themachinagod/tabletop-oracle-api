"""Unit tests for ConceptExtractionService.

Tests LLM-driven concept extraction with a mocked ModelClient.
Covers successful extraction, field validation, alias handling,
edge cases, malformed responses, and prompt construction.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ingestion.models import Chunk
from tabletop_oracle.services.kg.extraction import (
    ConceptExtractionService,
    ExtractedConcept,
    GameContext,
)
from tabletop_oracle.services.model.token_usage import TokenAttribution
from tabletop_oracle.services.model.types import CompletionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GAME_CONTEXT = GameContext(
    game_name="Catan",
    game_description="A resource trading and settlement building game.",
)


def _make_chunk(
    *,
    content: str = "Players collect resources by rolling dice.",
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
        section_path="Rules > Resources",
        heading="Resources",
        page_number=3,
        token_estimate=10,
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
        input_tokens=100,
        output_tokens=50,
        model_used="openai/gpt-4o",
        is_fallback=False,
    )


def _make_service(
    complete_return: CompletionResult | None = None,
    prompt_template: str | None = None,
) -> tuple[ConceptExtractionService, AsyncMock]:
    """Build a ConceptExtractionService with a mocked ModelClient."""
    mock_client = AsyncMock()
    if complete_return is not None:
        mock_client.complete.return_value = complete_return
    service = ConceptExtractionService(
        model_client=mock_client,
        prompt_template=prompt_template,
    )
    return service, mock_client


def _valid_concepts_json(
    concepts: list[dict] | None = None,
) -> str:
    """Return a valid JSON string of extracted concepts."""
    if concepts is None:
        concepts = [
            {
                "name": "resource",
                "semantic_type": "game_mechanic",
                "description": "Materials collected by players to build.",
                "source_text": "Players collect resources by rolling dice.",
                "aliases": ["material", "goods"],
            },
            {
                "name": "dice roll",
                "semantic_type": "player_action",
                "description": "Rolling dice to determine resource production.",
                "source_text": "Players collect resources by rolling dice.",
                "aliases": [],
            },
        ]
    return json.dumps(concepts)


# ---------------------------------------------------------------------------
# Tests — Successful extraction
# ---------------------------------------------------------------------------


class TestExtractFromChunkSuccess:
    """Tests for successful concept extraction."""

    @pytest.mark.asyncio
    async def test_returns_list_of_extracted_concepts(self) -> None:
        """extract_from_chunk returns a list of ExtractedConcept."""
        service, _ = _make_service(_make_completion(_valid_concepts_json()))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(c, ExtractedConcept) for c in result)

    @pytest.mark.asyncio
    async def test_concept_fields_set_correctly(self) -> None:
        """Extracted concepts have correct field values."""
        service, _ = _make_service(_make_completion(_valid_concepts_json()))
        chunk_id = uuid4()
        chunk = _make_chunk(chunk_db_id=chunk_id)

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        concept = result[0]
        assert concept.name == "resource"
        assert concept.semantic_type == "game_mechanic"
        assert concept.description == "Materials collected by players to build."
        assert concept.source_text == "Players collect resources by rolling dice."
        assert concept.source_chunk_id == chunk_id

    @pytest.mark.asyncio
    async def test_aliases_extracted_correctly(self) -> None:
        """Aliases are extracted as a list of strings."""
        service, _ = _make_service(_make_completion(_valid_concepts_json()))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result[0].aliases == ["material", "goods"]
        assert result[1].aliases == []

    @pytest.mark.asyncio
    async def test_source_chunk_id_from_metadata(self) -> None:
        """source_chunk_id uses chunk_db_id from metadata when available."""
        expected_id = uuid4()
        service, _ = _make_service(_make_completion(_valid_concepts_json()))
        chunk = _make_chunk(chunk_db_id=expected_id)

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        for concept in result:
            assert concept.source_chunk_id == expected_id

    @pytest.mark.asyncio
    async def test_source_chunk_id_deterministic_fallback(self) -> None:
        """source_chunk_id is deterministic when no chunk_db_id in metadata."""
        service, _ = _make_service(_make_completion(_valid_concepts_json()))
        chunk = _make_chunk(content="Some text")

        result1 = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )
        result2 = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result1[0].source_chunk_id == result2[0].source_chunk_id


# ---------------------------------------------------------------------------
# Tests — Empty / no-concept cases
# ---------------------------------------------------------------------------


class TestExtractFromChunkEmptyInput:
    """Tests for empty or trivial input."""

    @pytest.mark.asyncio
    async def test_empty_chunk_returns_empty_list(self) -> None:
        """An empty chunk content returns an empty list without calling LLM."""
        service, mock_client = _make_service()
        chunk = _make_chunk(content="")

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_chunk_returns_empty_list(self) -> None:
        """A whitespace-only chunk returns an empty list without calling LLM."""
        service, mock_client = _make_service()
        chunk = _make_chunk(content="   \n\t  ")

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_returns_empty_array(self) -> None:
        """LLM returning [] produces an empty result list."""
        service, _ = _make_service(_make_completion("[]"))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []


# ---------------------------------------------------------------------------
# Tests — Malformed LLM responses
# ---------------------------------------------------------------------------


class TestExtractFromChunkMalformedResponse:
    """Tests for graceful handling of malformed LLM output."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty_list(self) -> None:
        """Completely invalid JSON returns an empty list."""
        service, _ = _make_service(_make_completion("not json at all"))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_json_object_instead_of_array_returns_empty(self) -> None:
        """A JSON object (not array) returns an empty list."""
        service, _ = _make_service(
            _make_completion('{"name": "resource", "semantic_type": "thing"}')
        )
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_entry_missing_required_field_skipped(self) -> None:
        """Entries missing required fields are skipped; valid ones kept."""
        concepts = [
            {
                "name": "resource",
                "semantic_type": "game_mechanic",
                "description": "A resource.",
                "source_text": "Collect resources.",
                "aliases": [],
            },
            {
                "name": "broken",
                # missing semantic_type
                "description": "Incomplete.",
                "source_text": "Something.",
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(concepts)))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert len(result) == 1
        assert result[0].name == "resource"

    @pytest.mark.asyncio
    async def test_entry_with_empty_required_field_skipped(self) -> None:
        """Entries with empty string required fields are skipped."""
        concepts = [
            {
                "name": "",
                "semantic_type": "type",
                "description": "Desc.",
                "source_text": "Text.",
                "aliases": [],
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(concepts)))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_non_dict_entries_skipped(self) -> None:
        """Non-dict entries in the array are skipped."""
        payload = json.dumps(["not a dict", 42, None])
        service, _ = _make_service(_make_completion(payload))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_parsed(self) -> None:
        """JSON wrapped in markdown code fences is parsed correctly."""
        raw = "```json\n" + _valid_concepts_json() + "\n```"
        service, _ = _make_service(_make_completion(raw))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_non_list_aliases_treated_as_empty(self) -> None:
        """Non-list aliases field defaults to empty list."""
        concepts = [
            {
                "name": "resource",
                "semantic_type": "type",
                "description": "Desc.",
                "source_text": "Text.",
                "aliases": "not a list",
            },
        ]
        service, _ = _make_service(_make_completion(json.dumps(concepts)))
        chunk = _make_chunk()

        result = await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        assert len(result) == 1
        assert result[0].aliases == []


# ---------------------------------------------------------------------------
# Tests — Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Tests for prompt building and context inclusion."""

    @pytest.mark.asyncio
    async def test_existing_concepts_included_in_prompt(self) -> None:
        """existing_concepts list appears in the LLM prompt."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()
        existing = ["resource", "settlement", "road"]

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
            existing_concepts=existing,
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "resource, settlement, road" in prompt_text
        assert "Existing Concepts" in prompt_text

    @pytest.mark.asyncio
    async def test_no_existing_concepts_section_when_none(self) -> None:
        """No existing concepts section when list is None."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
            existing_concepts=None,
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "Existing Concepts" not in prompt_text

    @pytest.mark.asyncio
    async def test_game_context_included_in_prompt(self) -> None:
        """Game name and description appear in the prompt."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "Catan" in prompt_text
        assert "resource trading and settlement building" in prompt_text

    @pytest.mark.asyncio
    async def test_chunk_text_included_in_prompt(self) -> None:
        """Chunk content appears in the prompt."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk(content="The robber blocks resource production.")

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "The robber blocks resource production." in prompt_text

    @pytest.mark.asyncio
    async def test_uses_concept_extraction_capability(self) -> None:
        """The service uses the CONCEPT_EXTRACTION model capability."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["capability"] == ModelCapability.CONCEPT_EXTRACTION

    @pytest.mark.asyncio
    async def test_attribution_passed_to_model_client(self) -> None:
        """The attribution is forwarded to the ModelClient."""
        service, mock_client = _make_service(_make_completion("[]"))
        chunk = _make_chunk()
        attribution = _make_attribution()

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=attribution,
        )

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["attribution"] is attribution

    @pytest.mark.asyncio
    async def test_custom_prompt_template_used(self) -> None:
        """A custom prompt template replaces the default."""
        custom = (
            "Game: {game_name}. Desc: {game_description}. "
            "{existing_concepts_section} Text: {chunk_text}"
        )
        service, mock_client = _make_service(
            _make_completion("[]"),
            prompt_template=custom,
        )
        chunk = _make_chunk(content="Test content")

        await service.extract_from_chunk(
            chunk=chunk,
            game_context=_GAME_CONTEXT,
            attribution=_make_attribution(),
        )

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert prompt_text.startswith("Game: Catan.")
        assert "Test content" in prompt_text

"""Unit tests for GraphIntegrationService.

Tests concept integration (new creation, duplicate merging, alias resolution),
association integration (creation, deduplication, unresolvable concepts),
and the concept_map builder.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tabletop_oracle.services.kg.integration import GraphIntegrationService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extracted_concept(
    *,
    name: str = "initiative",
    semantic_type: str = "game_mechanic",
    description: str = "Determines turn order",
    source_chunk_id: None = None,
    source_text: str = "Roll for initiative",
    aliases: list[str] | None = None,
) -> MagicMock:
    """Build a mock ExtractedConcept."""
    ec = MagicMock()
    ec.name = name
    ec.semantic_type = semantic_type
    ec.description = description
    ec.source_chunk_id = source_chunk_id or uuid4()
    ec.source_text = source_text
    ec.aliases = aliases or []
    return ec


def _make_kg_concept(
    *,
    concept_id: None = None,
    name: str = "initiative",
    canonical_id: None = None,
    game_id: None = None,
) -> MagicMock:
    """Build a mock KGConcept."""
    concept = MagicMock()
    concept.id = concept_id or uuid4()
    concept.name = name
    concept.canonical_id = canonical_id
    concept.game_id = game_id or uuid4()
    return concept


def _make_discovered_association(
    *,
    source_name: str = "initiative",
    target_name: str = "combat",
    label: str = "determines",
    description: str = "Initiative determines combat order",
    source_chunk_id: None = None,
    source_text: str = "Initiative determines combat order.",
) -> MagicMock:
    """Build a mock DiscoveredAssociation."""
    assoc = MagicMock()
    assoc.source_concept_name = source_name
    assoc.target_concept_name = target_name
    assoc.relationship_label = label
    assoc.description = description
    assoc.source_chunk_id = source_chunk_id or uuid4()
    assoc.source_text = source_text
    return assoc


def _make_service() -> tuple[GraphIntegrationService, AsyncMock, AsyncMock, AsyncMock]:
    """Create a service with mocked repositories."""
    concept_repo = AsyncMock()
    association_repo = AsyncMock()
    source_repo = AsyncMock()

    service = GraphIntegrationService(
        concept_repo=concept_repo,
        association_repo=association_repo,
        source_repo=source_repo,
    )
    return service, concept_repo, association_repo, source_repo


# ---------------------------------------------------------------------------
# Concept Integration Tests
# ---------------------------------------------------------------------------


class TestIntegrateConcepts:
    """Tests for integrate_concepts."""

    @pytest.mark.asyncio
    async def test_new_concept_created(self) -> None:
        """New concept is created with source reference when not found."""
        service, concept_repo, _, source_repo = _make_service()
        game_id = uuid4()
        doc_id = uuid4()
        version_id = uuid4()

        concept_repo.find_by_name.return_value = None
        created_concept = _make_kg_concept(name="initiative")
        concept_repo.create.return_value = created_concept
        source_repo.create.return_value = MagicMock()

        extracted = _make_extracted_concept()

        result = await service.integrate_concepts(
            game_id=game_id,
            document_id=doc_id,
            version_id=version_id,
            document_type="core_rules",
            document_name="Core Rulebook",
            expansion_id=None,
            extracted_concepts=[extracted],
        )

        assert len(result) == 1
        assert result[0] is created_concept
        concept_repo.create.assert_called_once()
        source_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_concept_merges_source(self) -> None:
        """Existing concept gets new source reference instead of duplicate."""
        service, concept_repo, _, source_repo = _make_service()
        game_id = uuid4()

        existing = _make_kg_concept(name="initiative")
        concept_repo.find_by_name.return_value = existing
        source_repo.find_by_concept_and_chunk.return_value = None
        source_repo.create.return_value = MagicMock()

        extracted = _make_extracted_concept()

        result = await service.integrate_concepts(
            game_id=game_id,
            document_id=uuid4(),
            version_id=uuid4(),
            document_type="core_rules",
            document_name="Core Rulebook",
            expansion_id=None,
            extracted_concepts=[extracted],
        )

        assert len(result) == 1
        assert result[0] is existing
        # Should not create a new concept
        concept_repo.create.assert_not_called()
        # Should add a source
        source_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_source_not_added_twice(self) -> None:
        """Source reference is not duplicated for same concept-chunk pair."""
        service, concept_repo, _, source_repo = _make_service()

        existing = _make_kg_concept(name="initiative")
        concept_repo.find_by_name.return_value = existing
        # Source already exists for this chunk
        source_repo.find_by_concept_and_chunk.return_value = MagicMock()

        extracted = _make_extracted_concept()

        await service.integrate_concepts(
            game_id=uuid4(),
            document_id=uuid4(),
            version_id=uuid4(),
            document_type="core_rules",
            document_name="Core Rulebook",
            expansion_id=None,
            extracted_concepts=[extracted],
        )

        source_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_alias_concepts_created(self) -> None:
        """Alias concepts are created pointing to canonical concept."""
        service, concept_repo, _, source_repo = _make_service()

        concept_repo.find_by_name.return_value = None
        canonical = _make_kg_concept(name="initiative")
        alias_concept = _make_kg_concept(name="init")

        # First call creates canonical, subsequent calls create aliases
        concept_repo.create.side_effect = [canonical, alias_concept]
        source_repo.create.return_value = MagicMock()

        extracted = _make_extracted_concept(aliases=["init"])

        result = await service.integrate_concepts(
            game_id=uuid4(),
            document_id=uuid4(),
            version_id=uuid4(),
            document_type="core_rules",
            document_name="Core Rulebook",
            expansion_id=None,
            extracted_concepts=[extracted],
        )

        assert len(result) == 1
        assert result[0] is canonical
        # concept_repo.create called for canonical + alias
        assert concept_repo.create.call_count == 2
        alias_call_args = concept_repo.create.call_args_list[1]
        alias_obj = alias_call_args[0][0]
        assert alias_obj.canonical_id == canonical.id

    @pytest.mark.asyncio
    async def test_alias_not_created_if_name_exists(self) -> None:
        """Alias concept is skipped if name already exists in the game."""
        service, concept_repo, _, source_repo = _make_service()

        # First find_by_name for main concept returns None (new)
        # Second find_by_name for alias returns existing
        concept_repo.find_by_name.side_effect = [
            None,
            _make_kg_concept(name="init"),
        ]
        canonical = _make_kg_concept(name="initiative")
        concept_repo.create.return_value = canonical
        source_repo.create.return_value = MagicMock()

        extracted = _make_extracted_concept(aliases=["init"])

        await service.integrate_concepts(
            game_id=uuid4(),
            document_id=uuid4(),
            version_id=uuid4(),
            document_type="core_rules",
            document_name="Core Rulebook",
            expansion_id=None,
            extracted_concepts=[extracted],
        )

        # Only the canonical concept should be created
        assert concept_repo.create.call_count == 1

    @pytest.mark.asyncio
    async def test_synonym_resolved_to_canonical(self) -> None:
        """When existing concept is an alias, resolves to canonical."""
        service, concept_repo, _, source_repo = _make_service()

        canonical = _make_kg_concept(name="initiative")
        alias = _make_kg_concept(name="init", canonical_id=canonical.id)

        concept_repo.find_by_name.return_value = alias
        concept_repo.get_by_id.return_value = canonical
        source_repo.find_by_concept_and_chunk.return_value = None
        source_repo.create.return_value = MagicMock()

        extracted = _make_extracted_concept(name="init")

        result = await service.integrate_concepts(
            game_id=uuid4(),
            document_id=uuid4(),
            version_id=uuid4(),
            document_type="core_rules",
            document_name="Core Rulebook",
            expansion_id=None,
            extracted_concepts=[extracted],
        )

        assert result[0] is canonical
        # Source should be added to the canonical concept, not the alias
        source_repo.find_by_concept_and_chunk.assert_called_once_with(
            canonical.id, extracted.source_chunk_id
        )


# ---------------------------------------------------------------------------
# Association Integration Tests
# ---------------------------------------------------------------------------


class TestIntegrateAssociations:
    """Tests for integrate_associations."""

    @pytest.mark.asyncio
    async def test_new_association_created(self) -> None:
        """New association is created with source when not found."""
        service, _, assoc_repo, _ = _make_service()
        game_id = uuid4()
        doc_id = uuid4()
        init_id = uuid4()
        combat_id = uuid4()

        concept_map = {"initiative": init_id, "combat": combat_id}

        assoc_repo.find_existing.return_value = None
        created = MagicMock()
        created.id = uuid4()
        assoc_repo.create.return_value = created
        assoc_repo.add_source.return_value = MagicMock()

        discovered = _make_discovered_association()

        result = await service.integrate_associations(
            game_id=game_id,
            document_id=doc_id,
            associations=[discovered],
            concept_map=concept_map,
        )

        assert len(result) == 1
        assoc_repo.create.assert_called_once()
        assoc_repo.add_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_existing_association_gets_new_source(self) -> None:
        """Existing association gets new source reference."""
        service, _, assoc_repo, _ = _make_service()
        init_id = uuid4()
        combat_id = uuid4()
        concept_map = {"initiative": init_id, "combat": combat_id}

        existing = MagicMock()
        existing.id = uuid4()
        assoc_repo.find_existing.return_value = existing
        assoc_repo.find_source_by_chunk.return_value = None
        assoc_repo.add_source.return_value = MagicMock()

        discovered = _make_discovered_association()

        result = await service.integrate_associations(
            game_id=uuid4(),
            document_id=uuid4(),
            associations=[discovered],
            concept_map=concept_map,
        )

        assert len(result) == 1
        assert result[0] is existing
        assoc_repo.create.assert_not_called()
        assoc_repo.add_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_existing_association_duplicate_source_skipped(self) -> None:
        """Source not added if already exists for association-chunk pair."""
        service, _, assoc_repo, _ = _make_service()
        init_id = uuid4()
        combat_id = uuid4()
        concept_map = {"initiative": init_id, "combat": combat_id}

        existing = MagicMock()
        existing.id = uuid4()
        assoc_repo.find_existing.return_value = existing
        assoc_repo.find_source_by_chunk.return_value = MagicMock()

        discovered = _make_discovered_association()

        await service.integrate_associations(
            game_id=uuid4(),
            document_id=uuid4(),
            associations=[discovered],
            concept_map=concept_map,
        )

        assoc_repo.add_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_unresolvable_concept_skipped(self) -> None:
        """Association is skipped when concept name cannot be resolved."""
        service, _, assoc_repo, _ = _make_service()
        concept_map = {"initiative": uuid4()}  # "combat" missing

        discovered = _make_discovered_association()

        result = await service.integrate_associations(
            game_id=uuid4(),
            document_id=uuid4(),
            associations=[discovered],
            concept_map=concept_map,
        )

        assert len(result) == 0
        assoc_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_associations_integrated(self) -> None:
        """Multiple associations are processed independently."""
        service, _, assoc_repo, _ = _make_service()
        init_id = uuid4()
        combat_id = uuid4()
        concept_map = {"initiative": init_id, "combat": combat_id}

        assoc_repo.find_existing.return_value = None
        assoc1 = MagicMock()
        assoc1.id = uuid4()
        assoc2 = MagicMock()
        assoc2.id = uuid4()
        assoc_repo.create.side_effect = [assoc1, assoc2]
        assoc_repo.add_source.return_value = MagicMock()

        d1 = _make_discovered_association(label="determines")
        d2 = _make_discovered_association(label="requires")

        result = await service.integrate_associations(
            game_id=uuid4(),
            document_id=uuid4(),
            associations=[d1, d2],
            concept_map=concept_map,
        )

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Concept Map Tests
# ---------------------------------------------------------------------------


class TestBuildConceptMap:
    """Tests for build_concept_map."""

    def test_builds_lowercase_map(self) -> None:
        """Map keys are lowercase concept names."""
        service, _, _, _ = _make_service()
        c1 = _make_kg_concept(name="Initiative")
        c2 = _make_kg_concept(name="Combat Phase")

        result = service.build_concept_map([c1, c2])

        assert "initiative" in result
        assert "combat phase" in result
        assert result["initiative"] == c1.id
        assert result["combat phase"] == c2.id

    def test_empty_list_returns_empty_map(self) -> None:
        """Empty concept list yields empty map."""
        service, _, _, _ = _make_service()
        assert service.build_concept_map([]) == {}

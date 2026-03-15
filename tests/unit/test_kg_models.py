"""Unit tests for Knowledge Graph ORM models.

Validates model metadata registration, table names, column definitions,
unique constraints, foreign keys, and relationships for all 6 KG tables
without requiring a database connection.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import inspect as sa_inspect

from tabletop_oracle.models import (
    ConceptEmbedding,
    KGAssociation,
    KGAssociationSource,
    KGAuditLog,
    KGConcept,
    KGConceptSource,
    MappedBase,
)


def _inspect_mapper(model: type) -> Any:
    """Inspect a model's mapper (returns Mapper, typed as Any for mypy)."""
    return sa_inspect(model)


def _get_column_names(model: type) -> set[str]:
    """Extract column names from a model's table."""
    mapper = _inspect_mapper(model)
    return {col.name for col in mapper.columns}


def _get_relationship_names(model: type) -> set[str]:
    """Extract relationship property names from a model."""
    mapper = _inspect_mapper(model)
    return {r.key for r in mapper.relationships}


# ---------------------------------------------------------------------------
# Model import and metadata registration
# ---------------------------------------------------------------------------


class TestKGModelImports:
    """All KG models can be imported and are registered in shared metadata."""

    def test_all_kg_models_importable(self) -> None:
        """Every KG model class can be imported from the models package."""
        model_classes = [
            KGConcept,
            KGConceptSource,
            KGAssociation,
            KGAssociationSource,
            ConceptEmbedding,
            KGAuditLog,
        ]
        for cls in model_classes:
            assert cls is not None

    def test_kg_tables_registered_in_metadata(self) -> None:
        """All 6 KG tables are registered in the shared metadata."""
        expected_kg_tables = {
            "kg_concepts",
            "kg_concept_sources",
            "kg_associations",
            "kg_association_sources",
            "concept_embeddings",
            "kg_audit_log",
        }
        actual_tables = set(MappedBase.metadata.tables.keys())
        assert expected_kg_tables <= actual_tables

    def test_total_table_count(self) -> None:
        """Total metadata table count is 23 (17 existing + 6 KG)."""
        actual_count = len(MappedBase.metadata.tables)
        assert actual_count == 23


# ---------------------------------------------------------------------------
# Table name mapping
# ---------------------------------------------------------------------------


class TestKGTableNames:
    """Each KG model maps to the correct database table."""

    @pytest.mark.parametrize(
        ("model", "expected_table"),
        [
            (KGConcept, "kg_concepts"),
            (KGConceptSource, "kg_concept_sources"),
            (KGAssociation, "kg_associations"),
            (KGAssociationSource, "kg_association_sources"),
            (ConceptEmbedding, "concept_embeddings"),
            (KGAuditLog, "kg_audit_log"),
        ],
    )
    def test_table_name(self, model: type, expected_table: str) -> None:
        """Model __tablename__ matches the expected database table."""
        assert model.__tablename__ == expected_table  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------


class TestKGConceptColumns:
    """KGConcept has all expected columns."""

    def test_has_base_columns(self) -> None:
        """KGConcept inherits id, created_at, updated_at from Base."""
        cols = _get_column_names(KGConcept)
        assert {"id", "created_at", "updated_at"} <= cols

    def test_has_domain_columns(self) -> None:
        """KGConcept has all domain-specific columns."""
        cols = _get_column_names(KGConcept)
        expected = {
            "game_id",
            "name",
            "semantic_type",
            "description",
            "canonical_id",
            "session_id",
        }
        assert expected <= cols

    def test_game_id_not_nullable(self) -> None:
        """game_id is required (NOT NULL)."""
        table = MappedBase.metadata.tables["kg_concepts"]
        assert not table.c.game_id.nullable

    def test_canonical_id_nullable(self) -> None:
        """canonical_id is nullable (NULL for canonical concepts)."""
        table = MappedBase.metadata.tables["kg_concepts"]
        assert table.c.canonical_id.nullable

    def test_session_id_nullable(self) -> None:
        """session_id is nullable (NULL for V1 global KG)."""
        table = MappedBase.metadata.tables["kg_concepts"]
        assert table.c.session_id.nullable


class TestKGConceptSourceColumns:
    """KGConceptSource has all expected columns."""

    def test_has_id_and_created_at(self) -> None:
        """KGConceptSource has id and created_at (no updated_at)."""
        cols = _get_column_names(KGConceptSource)
        assert {"id", "created_at"} <= cols
        assert "updated_at" not in cols

    def test_has_domain_columns(self) -> None:
        """KGConceptSource has all domain-specific columns."""
        cols = _get_column_names(KGConceptSource)
        expected = {
            "concept_id",
            "document_id",
            "version_id",
            "chunk_id",
            "expansion_id",
            "document_type",
            "document_name",
            "section_path",
            "page_number",
            "source_text",
            "is_authoritative",
        }
        assert expected <= cols

    def test_expansion_id_nullable(self) -> None:
        """expansion_id is nullable (NULL for base-game documents)."""
        table = MappedBase.metadata.tables["kg_concept_sources"]
        assert table.c.expansion_id.nullable

    def test_is_authoritative_has_default(self) -> None:
        """is_authoritative defaults to TRUE."""
        table = MappedBase.metadata.tables["kg_concept_sources"]
        assert table.c.is_authoritative.server_default is not None


class TestKGAssociationColumns:
    """KGAssociation has all expected columns."""

    def test_has_base_columns(self) -> None:
        """KGAssociation inherits id, created_at, updated_at from Base."""
        cols = _get_column_names(KGAssociation)
        assert {"id", "created_at", "updated_at"} <= cols

    def test_has_domain_columns(self) -> None:
        """KGAssociation has all domain-specific columns."""
        cols = _get_column_names(KGAssociation)
        expected = {
            "game_id",
            "source_concept_id",
            "target_concept_id",
            "relationship_label",
            "description",
        }
        assert expected <= cols

    def test_description_nullable(self) -> None:
        """description is nullable."""
        table = MappedBase.metadata.tables["kg_associations"]
        assert table.c.description.nullable


class TestKGAssociationSourceColumns:
    """KGAssociationSource has all expected columns."""

    def test_has_id_and_created_at(self) -> None:
        """KGAssociationSource has id and created_at (no updated_at)."""
        cols = _get_column_names(KGAssociationSource)
        assert {"id", "created_at"} <= cols
        assert "updated_at" not in cols

    def test_has_domain_columns(self) -> None:
        """KGAssociationSource has all domain-specific columns."""
        cols = _get_column_names(KGAssociationSource)
        expected = {
            "association_id",
            "document_id",
            "chunk_id",
            "expansion_id",
            "source_text",
        }
        assert expected <= cols


class TestConceptEmbeddingColumns:
    """ConceptEmbedding has all expected columns."""

    def test_has_id_and_created_at(self) -> None:
        """ConceptEmbedding has id and created_at (no updated_at)."""
        cols = _get_column_names(ConceptEmbedding)
        assert {"id", "created_at"} <= cols
        assert "updated_at" not in cols

    def test_has_domain_columns(self) -> None:
        """ConceptEmbedding has concept_id, embedding, and model_id."""
        cols = _get_column_names(ConceptEmbedding)
        assert {"concept_id", "embedding", "model_id"} <= cols


class TestKGAuditLogColumns:
    """KGAuditLog has all expected columns."""

    def test_has_id_and_created_at(self) -> None:
        """KGAuditLog has id and created_at (no updated_at)."""
        cols = _get_column_names(KGAuditLog)
        assert {"id", "created_at"} <= cols
        assert "updated_at" not in cols

    def test_has_domain_columns(self) -> None:
        """KGAuditLog has all domain-specific columns."""
        cols = _get_column_names(KGAuditLog)
        expected = {
            "game_id",
            "event_type",
            "document_id",
            "concept_id",
            "details",
        }
        assert expected <= cols

    def test_document_id_nullable(self) -> None:
        """document_id is nullable."""
        table = MappedBase.metadata.tables["kg_audit_log"]
        assert table.c.document_id.nullable

    def test_concept_id_nullable(self) -> None:
        """concept_id is nullable (no FK — concept may be deleted)."""
        table = MappedBase.metadata.tables["kg_audit_log"]
        assert table.c.concept_id.nullable

    def test_details_has_default(self) -> None:
        """details column defaults to empty JSONB object."""
        table = MappedBase.metadata.tables["kg_audit_log"]
        assert table.c.details.server_default is not None


# ---------------------------------------------------------------------------
# Unique constraints
# ---------------------------------------------------------------------------


class TestKGUniqueConstraints:
    """Unique constraints are defined per the design spec."""

    def _get_unique_constraint_columns(self, table_name: str) -> list[frozenset[str]]:
        """Get column sets for all unique constraints on a table."""
        table = MappedBase.metadata.tables[table_name]
        result = []
        for constraint in table.constraints:
            is_unique = constraint.__class__.__name__ == "UniqueConstraint"
            if hasattr(constraint, "columns") and is_unique:
                result.append(frozenset(c.name for c in constraint.columns))
        return result

    def test_concept_name_unique_per_game(self) -> None:
        """kg_concepts has UNIQUE(game_id, name)."""
        uqs = self._get_unique_constraint_columns("kg_concepts")
        assert frozenset({"game_id", "name"}) in uqs

    def test_concept_source_unique(self) -> None:
        """kg_concept_sources has UNIQUE(concept_id, chunk_id)."""
        uqs = self._get_unique_constraint_columns("kg_concept_sources")
        assert frozenset({"concept_id", "chunk_id"}) in uqs

    def test_association_unique(self) -> None:
        """kg_associations has 4-column unique constraint."""
        uqs = self._get_unique_constraint_columns("kg_associations")
        expected = frozenset(
            {
                "game_id",
                "source_concept_id",
                "target_concept_id",
                "relationship_label",
            }
        )
        assert expected in uqs

    def test_assoc_source_unique(self) -> None:
        """kg_association_sources has UNIQUE(association_id, chunk_id)."""
        uqs = self._get_unique_constraint_columns("kg_association_sources")
        assert frozenset({"association_id", "chunk_id"}) in uqs

    def test_concept_embedding_unique(self) -> None:
        """concept_embeddings has UNIQUE(concept_id)."""
        uqs = self._get_unique_constraint_columns("concept_embeddings")
        assert frozenset({"concept_id"}) in uqs


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


class TestKGForeignKeys:
    """Foreign key relationships are correctly defined."""

    def _get_fk_targets(self, table_name: str) -> set[str]:
        """Get FK target table.column strings for a table."""
        table = MappedBase.metadata.tables[table_name]
        targets = set()
        for fk in table.foreign_keys:
            targets.add(f"{fk.column.table.name}.{fk.column.name}")
        return targets

    def test_kg_concepts_fks(self) -> None:
        """kg_concepts references games, kg_concepts (self), sessions."""
        fks = self._get_fk_targets("kg_concepts")
        assert "games.id" in fks
        assert "kg_concepts.id" in fks
        assert "sessions.id" in fks

    def test_kg_concept_sources_fks(self) -> None:
        """kg_concept_sources references all required parent tables."""
        fks = self._get_fk_targets("kg_concept_sources")
        assert "kg_concepts.id" in fks
        assert "documents.id" in fks
        assert "document_versions.id" in fks
        assert "document_chunks.id" in fks
        assert "expansions.id" in fks

    def test_kg_associations_fks(self) -> None:
        """kg_associations references games and kg_concepts (twice)."""
        fks = self._get_fk_targets("kg_associations")
        assert "games.id" in fks
        assert "kg_concepts.id" in fks

    def test_kg_association_sources_fks(self) -> None:
        """kg_association_sources references all required parent tables."""
        fks = self._get_fk_targets("kg_association_sources")
        assert "kg_associations.id" in fks
        assert "documents.id" in fks
        assert "document_chunks.id" in fks
        assert "expansions.id" in fks

    def test_concept_embeddings_fks(self) -> None:
        """concept_embeddings references kg_concepts."""
        fks = self._get_fk_targets("concept_embeddings")
        assert "kg_concepts.id" in fks

    def test_kg_audit_log_fks(self) -> None:
        """kg_audit_log references games and documents."""
        fks = self._get_fk_targets("kg_audit_log")
        assert "games.id" in fks
        assert "documents.id" in fks

    def test_kg_audit_log_concept_id_has_no_fk(self) -> None:
        """kg_audit_log.concept_id has no FK (concept may be deleted)."""
        fks = self._get_fk_targets("kg_audit_log")
        # concept_id should not reference kg_concepts
        fk_targets_with_concept = [fk for fk in fks if "kg_concepts" in fk]
        assert len(fk_targets_with_concept) == 0


# ---------------------------------------------------------------------------
# CASCADE delete behaviour
# ---------------------------------------------------------------------------


class TestKGCascadeDeletes:
    """CASCADE delete constraints are correctly defined."""

    def _get_cascade_fks(self, table_name: str) -> dict[str, str | None]:
        """Get FK column -> ondelete setting for a table."""
        table = MappedBase.metadata.tables[table_name]
        result = {}
        for fk in table.foreign_keys:
            col_name = fk.parent.name
            result[col_name] = fk.ondelete
        return result

    def test_concept_source_cascades_on_concept_delete(self) -> None:
        """Deleting a concept cascades to concept_sources."""
        cascades = self._get_cascade_fks("kg_concept_sources")
        assert cascades.get("concept_id") == "CASCADE"

    def test_association_cascades_on_concept_delete(self) -> None:
        """Deleting a concept cascades to associations."""
        cascades = self._get_cascade_fks("kg_associations")
        assert cascades.get("source_concept_id") == "CASCADE"
        assert cascades.get("target_concept_id") == "CASCADE"

    def test_association_source_cascades_on_association_delete(self) -> None:
        """Deleting an association cascades to association_sources."""
        cascades = self._get_cascade_fks("kg_association_sources")
        assert cascades.get("association_id") == "CASCADE"

    def test_embedding_cascades_on_concept_delete(self) -> None:
        """Deleting a concept cascades to embeddings."""
        cascades = self._get_cascade_fks("concept_embeddings")
        assert cascades.get("concept_id") == "CASCADE"


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestKGRelationships:
    """ORM relationships are correctly defined."""

    def test_kg_concept_relationships(self) -> None:
        """KGConcept has game, canonical, session, sources, embeddings."""
        rels = _get_relationship_names(KGConcept)
        assert {"game", "canonical", "session", "sources", "embeddings"} <= rels

    def test_kg_concept_source_relationships(self) -> None:
        """KGConceptSource has concept, document, version, chunk, expansion."""
        rels = _get_relationship_names(KGConceptSource)
        assert {"concept", "document", "version", "chunk", "expansion"} <= rels

    def test_kg_association_relationships(self) -> None:
        """KGAssociation has game, source_concept, target_concept, sources."""
        rels = _get_relationship_names(KGAssociation)
        assert {"game", "source_concept", "target_concept", "sources"} <= rels

    def test_kg_association_source_relationships(self) -> None:
        """KGAssociationSource has association, document, chunk, expansion."""
        rels = _get_relationship_names(KGAssociationSource)
        assert {"association", "document", "chunk", "expansion"} <= rels

    def test_concept_embedding_relationships(self) -> None:
        """ConceptEmbedding has concept."""
        rels = _get_relationship_names(ConceptEmbedding)
        assert "concept" in rels

    def test_kg_audit_log_relationships(self) -> None:
        """KGAuditLog has game and document."""
        rels = _get_relationship_names(KGAuditLog)
        assert {"game", "document"} <= rels

"""Unit tests for SQLAlchemy ORM models.

Validates model metadata registration, table names, column types,
relationships, composite primary keys, and enum definitions without
requiring a database connection.
"""

from __future__ import annotations

import enum
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect

from tabletop_oracle.models import (
    AuthSession,
    Base,
    Citation,
    ContextAttachment,
    Document,
    DocumentChunk,
    DocumentVersion,
    Expansion,
    Game,
    GameTag,
    GuardrailConfig,
    MappedBase,
    Message,
    ModelSlot,
    PlayerFeedback,
    Session,
    SessionExpansion,
    TokenUsageLog,
    User,
)
from tabletop_oracle.models.enums import (
    ContextAttachmentType,
    DocumentFormat,
    DocumentStatus,
    DocumentType,
    FeedbackRating,
    GameComplexity,
    MessageType,
    ModelCapability,
    OAuthProvider,
    SessionStatus,
    UserRole,
    UserStatus,
)

# ---------------------------------------------------------------------------
# Model import and metadata registration
# ---------------------------------------------------------------------------


class TestModelImports:
    """All models can be imported and are registered in shared metadata."""

    def test_all_models_importable(self) -> None:
        """Every model class can be imported from the models package."""
        model_classes = [
            AuthSession,
            User,
            Game,
            GameTag,
            Expansion,
            Document,
            DocumentChunk,
            DocumentVersion,
            Session,
            SessionExpansion,
            Message,
            Citation,
            ContextAttachment,
            PlayerFeedback,
            ModelSlot,
            TokenUsageLog,
            GuardrailConfig,
        ]
        for cls in model_classes:
            assert cls is not None

    def test_metadata_shared_between_base_and_mapped_base(self) -> None:
        """Base and MappedBase share the same MetaData instance."""
        assert Base.metadata is MappedBase.metadata

    def test_all_tables_registered_in_metadata(self) -> None:
        """All 17 tables are registered in the shared metadata."""
        expected_tables = {
            "auth_sessions",
            "users",
            "games",
            "game_tags",
            "expansions",
            "documents",
            "document_chunks",
            "document_versions",
            "sessions",
            "session_expansions",
            "messages",
            "citations",
            "context_attachments",
            "player_feedback",
            "model_slots",
            "token_usage_log",
            "guardrail_config",
        }
        actual_tables = set(MappedBase.metadata.tables.keys())
        assert expected_tables == actual_tables


# ---------------------------------------------------------------------------
# Table name mapping
# ---------------------------------------------------------------------------


class TestTableNames:
    """Each model maps to the correct database table."""

    @pytest.mark.parametrize(
        ("model", "expected_table"),
        [
            (AuthSession, "auth_sessions"),
            (User, "users"),
            (Game, "games"),
            (GameTag, "game_tags"),
            (Expansion, "expansions"),
            (Document, "documents"),
            (DocumentChunk, "document_chunks"),
            (DocumentVersion, "document_versions"),
            (Session, "sessions"),
            (SessionExpansion, "session_expansions"),
            (Message, "messages"),
            (Citation, "citations"),
            (ContextAttachment, "context_attachments"),
            (PlayerFeedback, "player_feedback"),
            (ModelSlot, "model_slots"),
            (TokenUsageLog, "token_usage_log"),
            (GuardrailConfig, "guardrail_config"),
        ],
    )
    def test_table_name(self, model: type, expected_table: str) -> None:
        """Model __tablename__ matches the expected database table."""
        assert model.__tablename__ == expected_table  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Enum definitions
# ---------------------------------------------------------------------------

ALL_ENUM_CLASSES: list[type[enum.Enum]] = [
    OAuthProvider,
    UserRole,
    UserStatus,
    GameComplexity,
    DocumentType,
    DocumentFormat,
    DocumentStatus,
    SessionStatus,
    MessageType,
    ContextAttachmentType,
    FeedbackRating,
    ModelCapability,
]


class TestEnums:
    """Python enums match PostgreSQL enum type definitions."""

    def test_all_enums_are_enum_subclass(self) -> None:
        """All enum classes are proper enum.Enum subclasses."""
        for cls in ALL_ENUM_CLASSES:
            assert issubclass(cls, enum.Enum)

    def test_enum_count(self) -> None:
        """There are exactly 12 enum types matching the 12 PostgreSQL enums."""
        assert len(ALL_ENUM_CLASSES) == 12

    @pytest.mark.parametrize(
        ("enum_class", "expected_values"),
        [
            (OAuthProvider, ["google", "microsoft"]),
            (UserRole, ["player", "curator"]),
            (UserStatus, ["active", "disabled"]),
            (GameComplexity, ["light", "medium", "heavy"]),
            (
                DocumentType,
                ["core_rules", "faq", "errata", "expansion_rules", "strategy", "other"],
            ),
            (DocumentFormat, ["pdf", "markdown", "text", "html", "docx"]),
            (DocumentStatus, ["uploaded", "parsing", "processed", "error"]),
            (SessionStatus, ["active", "archived"]),
            (
                MessageType,
                ["user_question", "ai_answer", "ai_clarification", "system"],
            ),
            (ContextAttachmentType, ["text", "image"]),
            (FeedbackRating, ["positive", "negative"]),
            (
                ModelCapability,
                [
                    "intent_analysis",
                    "retrieval_augmentation",
                    "answer_synthesis",
                    "clarification_generation",
                    "concept_extraction",
                    "vision_processing",
                ],
            ),
        ],
    )
    def test_enum_values_match_postgres(
        self,
        enum_class: type[enum.Enum],
        expected_values: list[str],
    ) -> None:
        """Enum values match the PostgreSQL CREATE TYPE ... AS ENUM definition."""
        actual_values = [member.value for member in enum_class]
        assert actual_values == expected_values


# ---------------------------------------------------------------------------
# Composite primary keys
# ---------------------------------------------------------------------------


def _inspect_mapper(model: type) -> Any:
    """Inspect a model's mapper (returns Mapper, typed as Any for mypy)."""
    return sa_inspect(model)


class TestCompositePrimaryKeys:
    """Join tables use composite primary keys as designed."""

    def test_game_tag_composite_pk(self) -> None:
        """GameTag has composite PK (game_id, tag)."""
        mapper = _inspect_mapper(GameTag)
        pk_cols = [col.name for col in mapper.primary_key]
        assert pk_cols == ["game_id", "tag"]

    def test_session_expansion_composite_pk(self) -> None:
        """SessionExpansion has composite PK (session_id, expansion_id)."""
        mapper = _inspect_mapper(SessionExpansion)
        pk_cols = [col.name for col in mapper.primary_key]
        assert pk_cols == ["session_id", "expansion_id"]


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    """Bidirectional relationships are correctly defined."""

    def _get_relationship_names(self, model: type) -> set[str]:
        """Extract relationship property names from a model."""
        mapper = _inspect_mapper(model)
        return {r.key for r in mapper.relationships}

    def test_user_relationships(self) -> None:
        """User has games, sessions, auth_sessions, and document_versions relationships."""
        rels = self._get_relationship_names(User)
        assert {"games", "sessions", "auth_sessions", "document_versions"} <= rels

    def test_auth_session_relationships(self) -> None:
        """AuthSession has user relationship."""
        rels = self._get_relationship_names(AuthSession)
        assert "user" in rels

    def test_game_relationships(self) -> None:
        """Game has creator, expansions, documents, tags, sessions."""
        rels = self._get_relationship_names(Game)
        assert {"creator", "expansions", "documents", "tags", "sessions"} <= rels

    def test_expansion_relationships(self) -> None:
        """Expansion has game, documents, session_expansions."""
        rels = self._get_relationship_names(Expansion)
        assert {"game", "documents", "session_expansions"} <= rels

    def test_document_relationships(self) -> None:
        """Document has game, expansion, versions, chunks."""
        rels = self._get_relationship_names(Document)
        assert {"game", "expansion", "versions", "chunks"} <= rels

    def test_document_version_relationships(self) -> None:
        """DocumentVersion has document, uploader, chunks."""
        rels = self._get_relationship_names(DocumentVersion)
        assert {"document", "uploader", "chunks"} <= rels

    def test_document_chunk_relationships(self) -> None:
        """DocumentChunk has document and version relationships."""
        rels = self._get_relationship_names(DocumentChunk)
        assert {"document", "version"} <= rels

    def test_session_relationships(self) -> None:
        """Session has user, game, session_expansions, messages."""
        rels = self._get_relationship_names(Session)
        assert {"user", "game", "session_expansions", "messages"} <= rels

    def test_message_relationships(self) -> None:
        """Message has session, reply_to, citations, context_attachments."""
        rels = self._get_relationship_names(Message)
        assert {"session", "reply_to", "citations", "context_attachments"} <= rels

    def test_message_self_referential_relationship(self) -> None:
        """Message.reply_to is a self-referential relationship to Message."""
        mapper = _inspect_mapper(Message)
        reply_to = mapper.relationships["reply_to"]
        assert reply_to.mapper.class_ is Message

    def test_citation_relationships(self) -> None:
        """Citation has message relationship."""
        rels = self._get_relationship_names(Citation)
        assert "message" in rels

    def test_context_attachment_relationships(self) -> None:
        """ContextAttachment has message relationship."""
        rels = self._get_relationship_names(ContextAttachment)
        assert "message" in rels

    def test_game_tag_relationships(self) -> None:
        """GameTag has game relationship."""
        rels = self._get_relationship_names(GameTag)
        assert "game" in rels

    def test_session_expansion_relationships(self) -> None:
        """SessionExpansion has session and expansion relationships."""
        rels = self._get_relationship_names(SessionExpansion)
        assert {"session", "expansion"} <= rels


# ---------------------------------------------------------------------------
# Foreign key cascade settings
# ---------------------------------------------------------------------------


class TestForeignKeys:
    """Foreign keys have correct ondelete cascade settings."""

    def _get_fk_ondelete(self, model: type, column_name: str) -> str | None:
        """Get the ON DELETE setting for a FK column."""
        table = model.__table__  # type: ignore[attr-defined]
        col = table.c[column_name]
        for fk in col.foreign_keys:
            return str(fk.ondelete) if fk.ondelete else None
        return None

    def test_game_tag_game_id_cascade(self) -> None:
        """GameTag.game_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(GameTag, "game_id") == "CASCADE"

    def test_expansion_game_id_cascade(self) -> None:
        """Expansion.game_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(Expansion, "game_id") == "CASCADE"

    def test_document_version_document_id_cascade(self) -> None:
        """DocumentVersion.document_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(DocumentVersion, "document_id") == "CASCADE"

    def test_session_expansion_session_id_cascade(self) -> None:
        """SessionExpansion.session_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(SessionExpansion, "session_id") == "CASCADE"

    def test_citation_message_id_cascade(self) -> None:
        """Citation.message_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(Citation, "message_id") == "CASCADE"

    def test_context_attachment_message_id_cascade(self) -> None:
        """ContextAttachment.message_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(ContextAttachment, "message_id") == "CASCADE"

    def test_auth_session_user_id_cascade(self) -> None:
        """AuthSession.user_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(AuthSession, "user_id") == "CASCADE"

    def test_document_chunk_document_id_cascade(self) -> None:
        """DocumentChunk.document_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(DocumentChunk, "document_id") == "CASCADE"

    def test_document_chunk_version_id_cascade(self) -> None:
        """DocumentChunk.version_id has ON DELETE CASCADE."""
        assert self._get_fk_ondelete(DocumentChunk, "version_id") == "CASCADE"

    def test_document_game_id_no_cascade(self) -> None:
        """Document.game_id has no ON DELETE CASCADE (restrict)."""
        assert self._get_fk_ondelete(Document, "game_id") is None

    def test_session_user_id_no_cascade(self) -> None:
        """Session.user_id has no ON DELETE CASCADE (restrict)."""
        assert self._get_fk_ondelete(Session, "user_id") is None


# ---------------------------------------------------------------------------
# Column nullability for key nullable fields
# ---------------------------------------------------------------------------


class TestNullability:
    """Nullable columns are correctly annotated."""

    def _is_nullable(self, model: type, column_name: str) -> bool:
        """Check if a column is nullable in the table definition."""
        table = model.__table__  # type: ignore[attr-defined]
        return bool(table.c[column_name].nullable)

    def test_user_oauth_subject_id_nullable(self) -> None:
        """User.oauth_subject_id is nullable."""
        assert self._is_nullable(User, "oauth_subject_id")

    def test_game_publisher_nullable(self) -> None:
        """Game.publisher is nullable."""
        assert self._is_nullable(Game, "publisher")

    def test_game_complexity_nullable(self) -> None:
        """Game.complexity is nullable."""
        assert self._is_nullable(Game, "complexity")

    def test_game_archived_at_nullable(self) -> None:
        """Game.archived_at is nullable."""
        assert self._is_nullable(Game, "archived_at")

    def test_document_expansion_id_nullable(self) -> None:
        """Document.expansion_id is nullable."""
        assert self._is_nullable(Document, "expansion_id")

    def test_document_error_message_nullable(self) -> None:
        """Document.error_message is nullable."""
        assert self._is_nullable(Document, "error_message")

    def test_document_archived_at_nullable(self) -> None:
        """Document.archived_at is nullable (null = active)."""
        assert self._is_nullable(Document, "archived_at")

    def test_document_chunk_section_path_nullable(self) -> None:
        """DocumentChunk.section_path is nullable."""
        assert self._is_nullable(DocumentChunk, "section_path")

    def test_document_chunk_heading_nullable(self) -> None:
        """DocumentChunk.heading is nullable."""
        assert self._is_nullable(DocumentChunk, "heading")

    def test_document_chunk_page_number_nullable(self) -> None:
        """DocumentChunk.page_number is nullable."""
        assert self._is_nullable(DocumentChunk, "page_number")

    def test_document_chunk_content_not_nullable(self) -> None:
        """DocumentChunk.content is NOT nullable."""
        assert not self._is_nullable(DocumentChunk, "content")

    def test_document_chunk_token_estimate_not_nullable(self) -> None:
        """DocumentChunk.token_estimate is NOT nullable."""
        assert not self._is_nullable(DocumentChunk, "token_estimate")

    def test_message_in_reply_to_id_nullable(self) -> None:
        """Message.in_reply_to_id is nullable."""
        assert self._is_nullable(Message, "in_reply_to_id")

    def test_message_confidence_score_nullable(self) -> None:
        """Message.confidence_score is nullable."""
        assert self._is_nullable(Message, "confidence_score")

    def test_auth_session_user_agent_nullable(self) -> None:
        """AuthSession.user_agent is nullable."""
        assert self._is_nullable(AuthSession, "user_agent")

    def test_auth_session_ip_address_nullable(self) -> None:
        """AuthSession.ip_address is nullable."""
        assert self._is_nullable(AuthSession, "ip_address")

    def test_auth_session_expires_at_not_nullable(self) -> None:
        """AuthSession.expires_at is NOT nullable."""
        assert not self._is_nullable(AuthSession, "expires_at")

    def test_auth_session_user_id_not_nullable(self) -> None:
        """AuthSession.user_id is NOT nullable."""
        assert not self._is_nullable(AuthSession, "user_id")

    def test_user_email_not_nullable(self) -> None:
        """User.email is NOT nullable."""
        assert not self._is_nullable(User, "email")

    def test_game_name_not_nullable(self) -> None:
        """Game.name is NOT nullable."""
        assert not self._is_nullable(Game, "name")


# ---------------------------------------------------------------------------
# Base class inheritance
# ---------------------------------------------------------------------------


class TestBaseInheritance:
    """Models inherit from the correct base class."""

    @pytest.mark.parametrize(
        "model",
        [Game, Expansion],
    )
    def test_standard_models_inherit_base(self, model: type) -> None:
        """Models with id + created_at + updated_at inherit from Base."""
        assert issubclass(model, Base)

    @pytest.mark.parametrize(
        "model",
        [
            AuthSession,
            User,
            Document,
            DocumentChunk,
            DocumentVersion,
            Session,
            Message,
            Citation,
            ContextAttachment,
            PlayerFeedback,
            ModelSlot,
            TokenUsageLog,
            GuardrailConfig,
            GameTag,
            SessionExpansion,
        ],
    )
    def test_nonstandard_models_inherit_mapped_base(self, model: type) -> None:
        """Models with non-standard columns inherit from MappedBase."""
        assert issubclass(model, MappedBase)

    @pytest.mark.parametrize(
        "model",
        [Game, Expansion],
    )
    def test_base_models_have_common_columns(self, model: type) -> None:
        """Base-inheriting models have id, created_at, updated_at."""
        table = model.__table__  # type: ignore[attr-defined]
        col_names = {c.name for c in table.columns}
        assert {"id", "created_at", "updated_at"} <= col_names


# ---------------------------------------------------------------------------
# Server defaults
# ---------------------------------------------------------------------------


class TestAuthSessionPrimaryKey:
    """AuthSession uses a TEXT primary key, not UUID."""

    def test_auth_session_id_is_text_type(self) -> None:
        """AuthSession.id column is TEXT (not UUID)."""
        table = AuthSession.__table__  # type: ignore[attr-defined]
        col = table.c.id
        assert isinstance(col.type, sa.Text)

    def test_auth_session_id_is_primary_key(self) -> None:
        """AuthSession.id is the primary key."""
        table = AuthSession.__table__  # type: ignore[attr-defined]
        pk_cols = [c.name for c in table.primary_key.columns]
        assert pk_cols == ["id"]


class TestServerDefaults:
    """Key columns have correct server defaults."""

    def test_user_role_default_player(self) -> None:
        """User.role defaults to 'player'."""
        table = User.__table__
        default = table.c.role.server_default
        assert default is not None
        assert "player" in str(default.arg)

    def test_user_status_default_active(self) -> None:
        """User.status defaults to 'active'."""
        table = User.__table__
        default = table.c.status.server_default
        assert default is not None
        assert "active" in str(default.arg)

    def test_document_status_default_uploaded(self) -> None:
        """Document.status defaults to 'uploaded'."""
        table = Document.__table__
        default = table.c.status.server_default
        assert default is not None
        assert "uploaded" in str(default.arg)

    def test_session_status_default_active(self) -> None:
        """Session.status defaults to 'active'."""
        table = Session.__table__
        default = table.c.status.server_default
        assert default is not None
        assert "active" in str(default.arg)

    def test_guardrail_enforcement_enabled_default_true(self) -> None:
        """GuardrailConfig.enforcement_enabled defaults to TRUE."""
        table = GuardrailConfig.__table__
        default = table.c.enforcement_enabled.server_default
        assert default is not None
        assert "true" in str(default.arg).lower()

    def test_document_chunk_type_default_text(self) -> None:
        """DocumentChunk.chunk_type defaults to 'text'."""
        table = DocumentChunk.__table__  # type: ignore[attr-defined]
        default = table.c.chunk_type.server_default
        assert default is not None
        assert "text" in str(default.arg)

    def test_document_version_is_active_default_true(self) -> None:
        """DocumentVersion.is_active defaults to TRUE."""
        table = DocumentVersion.__table__
        default = table.c.is_active.server_default
        assert default is not None
        assert "true" in str(default.arg).lower()


# ---------------------------------------------------------------------------
# Unique constraints
# ---------------------------------------------------------------------------


class TestUniqueConstraints:
    """Unique constraints are correctly defined."""

    def test_user_email_unique(self) -> None:
        """User.email has a unique constraint."""
        table = User.__table__
        assert table.c.email.unique

    def test_model_slot_capability_unique(self) -> None:
        """ModelSlot.capability has a unique constraint."""
        table = ModelSlot.__table__
        assert table.c.capability.unique

    def test_player_feedback_message_id_unique(self) -> None:
        """PlayerFeedback.message_id has a unique constraint."""
        table = PlayerFeedback.__table__
        assert table.c.message_id.unique

    def test_document_chunk_composite_unique_constraint(self) -> None:
        """DocumentChunk has unique constraint on (document_id, version_id, chunk_index)."""
        table = DocumentChunk.__table__  # type: ignore[attr-defined]
        uq_names = {c.name for c in table.constraints if isinstance(c, sa.UniqueConstraint)}
        assert "uq_chunk_index" in uq_names

    def test_document_chunk_unique_constraint_columns(self) -> None:
        """uq_chunk_index covers document_id, version_id, chunk_index."""
        table = DocumentChunk.__table__  # type: ignore[attr-defined]
        for constraint in table.constraints:
            if isinstance(constraint, sa.UniqueConstraint) and constraint.name == "uq_chunk_index":
                col_names = {c.name for c in constraint.columns}
                assert col_names == {"document_id", "version_id", "chunk_index"}
                return
        pytest.fail("uq_chunk_index constraint not found")

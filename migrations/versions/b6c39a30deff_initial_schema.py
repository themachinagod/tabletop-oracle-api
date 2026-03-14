"""Initial schema: 14 tables, 12 enums, indexes, triggers.

Creates the complete F001 relational schema for tabletop-oracle.

Enum types (12): user_role, user_status, oauth_provider, game_complexity,
    document_type, document_format, document_status, session_status,
    message_type, context_attachment_type, feedback_rating, model_capability

Tables (14): users, games, game_tags, expansions, documents,
    document_versions, sessions, session_expansions, messages, citations,
    context_attachments, player_feedback, model_slots, token_usage_log,
    guardrail_config

Additional DDL: update_updated_at() trigger function, per-table triggers,
    partial indexes, GIN full-text index, CHECK constraints.

Design reference: docs/architecture/f001-data-model-api-conventions/design.md
Epic: themachinagod/tabletop-oracle-docs#4

Revision ID: b6c39a30deff
Revises:
Create Date: 2026-03-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6c39a30deff"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# ---------------------------------------------------------------------------
# Enum type definitions — order does not matter, they have no dependencies
# ---------------------------------------------------------------------------
ENUM_TYPES = [
    ("user_role", ["player", "curator"]),
    ("user_status", ["active", "disabled"]),
    ("oauth_provider", ["google", "microsoft"]),
    ("game_complexity", ["light", "medium", "heavy"]),
    (
        "document_type",
        ["core_rules", "faq", "errata", "expansion_rules", "strategy", "other"],
    ),
    ("document_format", ["pdf", "markdown", "text", "html", "docx"]),
    ("document_status", ["uploaded", "parsing", "processed", "error"]),
    ("session_status", ["active", "archived"]),
    ("message_type", ["user_question", "ai_answer", "ai_clarification", "system"]),
    ("context_attachment_type", ["text", "image"]),
    ("feedback_rating", ["positive", "negative"]),
    (
        "model_capability",
        [
            "intent_analysis",
            "retrieval_augmentation",
            "answer_synthesis",
            "clarification_generation",
            "concept_extraction",
            "vision_processing",
        ],
    ),
]

# Tables that receive the update_updated_at trigger
TRIGGER_TABLES = ["games", "expansions", "documents", "model_slots", "guardrail_config"]


def _create_enums() -> None:
    """Create all PostgreSQL enum types."""
    for name, values in ENUM_TYPES:
        values_sql = ", ".join(f"'{v}'" for v in values)
        op.execute(sa.text(f"CREATE TYPE {name} AS ENUM ({values_sql})"))


def _drop_enums() -> None:
    """Drop all PostgreSQL enum types in reverse order."""
    for name, _ in reversed(ENUM_TYPES):
        op.execute(sa.text(f"DROP TYPE IF EXISTS {name}"))


def _create_trigger_function() -> None:
    """Create the shared update_updated_at() PL/pgSQL function."""
    op.execute(
        sa.text("""
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    )


def _apply_triggers() -> None:
    """Apply update_updated_at trigger to designated tables."""
    for table in TRIGGER_TABLES:
        op.execute(
            sa.text(f"""
            CREATE TRIGGER trg_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at()
        """)
        )


def _drop_triggers() -> None:
    """Drop all update_updated_at triggers."""
    for table in reversed(TRIGGER_TABLES):
        op.execute(sa.text(f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table}"))


def _create_tables() -> None:
    """Create all 14 tables with columns, defaults, and inline constraints."""
    # ---- users ----
    op.create_table(
        "users",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "oauth_provider",
            sa.Enum("google", "microsoft", name="oauth_provider", create_type=False),
            nullable=False,
        ),
        sa.Column("oauth_subject_id", sa.Text(), nullable=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("player", "curator", name="user_role", create_type=False),
            nullable=False,
            server_default="player",
        ),
        sa.Column(
            "status",
            sa.Enum("active", "disabled", name="user_status", create_type=False),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_login_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    # Partial unique index: only enforce uniqueness where oauth_subject_id is not null
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_users_oauth "
            "ON users (oauth_provider, oauth_subject_id) "
            "WHERE oauth_subject_id IS NOT NULL"
        )
    )

    # ---- games ----
    op.create_table(
        "games",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("created_by", sa.UUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("publisher", sa.Text(), nullable=True),
        sa.Column("year_published", sa.Integer(), nullable=True),
        sa.Column("edition", sa.Text(), nullable=True),
        sa.Column("min_players", sa.Integer(), nullable=True),
        sa.Column("max_players", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("cover_image_url", sa.Text(), nullable=True),
        sa.Column(
            "complexity",
            sa.Enum("light", "medium", "heavy", name="game_complexity", create_type=False),
            nullable=True,
        ),
        sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "(min_players IS NULL AND max_players IS NULL) "
            "OR (min_players IS NOT NULL AND max_players IS NOT NULL "
            "AND min_players <= max_players)",
            name="ck_games_players",
        ),
    )
    op.execute(
        sa.text("CREATE INDEX idx_games_archived ON games (archived_at) WHERE archived_at IS NULL")
    )
    op.execute(
        sa.text("CREATE INDEX idx_games_name ON games USING gin (to_tsvector('english', name))")
    )

    # ---- game_tags ----
    op.create_table(
        "game_tags",
        sa.Column(
            "game_id",
            sa.UUID(),
            sa.ForeignKey("games.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("game_id", "tag"),
    )
    op.create_index("idx_game_tags_tag", "game_tags", ["tag"])

    # ---- expansions ----
    op.create_table(
        "expansions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "game_id",
            sa.UUID(),
            sa.ForeignKey("games.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("year_published", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_expansions_game", "expansions", ["game_id"])

    # ---- documents ----
    op.create_table(
        "documents",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("game_id", sa.UUID(), sa.ForeignKey("games.id"), nullable=False),
        sa.Column(
            "expansion_id",
            sa.UUID(),
            sa.ForeignKey("expansions.id"),
            nullable=True,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "core_rules",
                "faq",
                "errata",
                "expansion_rules",
                "strategy",
                "other",
                name="document_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "format",
            sa.Enum(
                "pdf",
                "markdown",
                "text",
                "html",
                "docx",
                name="document_format",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "parsing",
                "processed",
                "error",
                name="document_status",
                create_type=False,
            ),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("current_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "uploaded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_documents_game", "documents", ["game_id"])
    op.execute(
        sa.text(
            "CREATE INDEX idx_documents_expansion ON documents (expansion_id) "
            "WHERE expansion_id IS NOT NULL"
        )
    )
    op.create_index("idx_documents_status", "documents", ["status"])

    # ---- document_versions ----
    op.create_table(
        "document_versions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column(
            "uploaded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("uploaded_by", sa.UUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.UniqueConstraint("document_id", "version_number", name="uq_docver_version"),
    )
    op.create_index("idx_docver_document", "document_versions", ["document_id"])
    op.execute(
        sa.text(
            "CREATE INDEX idx_docver_active ON document_versions (document_id) "
            "WHERE is_active = TRUE"
        )
    )

    # ---- sessions ----
    op.create_table(
        "sessions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("game_id", sa.UUID(), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("player_count", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("active", "archived", name="session_status", create_type=False),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_active_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_sessions_user", "sessions", ["user_id"])
    op.create_index("idx_sessions_game", "sessions", ["game_id"])
    op.execute(
        sa.text(
            "CREATE INDEX idx_sessions_user_active ON sessions (user_id, status) "
            "WHERE status = 'active'"
        )
    )
    op.execute(
        sa.text("CREATE INDEX idx_sessions_last_active ON sessions (user_id, last_active_at DESC)")
    )

    # ---- session_expansions ----
    op.create_table(
        "session_expansions",
        sa.Column(
            "session_id",
            sa.UUID(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "expansion_id",
            sa.UUID(),
            sa.ForeignKey("expansions.id"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("session_id", "expansion_id"),
    )

    # ---- messages ----
    op.create_table(
        "messages",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("session_id", sa.UUID(), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "user_question",
                "ai_answer",
                "ai_clarification",
                "system",
                name="message_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column(
            "in_reply_to_id",
            sa.UUID(),
            sa.ForeignKey("messages.id"),
            nullable=True,
        ),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("processing_duration_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("session_id", "sequence", name="uq_messages_sequence"),
        sa.CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_messages_confidence",
        ),
    )
    op.create_index("idx_messages_session", "messages", ["session_id", "sequence"])

    # ---- citations ----
    op.create_table(
        "citations",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "message_id",
            sa.UUID(),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id"),
            nullable=False,
        ),
        sa.Column("document_name", sa.Text(), nullable=False),
        sa.Column(
            "document_type",
            sa.Enum(
                "core_rules",
                "faq",
                "errata",
                "expansion_rules",
                "strategy",
                "other",
                name="document_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("excerpt", sa.Text(), nullable=False),
    )
    op.create_index("idx_citations_message", "citations", ["message_id"])

    # ---- context_attachments ----
    op.create_table(
        "context_attachments",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "message_id",
            sa.UUID(),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "type",
            sa.Enum("text", "image", name="context_attachment_type", create_type=False),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("file_name", sa.Text(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("processed_description", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "(type = 'text' AND content IS NOT NULL) "
            "OR (type = 'image' AND file_path IS NOT NULL)",
            name="ck_context_type",
        ),
    )
    op.create_index("idx_context_message", "context_attachments", ["message_id"])

    # ---- player_feedback ----
    op.create_table(
        "player_feedback",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "message_id",
            sa.UUID(),
            sa.ForeignKey("messages.id"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.UUID(),
            sa.ForeignKey("sessions.id"),
            nullable=False,
        ),
        sa.Column("game_id", sa.UUID(), sa.ForeignKey("games.id"), nullable=False),
        sa.Column(
            "rating",
            sa.Enum("positive", "negative", name="feedback_rating", create_type=False),
            nullable=False,
        ),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("message_id", name="uq_feedback_message"),
    )
    op.create_index("idx_feedback_game", "player_feedback", ["game_id"])
    op.create_index("idx_feedback_session", "player_feedback", ["session_id"])
    op.create_index("idx_feedback_rating", "player_feedback", ["game_id", "rating"])

    # ---- model_slots ----
    op.create_table(
        "model_slots",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "capability",
            sa.Enum(
                "intent_analysis",
                "retrieval_augmentation",
                "answer_synthesis",
                "clarification_generation",
                "concept_extraction",
                "vision_processing",
                name="model_capability",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("max_tokens_per_call", sa.Integer(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("fallback_provider", sa.Text(), nullable=True),
        sa.Column("fallback_model_id", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("capability", name="uq_model_slot_capability"),
    )

    # ---- token_usage_log ----
    op.create_table(
        "token_usage_log",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "capability",
            sa.Enum(
                "intent_analysis",
                "retrieval_augmentation",
                "answer_synthesis",
                "clarification_generation",
                "concept_extraction",
                "vision_processing",
                name="model_capability",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column(
            "session_id",
            sa.UUID(),
            sa.ForeignKey("sessions.id"),
            nullable=True,
        ),
        sa.Column(
            "message_id",
            sa.UUID(),
            sa.ForeignKey("messages.id"),
            nullable=True,
        ),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_token_usage_user", "token_usage_log", ["user_id", "created_at"])
    op.create_index("idx_token_usage_session", "token_usage_log", ["session_id"])
    op.create_index("idx_token_usage_document", "token_usage_log", ["document_id"])
    op.create_index("idx_token_usage_daily", "token_usage_log", ["created_at"])
    op.create_index(
        "idx_token_usage_game",
        "token_usage_log",
        ["session_id", "created_at"],
    )

    # ---- guardrail_config ----
    op.create_table(
        "guardrail_config",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "enforcement_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column("max_tokens_per_query", sa.Integer(), nullable=True),
        sa.Column("max_model_calls_per_query", sa.Integer(), nullable=True),
        sa.Column("max_queries_per_session", sa.Integer(), nullable=True),
        sa.Column("daily_token_budget", sa.Integer(), nullable=True),
        sa.Column("daily_query_budget", sa.Integer(), nullable=True),
        sa.Column("per_document_ingestion_limit", sa.Integer(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


# ---------------------------------------------------------------------------
# Tables in reverse dependency order for downgrade
# ---------------------------------------------------------------------------
TABLES_DROP_ORDER = [
    "guardrail_config",
    "token_usage_log",
    "model_slots",
    "player_feedback",
    "context_attachments",
    "citations",
    "messages",
    "session_expansions",
    "sessions",
    "document_versions",
    "documents",
    "expansions",
    "game_tags",
    "games",
    "users",
]


def upgrade() -> None:
    """Create the complete F001 relational schema."""
    _create_enums()
    _create_tables()
    _create_trigger_function()
    _apply_triggers()


def downgrade() -> None:
    """Drop the complete F001 relational schema in reverse dependency order."""
    _drop_triggers()
    op.execute(sa.text("DROP FUNCTION IF EXISTS update_updated_at()"))

    for table in TABLES_DROP_ORDER:
        op.drop_table(table)

    _drop_enums()

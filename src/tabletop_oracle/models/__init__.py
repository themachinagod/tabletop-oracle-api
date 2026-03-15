"""SQLAlchemy ORM models.

Re-exports all model classes and enums. Importing this module triggers
SQLAlchemy metadata registration for all models, which is required for
Alembic auto-detection of schema changes.
"""

from tabletop_oracle.models.auth_session import AuthSession
from tabletop_oracle.models.base import Base, MappedBase
from tabletop_oracle.models.document import Document, DocumentChunk, DocumentVersion
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
from tabletop_oracle.models.expansion import Expansion
from tabletop_oracle.models.feedback import PlayerFeedback
from tabletop_oracle.models.game import Game, GameTag
from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.models.message import Citation, ContextAttachment, Message
from tabletop_oracle.models.model_slot import ModelSlot, TokenUsageLog
from tabletop_oracle.models.session import Session, SessionExpansion
from tabletop_oracle.models.user import User

__all__ = [
    "AuthSession",
    "Base",
    "Citation",
    "ContextAttachment",
    "ContextAttachmentType",
    "Document",
    "DocumentChunk",
    "DocumentFormat",
    "DocumentStatus",
    "DocumentType",
    "DocumentVersion",
    "Expansion",
    "FeedbackRating",
    "Game",
    "GameComplexity",
    "GameTag",
    "GuardrailConfig",
    "MappedBase",
    "Message",
    "MessageType",
    "ModelCapability",
    "ModelSlot",
    "OAuthProvider",
    "PlayerFeedback",
    "Session",
    "SessionExpansion",
    "SessionStatus",
    "TokenUsageLog",
    "User",
    "UserRole",
    "UserStatus",
]

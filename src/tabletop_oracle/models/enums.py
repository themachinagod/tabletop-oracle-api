"""Python enum classes matching the 12 PostgreSQL enum types.

Each enum mirrors a ``CREATE TYPE ... AS ENUM (...)`` definition from the
initial schema migration. Values MUST stay in sync with the database types.
The enum names match the PostgreSQL type names in snake_case; the Python
class names use PascalCase.
"""

import enum


class OAuthProvider(enum.Enum):
    """OAuth identity providers (``oauth_provider`` type)."""

    GOOGLE = "google"
    MICROSOFT = "microsoft"


class UserRole(enum.Enum):
    """Application-level user roles (``user_role`` type)."""

    PLAYER = "player"
    CURATOR = "curator"


class UserStatus(enum.Enum):
    """User account status (``user_status`` type)."""

    ACTIVE = "active"
    DISABLED = "disabled"


class GameComplexity(enum.Enum):
    """Game complexity classification (``game_complexity`` type)."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class DocumentType(enum.Enum):
    """Document content classification (``document_type`` type)."""

    CORE_RULES = "core_rules"
    FAQ = "faq"
    ERRATA = "errata"
    EXPANSION_RULES = "expansion_rules"
    STRATEGY = "strategy"
    OTHER = "other"


class DocumentFormat(enum.Enum):
    """Document file format (``document_format`` type)."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    DOCX = "docx"


class DocumentStatus(enum.Enum):
    """Document processing status (``document_status`` type)."""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    PROCESSED = "processed"
    ERROR = "error"


class SessionStatus(enum.Enum):
    """Game session lifecycle status (``session_status`` type)."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class MessageType(enum.Enum):
    """Chat message type (``message_type`` type)."""

    USER_QUESTION = "user_question"
    AI_ANSWER = "ai_answer"
    AI_CLARIFICATION = "ai_clarification"
    SYSTEM = "system"


class ContextAttachmentType(enum.Enum):
    """Context attachment content type (``context_attachment_type`` type)."""

    TEXT = "text"
    IMAGE = "image"


class FeedbackRating(enum.Enum):
    """Player feedback rating (``feedback_rating`` type)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ModelCapability(enum.Enum):
    """AI model capability slot (``model_capability`` type)."""

    INTENT_ANALYSIS = "intent_analysis"
    RETRIEVAL_AUGMENTATION = "retrieval_augmentation"
    ANSWER_SYNTHESIS = "answer_synthesis"
    CLARIFICATION_GENERATION = "clarification_generation"
    CONCEPT_EXTRACTION = "concept_extraction"
    VISION_PROCESSING = "vision_processing"
    EMBEDDING = "embedding"

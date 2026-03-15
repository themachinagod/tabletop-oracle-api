"""Pydantic schemas for Message, Citation, and ContextAttachment responses.

Covers response schemas for the message history and cancellation endpoints.
Citations are included on ai_answer messages; context attachments on
user_question messages. All field types match the ORM models and F001
design conventions.
"""

from __future__ import annotations

import uuid  # noqa: TC003
from datetime import datetime  # noqa: TC003

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CitationResponse(BaseModel):
    """Response schema for a citation attached to an AI answer message.

    Attributes:
        id: Citation UUID.
        document_id: FK to the cited document.
        document_name: Denormalised document name at citation time.
        document_type: Denormalised document type at citation time.
        section_path: Document section path (optional).
        page_number: Page number reference (optional).
        excerpt: Quoted excerpt from the source.
    """

    id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    document_type: str
    section_path: str | None
    page_number: int | None
    excerpt: str

    model_config = {"from_attributes": True}


class ContextAttachmentResponse(BaseModel):
    """Response schema for a context attachment on a user question message.

    Attributes:
        id: Attachment UUID.
        type: Attachment type (text or image).
        content: Inline text content (for text attachments).
        file_path: Blob store path (for image attachments).
        file_name: Original file name (optional).
        file_size: File size in bytes (optional).
        processed_description: AI-generated description (optional).
    """

    id: uuid.UUID
    type: str
    content: str | None
    file_path: str | None
    file_name: str | None
    file_size: int | None
    processed_description: str | None

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    """Response schema for a single message in conversation history.

    Citations are populated for ai_answer messages (empty list otherwise).
    Context attachments are populated for user_question messages (empty
    list otherwise).

    Attributes:
        id: Message UUID.
        session_id: FK to the owning game session.
        type: Message type (user_question, ai_answer, ai_clarification, system).
        content: Message text content.
        sequence: Monotonically increasing sequence within the session.
        in_reply_to_id: Self-referential FK for threading (optional).
        confidence_score: AI confidence 0.0-1.0 (optional, ai_answer only).
        processing_duration_ms: Response generation time in ms (optional).
        created_at: Message creation timestamp.
        cancelled_at: Cancellation timestamp, if cancelled (optional).
        citations: Source citations (ai_answer messages only).
        context_attachments: User-provided context (user_question messages only).
    """

    id: uuid.UUID
    session_id: uuid.UUID
    type: str
    content: str
    sequence: int
    in_reply_to_id: uuid.UUID | None
    confidence_score: float | None
    processing_duration_ms: int | None
    created_at: datetime
    cancelled_at: datetime | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    context_attachments: list[ContextAttachmentResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class CancelResponse(BaseModel):
    """Response schema for the cancellation endpoint.

    Returned in the data envelope when a message cancellation is accepted.

    Attributes:
        id: The message UUID that was cancelled.
        status: Always ``"cancelling"`` to indicate the cancellation is
            in progress.
    """

    id: uuid.UUID
    status: str = "cancelling"

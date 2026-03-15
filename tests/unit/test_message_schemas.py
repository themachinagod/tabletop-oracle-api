"""Unit tests for message response schemas.

Tests Pydantic schema validation and serialization for MessageResponse,
CitationResponse, and ContextAttachmentResponse.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from tabletop_oracle.schemas.message import (
    CitationResponse,
    ContextAttachmentResponse,
    MessageResponse,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 14, 12, 0, 0, tzinfo=UTC)
MSG_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()
CITATION_ID = uuid.uuid4()
ATTACHMENT_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Tests: CitationResponse
# ---------------------------------------------------------------------------


class TestCitationResponse:
    """Tests for CitationResponse schema."""

    def test_creates_with_all_fields(self) -> None:
        """CitationResponse accepts all fields."""
        citation = CitationResponse(
            id=CITATION_ID,
            document_id=DOC_ID,
            document_name="Core Rulebook",
            document_type="core_rules",
            section_path="Chapter 3 > Combat",
            page_number=42,
            excerpt="Roll 2d6 for initiative.",
        )
        assert citation.document_name == "Core Rulebook"
        assert citation.page_number == 42

    def test_optional_fields_accept_none(self) -> None:
        """CitationResponse accepts None for optional fields."""
        citation = CitationResponse(
            id=CITATION_ID,
            document_id=DOC_ID,
            document_name="FAQ",
            document_type="faq",
            section_path=None,
            page_number=None,
            excerpt="Yes, you can.",
        )
        assert citation.section_path is None
        assert citation.page_number is None


# ---------------------------------------------------------------------------
# Tests: ContextAttachmentResponse
# ---------------------------------------------------------------------------


class TestContextAttachmentResponse:
    """Tests for ContextAttachmentResponse schema."""

    def test_creates_text_attachment(self) -> None:
        """ContextAttachmentResponse represents a text attachment."""
        attachment = ContextAttachmentResponse(
            id=ATTACHMENT_ID,
            type="text",
            content="Additional context here",
            file_path=None,
            file_name=None,
            file_size=None,
            processed_description=None,
        )
        assert attachment.type == "text"
        assert attachment.content == "Additional context here"

    def test_creates_image_attachment(self) -> None:
        """ContextAttachmentResponse represents an image attachment."""
        attachment = ContextAttachmentResponse(
            id=ATTACHMENT_ID,
            type="image",
            content=None,
            file_path="/blobs/img.png",
            file_name="board_photo.png",
            file_size=204800,
            processed_description="A game board setup",
        )
        assert attachment.type == "image"
        assert attachment.file_path == "/blobs/img.png"
        assert attachment.file_size == 204800


# ---------------------------------------------------------------------------
# Tests: MessageResponse
# ---------------------------------------------------------------------------


class TestMessageResponse:
    """Tests for MessageResponse schema."""

    def test_creates_user_question(self) -> None:
        """MessageResponse represents a user question with context attachments."""
        msg = MessageResponse(
            id=MSG_ID,
            session_id=SESSION_ID,
            type="user_question",
            content="How does combat work?",
            sequence=1,
            in_reply_to_id=None,
            confidence_score=None,
            processing_duration_ms=None,
            created_at=NOW,
            citations=[],
            context_attachments=[
                ContextAttachmentResponse(
                    id=ATTACHMENT_ID,
                    type="text",
                    content="Page from rulebook",
                    file_path=None,
                    file_name=None,
                    file_size=None,
                    processed_description=None,
                )
            ],
        )
        assert msg.type == "user_question"
        assert len(msg.context_attachments) == 1
        assert len(msg.citations) == 0

    def test_creates_ai_answer(self) -> None:
        """MessageResponse represents an AI answer with citations."""
        msg = MessageResponse(
            id=MSG_ID,
            session_id=SESSION_ID,
            type="ai_answer",
            content="Combat uses a d20 system.",
            sequence=2,
            in_reply_to_id=None,
            confidence_score=0.92,
            processing_duration_ms=1500,
            created_at=NOW,
            citations=[
                CitationResponse(
                    id=CITATION_ID,
                    document_id=DOC_ID,
                    document_name="Core Rulebook",
                    document_type="core_rules",
                    section_path="Chapter 3",
                    page_number=42,
                    excerpt="Roll d20 + modifier.",
                )
            ],
            context_attachments=[],
        )
        assert msg.type == "ai_answer"
        assert msg.confidence_score == 0.92
        assert len(msg.citations) == 1

    def test_defaults_to_empty_lists(self) -> None:
        """MessageResponse defaults citations and context_attachments to empty."""
        msg = MessageResponse(
            id=MSG_ID,
            session_id=SESSION_ID,
            type="system",
            content="Session started.",
            sequence=0,
            in_reply_to_id=None,
            confidence_score=None,
            processing_duration_ms=None,
            created_at=NOW,
        )
        assert msg.citations == []
        assert msg.context_attachments == []

    def test_serialization_round_trip(self) -> None:
        """MessageResponse serializes and deserializes consistently."""
        msg = MessageResponse(
            id=MSG_ID,
            session_id=SESSION_ID,
            type="ai_answer",
            content="Answer text.",
            sequence=5,
            in_reply_to_id=None,
            confidence_score=0.85,
            processing_duration_ms=800,
            created_at=NOW,
            citations=[],
            context_attachments=[],
        )
        data = msg.model_dump(mode="json")
        restored = MessageResponse.model_validate(data)
        assert restored.id == msg.id
        assert restored.confidence_score == 0.85
        assert restored.sequence == 5

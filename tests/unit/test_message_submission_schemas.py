"""Unit tests for message submission Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tabletop_oracle.schemas.message import (
    ContextAttachmentRequest,
    MessageSubmitRequest,
)


class TestMessageSubmitRequest:
    """Tests for MessageSubmitRequest validation."""

    def test_valid_text_only(self) -> None:
        """Minimal valid request with text content only."""
        req = MessageSubmitRequest(content="How does combat work?")
        assert req.content == "How does combat work?"
        assert req.context_attachments == []

    def test_valid_with_text_attachment(self) -> None:
        """Request with a text context attachment."""
        req = MessageSubmitRequest(
            content="Q?",
            context_attachments=[ContextAttachmentRequest(type="text", content="ctx")],
        )
        assert len(req.context_attachments) == 1

    def test_valid_with_image_attachment(self) -> None:
        """Request with an image context attachment."""
        req = MessageSubmitRequest(
            content="What is on the board?",
            context_attachments=[ContextAttachmentRequest(type="image", file_name="b.jpg")],
        )
        assert req.context_attachments[0].file_name == "b.jpg"

    def test_empty_content_rejected(self) -> None:
        """Empty string content fails validation."""
        with pytest.raises(ValidationError):
            MessageSubmitRequest(content="")

    def test_invalid_attachment_type_rejected(self) -> None:
        """Attachment type not matching text|image fails."""
        with pytest.raises(ValidationError):
            MessageSubmitRequest(
                content="Q",
                context_attachments=[ContextAttachmentRequest(type="video", content="d")],
            )

    def test_content_max_length(self) -> None:
        """Content exceeding 10000 characters fails."""
        with pytest.raises(ValidationError):
            MessageSubmitRequest(content="x" * 10001)

    def test_content_at_max_length(self) -> None:
        """Content exactly at 10000 characters succeeds."""
        req = MessageSubmitRequest(content="x" * 10000)
        assert len(req.content) == 10000

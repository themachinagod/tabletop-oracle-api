"""Unit tests for ContextPreparationStage.

Tests:
- Conversation history loaded correctly via ConversationContextService
- Text context passed through from context_attachments (AC-605)
- Image context processed via VisionService (AC-605)
- Processed descriptions stored on context_attachments
- Graceful handling of image processing failures
- No events yielded (silent stage)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from tabletop_oracle.models.enums import ContextAttachmentType
from tabletop_oracle.services.ai.stages.context_preparation import (
    ContextPreparationStage,
    _extract_format,
)
from tabletop_oracle.services.vision.models import (
    ImageDescription,
    VisionProcessingError,
    VisionUnavailableError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()


def _make_attachment(
    *,
    att_type: ContextAttachmentType = ContextAttachmentType.TEXT,
    content: str | None = None,
    file_path: str | None = None,
    file_name: str | None = None,
) -> MagicMock:
    """Build a mock ContextAttachment.

    Args:
        att_type: Attachment type (text or image).
        content: Text content (for text attachments).
        file_path: Blob storage path (for image attachments).
        file_name: Original file name (for image attachments).

    Returns:
        MagicMock mimicking a ContextAttachment ORM instance.
    """
    mock = MagicMock()
    mock.id = uuid.uuid4()
    mock.type = att_type
    mock.content = content
    mock.file_path = file_path
    mock.file_name = file_name
    mock.processed_description = None
    return mock


def _make_ctx(
    attachments: list[Any] | None = None,
) -> MagicMock:
    """Build a mock PipelineContext.

    Args:
        attachments: List of context attachments on the user_message.

    Returns:
        MagicMock mimicking a PipelineContext dataclass.
    """
    ctx = MagicMock()
    ctx.session.id = _SESSION_ID
    ctx.user_message.id = _MESSAGE_ID
    ctx.user_message.context_attachments = attachments or []
    ctx.conversation_history = []
    ctx.ad_hoc_text_context = None
    ctx.ad_hoc_image_descriptions = []
    return ctx


def _make_stage(
    *,
    conversation_messages: list[Any] | None = None,
    conversation_summary: str | None = None,
    vision_result: ImageDescription | None = None,
    vision_error: Exception | None = None,
    storage_data: bytes = b"fake-image-bytes",
    storage_error: Exception | None = None,
) -> tuple[ContextPreparationStage, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build a ContextPreparationStage with mocked dependencies.

    Args:
        conversation_messages: Messages returned by get_context_with_summary.
        conversation_summary: Summary returned by get_context_with_summary.
        vision_result: Result from VisionService.describe_image.
        vision_error: Exception raised by VisionService.describe_image.
        storage_data: Bytes returned by blob storage retrieve.
        storage_error: Exception raised by blob storage retrieve.

    Returns:
        Tuple of (stage, conversation_service, vision_service, blob_storage, db).
    """
    conversation_service = AsyncMock()
    conversation_service.get_context_with_summary.return_value = (
        conversation_messages or [],
        conversation_summary,
    )

    vision_service = AsyncMock()
    if vision_error:
        vision_service.describe_image.side_effect = vision_error
    elif vision_result:
        vision_service.describe_image.return_value = vision_result
    else:
        vision_service.describe_image.return_value = ImageDescription(
            text="A game board diagram",
            confidence="high",
            input_tokens=100,
            output_tokens=50,
        )

    blob_storage = AsyncMock()
    if storage_error:
        blob_storage.retrieve.side_effect = storage_error
    else:
        blob_storage.retrieve.return_value = storage_data

    db = AsyncMock()

    stage = ContextPreparationStage(
        conversation_service=conversation_service,
        vision_service=vision_service,
        blob_storage=blob_storage,
        db=db,
    )

    return stage, conversation_service, vision_service, blob_storage, db


# ---------------------------------------------------------------------------
# Tests: Conversation History
# ---------------------------------------------------------------------------


class TestConversationHistoryLoading:
    """Tests for conversation history loading."""

    async def test_loads_conversation_history_into_context(self) -> None:
        """Conversation history from service is set on pipeline context."""
        messages = [MagicMock(), MagicMock()]
        stage, conv_svc, _, _, _ = _make_stage(conversation_messages=messages)
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        conv_svc.get_context_with_summary.assert_awaited_once_with(
            session_id=_SESSION_ID,
        )
        assert ctx.conversation_history == messages

    async def test_empty_history_sets_empty_list(self) -> None:
        """Empty conversation history results in empty list on context."""
        stage, _, _, _, _ = _make_stage(conversation_messages=[])
        ctx = _make_ctx()

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.conversation_history == []


# ---------------------------------------------------------------------------
# Tests: Text Context
# ---------------------------------------------------------------------------


class TestTextContextExtraction:
    """Tests for text ad-hoc context extraction."""

    async def test_single_text_attachment_sets_ad_hoc_text(self) -> None:
        """Single text attachment content is set on context."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.TEXT,
            content="The ranger has 3 HP left.",
        )
        stage, _, _, _, _ = _make_stage()
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_text_context == "The ranger has 3 HP left."

    async def test_multiple_text_attachments_concatenated_with_separator(
        self,
    ) -> None:
        """Multiple text attachments are joined with double newline."""
        attachments = [
            _make_attachment(
                att_type=ContextAttachmentType.TEXT,
                content="First context",
            ),
            _make_attachment(
                att_type=ContextAttachmentType.TEXT,
                content="Second context",
            ),
        ]
        stage, _, _, _, _ = _make_stage()
        ctx = _make_ctx(attachments=attachments)

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_text_context == "First context\n\nSecond context"

    async def test_text_attachment_with_empty_content_skipped(self) -> None:
        """Text attachment with empty/None content is skipped."""
        attachments = [
            _make_attachment(att_type=ContextAttachmentType.TEXT, content=None),
            _make_attachment(att_type=ContextAttachmentType.TEXT, content="Valid"),
        ]
        stage, _, _, _, _ = _make_stage()
        ctx = _make_ctx(attachments=attachments)

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_text_context == "Valid"

    async def test_no_text_attachments_leaves_none(self) -> None:
        """No text attachments leaves ad_hoc_text_context as None."""
        stage, _, _, _, _ = _make_stage()
        ctx = _make_ctx(attachments=[])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_text_context is None

    async def test_image_attachments_do_not_affect_text_context(self) -> None:
        """Image attachments are not included in text context."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path="images/board.png",
            file_name="board.png",
        )
        stage, _, _, _, _ = _make_stage()
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_text_context is None


# ---------------------------------------------------------------------------
# Tests: Image Context
# ---------------------------------------------------------------------------


class TestImageContextProcessing:
    """Tests for image ad-hoc context via VisionService."""

    async def test_image_attachment_processed_via_vision_service(self) -> None:
        """Image attachment is described by VisionService."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path="images/board.png",
            file_name="board.png",
        )
        vision_result = ImageDescription(
            text="A hexagonal game board",
            confidence="high",
            input_tokens=200,
            output_tokens=80,
        )
        stage, _, vision_svc, storage, _ = _make_stage(vision_result=vision_result)
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        storage.retrieve.assert_awaited_once_with("images/board.png")
        vision_svc.describe_image.assert_awaited_once()
        call_kwargs = vision_svc.describe_image.call_args
        assert call_kwargs.kwargs["image_data"] == b"fake-image-bytes"
        assert call_kwargs.kwargs["image_format"] == "png"
        assert call_kwargs.kwargs["attribution"].session_id == _SESSION_ID
        assert call_kwargs.kwargs["attribution"].message_id == _MESSAGE_ID
        assert ctx.ad_hoc_image_descriptions == ["A hexagonal game board"]

    async def test_image_description_persisted_to_attachment(self) -> None:
        """Processed description is written back to attachment record."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path="images/card.jpg",
            file_name="card.jpg",
        )
        vision_result = ImageDescription(
            text="A spell card showing Fireball",
            confidence="high",
            input_tokens=150,
            output_tokens=60,
        )
        stage, _, _, _, db = _make_stage(vision_result=vision_result)
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert attachment.processed_description == "A spell card showing Fireball"
        db.flush.assert_awaited_once()

    async def test_multiple_images_all_processed(self) -> None:
        """Multiple image attachments are all processed independently."""
        attachments = [
            _make_attachment(
                att_type=ContextAttachmentType.IMAGE,
                file_path="images/a.png",
                file_name="a.png",
            ),
            _make_attachment(
                att_type=ContextAttachmentType.IMAGE,
                file_path="images/b.jpeg",
                file_name="b.jpeg",
            ),
        ]
        descriptions = [
            ImageDescription(text="Image A", confidence="high"),
            ImageDescription(text="Image B", confidence="medium"),
        ]
        stage, _, vision_svc, _, _ = _make_stage()
        vision_svc.describe_image.side_effect = descriptions
        ctx = _make_ctx(attachments=attachments)

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert len(ctx.ad_hoc_image_descriptions) == 2
        assert ctx.ad_hoc_image_descriptions == ["Image A", "Image B"]

    async def test_image_with_no_file_path_skipped(self) -> None:
        """Image attachment without file_path is skipped gracefully."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path=None,
            file_name=None,
        )
        stage, _, vision_svc, _, _ = _make_stage()
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        vision_svc.describe_image.assert_not_awaited()
        assert ctx.ad_hoc_image_descriptions == []


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestImageErrorHandling:
    """Tests for graceful error handling during image processing."""

    async def test_storage_retrieval_failure_skips_image(self) -> None:
        """Blob storage failure skips the image without crashing."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path="images/missing.png",
            file_name="missing.png",
        )
        stage, _, vision_svc, _, _ = _make_stage(
            storage_error=FileNotFoundError("not found"),
        )
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        vision_svc.describe_image.assert_not_awaited()
        assert ctx.ad_hoc_image_descriptions == []

    async def test_vision_processing_failure_skips_image(self) -> None:
        """VisionProcessingError skips the image without crashing."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path="images/corrupt.png",
            file_name="corrupt.png",
        )
        stage, _, _, _, _ = _make_stage(
            vision_error=VisionProcessingError("model failed", provider="openai"),
        )
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_image_descriptions == []

    async def test_vision_unavailable_skips_image(self) -> None:
        """VisionUnavailableError skips the image without crashing."""
        attachment = _make_attachment(
            att_type=ContextAttachmentType.IMAGE,
            file_path="images/board.png",
            file_name="board.png",
        )
        stage, _, _, _, _ = _make_stage(
            vision_error=VisionUnavailableError(),
        )
        ctx = _make_ctx(attachments=[attachment])

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_image_descriptions == []

    async def test_partial_failure_keeps_successful_images(self) -> None:
        """When some images fail, successful ones are still kept."""
        attachments = [
            _make_attachment(
                att_type=ContextAttachmentType.IMAGE,
                file_path="images/good.png",
                file_name="good.png",
            ),
            _make_attachment(
                att_type=ContextAttachmentType.IMAGE,
                file_path="images/bad.png",
                file_name="bad.png",
            ),
        ]
        good_result = ImageDescription(text="Good image", confidence="high")
        stage, _, vision_svc, _, _ = _make_stage()
        vision_svc.describe_image.side_effect = [
            good_result,
            VisionProcessingError("failed", provider="openai"),
        ]
        ctx = _make_ctx(attachments=attachments)

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_image_descriptions == ["Good image"]


# ---------------------------------------------------------------------------
# Tests: No Events Yielded
# ---------------------------------------------------------------------------


class TestNoEventsYielded:
    """Tests that the stage yields no SSE events."""

    async def test_no_attachments_yields_nothing(self) -> None:
        """Stage produces no SSE events."""
        stage, _, _, _, _ = _make_stage()
        ctx = _make_ctx()
        events = [event async for event in stage.execute(ctx, MagicMock())]
        assert events == []


# ---------------------------------------------------------------------------
# Tests: Mixed Attachments
# ---------------------------------------------------------------------------


class TestMixedAttachments:
    """Tests for messages with both text and image attachments."""

    async def test_mixed_attachments_both_processed(self) -> None:
        """Text and image attachments are processed independently."""
        attachments = [
            _make_attachment(
                att_type=ContextAttachmentType.TEXT,
                content="We are on round 3.",
            ),
            _make_attachment(
                att_type=ContextAttachmentType.IMAGE,
                file_path="images/board.png",
                file_name="board.png",
            ),
        ]
        vision_result = ImageDescription(
            text="Board state at round 3",
            confidence="high",
        )
        stage, _, _, _, _ = _make_stage(vision_result=vision_result)
        ctx = _make_ctx(attachments=attachments)

        async for _ in stage.execute(ctx, MagicMock()):
            pass

        assert ctx.ad_hoc_text_context == "We are on round 3."
        assert ctx.ad_hoc_image_descriptions == ["Board state at round 3"]


# ---------------------------------------------------------------------------
# Tests: _extract_format helper
# ---------------------------------------------------------------------------


class TestExtractFormat:
    """Tests for the _extract_format helper function."""

    def test_png_extension(self) -> None:
        assert _extract_format("board.png") == "png"

    def test_jpeg_extension(self) -> None:
        assert _extract_format("photo.jpeg") == "jpeg"

    def test_jpg_extension(self) -> None:
        assert _extract_format("photo.jpg") == "jpg"

    def test_upper_case_returns_lower(self) -> None:
        assert _extract_format("image.PNG") == "png"

    def test_path_with_directories(self) -> None:
        assert _extract_format("images/subfolder/board.webp") == "webp"

    def test_no_extension_defaults_to_png(self) -> None:
        assert _extract_format("noextension") == "png"

    def test_trailing_dot_defaults_to_png(self) -> None:
        assert _extract_format("file.") == "png"

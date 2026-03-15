"""Stage 1: Context preparation for the query pipeline.

Loads conversation history, passes through text ad-hoc context, and
processes image ad-hoc context via VisionService. Image descriptions
are written back to the ``context_attachments.processed_description``
column for persistence.

Design reference: EPIC-004 design.md, Stage 1: Context Preparation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabletop_oracle.models.enums import ContextAttachmentType
from tabletop_oracle.services.vision.models import (
    TokenAttribution,
    VisionProcessingError,
    VisionUnavailableError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.models.message import ContextAttachment
    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.ai.conversation import ConversationContextService
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.vision.protocols import VisionServiceProtocol
    from tabletop_oracle.storage.interface import BlobStorageInterface
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)


class ContextPreparationStage:
    """Prepares all context needed by downstream pipeline stages.

    Responsibilities:
    - Load relevant conversation history via ConversationContextService.
    - Pass through text ad-hoc context from context_attachments.
    - Process image ad-hoc context via VisionService.describe_image().
    - Write processed descriptions back to context_attachments.

    Args:
        conversation_service: Service for retrieving conversation history.
        vision_service: Shared vision processing service for images.
        blob_storage: Storage backend for retrieving image file data.
        db: Async database session for persisting processed descriptions.
    """

    def __init__(
        self,
        conversation_service: ConversationContextService,
        vision_service: VisionServiceProtocol,
        blob_storage: BlobStorageInterface,
        db: AsyncSession,
    ) -> None:
        self._conversation_service = conversation_service
        self._vision_service = vision_service
        self._blob_storage = blob_storage
        self._db = db

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Prepare context for downstream stages.

        Populates ``ctx.conversation_history``, ``ctx.ad_hoc_text_context``,
        and ``ctx.ad_hoc_image_descriptions``. Image descriptions are also
        persisted to the ``context_attachments.processed_description`` column.

        Args:
            ctx: Pipeline context with session and message metadata.
            model_client: Model client (unused by this stage, required
                by the PipelineStage protocol).

        Yields:
            No SSE events. This stage is silent.
        """
        await self._load_conversation_history(ctx)
        self._extract_text_context(ctx)
        await self._process_image_context(ctx)

        # Async generator must contain at least one yield (even if unreachable)
        # to satisfy the protocol's AsyncGenerator return type.
        return
        yield  # pragma: no cover — makes this function an async generator

    async def _load_conversation_history(self, ctx: PipelineContext) -> None:
        """Load relevant conversation history for follow-up context.

        Uses ConversationContextService to retrieve prior messages with
        sliding window + summary strategy.

        Args:
            ctx: Pipeline context to populate with conversation history.
        """
        session_id = ctx.session.id

        logger.debug(
            "Loading conversation history",
            extra={"session_id": str(session_id)},
        )

        messages, _ = await self._conversation_service.get_context_with_summary(
            session_id=session_id,
        )

        ctx.conversation_history = messages

        logger.info(
            "Conversation history loaded",
            extra={
                "session_id": str(session_id),
                "message_count": len(messages),
            },
        )

    def _extract_text_context(self, ctx: PipelineContext) -> None:
        """Extract text ad-hoc context from context attachments.

        Concatenates all text-type context attachments into a single
        string on the pipeline context.

        Args:
            ctx: Pipeline context with user_message containing attachments.
        """
        attachments: list[ContextAttachment] = ctx.user_message.context_attachments
        text_parts: list[str] = []

        for attachment in attachments:
            if attachment.type == ContextAttachmentType.TEXT and attachment.content:
                text_parts.append(attachment.content)

        if text_parts:
            ctx.ad_hoc_text_context = "\n\n".join(text_parts)
            logger.debug(
                "Text context extracted",
                extra={
                    "message_id": str(ctx.user_message.id),
                    "attachment_count": len(text_parts),
                },
            )

    async def _process_image_context(self, ctx: PipelineContext) -> None:
        """Process image ad-hoc context via VisionService.

        For each image-type context attachment:
        1. Retrieve image data from blob storage.
        2. Call VisionService.describe_image() with session/message attribution.
        3. Store the description in ctx.ad_hoc_image_descriptions.
        4. Write the description back to context_attachments.processed_description.

        Failures on individual images are logged and skipped — one bad
        image should not block the entire query.

        Args:
            ctx: Pipeline context with user_message containing attachments.
        """
        attachments: list[ContextAttachment] = ctx.user_message.context_attachments
        image_attachments = [a for a in attachments if a.type == ContextAttachmentType.IMAGE]

        if not image_attachments:
            return

        session_id = ctx.session.id
        message_id = ctx.user_message.id
        attribution = TokenAttribution(session_id=session_id, message_id=message_id)

        for attachment in image_attachments:
            description = await self._process_single_image(attachment, attribution)
            if description:
                ctx.ad_hoc_image_descriptions.append(description)

        if ctx.ad_hoc_image_descriptions:
            logger.info(
                "Image context processed",
                extra={
                    "message_id": str(message_id),
                    "processed_count": len(ctx.ad_hoc_image_descriptions),
                    "total_images": len(image_attachments),
                },
            )

    async def _process_single_image(
        self,
        attachment: ContextAttachment,
        attribution: TokenAttribution,
    ) -> str | None:
        """Process a single image attachment through VisionService.

        Retrieves image data from blob storage, calls VisionService, and
        persists the description back to the attachment record.

        Args:
            attachment: The image context attachment to process.
            attribution: Token usage attribution for tracking.

        Returns:
            The generated description text, or None if processing failed.
        """
        if not attachment.file_path:
            logger.warning(
                "Image attachment has no file_path, skipping",
                extra={"attachment_id": str(attachment.id)},
            )
            return None

        try:
            image_data = await self._blob_storage.retrieve(attachment.file_path)
        except Exception:
            logger.warning(
                "Failed to retrieve image from storage, skipping",
                extra={
                    "attachment_id": str(attachment.id),
                    "file_path": attachment.file_path,
                },
                exc_info=True,
            )
            return None

        image_format = _extract_format(attachment.file_name or attachment.file_path)

        try:
            result = await self._vision_service.describe_image(
                image_data=image_data,
                image_format=image_format,
                attribution=attribution,
            )
        except (VisionProcessingError, VisionUnavailableError, ValueError):
            logger.warning(
                "Vision processing failed for image attachment, skipping",
                extra={
                    "attachment_id": str(attachment.id),
                    "image_format": image_format,
                },
                exc_info=True,
            )
            return None

        # Persist description back to the attachment record
        attachment.processed_description = result.text
        await self._db.flush()

        return result.text


def _extract_format(filename: str) -> str:
    """Extract image format from a filename or path.

    Args:
        filename: File name or path to extract the extension from.

    Returns:
        Lowercase extension without the leading dot (e.g., "png", "jpeg").
        Defaults to "png" if no extension is found.
    """
    dot_idx = filename.rfind(".")
    if dot_idx == -1 or dot_idx == len(filename) - 1:
        return "png"
    return filename[dot_idx + 1 :].lower()

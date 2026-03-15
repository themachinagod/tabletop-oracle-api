"""Business logic for Document entity management.

Encapsulates upload orchestration (validate, store, create record, enqueue
processing), CRUD operations, reclassification, expansion re-association,
soft-delete, and repository delegation. All domain errors use the F001
exception hierarchy.
"""

from __future__ import annotations

import logging
import mimetypes
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tabletop_oracle.errors.exceptions import (
    ContentTooLargeError,
    NotFoundError,
    UnprocessableEntityError,
    ValidationError,
)
from tabletop_oracle.models.document import Document, DocumentVersion
from tabletop_oracle.models.enums import DocumentFormat, DocumentStatus, DocumentType

if TYPE_CHECKING:
    import uuid

    from fastapi import UploadFile

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.repositories.document_repository import DocumentRepository
    from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
    from tabletop_oracle.repositories.game_repository import GameRepository
    from tabletop_oracle.schemas.document import (
        DocumentFilters,
        DocumentTypeFilter,
    )
    from tabletop_oracle.storage.interface import BlobStorageInterface

logger = logging.getLogger(__name__)

#: Maximum file size in bytes (50 MB).
MAX_FILE_SIZE = 50 * 1024 * 1024

#: Map from file extension to DocumentFormat enum.
_EXTENSION_FORMAT_MAP: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".txt": DocumentFormat.TEXT,
    ".text": DocumentFormat.TEXT,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
    ".docx": DocumentFormat.DOCX,
}

#: Map from MIME type to DocumentFormat enum.
_MIME_FORMAT_MAP: dict[str, DocumentFormat] = {
    "application/pdf": DocumentFormat.PDF,
    "text/markdown": DocumentFormat.MARKDOWN,
    "text/plain": DocumentFormat.TEXT,
    "text/html": DocumentFormat.HTML,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.DOCX,
}


def _detect_format(filename: str, content_type: str | None) -> DocumentFormat | None:
    """Detect document format from filename extension and MIME type.

    Args:
        filename: Original filename with extension.
        content_type: MIME content type from the upload, if available.

    Returns:
        Detected DocumentFormat, or None if unrecognised.
    """
    # Try extension first (more reliable than browser-supplied MIME type)
    ext = "." + filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    fmt = _EXTENSION_FORMAT_MAP.get(ext)
    if fmt is not None:
        return fmt

    # Fall back to content type
    if content_type:
        fmt = _MIME_FORMAT_MAP.get(content_type)
        if fmt is not None:
            return fmt

    return None


class DocumentService:
    """Business logic for Document entity management and upload orchestration.

    Orchestrates parent game validation, file validation, blob storage,
    DB record creation, processing task enqueue (stub), CRUD operations,
    reclassification, expansion re-association, and soft-delete.

    Args:
        document_repo: Repository for document data access.
        game_repo: Repository for parent game validation.
        expansion_repo: Repository for expansion validation.
        blob_store: Blob storage interface for file persistence.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        game_repo: GameRepository,
        expansion_repo: ExpansionRepository,
        blob_store: BlobStorageInterface,
    ) -> None:
        self._document_repo = document_repo
        self._game_repo = game_repo
        self._expansion_repo = expansion_repo
        self._blob_store = blob_store

    async def upload(
        self,
        game_id: uuid.UUID,
        file: UploadFile,
        doc_type: DocumentTypeFilter,
        user_id: uuid.UUID,
        *,
        name: str | None = None,
        expansion_id: uuid.UUID | None = None,
    ) -> Document:
        """Upload a single document: validate, store file, create record, enqueue processing.

        Args:
            game_id: UUID of the parent game.
            file: The uploaded file.
            doc_type: Document type classification.
            user_id: UUID of the authenticated curator.
            name: Display name override (defaults to filename).
            expansion_id: Optional expansion to associate with.

        Returns:
            The created Document entity.

        Raises:
            NotFoundError: If the parent game does not exist.
            ContentTooLargeError: If the file exceeds 50 MB.
            ValidationError: If the file format is unsupported.
            UnprocessableEntityError: If expansion_id does not belong to game.
        """
        await self._validate_game_exists(game_id)

        if expansion_id is not None:
            await self._validate_expansion_belongs_to_game(game_id, expansion_id)

        # Read file content
        file_data = await file.read()
        file_size = len(file_data)

        if file_size > MAX_FILE_SIZE:
            raise ContentTooLargeError(
                f"File size {file_size} bytes exceeds maximum of {MAX_FILE_SIZE} bytes (50 MB)"
            )

        if file_size == 0:
            raise ValidationError("Uploaded file is empty")

        filename = file.filename or "unnamed"
        doc_format = _detect_format(filename, file.content_type)
        if doc_format is None:
            raise ValidationError(
                f"Unsupported file format. Supported: pdf, md, txt, html, docx. "
                f"Got filename: {filename}"
            )

        display_name = name or filename
        orm_type = DocumentType(doc_type.value)

        # Create the document record (let DB generate UUID)
        document = Document(
            game_id=game_id,
            expansion_id=expansion_id,
            name=display_name,
            type=orm_type,
            format=doc_format,
            status=DocumentStatus.UPLOADED,
            current_version=1,
            file_path="pending",  # Placeholder, updated after storage
            file_size=file_size,
        )
        document = await self._document_repo.create(document)

        # Store file in blob storage
        content_type = (
            file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        storage_path = f"documents/{game_id}/{document.id}/v1/{filename}"
        await self._blob_store.store(storage_path, file_data, content_type)

        # Update file_path with actual storage location
        document.file_path = storage_path
        document = await self._document_repo.update(document)

        # Create initial version record
        version = DocumentVersion(
            document_id=document.id,
            version_number=1,
            file_path=storage_path,
            file_size=file_size,
            is_active=True,
            uploaded_by=user_id,
        )
        await self._document_repo.create_version(version)

        # Stub: enqueue processing task (actual Celery integration is #31)
        logger.info(
            "Processing enqueue stub: document_id=%s, version=1",
            document.id,
        )

        return document

    async def get_document(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> Document:
        """Retrieve a document scoped to a specific game.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to retrieve.

        Returns:
            The Document entity.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
        """
        doc = await self._document_repo.get_by_id(game_id, document_id)
        if doc is None:
            raise NotFoundError("Document", str(document_id))
        return doc

    async def get_document_detail(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Retrieve document with chunk_count and version_count.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to retrieve.

        Returns:
            Dict with document, chunk_count, version_count.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
        """
        detail = await self._document_repo.get_detail(game_id, document_id)
        if detail is None:
            raise NotFoundError("Document", str(document_id))
        return detail

    async def list_documents(
        self,
        game_id: uuid.UUID,
        filters: DocumentFilters,
        pagination: PaginationParams,
    ) -> tuple[list[Document], int]:
        """List documents for a game with filtering and pagination.

        Args:
            game_id: UUID of the parent game (silo boundary).
            filters: Parsed filter parameters.
            pagination: Page number and page size.

        Returns:
            Tuple of (list of Document entities, total count).

        Raises:
            NotFoundError: If the parent game does not exist.
        """
        await self._validate_game_exists(game_id)
        return await self._document_repo.list_by_game(game_id, filters, pagination)

    async def reclassify(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        new_type: DocumentTypeFilter,
    ) -> Document:
        """Change document type classification. Metadata-only, no reprocessing.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to reclassify.
            new_type: New document type.

        Returns:
            The updated Document entity.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
        """
        doc = await self.get_document(game_id, document_id)
        orm_type = DocumentType(new_type.value)
        doc.type = orm_type
        return await self._document_repo.update(doc)

    async def reassociate_expansion(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        expansion_id: uuid.UUID | None,
    ) -> Document:
        """Change document expansion association. Metadata-only.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to update.
            expansion_id: New expansion UUID, or None for base game.

        Returns:
            The updated Document entity.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
            UnprocessableEntityError: If expansion does not belong to game.
        """
        doc = await self.get_document(game_id, document_id)

        if expansion_id is not None:
            await self._validate_expansion_belongs_to_game(game_id, expansion_id)

        doc.expansion_id = expansion_id
        return await self._document_repo.update(doc)

    async def delete_document(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Soft-delete a document by setting archived_at.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to delete.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
        """
        doc = await self.get_document(game_id, document_id)
        doc.archived_at = datetime.now(UTC)
        await self._document_repo.update(doc)

        # Stub: KG cleanup (actual integration is downstream)
        logger.info(
            "KG cleanup stub: document_id=%s, game_id=%s",
            document_id,
            game_id,
        )

    async def _validate_game_exists(self, game_id: uuid.UUID) -> None:
        """Verify that the parent game exists.

        Args:
            game_id: UUID of the game to validate.

        Raises:
            NotFoundError: If no game exists with the given ID.
        """
        game = await self._game_repo.get_by_id(game_id)
        if game is None:
            raise NotFoundError("Game", str(game_id))

    async def _validate_expansion_belongs_to_game(
        self,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Verify that an expansion belongs to the specified game.

        Args:
            game_id: UUID of the parent game.
            expansion_id: UUID of the expansion to validate.

        Raises:
            UnprocessableEntityError: If expansion does not exist or
                belongs to a different game.
        """
        expansion = await self._expansion_repo.get_by_id(game_id, expansion_id)
        if expansion is None:
            raise UnprocessableEntityError(
                f"Expansion '{expansion_id}' does not exist or does not belong to game '{game_id}'"
            )

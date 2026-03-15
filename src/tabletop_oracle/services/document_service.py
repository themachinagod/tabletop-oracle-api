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
    ConflictError,
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

    async def upload_version(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        file: UploadFile,
        user_id: uuid.UUID,
    ) -> DocumentVersion:
        """Upload a new version of an existing document.

        Deactivates previous versions, creates a new active version,
        increments the document's current_version, resets status to
        uploaded, and enqueues processing.

        Args:
            game_id: UUID of the parent game.
            document_id: UUID of the document.
            file: The replacement file upload.
            user_id: UUID of the authenticated curator.

        Returns:
            The created DocumentVersion entity.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
            ContentTooLargeError: If the file exceeds 50 MB.
            ValidationError: If file is empty or format does not match.
        """
        doc = await self.get_document(game_id, document_id)

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

        if doc_format != doc.format:
            raise ValidationError(
                f"New version format '{doc_format.value}' does not match "
                f"original document format '{doc.format.value}'"
            )

        new_version_number = doc.current_version + 1
        storage_path = f"documents/{game_id}/{document_id}/v{new_version_number}/{filename}"

        content_type = (
            file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        await self._blob_store.store(storage_path, file_data, content_type)

        # Deactivate all existing versions
        await self._document_repo.deactivate_versions(document_id)

        # Create new version
        version = DocumentVersion(
            document_id=document_id,
            version_number=new_version_number,
            file_path=storage_path,
            file_size=file_size,
            is_active=True,
            uploaded_by=user_id,
        )
        version = await self._document_repo.create_version(version)

        # Update document metadata
        doc.current_version = new_version_number
        doc.file_path = storage_path
        doc.file_size = file_size
        doc.status = DocumentStatus.UPLOADED
        doc.processed_at = None
        doc.error_message = None
        await self._document_repo.update(doc)

        # Stub: enqueue processing task (actual Celery integration is #31)
        logger.info(
            "Processing enqueue stub: document_id=%s, version=%d",
            document_id,
            new_version_number,
        )

        return version

    async def get_versions(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        pagination: PaginationParams,
    ) -> tuple[list[DocumentVersion], int]:
        """List version history for a document.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document.
            pagination: Page number and page size.

        Returns:
            Tuple of (list of DocumentVersion entities, total count).

        Raises:
            NotFoundError: If the document does not exist or wrong game.
        """
        await self.get_document(game_id, document_id)
        return await self._document_repo.list_versions(document_id, pagination)

    async def reprocess(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> Document:
        """Trigger reprocessing of a document's active version.

        Clears existing chunks, resets status to uploaded, and enqueues
        the active version for processing.

        Args:
            game_id: UUID of the parent game.
            document_id: UUID of the document.

        Returns:
            The updated Document entity with reset status.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
            ConflictError: If the document is currently processing.
        """
        doc = await self.get_document(game_id, document_id)

        if doc.status == DocumentStatus.PARSING:
            raise ConflictError(
                "Document is currently processing. Wait for processing to complete."
            )

        # Clear existing chunks
        deleted_count = await self._document_repo.delete_chunks_for_document(document_id)
        logger.info(
            "Reprocess: cleared %d chunks for document_id=%s",
            deleted_count,
            document_id,
        )

        # Reset status
        doc.status = DocumentStatus.UPLOADED
        doc.processed_at = None
        doc.error_message = None
        doc = await self._document_repo.update(doc)

        # Stub: enqueue processing task (actual Celery integration is #31)
        logger.info(
            "Processing enqueue stub (reprocess): document_id=%s, version=%d",
            document_id,
            doc.current_version,
        )

        return doc

    async def get_preview(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Return extracted content and structure for preview.

        Retrieves all chunks for the document and constructs a preview
        with section structure and statistics.

        Args:
            game_id: UUID of the parent game.
            document_id: UUID of the document.

        Returns:
            Dict with document_id, format, sections, chunks, and stats.

        Raises:
            NotFoundError: If the document does not exist or wrong game.
            ConflictError: If the document has not been processed yet.
        """
        doc = await self.get_document(game_id, document_id)

        if doc.status != DocumentStatus.PROCESSED:
            raise ConflictError(
                f"Document has not been processed yet. Current status: '{doc.status.value}'"
            )

        chunks = await self._document_repo.get_chunks_for_preview(document_id)

        # Build section tree from chunks
        sections = self._build_section_tree(chunks)

        # Build chunk summaries for flat preview
        chunk_data = [
            {
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type,
                "content": c.content,
                "section_path": c.section_path,
                "heading": c.heading,
                "page_number": c.page_number,
                "token_estimate": c.token_estimate,
            }
            for c in chunks
        ]

        total_characters = sum(len(c.content) for c in chunks)
        total_tokens = sum(c.token_estimate for c in chunks)

        return {
            "document_id": document_id,
            "format": doc.format.value,
            "sections": sections,
            "chunks": chunk_data,
            "stats": {
                "total_chunks": len(chunks),
                "total_characters": total_characters,
                "estimated_tokens": total_tokens,
            },
        }

    @staticmethod
    def _build_section_tree(
        chunks: list[Any],
    ) -> list[dict[str, Any]]:
        """Build a hierarchical section tree from flat chunks.

        Groups chunks by heading to reconstruct the document's
        section hierarchy for preview display.

        Args:
            chunks: Ordered list of DocumentChunk entities.

        Returns:
            List of section dicts with title, level, content, children.
        """
        sections: list[dict[str, Any]] = []
        seen_headings: set[str] = set()

        for chunk in chunks:
            heading = chunk.heading or ""
            if heading and heading not in seen_headings:
                seen_headings.add(heading)
                sections.append(
                    {
                        "title": heading,
                        "level": 1,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "children": [],
                    }
                )
            elif sections:
                # Append content to the last section
                sections[-1]["content"] += "\n" + chunk.content
            else:
                # No heading yet, create untitled root section
                sections.append(
                    {
                        "title": "",
                        "level": 1,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "children": [],
                    }
                )

        return sections

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

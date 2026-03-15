"""Unit tests for DocumentService business logic.

Tests all service methods with mocked repositories and blob storage.
Covers happy paths, game/expansion validation, file validation,
format detection, reclassification, soft-delete, and error cases.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import (
    ContentTooLargeError,
    NotFoundError,
    UnprocessableEntityError,
    ValidationError,
)
from tabletop_oracle.models.enums import DocumentFormat, DocumentStatus, DocumentType
from tabletop_oracle.schemas.document import (
    DocumentFilters,
    DocumentTypeFilter,
)
from tabletop_oracle.services.document_service import DocumentService, _detect_format

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_document_repo() -> AsyncMock:
    """Create a mock DocumentRepository."""
    return AsyncMock()


@pytest.fixture
def mock_game_repo() -> AsyncMock:
    """Create a mock GameRepository."""
    return AsyncMock()


@pytest.fixture
def mock_expansion_repo() -> AsyncMock:
    """Create a mock ExpansionRepository."""
    return AsyncMock()


@pytest.fixture
def mock_blob_store() -> AsyncMock:
    """Create a mock BlobStorageInterface."""
    store = AsyncMock()
    store.store.return_value = "documents/game/doc/v1/file.pdf"
    return store


@pytest.fixture
def service(
    mock_document_repo: AsyncMock,
    mock_game_repo: AsyncMock,
    mock_expansion_repo: AsyncMock,
    mock_blob_store: AsyncMock,
) -> DocumentService:
    """Create a DocumentService with all mocked dependencies."""
    return DocumentService(
        document_repo=mock_document_repo,
        game_repo=mock_game_repo,
        expansion_repo=mock_expansion_repo,
        blob_store=mock_blob_store,
    )


@pytest.fixture
def game_id() -> uuid.UUID:
    """Fixed game UUID for tests."""
    return uuid.UUID("550e8400-e29b-41d4-a716-446655440000")


@pytest.fixture
def document_id() -> uuid.UUID:
    """Fixed document UUID for tests."""
    return uuid.UUID("660f9511-f3a0-52e5-b827-557766551111")


@pytest.fixture
def user_id() -> uuid.UUID:
    """Fixed user UUID for tests."""
    return uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _make_document(
    *,
    doc_id: uuid.UUID | None = None,
    game_id: uuid.UUID | None = None,
    name: str = "Test Rules.pdf",
    doc_type: DocumentType = DocumentType.CORE_RULES,
    doc_format: DocumentFormat = DocumentFormat.PDF,
    status: DocumentStatus = DocumentStatus.UPLOADED,
    archived_at: datetime | None = None,
) -> MagicMock:
    """Create a mock Document entity with specified attributes."""
    doc = MagicMock()
    doc.id = doc_id or uuid.uuid4()
    doc.game_id = game_id or uuid.uuid4()
    doc.expansion_id = None
    doc.name = name
    doc.type = doc_type
    doc.format = doc_format
    doc.status = status
    doc.current_version = 1
    doc.file_path = "documents/game/doc/v1/file.pdf"
    doc.file_size = 1024
    doc.error_message = None
    doc.uploaded_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    doc.processed_at = None
    doc.updated_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    doc.archived_at = archived_at
    return doc


def _make_upload_file(
    *,
    filename: str = "rules.pdf",
    content: bytes = b"fake pdf content",
    content_type: str = "application/pdf",
) -> AsyncMock:
    """Create a mock UploadFile."""
    file = AsyncMock()
    file.filename = filename
    file.content_type = content_type
    file.read.return_value = content
    return file


def _make_game(*, game_id: uuid.UUID | None = None) -> MagicMock:
    """Create a mock Game entity."""
    game = MagicMock()
    game.id = game_id or uuid.uuid4()
    game.archived_at = None
    return game


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for _detect_format helper."""

    def test_pdf_extension(self) -> None:
        """PDF detected from .pdf extension."""
        assert _detect_format("rules.pdf", None) == DocumentFormat.PDF

    def test_markdown_extension(self) -> None:
        """Markdown detected from .md extension."""
        assert _detect_format("notes.md", None) == DocumentFormat.MARKDOWN

    def test_txt_extension(self) -> None:
        """Text detected from .txt extension."""
        assert _detect_format("readme.txt", None) == DocumentFormat.TEXT

    def test_html_extension(self) -> None:
        """HTML detected from .html extension."""
        assert _detect_format("page.html", None) == DocumentFormat.HTML

    def test_docx_extension(self) -> None:
        """DOCX detected from .docx extension."""
        assert _detect_format("report.docx", None) == DocumentFormat.DOCX

    def test_mime_type_fallback(self) -> None:
        """Format detected from MIME type when extension is unknown."""
        assert _detect_format("file", "application/pdf") == DocumentFormat.PDF

    def test_unsupported_format(self) -> None:
        """None returned for unsupported format."""
        assert _detect_format("image.png", "image/png") is None

    def test_extension_takes_priority(self) -> None:
        """Extension detection takes priority over MIME type."""
        result = _detect_format("rules.pdf", "text/plain")
        assert result == DocumentFormat.PDF


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class TestUpload:
    """Tests for DocumentService.upload."""

    @pytest.mark.anyio
    async def test_upload_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        mock_game_repo: AsyncMock,
        mock_blob_store: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Successful upload creates document and version records."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)

        created_doc = _make_document(game_id=game_id)
        mock_document_repo.create.return_value = created_doc
        mock_document_repo.update.return_value = created_doc

        file = _make_upload_file()
        result = await service.upload(
            game_id=game_id,
            file=file,
            doc_type=DocumentTypeFilter.CORE_RULES,
            user_id=user_id,
        )

        assert result == created_doc
        mock_document_repo.create.assert_called_once()
        mock_document_repo.create_version.assert_called_once()
        mock_blob_store.store.assert_called_once()

    @pytest.mark.anyio
    async def test_upload_game_not_found(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload raises NotFoundError when game does not exist."""
        mock_game_repo.get_by_id.return_value = None

        file = _make_upload_file()
        with pytest.raises(NotFoundError):
            await service.upload(
                game_id=game_id,
                file=file,
                doc_type=DocumentTypeFilter.CORE_RULES,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_file_too_large(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload raises ContentTooLargeError for files over 50MB."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)

        large_content = b"x" * (50 * 1024 * 1024 + 1)
        file = _make_upload_file(content=large_content)

        with pytest.raises(ContentTooLargeError):
            await service.upload(
                game_id=game_id,
                file=file,
                doc_type=DocumentTypeFilter.CORE_RULES,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_empty_file(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload raises ValidationError for empty files."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)

        file = _make_upload_file(content=b"")
        with pytest.raises(ValidationError):
            await service.upload(
                game_id=game_id,
                file=file,
                doc_type=DocumentTypeFilter.CORE_RULES,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_unsupported_format(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload raises ValidationError for unsupported file format."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)

        file = _make_upload_file(filename="image.png", content_type="image/png")
        with pytest.raises(ValidationError):
            await service.upload(
                game_id=game_id,
                file=file,
                doc_type=DocumentTypeFilter.CORE_RULES,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_with_expansion_validation(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        mock_expansion_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_blob_store: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload validates that expansion belongs to the game."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)
        mock_expansion_repo.get_by_id.return_value = None  # Expansion not found

        exp_id = uuid.uuid4()
        file = _make_upload_file()

        with pytest.raises(UnprocessableEntityError):
            await service.upload(
                game_id=game_id,
                file=file,
                doc_type=DocumentTypeFilter.CORE_RULES,
                user_id=user_id,
                expansion_id=exp_id,
            )

    @pytest.mark.anyio
    async def test_upload_uses_filename_as_default_name(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_blob_store: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload uses filename as display name when name is not provided."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)

        created_doc = _make_document(game_id=game_id)
        mock_document_repo.create.return_value = created_doc
        mock_document_repo.update.return_value = created_doc

        file = _make_upload_file(filename="MyRules.pdf")
        await service.upload(
            game_id=game_id,
            file=file,
            doc_type=DocumentTypeFilter.CORE_RULES,
            user_id=user_id,
        )

        # Verify the Document was created with filename as name
        call_args = mock_document_repo.create.call_args
        doc_arg = call_args[0][0]
        assert doc_arg.name == "MyRules.pdf"


# ---------------------------------------------------------------------------
# Get document
# ---------------------------------------------------------------------------


class TestGetDocument:
    """Tests for DocumentService.get_document."""

    @pytest.mark.anyio
    async def test_get_existing_document(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Returns document when it exists."""
        expected = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = expected

        result = await service.get_document(game_id, document_id)
        assert result == expected

    @pytest.mark.anyio
    async def test_get_nonexistent_document(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when document does not exist."""
        mock_document_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.get_document(game_id, document_id)


# ---------------------------------------------------------------------------
# Get document detail
# ---------------------------------------------------------------------------


class TestGetDocumentDetail:
    """Tests for DocumentService.get_document_detail."""

    @pytest.mark.anyio
    async def test_get_detail_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Returns detail dict with chunk_count and version_count."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_detail.return_value = {
            "document": doc,
            "chunk_count": 10,
            "version_count": 2,
        }

        result = await service.get_document_detail(game_id, document_id)
        assert result["chunk_count"] == 10
        assert result["version_count"] == 2

    @pytest.mark.anyio
    async def test_get_detail_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when document does not exist."""
        mock_document_repo.get_detail.return_value = None

        with pytest.raises(NotFoundError):
            await service.get_document_detail(game_id, document_id)


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    """Tests for DocumentService.list_documents."""

    @pytest.mark.anyio
    async def test_list_validates_game_exists(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
    ) -> None:
        """List raises NotFoundError when game does not exist."""
        mock_game_repo.get_by_id.return_value = None
        filters = DocumentFilters()
        pagination = MagicMock()

        with pytest.raises(NotFoundError):
            await service.list_documents(game_id, filters, pagination)

    @pytest.mark.anyio
    async def test_list_delegates_to_repo(
        self,
        service: DocumentService,
        mock_game_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
    ) -> None:
        """List delegates to repository after game validation."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)
        mock_document_repo.list_by_game.return_value = ([], 0)

        filters = DocumentFilters()
        pagination = MagicMock()

        docs, total = await service.list_documents(game_id, filters, pagination)
        assert docs == []
        assert total == 0
        mock_document_repo.list_by_game.assert_called_once()


# ---------------------------------------------------------------------------
# Reclassify
# ---------------------------------------------------------------------------


class TestReclassify:
    """Tests for DocumentService.reclassify."""

    @pytest.mark.anyio
    async def test_reclassify_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reclassify updates document type."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.update.return_value = doc

        result = await service.reclassify(game_id, document_id, DocumentTypeFilter.ERRATA)
        assert result.type == DocumentType.ERRATA
        mock_document_repo.update.assert_called_once()

    @pytest.mark.anyio
    async def test_reclassify_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reclassify raises NotFoundError for nonexistent document."""
        mock_document_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.reclassify(game_id, document_id, DocumentTypeFilter.FAQ)


# ---------------------------------------------------------------------------
# Reassociate expansion
# ---------------------------------------------------------------------------


class TestReassociateExpansion:
    """Tests for DocumentService.reassociate_expansion."""

    @pytest.mark.anyio
    async def test_reassociate_to_expansion(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reassociate sets expansion_id when expansion belongs to game."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.update.return_value = doc

        exp_id = uuid.uuid4()
        expansion = MagicMock()
        expansion.id = exp_id
        mock_expansion_repo.get_by_id.return_value = expansion

        result = await service.reassociate_expansion(game_id, document_id, exp_id)
        assert result.expansion_id == exp_id

    @pytest.mark.anyio
    async def test_reassociate_to_base_game(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reassociate with None removes expansion association."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.update.return_value = doc

        result = await service.reassociate_expansion(game_id, document_id, None)
        assert result.expansion_id is None

    @pytest.mark.anyio
    async def test_reassociate_invalid_expansion(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reassociate raises UnprocessableEntityError for wrong-game expansion."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(UnprocessableEntityError):
            await service.reassociate_expansion(game_id, document_id, uuid.uuid4())


# ---------------------------------------------------------------------------
# Delete (soft-delete)
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    """Tests for DocumentService.delete_document."""

    @pytest.mark.anyio
    async def test_delete_sets_archived_at(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Delete sets archived_at on the document."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.update.return_value = doc

        await service.delete_document(game_id, document_id)

        assert doc.archived_at is not None
        mock_document_repo.update.assert_called_once()

    @pytest.mark.anyio
    async def test_delete_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Delete raises NotFoundError for nonexistent document."""
        mock_document_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.delete_document(game_id, document_id)

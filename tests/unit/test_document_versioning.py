"""Unit tests for document versioning service methods.

Tests upload_version, get_versions, reprocess, and get_preview service
methods with mocked repositories and blob storage. Covers happy paths,
format mismatch, size limits, conflict detection, and preview construction.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import (
    ConflictError,
    ContentTooLargeError,
    NotFoundError,
    ValidationError,
)
from tabletop_oracle.models.enums import DocumentFormat, DocumentStatus, DocumentType
from tabletop_oracle.services.document_service import DocumentService

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
    store.store.return_value = "documents/game/doc/v2/file.pdf"
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
    current_version: int = 1,
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
    doc.current_version = current_version
    doc.file_path = "documents/game/doc/v1/file.pdf"
    doc.file_size = 1024
    doc.error_message = None
    doc.uploaded_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    doc.processed_at = None
    doc.updated_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    doc.archived_at = None
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


def _make_version(
    *,
    version_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    version_number: int = 1,
    is_active: bool = True,
) -> MagicMock:
    """Create a mock DocumentVersion entity."""
    version = MagicMock()
    version.id = version_id or uuid.uuid4()
    version.document_id = document_id or uuid.uuid4()
    version.version_number = version_number
    version.file_path = f"documents/game/doc/v{version_number}/file.pdf"
    version.file_size = 1024
    version.is_active = is_active
    version.uploaded_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    version.processed_at = None
    version.uploaded_by = uuid.uuid4()
    return version


def _make_chunk(
    *,
    chunk_index: int = 0,
    content: str = "Test content",
    heading: str | None = "Test Heading",
    section_path: str | None = "Test > Path",
    page_number: int | None = 1,
    chunk_type: str = "text",
    token_estimate: int = 10,
) -> MagicMock:
    """Create a mock DocumentChunk entity."""
    chunk = MagicMock()
    chunk.chunk_index = chunk_index
    chunk.content = content
    chunk.heading = heading
    chunk.section_path = section_path
    chunk.page_number = page_number
    chunk.chunk_type = chunk_type
    chunk.token_estimate = token_estimate
    return chunk


# ---------------------------------------------------------------------------
# Upload version
# ---------------------------------------------------------------------------


class TestUploadVersion:
    """Tests for DocumentService.upload_version."""

    @pytest.mark.anyio
    async def test_upload_version_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        mock_blob_store: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Successful version upload creates new version and updates document."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc

        new_version = _make_version(
            document_id=document_id,
            version_number=2,
        )
        mock_document_repo.create_version.return_value = new_version
        mock_document_repo.update.return_value = doc

        file = _make_upload_file()
        result = await service.upload_version(
            game_id=game_id,
            document_id=document_id,
            file=file,
            user_id=user_id,
        )

        assert result == new_version
        mock_document_repo.deactivate_versions.assert_called_once_with(document_id)
        mock_document_repo.create_version.assert_called_once()
        mock_blob_store.store.assert_called_once()
        mock_document_repo.update.assert_called_once()

    @pytest.mark.anyio
    async def test_upload_version_increments_version_number(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        mock_blob_store: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """New version gets version_number = current_version + 1."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            current_version=3,
        )
        mock_document_repo.get_by_id.return_value = doc

        new_version = _make_version(version_number=4)
        mock_document_repo.create_version.return_value = new_version
        mock_document_repo.update.return_value = doc

        file = _make_upload_file()
        await service.upload_version(
            game_id=game_id,
            document_id=document_id,
            file=file,
            user_id=user_id,
        )

        version_arg = mock_document_repo.create_version.call_args[0][0]
        assert version_arg.version_number == 4

    @pytest.mark.anyio
    async def test_upload_version_document_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload version raises NotFoundError when document does not exist."""
        mock_document_repo.get_by_id.return_value = None

        file = _make_upload_file()
        with pytest.raises(NotFoundError):
            await service.upload_version(
                game_id=game_id,
                document_id=document_id,
                file=file,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_version_file_too_large(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload version raises ContentTooLargeError for files over 50MB."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc

        large_content = b"x" * (50 * 1024 * 1024 + 1)
        file = _make_upload_file(content=large_content)

        with pytest.raises(ContentTooLargeError):
            await service.upload_version(
                game_id=game_id,
                document_id=document_id,
                file=file,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_version_empty_file(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload version raises ValidationError for empty files."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc

        file = _make_upload_file(content=b"")
        with pytest.raises(ValidationError):
            await service.upload_version(
                game_id=game_id,
                document_id=document_id,
                file=file,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_version_format_mismatch(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload version raises ValidationError when format does not match original."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            doc_format=DocumentFormat.PDF,
        )
        mock_document_repo.get_by_id.return_value = doc

        file = _make_upload_file(
            filename="rules.md",
            content_type="text/markdown",
        )
        with pytest.raises(ValidationError, match="does not match"):
            await service.upload_version(
                game_id=game_id,
                document_id=document_id,
                file=file,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_version_unsupported_format(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload version raises ValidationError for unsupported file format."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc

        file = _make_upload_file(
            filename="image.png",
            content_type="image/png",
        )
        with pytest.raises(ValidationError):
            await service.upload_version(
                game_id=game_id,
                document_id=document_id,
                file=file,
                user_id=user_id,
            )

    @pytest.mark.anyio
    async def test_upload_version_resets_status(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        mock_blob_store: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Upload version resets document status to UPLOADED."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PROCESSED,
        )
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.create_version.return_value = _make_version()
        mock_document_repo.update.return_value = doc

        file = _make_upload_file()
        await service.upload_version(
            game_id=game_id,
            document_id=document_id,
            file=file,
            user_id=user_id,
        )

        assert doc.status == DocumentStatus.UPLOADED
        assert doc.processed_at is None
        assert doc.error_message is None


# ---------------------------------------------------------------------------
# Get versions
# ---------------------------------------------------------------------------


class TestGetVersions:
    """Tests for DocumentService.get_versions."""

    @pytest.mark.anyio
    async def test_get_versions_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Returns paginated version list."""
        doc = _make_document(doc_id=document_id, game_id=game_id)
        mock_document_repo.get_by_id.return_value = doc

        versions = [
            _make_version(version_number=2, is_active=True),
            _make_version(version_number=1, is_active=False),
        ]
        mock_document_repo.list_versions.return_value = (versions, 2)

        pagination = MagicMock()
        result_versions, total = await service.get_versions(
            game_id,
            document_id,
            pagination,
        )

        assert len(result_versions) == 2
        assert total == 2
        mock_document_repo.list_versions.assert_called_once_with(
            document_id,
            pagination,
        )

    @pytest.mark.anyio
    async def test_get_versions_document_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when document does not exist."""
        mock_document_repo.get_by_id.return_value = None
        pagination = MagicMock()

        with pytest.raises(NotFoundError):
            await service.get_versions(game_id, document_id, pagination)


# ---------------------------------------------------------------------------
# Reprocess
# ---------------------------------------------------------------------------


class TestReprocess:
    """Tests for DocumentService.reprocess."""

    @pytest.mark.anyio
    async def test_reprocess_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reprocess clears chunks and resets status to UPLOADED."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PROCESSED,
        )
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.delete_chunks_for_document.return_value = 10
        mock_document_repo.update.return_value = doc

        result = await service.reprocess(game_id, document_id)

        assert result.status == DocumentStatus.UPLOADED
        assert result.processed_at is None
        assert result.error_message is None
        mock_document_repo.delete_chunks_for_document.assert_called_once_with(
            document_id,
        )
        mock_document_repo.update.assert_called_once()

    @pytest.mark.anyio
    async def test_reprocess_from_error_status(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reprocess works on documents in error status."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.ERROR,
        )
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.delete_chunks_for_document.return_value = 0
        mock_document_repo.update.return_value = doc

        result = await service.reprocess(game_id, document_id)
        assert result.status == DocumentStatus.UPLOADED

    @pytest.mark.anyio
    async def test_reprocess_conflict_when_parsing(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reprocess raises ConflictError when document is currently processing."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PARSING,
        )
        mock_document_repo.get_by_id.return_value = doc

        with pytest.raises(ConflictError, match="currently processing"):
            await service.reprocess(game_id, document_id)

    @pytest.mark.anyio
    async def test_reprocess_document_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reprocess raises NotFoundError when document does not exist."""
        mock_document_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.reprocess(game_id, document_id)

    @pytest.mark.anyio
    async def test_reprocess_from_uploaded_status(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Reprocess works on documents in uploaded status (not yet processed)."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.UPLOADED,
        )
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.delete_chunks_for_document.return_value = 0
        mock_document_repo.update.return_value = doc

        result = await service.reprocess(game_id, document_id)
        assert result.status == DocumentStatus.UPLOADED


# ---------------------------------------------------------------------------
# Get preview
# ---------------------------------------------------------------------------


class TestGetPreview:
    """Tests for DocumentService.get_preview."""

    @pytest.mark.anyio
    async def test_get_preview_success(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Returns preview data with sections, chunks, and stats."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PROCESSED,
        )
        mock_document_repo.get_by_id.return_value = doc

        chunks = [
            _make_chunk(chunk_index=0, heading="Setup", content="First section"),
            _make_chunk(chunk_index=1, heading="Rules", content="Second section"),
        ]
        mock_document_repo.get_chunks_for_preview.return_value = chunks

        result = await service.get_preview(game_id, document_id)

        assert result["document_id"] == document_id
        assert result["format"] == "pdf"
        assert len(result["sections"]) == 2
        assert len(result["chunks"]) == 2
        assert result["stats"]["total_chunks"] == 2
        assert result["stats"]["total_characters"] == len("First section") + len("Second section")
        assert result["stats"]["estimated_tokens"] == 20

    @pytest.mark.anyio
    async def test_get_preview_not_processed(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Raises ConflictError when document is not in processed status."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.UPLOADED,
        )
        mock_document_repo.get_by_id.return_value = doc

        with pytest.raises(ConflictError, match="not been processed"):
            await service.get_preview(game_id, document_id)

    @pytest.mark.anyio
    async def test_get_preview_document_not_found(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when document does not exist."""
        mock_document_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.get_preview(game_id, document_id)

    @pytest.mark.anyio
    async def test_get_preview_empty_chunks(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Returns empty sections and zero stats when no chunks exist."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PROCESSED,
        )
        mock_document_repo.get_by_id.return_value = doc
        mock_document_repo.get_chunks_for_preview.return_value = []

        result = await service.get_preview(game_id, document_id)

        assert result["sections"] == []
        assert result["chunks"] == []
        assert result["stats"]["total_chunks"] == 0

    @pytest.mark.anyio
    async def test_get_preview_section_grouping(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Chunks with same heading are grouped into one section."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PROCESSED,
        )
        mock_document_repo.get_by_id.return_value = doc

        chunks = [
            _make_chunk(chunk_index=0, heading="Setup", content="Part 1"),
            _make_chunk(chunk_index=1, heading="Setup", content="Part 2"),
            _make_chunk(chunk_index=2, heading="Rules", content="Part 3"),
        ]
        mock_document_repo.get_chunks_for_preview.return_value = chunks

        result = await service.get_preview(game_id, document_id)

        # "Setup" chunks grouped, "Rules" is separate
        assert len(result["sections"]) == 2
        assert result["sections"][0]["title"] == "Setup"
        assert "Part 1" in result["sections"][0]["content"]
        assert "Part 2" in result["sections"][0]["content"]
        assert result["sections"][1]["title"] == "Rules"

    @pytest.mark.anyio
    async def test_get_preview_parsing_status_conflict(
        self,
        service: DocumentService,
        mock_document_repo: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Raises ConflictError when document is in parsing status."""
        doc = _make_document(
            doc_id=document_id,
            game_id=game_id,
            status=DocumentStatus.PARSING,
        )
        mock_document_repo.get_by_id.return_value = doc

        with pytest.raises(ConflictError):
            await service.get_preview(game_id, document_id)


# ---------------------------------------------------------------------------
# Build section tree (static method)
# ---------------------------------------------------------------------------


class TestBuildSectionTree:
    """Tests for DocumentService._build_section_tree static method."""

    def test_single_heading(self) -> None:
        """Single chunk with heading produces one section."""
        chunks = [_make_chunk(heading="Title", content="Content")]
        result = DocumentService._build_section_tree(chunks)

        assert len(result) == 1
        assert result[0]["title"] == "Title"
        assert result[0]["content"] == "Content"

    def test_no_heading(self) -> None:
        """Chunk without heading creates untitled section."""
        chunks = [_make_chunk(heading=None, content="Orphan content")]
        result = DocumentService._build_section_tree(chunks)

        assert len(result) == 1
        assert result[0]["title"] == ""

    def test_empty_chunks(self) -> None:
        """Empty chunk list returns empty sections."""
        result = DocumentService._build_section_tree([])
        assert result == []

    def test_consecutive_same_heading(self) -> None:
        """Chunks with same heading are merged into one section."""
        chunks = [
            _make_chunk(heading="H1", content="A"),
            _make_chunk(heading="H1", content="B"),
        ]
        result = DocumentService._build_section_tree(chunks)

        assert len(result) == 1
        assert "A" in result[0]["content"]
        assert "B" in result[0]["content"]

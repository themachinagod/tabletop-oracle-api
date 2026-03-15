"""Document CRUD API router.

All endpoints are game-scoped under ``/games/{game_id}/documents``.
Curator role required for all operations. Follows three-layer
architecture: router -> service -> repository.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from starlette.responses import Response

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.documents.dependencies import get_document_service
from tabletop_oracle.api.pagination import PaginationParams, get_pagination_params
from tabletop_oracle.api.response import ok, ok_list
from tabletop_oracle.auth.dependencies import CurrentUserDep, require_role
from tabletop_oracle.schemas.common import DataEnvelope, ListEnvelope  # noqa: TC001
from tabletop_oracle.schemas.document import (
    DocumentDetailResponse,
    DocumentExpansionUpdate,
    DocumentFilters,
    DocumentResponse,
    DocumentTypeFilter,
    DocumentTypeUpdate,
)
from tabletop_oracle.services.document_service import DocumentService

router = APIRouter(tags=["documents"])

#: Module-level dependency singletons for FastAPI injection.
_curator_dep = Depends(require_role("curator"))
_service_dep = Depends(get_document_service)
_pagination_dep = Depends(get_pagination_params)

#: Annotated type aliases for clean route signatures.
ServiceDep = Annotated[DocumentService, _service_dep]
PaginationDep = Annotated[PaginationParams, _pagination_dep]


@router.post(
    "",
    status_code=201,
    dependencies=[_curator_dep],
)
async def upload_documents(
    game_id: uuid.UUID,
    request: Request,
    user: CurrentUserDep,
    db: DbSession,
    service: ServiceDep,
    files: Annotated[list[UploadFile], File(...)],
    doc_type: Annotated[DocumentTypeFilter, Form(alias="type")],
    expansion_id: Annotated[uuid.UUID | None, Form()] = None,
    name: Annotated[str | None, Form()] = None,
) -> DataEnvelope[Any]:
    """Upload one or more documents for a game.

    Multipart form upload supporting single or bulk file upload.
    Each uploaded document is enqueued for background processing.

    Args:
        game_id: UUID of the parent game.
        request: The HTTP request.
        user: Authenticated curator.
        db: Database session.
        service: Document service (injected).
        files: One or more files to upload.
        doc_type: Document type classification (form field name: "type").
        expansion_id: Optional expansion UUID.
        name: Optional display name (used for single file; ignored for bulk).

    Returns:
        201 with DataEnvelope containing single or list of DocumentResponse.
    """
    documents = []
    for uploaded_file in files:
        doc = await service.upload(
            game_id=game_id,
            file=uploaded_file,
            doc_type=doc_type,
            user_id=user.user_id,
            name=name if len(files) == 1 else None,
            expansion_id=expansion_id,
        )
        documents.append(DocumentResponse.model_validate(doc))

    if len(documents) == 1:
        return ok(documents[0], request)

    return ok(documents, request)


@router.get(
    "",
    dependencies=[_curator_dep],
)
async def list_documents(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
    service: ServiceDep,
    pagination: PaginationDep,
    status: Annotated[str | None, Query()] = None,
    doc_type: Annotated[str | None, Query(alias="type")] = None,
    expansion_id: Annotated[str | None, Query()] = None,
    sort: Annotated[str, Query()] = "-uploaded_at",
) -> ListEnvelope[Any]:
    """List documents for a game with filtering and pagination.

    Args:
        game_id: UUID of the parent game.
        request: The HTTP request.
        db: Database session.
        service: Document service (injected).
        pagination: Pagination parameters.
        status: Comma-separated status filter.
        doc_type: Comma-separated type filter (query param name: "type").
        expansion_id: Expansion UUID filter (use "null" for base game only).
        sort: Sort field with optional - prefix.

    Returns:
        200 with ListEnvelope containing DocumentResponse items.
    """
    parsed_expansion_id: uuid.UUID | str | None = None
    if expansion_id == "null":
        parsed_expansion_id = "null"
    elif expansion_id is not None:
        parsed_expansion_id = uuid.UUID(expansion_id)

    filters = DocumentFilters(
        status=status,
        type=doc_type,
        expansion_id=parsed_expansion_id,
        sort=sort,
    )

    documents, total = await service.list_documents(game_id, filters, pagination)
    items: list[object] = [DocumentResponse.model_validate(doc) for doc in documents]
    return ok_list(items, total, pagination, request)


@router.get(
    "/{document_id}",
    dependencies=[_curator_dep],
)
async def get_document(
    game_id: uuid.UUID,
    document_id: uuid.UUID,
    request: Request,
    db: DbSession,
    service: ServiceDep,
) -> DataEnvelope[Any]:
    """Get document detail with processing stats.

    Args:
        game_id: UUID of the parent game.
        document_id: UUID of the document.
        request: The HTTP request.
        db: Database session.
        service: Document service (injected).

    Returns:
        200 with DataEnvelope containing DocumentDetailResponse.
    """
    detail = await service.get_document_detail(game_id, document_id)
    doc = detail["document"]
    response = DocumentDetailResponse(
        id=doc.id,
        game_id=doc.game_id,
        expansion_id=doc.expansion_id,
        name=doc.name,
        type=doc.type.value,
        format=doc.format.value,
        status=doc.status.value,
        current_version=doc.current_version,
        file_size=doc.file_size,
        error_message=doc.error_message,
        uploaded_at=doc.uploaded_at,
        processed_at=doc.processed_at,
        updated_at=doc.updated_at,
        archived_at=doc.archived_at,
        chunk_count=detail["chunk_count"],
        version_count=detail["version_count"],
    )
    return ok(response, request)


@router.put(
    "/{document_id}/type",
    dependencies=[_curator_dep],
)
async def reclassify_document(
    game_id: uuid.UUID,
    document_id: uuid.UUID,
    body: DocumentTypeUpdate,
    request: Request,
    db: DbSession,
    service: ServiceDep,
) -> DataEnvelope[Any]:
    """Reclassify a document's type. Metadata-only, no reprocessing.

    Args:
        game_id: UUID of the parent game.
        document_id: UUID of the document.
        body: Request body with new type.
        request: The HTTP request.
        db: Database session.
        service: Document service (injected).

    Returns:
        200 with DataEnvelope containing updated DocumentResponse.
    """
    doc = await service.reclassify(game_id, document_id, body.type)
    return ok(DocumentResponse.model_validate(doc), request)


@router.put(
    "/{document_id}/expansion",
    dependencies=[_curator_dep],
)
async def reassociate_expansion(
    game_id: uuid.UUID,
    document_id: uuid.UUID,
    body: DocumentExpansionUpdate,
    request: Request,
    db: DbSession,
    service: ServiceDep,
) -> DataEnvelope[Any]:
    """Change a document's expansion association. Metadata-only.

    Args:
        game_id: UUID of the parent game.
        document_id: UUID of the document.
        body: Request body with expansion_id (or null for base game).
        request: The HTTP request.
        db: Database session.
        service: Document service (injected).

    Returns:
        200 with DataEnvelope containing updated DocumentResponse.
    """
    doc = await service.reassociate_expansion(game_id, document_id, body.expansion_id)
    return ok(DocumentResponse.model_validate(doc), request)


@router.delete(
    "/{document_id}",
    status_code=204,
    dependencies=[_curator_dep],
)
async def delete_document(
    game_id: uuid.UUID,
    document_id: uuid.UUID,
    db: DbSession,
    service: ServiceDep,
) -> Response:
    """Soft-delete a document by setting archived_at.

    Args:
        game_id: UUID of the parent game.
        document_id: UUID of the document.
        db: Database session.
        service: Document service (injected).

    Returns:
        204 No Content.
    """
    await service.delete_document(game_id, document_id)
    return Response(status_code=204)

"""FastAPI dependencies for the document API layer.

Provides factory functions for constructing the DocumentService with
its repository and storage dependencies.
"""

from __future__ import annotations

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.repositories.document_repository import DocumentRepository
from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
from tabletop_oracle.repositories.game_repository import GameRepository
from tabletop_oracle.services.document_service import DocumentService
from tabletop_oracle.storage.local import LocalBlobStorage


def get_document_service(db: DbSession) -> DocumentService:
    """Build a DocumentService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        Configured DocumentService instance.
    """
    return DocumentService(
        document_repo=DocumentRepository(db),
        game_repo=GameRepository(db),
        expansion_repo=ExpansionRepository(db),
        blob_store=LocalBlobStorage(),
    )

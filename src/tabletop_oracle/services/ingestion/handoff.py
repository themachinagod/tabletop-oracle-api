"""KG handoff interface and V1 stub implementation.

Defines the contract for handing off processed document chunks to the
Knowledge Graph engine. The V1 stub logs the handoff and returns
successfully -- replaced via DI when EPIC-003 implements the real KG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.services.ingestion.models import Chunk

logger = logging.getLogger(__name__)


class KGIngestionError(Exception):
    """Raised when the KG engine fails to ingest or remove document data.

    Attributes:
        message: Human-readable error description.
        document_id: UUID of the document that failed.
    """

    def __init__(self, message: str, *, document_id: UUID | None = None) -> None:
        self.message = message
        self.document_id = document_id
        super().__init__(message)


@dataclass(frozen=True)
class KGHandoffPayload:
    """Payload passed from the ingestion pipeline to the KG engine.

    Immutable snapshot of all data the KG needs to process a document's
    chunks. Built by the handoff stage from PipelineContext fields.

    Attributes:
        document_id: UUID of the source document.
        version_id: UUID of the specific version processed.
        game_id: UUID of the owning game (for KG scoping).
        document_type: Document classification (affects conflict priority).
        expansion_id: Expansion association (for session-level filtering).
        chunks: All chunks produced by the chunking engine.
        replaces_version_id: If this is a version replacement, the previous version ID.
    """

    document_id: UUID
    version_id: UUID
    game_id: UUID
    document_type: str
    expansion_id: UUID | None
    chunks: tuple[Chunk, ...] = field(default_factory=tuple)
    replaces_version_id: UUID | None = None


@runtime_checkable
class KGHandoffService(Protocol):
    """Interface for handing off processed chunks to the KG engine.

    Implementations must handle both ingestion (new/updated documents)
    and removal (document deletion). The stub returns successfully;
    the real implementation (EPIC-003) will write to the KG store.
    """

    async def ingest_document(self, payload: KGHandoffPayload) -> None:
        """Ingest processed chunks into the KG.

        Args:
            payload: Complete handoff payload with document metadata and chunks.

        Raises:
            KGIngestionError: On failure to ingest.
        """
        ...

    async def remove_document(self, document_id: UUID, game_id: UUID) -> None:
        """Remove a document's contribution from the KG.

        Args:
            document_id: UUID of the document to remove.
            game_id: UUID of the owning game (for KG scoping).

        Raises:
            KGIngestionError: On failure to remove.
        """
        ...


class StubKGHandoffService:
    """V1 stub that logs handoff operations and returns successfully.

    Placeholder implementation for use until EPIC-003 delivers the real
    KG integration. Satisfies the KGHandoffService protocol.
    """

    async def ingest_document(self, payload: KGHandoffPayload) -> None:
        """Log the ingestion handoff and return successfully.

        Args:
            payload: Complete handoff payload with document metadata and chunks.
        """
        logger.info(
            "KG handoff stub: ingest_document called — "
            "document_id=%s, version_id=%s, game_id=%s, "
            "document_type=%s, chunk_count=%d, replaces_version_id=%s",
            payload.document_id,
            payload.version_id,
            payload.game_id,
            payload.document_type,
            len(payload.chunks),
            payload.replaces_version_id,
        )

    async def remove_document(self, document_id: UUID, game_id: UUID) -> None:
        """Log the removal request and return successfully.

        Args:
            document_id: UUID of the document to remove.
            game_id: UUID of the owning game.
        """
        logger.info(
            "KG handoff stub: remove_document called — document_id=%s, game_id=%s",
            document_id,
            game_id,
        )

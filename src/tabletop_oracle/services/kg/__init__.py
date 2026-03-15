"""Knowledge Graph engine services.

Contains concept extraction, association discovery, embedding, retrieval,
audit logging, graph integration, conflict resolution, and supporting
infrastructure for the KG pipeline (EPIC-003).
"""

from tabletop_oracle.services.kg.associations import (
    AssociationDiscoveryService,
    ConceptSummary,
    DiscoveredAssociation,
)
from tabletop_oracle.services.kg.audit import KGAuditService
from tabletop_oracle.services.kg.conflicts import ConflictResolutionService
from tabletop_oracle.services.kg.extraction import (
    ConceptExtractionService,
    ExtractedConcept,
    GameContext,
)
from tabletop_oracle.services.kg.integration import GraphIntegrationService

__all__ = [
    "AssociationDiscoveryService",
    "ConceptExtractionService",
    "ConceptSummary",
    "ConflictResolutionService",
    "DiscoveredAssociation",
    "ExtractedConcept",
    "GameContext",
    "GraphIntegrationService",
    "KGAuditService",
]

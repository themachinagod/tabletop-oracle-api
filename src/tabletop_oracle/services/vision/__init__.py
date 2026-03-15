"""Shared vision processing service for image-to-text conversion.

Used by the ingestion pipeline (document images) and the AI workflow
(ad-hoc context images). Uses the vision_processing model slot from
the model_slots configuration table (PRD-007).
"""

from tabletop_oracle.services.vision.models import (
    ImageDescription,
    TokenAttribution,
    VisionProcessingError,
    VisionUnavailableError,
)
from tabletop_oracle.services.vision.protocols import VisionServiceProtocol
from tabletop_oracle.services.vision.service import VisionService

__all__ = [
    "ImageDescription",
    "TokenAttribution",
    "VisionProcessingError",
    "VisionService",
    "VisionServiceProtocol",
    "VisionUnavailableError",
]

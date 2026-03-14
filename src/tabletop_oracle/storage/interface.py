"""Abstract blob storage interface."""

from abc import ABC, abstractmethod


class BlobStorageInterface(ABC):
    """Contract for blob storage backends."""

    @abstractmethod
    async def store(self, key: str, data: bytes, content_type: str) -> str:
        """Store a blob and return its retrieval key."""

    @abstractmethod
    async def retrieve(self, key: str) -> bytes:
        """Retrieve a blob by key."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a blob by key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check whether a blob exists."""

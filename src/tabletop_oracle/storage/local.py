"""Local filesystem blob storage implementation."""

from pathlib import Path

from tabletop_oracle.config import settings
from tabletop_oracle.storage.interface import BlobStorageInterface


class LocalBlobStorage(BlobStorageInterface):
    """Store blobs on the local filesystem."""

    def __init__(self, base_path: str | None = None) -> None:
        self._base = Path(base_path or settings.blob_storage_local_path)
        self._base.mkdir(parents=True, exist_ok=True)

    async def store(self, key: str, data: bytes, content_type: str) -> str:
        """Write blob to local filesystem."""
        path = self._base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    async def retrieve(self, key: str) -> bytes:
        """Read blob from local filesystem."""
        path = self._base / key
        if not path.exists():
            msg = f"Blob not found: {key}"
            raise FileNotFoundError(msg)
        return path.read_bytes()

    async def delete(self, key: str) -> None:
        """Delete blob from local filesystem."""
        path = self._base / key
        if path.exists():
            path.unlink()

    async def exists(self, key: str) -> bool:
        """Check if blob exists on local filesystem."""
        return (self._base / key).exists()

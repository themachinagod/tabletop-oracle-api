"""Local filesystem blob storage implementation."""

import json
from pathlib import Path

from tabletop_oracle.config import settings
from tabletop_oracle.storage.interface import BlobStorageInterface

_META_SUFFIX = ".meta"


class LocalBlobStorage(BlobStorageInterface):
    """Store blobs on the local filesystem."""

    def __init__(self, base_path: str | None = None) -> None:
        self._base = Path(base_path or settings.blob_storage_local_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, key: str) -> Path:
        """Resolve *key* to a path guaranteed to be inside the storage root.

        Raises:
            ValueError: If the key is empty, absolute, contains '..' segments,
                or otherwise resolves outside the storage root.
        """
        if not key or not key.strip():
            msg = f"Invalid storage key: {key!r}"
            raise ValueError(msg)

        if key.startswith("/") or key.startswith("\\"):
            msg = f"Storage key must be relative: {key!r}"
            raise ValueError(msg)

        # Reject any path component that is '..'
        if ".." in Path(key).parts:
            msg = f"Storage key must not contain '..': {key!r}"
            raise ValueError(msg)

        resolved = (self._base / key).resolve()

        # Belt-and-suspenders: ensure the resolved path is under _base
        if not resolved.is_relative_to(self._base):
            msg = f"Storage key resolves outside storage root: {key!r}"
            raise ValueError(msg)

        return resolved

    async def store(self, key: str, data: bytes, content_type: str) -> str:
        """Write blob and its metadata to local filesystem."""
        path = self._safe_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        meta_path = path.with_name(path.name + _META_SUFFIX)
        meta_path.write_text(json.dumps({"content_type": content_type}))

        return key

    async def retrieve(self, key: str) -> bytes:
        """Read blob from local filesystem."""
        path = self._safe_path(key)
        if not path.exists():
            msg = f"Blob not found: {key}"
            raise FileNotFoundError(msg)
        return path.read_bytes()

    async def delete(self, key: str) -> None:
        """Delete blob and its metadata from local filesystem."""
        path = self._safe_path(key)
        if path.exists():
            path.unlink()
        meta_path = path.with_name(path.name + _META_SUFFIX)
        if meta_path.exists():
            meta_path.unlink()

    async def exists(self, key: str) -> bool:
        """Check if blob exists on local filesystem."""
        return self._safe_path(key).exists()

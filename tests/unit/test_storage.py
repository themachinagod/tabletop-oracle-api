"""Unit tests for local blob storage."""

import json
from pathlib import Path

import pytest

from tabletop_oracle.storage.local import _META_SUFFIX, LocalBlobStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalBlobStorage:
    """Create a LocalBlobStorage using a temp directory."""
    return LocalBlobStorage(base_path=str(tmp_path))


# --- Basic operations ---


@pytest.mark.asyncio
async def test_store_and_retrieve(storage: LocalBlobStorage) -> None:
    """Stored blob can be retrieved."""
    key = await storage.store("test.txt", b"hello", "text/plain")
    assert key == "test.txt"
    data = await storage.retrieve("test.txt")
    assert data == b"hello"


@pytest.mark.asyncio
async def test_exists(storage: LocalBlobStorage) -> None:
    """exists() returns correct state."""
    assert await storage.exists("missing.txt") is False
    await storage.store("present.txt", b"data", "text/plain")
    assert await storage.exists("present.txt") is True


@pytest.mark.asyncio
async def test_delete(storage: LocalBlobStorage) -> None:
    """delete() removes the blob."""
    await storage.store("temp.txt", b"data", "text/plain")
    await storage.delete("temp.txt")
    assert await storage.exists("temp.txt") is False


@pytest.mark.asyncio
async def test_delete_missing_is_noop(storage: LocalBlobStorage) -> None:
    """delete() on missing key does not raise."""
    await storage.delete("nonexistent.txt")


@pytest.mark.asyncio
async def test_retrieve_missing_raises(storage: LocalBlobStorage) -> None:
    """retrieve() on missing key raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        await storage.retrieve("nonexistent.txt")


@pytest.mark.asyncio
async def test_store_nested_path(storage: LocalBlobStorage) -> None:
    """Storing with nested key creates parent directories."""
    await storage.store("a/b/c.txt", b"nested", "text/plain")
    data = await storage.retrieve("a/b/c.txt")
    assert data == b"nested"


# --- Path traversal prevention ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malicious_key",
    [
        "../../etc/passwd",
        "../secret.txt",
        "a/../../etc/shadow",
        "/etc/passwd",
        "\\etc\\passwd",
    ],
)
async def test_store_rejects_path_traversal(storage: LocalBlobStorage, malicious_key: str) -> None:
    """store() rejects keys that escape the storage root."""
    with pytest.raises(ValueError, match="Storage key"):
        await storage.store(malicious_key, b"data", "text/plain")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malicious_key",
    [
        "../../etc/passwd",
        "../secret.txt",
        "/etc/passwd",
    ],
)
async def test_retrieve_rejects_path_traversal(
    storage: LocalBlobStorage, malicious_key: str
) -> None:
    """retrieve() rejects keys that escape the storage root."""
    with pytest.raises(ValueError, match="Storage key"):
        await storage.retrieve(malicious_key)


@pytest.mark.asyncio
async def test_exists_rejects_path_traversal(storage: LocalBlobStorage) -> None:
    """exists() rejects keys that escape the storage root."""
    with pytest.raises(ValueError, match="Storage key"):
        await storage.exists("../../etc/passwd")


@pytest.mark.asyncio
async def test_delete_rejects_path_traversal(storage: LocalBlobStorage) -> None:
    """delete() rejects keys that escape the storage root."""
    with pytest.raises(ValueError, match="Storage key"):
        await storage.delete("../../etc/passwd")


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_key", ["", "   "])
async def test_store_rejects_empty_key(storage: LocalBlobStorage, empty_key: str) -> None:
    """store() rejects empty or whitespace-only keys."""
    with pytest.raises(ValueError, match="Invalid storage key"):
        await storage.store(empty_key, b"data", "text/plain")


# --- content_type persistence ---


@pytest.mark.asyncio
async def test_store_persists_content_type(storage: LocalBlobStorage, tmp_path: Path) -> None:
    """store() writes content_type to a sidecar metadata file."""
    await storage.store("doc.pdf", b"pdf-bytes", "application/pdf")

    meta_path = tmp_path / ("doc.pdf" + _META_SUFFIX)
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["content_type"] == "application/pdf"


@pytest.mark.asyncio
async def test_delete_removes_metadata(storage: LocalBlobStorage, tmp_path: Path) -> None:
    """delete() removes the sidecar metadata file alongside the blob."""
    await storage.store("doc.pdf", b"pdf-bytes", "application/pdf")
    await storage.delete("doc.pdf")

    meta_path = tmp_path / ("doc.pdf" + _META_SUFFIX)
    assert not meta_path.exists()

"""Unit tests for local blob storage."""

import pytest

from tabletop_oracle.storage.local import LocalBlobStorage


@pytest.fixture
def storage(tmp_path: object) -> LocalBlobStorage:
    """Create a LocalBlobStorage using a temp directory."""
    return LocalBlobStorage(base_path=str(tmp_path))


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

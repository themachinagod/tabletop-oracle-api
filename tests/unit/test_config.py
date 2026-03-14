"""Unit tests for application configuration."""

import os

from tabletop_oracle.config import Settings


def test_settings_has_required_fields() -> None:
    """Settings exposes all expected configuration fields."""
    s = Settings()
    assert hasattr(s, "database_url")
    assert hasattr(s, "database_url_async")
    assert hasattr(s, "log_level")
    assert hasattr(s, "bypass_auth")
    assert hasattr(s, "blob_storage_backend")
    assert hasattr(s, "auth_session_timeout_days")
    assert hasattr(s, "secret_key")
    assert hasattr(s, "frontend_origin")


def test_settings_reads_from_environment() -> None:
    """Settings picks up values from environment variables."""
    original = os.environ.get("LOG_LEVEL")
    os.environ["LOG_LEVEL"] = "DEBUG"
    try:
        s = Settings()
        assert s.log_level == "DEBUG"
    finally:
        if original is None:
            os.environ.pop("LOG_LEVEL", None)
        else:
            os.environ["LOG_LEVEL"] = original


def test_settings_auth_session_timeout_default() -> None:
    """Auth session timeout defaults to 30 days."""
    s = Settings()
    assert s.auth_session_timeout_days == 30


def test_settings_blob_storage_default() -> None:
    """Blob storage defaults to local backend."""
    s = Settings()
    assert s.blob_storage_backend == "local"

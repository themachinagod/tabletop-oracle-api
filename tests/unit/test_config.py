"""Unit tests for application configuration."""

from tabletop_oracle.config import Settings


def test_settings_defaults() -> None:
    """Settings has sensible defaults for development."""
    s = Settings()
    assert "postgresql" in s.database_url
    assert "asyncpg" in s.database_url_async
    assert s.log_level == "INFO"
    assert s.bypass_auth is False
    assert s.blob_storage_backend == "local"
    assert s.auth_session_timeout_days == 30


def test_settings_from_env(monkeypatch: object) -> None:
    """Settings can be overridden via environment variables."""
    import os

    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["BYPASS_AUTH"] = "true"
    try:
        s = Settings()
        assert s.log_level == "DEBUG"
        assert s.bypass_auth is True
    finally:
        os.environ.pop("LOG_LEVEL", None)
        os.environ.pop("BYPASS_AUTH", None)

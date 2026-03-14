"""Unit tests for structured logging configuration."""

import json
import logging
from io import StringIO

import pytest
import structlog

from tabletop_oracle.logging import _REDACTED, _SENSITIVE_KEYS, configure_logging


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog configuration between tests.

    structlog caches loggers; reset to avoid cross-test contamination.
    """
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _capture_log_output(log_level: str = "DEBUG") -> tuple[StringIO, structlog.stdlib.BoundLogger]:
    """Configure logging to capture output in a StringIO buffer.

    Returns the buffer and a bound logger for assertions.
    """
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
    )
    handler.setFormatter(formatter)

    configure_logging(log_level=log_level)

    # Replace all root handlers with our capture handler.
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    log: structlog.stdlib.BoundLogger = structlog.get_logger("test_logger")
    return buf, log


class TestLogOutputFormat:
    """Verify log output is valid JSON with required fields."""

    def test_log_entry_is_valid_json(self) -> None:
        """Each log line must be a parseable JSON object."""
        buf, log = _capture_log_output()
        log.info("test message")

        output = buf.getvalue().strip()
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self) -> None:
        """Every log entry must include timestamp, level, message, logger."""
        buf, log = _capture_log_output()
        log.info("hello world")

        parsed = json.loads(buf.getvalue().strip())
        assert "timestamp" in parsed
        assert parsed["level"] == "info"
        assert parsed["message"] == "hello world"
        assert parsed["logger"] == "test_logger"

    def test_timestamp_is_iso8601_utc(self) -> None:
        """Timestamp must be ISO 8601 and end with +00:00 or Z."""
        buf, log = _capture_log_output()
        log.info("ts check")

        parsed = json.loads(buf.getvalue().strip())
        ts = parsed["timestamp"]
        # structlog's ISO timestamper with utc=True produces "+00:00" suffix.
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_bound_context_appears_in_output(self) -> None:
        """Context variables bound via structlog appear in JSON output."""
        buf, log = _capture_log_output()
        structlog.contextvars.bind_contextvars(request_id="abc-123")
        log.info("with context")

        parsed = json.loads(buf.getvalue().strip())
        assert parsed["request_id"] == "abc-123"

    def test_log_level_filtering(self) -> None:
        """Messages below the configured level are suppressed."""
        buf, log = _capture_log_output(log_level="WARNING")
        log.info("should not appear")
        log.warning("should appear")

        output = buf.getvalue().strip()
        # Only the warning should be present.
        lines = [ln for ln in output.splitlines() if ln.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["message"] == "should appear"


class TestSensitiveDataFiltering:
    """Verify sensitive values are redacted from log output."""

    @pytest.mark.parametrize("key", sorted(_SENSITIVE_KEYS))
    def test_sensitive_key_redacted(self, key: str) -> None:
        """Each sensitive key must have its value replaced with [REDACTED]."""
        buf, log = _capture_log_output()
        log.info("sensitive test", **{key: "super-secret-value"})

        parsed = json.loads(buf.getvalue().strip())
        assert parsed[key] == _REDACTED

    def test_case_insensitive_filtering(self) -> None:
        """Sensitive key matching is case-insensitive."""
        buf, log = _capture_log_output()
        # The key in the event dict will be lowercase (Python kwargs), but
        # the filter checks key.lower() so mixed-case dict keys are caught.
        structlog.contextvars.bind_contextvars(Password="my-secret")
        log.info("case test")

        parsed = json.loads(buf.getvalue().strip())
        assert parsed.get("Password") == _REDACTED or parsed.get("password") == _REDACTED

    def test_non_sensitive_keys_unchanged(self) -> None:
        """Non-sensitive keys pass through unmodified."""
        buf, log = _capture_log_output()
        log.info("safe test", user_id="usr-42", game_id="gm-99")

        parsed = json.loads(buf.getvalue().strip())
        assert parsed["user_id"] == "usr-42"
        assert parsed["game_id"] == "gm-99"


class TestConfigureLogging:
    """Verify configure_logging sets up the environment correctly."""

    def test_default_log_level_is_info(self) -> None:
        """Without arguments, log level defaults to INFO."""
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_custom_log_level(self) -> None:
        """Passing a log level string configures the root logger."""
        configure_logging(log_level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_invalid_log_level_falls_back_to_info(self) -> None:
        """An unrecognised level string falls back to INFO."""
        configure_logging(log_level="NONEXISTENT")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_third_party_loggers_configured(self) -> None:
        """Uvicorn and sqlalchemy loggers propagate through structlog."""
        configure_logging()
        for name in ("uvicorn", "uvicorn.access", "sqlalchemy.engine"):
            lgr = logging.getLogger(name)
            assert lgr.propagate is True

"""Structured logging configuration using structlog.

Configures structlog with a JSON processor chain for production-ready
structured logging. All log entries include timestamp, level, request_id,
message, and logger fields. Third-party library logs (uvicorn, sqlalchemy,
asyncpg) are routed through structlog for consistent formatting.

Sensitive data (auth tokens, passwords, cookies) is filtered before output.
"""

import logging
import logging.config
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

# Keys whose values must never appear in log output.
_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "secret",
        "secret_key",
        "api_key",
        "client_secret",
        "set-cookie",
    }
)

_REDACTED = "[REDACTED]"


def _filter_sensitive_data(
    _logger: Any,
    _method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Remove sensitive values from log events.

    Checks all keys (case-insensitive) against a deny-list and replaces
    matching values with '[REDACTED]'.
    """
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Set up structlog and stdlib logging with JSON output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors used by both structlog and stdlib log entries.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _filter_sensitive_data,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog.
    structlog.configure(
        processors=[
            *shared_processors,
            # Prepare for final rendering — stdlib or structlog native.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Stdlib logging config — routes all stdlib loggers through structlog's
    # ProcessorFormatter so third-party libraries produce consistent JSON.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "structlog": {
                    "()": lambda: handler,
                },
            },
            "root": {
                "handlers": ["structlog"],
                "level": level,
            },
            "loggers": {
                "uvicorn": {"level": level, "propagate": True},
                "uvicorn.access": {"level": level, "propagate": True},
                "uvicorn.error": {"level": level, "propagate": True},
                "sqlalchemy.engine": {"level": level, "propagate": True},
                "asyncpg": {"level": level, "propagate": True},
            },
        }
    )

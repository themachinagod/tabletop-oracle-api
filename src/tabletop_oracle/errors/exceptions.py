"""Application exception hierarchy.

All application exceptions extend AppError and carry:
- message: human-readable description
- code: machine-readable error code (e.g. "NOT_FOUND")
- status_code: HTTP status code for the error response

Exception handlers in handlers.py map these to structured API responses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldError:
    """Per-field validation error detail."""

    field: str
    message: str
    code: str = "invalid"


class AppError(Exception):
    """Base application error.

    All domain exceptions inherit from this. The exception handler
    uses status_code and code to build the structured error response.

    Args:
        message: Human-readable error description.
        code: Machine-readable error code.
        status_code: HTTP status code.
    """

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An internal error occurred",
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class ValidationError(AppError):
    """Input validation failed.

    Supports an optional list of per-field errors for structured
    validation feedback.

    Args:
        message: Summary validation message.
        details: Per-field error details.
    """

    status_code: int = 400
    code: str = "VALIDATION_ERROR"

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        details: list[FieldError] | None = None,
    ) -> None:
        self.details: list[FieldError] = details or []
        super().__init__(message)


class NotFoundError(AppError):
    """Resource not found.

    Args:
        resource: The type of resource (e.g. "Game").
        identifier: The identifier that was looked up.
    """

    status_code: int = 404
    code: str = "NOT_FOUND"

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(message=f"{resource} '{identifier}' not found")


class AuthenticationError(AppError):
    """Authentication required or failed."""

    status_code: int = 401
    code: str = "AUTHENTICATION_REQUIRED"

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message=message)


class ForbiddenError(AppError):
    """Insufficient permissions."""

    status_code: int = 403
    code: str = "FORBIDDEN"

    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(message=message)


class ConflictError(AppError):
    """Resource conflict (e.g. duplicate, concurrent modification)."""

    status_code: int = 409
    code: str = "CONFLICT"

    def __init__(self, message: str = "Resource conflict") -> None:
        super().__init__(message=message)


class UnprocessableEntityError(AppError):
    """Request is syntactically valid but semantically incorrect."""

    status_code: int = 422
    code: str = "UNPROCESSABLE_ENTITY"

    def __init__(self, message: str = "Unprocessable entity") -> None:
        super().__init__(message=message)


class ContentTooLargeError(AppError):
    """Request payload exceeds size limit."""

    status_code: int = 413
    code: str = "CONTENT_TOO_LARGE"

    def __init__(self, message: str = "Content too large") -> None:
        super().__init__(message=message)


class RateLimitedError(AppError):
    """Client has exceeded rate limits."""

    status_code: int = 429
    code: str = "RATE_LIMITED"

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message=message)


class ConfigurationError(AppError):
    """Required system configuration is missing or invalid.

    Used when a required configuration entry (e.g. a model slot) is not
    present in the database. Maps to 500 because this is a server-side
    setup issue, not a client error.

    Args:
        message: Description of the missing or invalid configuration.
    """

    status_code: int = 500
    code: str = "CONFIGURATION_ERROR"

    def __init__(self, message: str = "Required configuration is missing") -> None:
        super().__init__(message=message)


class ServiceUnavailableError(AppError):
    """Downstream service or resource is unavailable."""

    status_code: int = 503
    code: str = "SERVICE_UNAVAILABLE"

    def __init__(self, message: str = "Service temporarily unavailable") -> None:
        super().__init__(message=message)


class GuardrailExceededError(AppError):
    """A token or query guardrail would be exceeded.

    Raised by ModelClient when a pre-call guardrail check fails.
    The message contains a player-friendly explanation.

    Args:
        message: Human-readable denial message for the player.
        guardrail_type: Which guardrail was hit (for logging/metrics).
    """

    status_code: int = 429
    code: str = "GUARDRAIL_EXCEEDED"

    def __init__(
        self,
        message: str = "Usage guardrail exceeded",
        *,
        guardrail_type: str = "",
    ) -> None:
        self.guardrail_type = guardrail_type
        super().__init__(message=message)


class ModelUnavailableError(AppError):
    """Both primary and fallback models failed.

    Raised by ModelClient when all model call attempts (including
    retries and fallback) have been exhausted.

    Args:
        message: Description of the failure.
        capability: The model capability that was requested.
    """

    status_code: int = 503
    code: str = "MODEL_UNAVAILABLE"

    def __init__(
        self,
        message: str = "AI model is currently unavailable",
        *,
        capability: str = "",
    ) -> None:
        self.capability = capability
        super().__init__(message=message)

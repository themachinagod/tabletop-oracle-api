"""Unit tests for the application exception hierarchy."""

from tabletop_oracle.errors.exceptions import (
    AppError,
    AuthenticationError,
    ConflictError,
    ContentTooLargeError,
    FieldError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ServiceUnavailableError,
    UnprocessableEntityError,
    ValidationError,
)

# --- AppError (base) ---


def test_app_error_defaults() -> None:
    """AppError stores message and defaults to 500 / INTERNAL_ERROR."""
    err = AppError("something broke")
    assert err.message == "something broke"
    assert err.code == "INTERNAL_ERROR"
    assert err.status_code == 500
    assert str(err) == "something broke"


def test_app_error_default_message() -> None:
    """AppError has a sensible default message."""
    err = AppError()
    assert err.message == "An internal error occurred"


def test_app_error_custom_code_and_status() -> None:
    """AppError accepts custom code and status_code overrides."""
    err = AppError("oops", code="CUSTOM", status_code=418)
    assert err.code == "CUSTOM"
    assert err.status_code == 418


def test_app_error_is_exception() -> None:
    """AppError is a proper Exception subclass."""
    err = AppError("test")
    assert isinstance(err, Exception)


# --- ValidationError ---


def test_validation_error_defaults() -> None:
    """ValidationError defaults to 400 / VALIDATION_ERROR with empty details."""
    err = ValidationError()
    assert err.status_code == 400
    assert err.code == "VALIDATION_ERROR"
    assert err.message == "Validation failed"
    assert err.details == []


def test_validation_error_with_details() -> None:
    """ValidationError carries per-field error details."""
    details = [
        FieldError(field="email", message="invalid format"),
        FieldError(field="name", message="required", code="required"),
    ]
    err = ValidationError("Input invalid", details=details)
    assert len(err.details) == 2
    assert err.details[0].field == "email"
    assert err.details[1].code == "required"


def test_validation_error_is_app_error() -> None:
    """ValidationError inherits from AppError."""
    assert issubclass(ValidationError, AppError)


# --- NotFoundError ---


def test_not_found_error() -> None:
    """NotFoundError formats resource and identifier."""
    err = NotFoundError("Game", "abc-123")
    assert err.status_code == 404
    assert err.code == "NOT_FOUND"
    assert "Game" in err.message
    assert "abc-123" in err.message


# --- AuthenticationError ---


def test_authentication_error_defaults() -> None:
    """AuthenticationError defaults to 401 / AUTHENTICATION_REQUIRED."""
    err = AuthenticationError()
    assert err.status_code == 401
    assert err.code == "AUTHENTICATION_REQUIRED"
    assert "Authentication" in err.message


def test_authentication_error_custom_message() -> None:
    """AuthenticationError accepts a custom message."""
    err = AuthenticationError("Token expired")
    assert err.message == "Token expired"


# --- ForbiddenError ---


def test_forbidden_error_defaults() -> None:
    """ForbiddenError defaults to 403 / FORBIDDEN."""
    err = ForbiddenError()
    assert err.status_code == 403
    assert err.code == "FORBIDDEN"
    assert "permissions" in err.message.lower()


# --- ConflictError ---


def test_conflict_error_defaults() -> None:
    """ConflictError defaults to 409 / CONFLICT."""
    err = ConflictError()
    assert err.status_code == 409
    assert err.code == "CONFLICT"


def test_conflict_error_custom_message() -> None:
    """ConflictError accepts a custom message."""
    err = ConflictError("Duplicate game name")
    assert err.message == "Duplicate game name"


# --- UnprocessableEntityError ---


def test_unprocessable_entity_error_defaults() -> None:
    """UnprocessableEntityError defaults to 422 / UNPROCESSABLE_ENTITY."""
    err = UnprocessableEntityError()
    assert err.status_code == 422
    assert err.code == "UNPROCESSABLE_ENTITY"


# --- ContentTooLargeError ---


def test_content_too_large_error_defaults() -> None:
    """ContentTooLargeError defaults to 413 / CONTENT_TOO_LARGE."""
    err = ContentTooLargeError()
    assert err.status_code == 413
    assert err.code == "CONTENT_TOO_LARGE"


# --- RateLimitedError ---


def test_rate_limited_error_defaults() -> None:
    """RateLimitedError defaults to 429 / RATE_LIMITED."""
    err = RateLimitedError()
    assert err.status_code == 429
    assert err.code == "RATE_LIMITED"


# --- ServiceUnavailableError ---


def test_service_unavailable_error_defaults() -> None:
    """ServiceUnavailableError defaults to 503 / SERVICE_UNAVAILABLE."""
    err = ServiceUnavailableError()
    assert err.status_code == 503
    assert err.code == "SERVICE_UNAVAILABLE"


# --- FieldError dataclass ---


def test_field_error_defaults() -> None:
    """FieldError defaults code to 'invalid'."""
    fe = FieldError(field="email", message="bad format")
    assert fe.field == "email"
    assert fe.message == "bad format"
    assert fe.code == "invalid"


def test_field_error_custom_code() -> None:
    """FieldError accepts a custom code."""
    fe = FieldError(field="name", message="required", code="required")
    assert fe.code == "required"

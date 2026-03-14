"""Unit tests for the application exception hierarchy."""

from tabletop_oracle.errors.exceptions import (
    AppError,
    AuthenticationError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)


def test_app_error_defaults() -> None:
    """AppError stores message and defaults code to INTERNAL_ERROR."""
    err = AppError("something broke")
    assert err.message == "something broke"
    assert err.code == "INTERNAL_ERROR"
    assert str(err) == "something broke"


def test_app_error_custom_code() -> None:
    """AppError accepts a custom error code."""
    err = AppError("oops", code="CUSTOM")
    assert err.code == "CUSTOM"


def test_not_found_error() -> None:
    """NotFoundError formats resource and identifier."""
    err = NotFoundError("Game", "abc-123")
    assert err.code == "NOT_FOUND"
    assert "Game" in err.message
    assert "abc-123" in err.message


def test_validation_error_with_field() -> None:
    """ValidationError stores optional field reference."""
    err = ValidationError("invalid email", field="email")
    assert err.code == "VALIDATION_ERROR"
    assert err.field == "email"
    assert err.message == "invalid email"


def test_validation_error_without_field() -> None:
    """ValidationError works without a field."""
    err = ValidationError("bad input")
    assert err.field is None


def test_authentication_error_default_message() -> None:
    """AuthenticationError has a sensible default message."""
    err = AuthenticationError()
    assert err.code == "AUTHENTICATION_ERROR"
    assert "Authentication" in err.message


def test_forbidden_error_default_message() -> None:
    """ForbiddenError has a sensible default message."""
    err = ForbiddenError()
    assert err.code == "FORBIDDEN"
    assert "permissions" in err.message.lower()

"""Unit tests for exception handlers."""

import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.exceptions import RequestValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

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
)
from tabletop_oracle.errors.exceptions import (
    ValidationError as AppValidationError,
)
from tabletop_oracle.errors.handlers import (
    app_error_handler,
    unhandled_exception_handler,
    validation_error_handler,
)


def _make_request(request_id: str = "test-req-id") -> MagicMock:
    """Create a mock Request with request_id on state."""
    request = MagicMock()
    request.state.request_id = request_id
    return request


def _parse_body(response: object) -> dict[str, object]:
    """Extract the JSON body from a JSONResponse."""
    return json.loads(getattr(response, "body", b"{}"))


# --- AppError handler ---


@pytest.mark.asyncio
async def test_app_error_handler_base() -> None:
    """AppError handler returns 500 with INTERNAL_ERROR code."""
    request = _make_request()
    exc = AppError("kaboom")
    response = await app_error_handler(request, exc)

    assert response.status_code == 500
    body = _parse_body(response)
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "kaboom"
    assert body["meta"]["request_id"] == "test-req-id"
    assert body["error"]["details"] == []


@pytest.mark.asyncio
async def test_app_error_handler_not_found() -> None:
    """NotFoundError handler returns 404."""
    request = _make_request("req-404")
    exc = NotFoundError("Game", "xyz")
    response = await app_error_handler(request, exc)

    assert response.status_code == 404
    body = _parse_body(response)
    assert body["error"]["code"] == "NOT_FOUND"
    assert "Game" in body["error"]["message"]
    assert body["meta"]["request_id"] == "req-404"


@pytest.mark.asyncio
async def test_app_error_handler_authentication() -> None:
    """AuthenticationError handler returns 401."""
    request = _make_request()
    exc = AuthenticationError()
    response = await app_error_handler(request, exc)

    assert response.status_code == 401
    body = _parse_body(response)
    assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"


@pytest.mark.asyncio
async def test_app_error_handler_forbidden() -> None:
    """ForbiddenError handler returns 403."""
    request = _make_request()
    response = await app_error_handler(request, ForbiddenError())
    assert response.status_code == 403
    body = _parse_body(response)
    assert body["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_app_error_handler_conflict() -> None:
    """ConflictError handler returns 409."""
    request = _make_request()
    response = await app_error_handler(request, ConflictError("dup"))
    assert response.status_code == 409
    body = _parse_body(response)
    assert body["error"]["code"] == "CONFLICT"


@pytest.mark.asyncio
async def test_app_error_handler_unprocessable() -> None:
    """UnprocessableEntityError handler returns 422."""
    request = _make_request()
    response = await app_error_handler(request, UnprocessableEntityError())
    assert response.status_code == 422
    body = _parse_body(response)
    assert body["error"]["code"] == "UNPROCESSABLE_ENTITY"


@pytest.mark.asyncio
async def test_app_error_handler_content_too_large() -> None:
    """ContentTooLargeError handler returns 413."""
    request = _make_request()
    response = await app_error_handler(request, ContentTooLargeError())
    assert response.status_code == 413
    body = _parse_body(response)
    assert body["error"]["code"] == "CONTENT_TOO_LARGE"


@pytest.mark.asyncio
async def test_app_error_handler_rate_limited() -> None:
    """RateLimitedError handler returns 429."""
    request = _make_request()
    response = await app_error_handler(request, RateLimitedError())
    assert response.status_code == 429
    body = _parse_body(response)
    assert body["error"]["code"] == "RATE_LIMITED"


@pytest.mark.asyncio
async def test_app_error_handler_service_unavailable() -> None:
    """ServiceUnavailableError handler returns 503."""
    request = _make_request()
    response = await app_error_handler(request, ServiceUnavailableError())
    assert response.status_code == 503
    body = _parse_body(response)
    assert body["error"]["code"] == "SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_app_error_handler_validation_with_details() -> None:
    """ValidationError handler includes per-field details."""
    request = _make_request()
    exc = AppValidationError(
        "Bad input",
        details=[
            FieldError(field="email", message="invalid format"),
            FieldError(field="age", message="must be positive", code="min_value"),
        ],
    )
    response = await app_error_handler(request, exc)

    assert response.status_code == 400
    body = _parse_body(response)
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert len(body["error"]["details"]) == 2
    assert body["error"]["details"][0]["field"] == "email"
    assert body["error"]["details"][1]["code"] == "min_value"


@pytest.mark.asyncio
async def test_app_error_handler_validation_without_details() -> None:
    """ValidationError without details returns empty details array."""
    request = _make_request()
    exc = AppValidationError("Bad input")
    response = await app_error_handler(request, exc)

    body = _parse_body(response)
    assert body["error"]["details"] == []


# --- RequestValidationError handler ---


@pytest.mark.asyncio
async def test_validation_error_handler() -> None:
    """Pydantic RequestValidationError maps to 400 with field details."""
    request = _make_request("req-val")

    # Build a realistic Pydantic validation error
    error_details: list[InitErrorDetails] = [
        {
            "type": PydanticCustomError("missing", "Field required"),
            "loc": ("body", "name"),
            "input": None,
        },
    ]
    pydantic_exc = ValidationError.from_exception_data(title="test", line_errors=error_details)
    exc = RequestValidationError(pydantic_exc.errors())

    response = await validation_error_handler(request, exc)

    assert response.status_code == 400
    body = _parse_body(response)
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["meta"]["request_id"] == "req-val"
    assert len(body["error"]["details"]) == 1
    assert body["error"]["details"][0]["field"] == "name"


@pytest.mark.asyncio
async def test_validation_error_handler_strips_body_prefix() -> None:
    """Field location strips 'body' prefix, joins remaining with dots."""
    request = _make_request()

    error_details: list[InitErrorDetails] = [
        {
            "type": PydanticCustomError("string_type", "String required"),
            "loc": ("body", "address", "city"),
            "input": 123,
        },
    ]
    pydantic_exc = ValidationError.from_exception_data(title="test", line_errors=error_details)
    exc = RequestValidationError(pydantic_exc.errors())

    response = await validation_error_handler(request, exc)

    body = _parse_body(response)
    assert body["error"]["details"][0]["field"] == "address.city"


# --- Unhandled exception handler ---


@pytest.mark.asyncio
async def test_unhandled_exception_handler_returns_500() -> None:
    """Catch-all returns 500 with generic message, no internal details."""
    request = _make_request("req-500")
    exc = RuntimeError("secret database error: password=hunter2")
    response = await unhandled_exception_handler(request, exc)

    assert response.status_code == 500
    body = _parse_body(response)
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["meta"]["request_id"] == "req-500"
    # Must NOT leak internal details
    assert "hunter2" not in body["error"]["message"]
    assert "database" not in body["error"]["message"]
    assert body["error"]["message"] == "An internal error occurred"


@pytest.mark.asyncio
async def test_unhandled_exception_handler_logs_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catch-all logs the full exception at ERROR level."""
    request = _make_request("req-log")
    exc = ValueError("something unexpected")

    with caplog.at_level(logging.ERROR):
        await unhandled_exception_handler(request, exc)

    assert "something unexpected" in caplog.text
    assert "req-log" in caplog.text


# --- Missing request_id fallback ---


@pytest.mark.asyncio
async def test_handler_fallback_when_no_request_id() -> None:
    """Handler uses 'unknown' when correlation middleware hasn't set request_id."""
    request = MagicMock()
    # Remove request_id from state
    del request.state.request_id
    exc = AppError("test")
    response = await app_error_handler(request, exc)

    body = _parse_body(response)
    assert body["meta"]["request_id"] == "unknown"

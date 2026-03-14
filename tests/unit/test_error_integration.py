"""Integration tests for error handling through the full FastAPI stack.

These tests verify that exceptions raised in endpoints produce the
correct structured error responses including status codes, error codes,
field details, and request_id from the correlation middleware.
"""

import pytest
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field

from tabletop_oracle.errors.exceptions import (
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
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware

# --- Test app with error-triggering endpoints ---

_router = APIRouter()


@_router.get("/trigger/not-found")
async def trigger_not_found() -> dict[str, str]:
    """Raise NotFoundError."""
    raise NotFoundError("Widget", "w-123")


@_router.get("/trigger/auth")
async def trigger_auth() -> dict[str, str]:
    """Raise AuthenticationError."""
    raise AuthenticationError()


@_router.get("/trigger/forbidden")
async def trigger_forbidden() -> dict[str, str]:
    """Raise ForbiddenError."""
    raise ForbiddenError()


@_router.get("/trigger/conflict")
async def trigger_conflict() -> dict[str, str]:
    """Raise ConflictError."""
    raise ConflictError("Duplicate name")


@_router.get("/trigger/unprocessable")
async def trigger_unprocessable() -> dict[str, str]:
    """Raise UnprocessableEntityError."""
    raise UnprocessableEntityError()


@_router.get("/trigger/too-large")
async def trigger_too_large() -> dict[str, str]:
    """Raise ContentTooLargeError."""
    raise ContentTooLargeError()


@_router.get("/trigger/rate-limited")
async def trigger_rate_limited() -> dict[str, str]:
    """Raise RateLimitedError."""
    raise RateLimitedError()


@_router.get("/trigger/unavailable")
async def trigger_unavailable() -> dict[str, str]:
    """Raise ServiceUnavailableError."""
    raise ServiceUnavailableError()


@_router.get("/trigger/validation")
async def trigger_validation() -> dict[str, str]:
    """Raise ValidationError with field details."""
    raise AppValidationError(
        "Input invalid",
        details=[
            FieldError(field="email", message="not a valid email"),
            FieldError(field="age", message="must be >= 0", code="min_value"),
        ],
    )


@_router.get("/trigger/unhandled")
async def trigger_unhandled() -> dict[str, str]:
    """Raise an unexpected exception."""
    msg = "secret: password=hunter2"
    raise RuntimeError(msg)


class _TestBody(BaseModel):
    """Request body for pydantic validation test."""

    name: str = Field(min_length=1)
    count: int = Field(gt=0)


@_router.post("/trigger/pydantic")
async def trigger_pydantic(body: _TestBody) -> dict[str, str]:
    """Endpoint requiring a valid body — triggers Pydantic validation."""
    return {"ok": body.name}


def _create_test_app() -> FastAPI:
    """Build a minimal FastAPI app with error handlers for testing."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.add_middleware(CorrelationMiddleware)
    test_app.include_router(_router)
    return test_app


@pytest.fixture
async def error_client() -> AsyncClient:
    """Async test client for error integration tests."""
    test_app = _create_test_app()
    transport = ASGITransport(app=test_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac  # type: ignore[misc]


# --- Integration tests ---


def _assert_error_structure(body: dict[str, object]) -> None:
    """Verify the response follows the ErrorEnvelope structure."""
    assert "error" in body
    assert "meta" in body
    error = body["error"]
    assert isinstance(error, dict)
    assert "code" in error
    assert "message" in error
    assert "details" in error
    meta = body["meta"]
    assert isinstance(meta, dict)
    assert "request_id" in meta
    assert meta["request_id"] != ""


@pytest.mark.asyncio
async def test_not_found_response(error_client: AsyncClient) -> None:
    """NotFoundError produces 404 with correct structure."""
    resp = await error_client.get("/trigger/not-found")
    assert resp.status_code == 404
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "NOT_FOUND"
    assert "Widget" in body["error"]["message"]


@pytest.mark.asyncio
async def test_auth_error_response(error_client: AsyncClient) -> None:
    """AuthenticationError produces 401."""
    resp = await error_client.get("/trigger/auth")
    assert resp.status_code == 401
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"


@pytest.mark.asyncio
async def test_forbidden_response(error_client: AsyncClient) -> None:
    """ForbiddenError produces 403."""
    resp = await error_client.get("/trigger/forbidden")
    assert resp.status_code == 403
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_conflict_response(error_client: AsyncClient) -> None:
    """ConflictError produces 409."""
    resp = await error_client.get("/trigger/conflict")
    assert resp.status_code == 409
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "CONFLICT"


@pytest.mark.asyncio
async def test_unprocessable_response(error_client: AsyncClient) -> None:
    """UnprocessableEntityError produces 422."""
    resp = await error_client.get("/trigger/unprocessable")
    assert resp.status_code == 422
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "UNPROCESSABLE_ENTITY"


@pytest.mark.asyncio
async def test_content_too_large_response(error_client: AsyncClient) -> None:
    """ContentTooLargeError produces 413."""
    resp = await error_client.get("/trigger/too-large")
    assert resp.status_code == 413
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "CONTENT_TOO_LARGE"


@pytest.mark.asyncio
async def test_rate_limited_response(error_client: AsyncClient) -> None:
    """RateLimitedError produces 429."""
    resp = await error_client.get("/trigger/rate-limited")
    assert resp.status_code == 429
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "RATE_LIMITED"


@pytest.mark.asyncio
async def test_service_unavailable_response(error_client: AsyncClient) -> None:
    """ServiceUnavailableError produces 503."""
    resp = await error_client.get("/trigger/unavailable")
    assert resp.status_code == 503
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_validation_error_with_details_response(error_client: AsyncClient) -> None:
    """ValidationError with field details produces 400 with details array."""
    resp = await error_client.get("/trigger/validation")
    assert resp.status_code == 400
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert len(body["error"]["details"]) == 2
    assert body["error"]["details"][0]["field"] == "email"
    assert body["error"]["details"][1]["code"] == "min_value"


@pytest.mark.asyncio
async def test_pydantic_validation_response(error_client: AsyncClient) -> None:
    """Pydantic RequestValidationError produces 400 with per-field details."""
    resp = await error_client.post(
        "/trigger/pydantic",
        json={"name": "", "count": -1},
    )
    assert resp.status_code == 400
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert len(body["error"]["details"]) >= 2


@pytest.mark.asyncio
async def test_pydantic_missing_fields_response(error_client: AsyncClient) -> None:
    """Missing required fields produce 400 with field details."""
    resp = await error_client.post("/trigger/pydantic", json={})
    assert resp.status_code == 400
    body = resp.json()
    _assert_error_structure(body)
    fields = [d["field"] for d in body["error"]["details"]]
    assert "name" in fields
    assert "count" in fields


@pytest.mark.asyncio
async def test_unhandled_exception_response(error_client: AsyncClient) -> None:
    """Unhandled exception produces 500 with no internal details."""
    resp = await error_client.get("/trigger/unhandled")
    assert resp.status_code == 500
    body = resp.json()
    _assert_error_structure(body)
    assert body["error"]["code"] == "INTERNAL_ERROR"
    # Must not leak internal details
    assert "hunter2" not in body["error"]["message"]
    assert "secret" not in body["error"]["message"]
    assert body["error"]["message"] == "An internal error occurred"


@pytest.mark.asyncio
async def test_request_id_propagated(error_client: AsyncClient) -> None:
    """Request ID from X-Request-ID header appears in error response."""
    resp = await error_client.get(
        "/trigger/not-found",
        headers={"X-Request-ID": "custom-req-id-123"},
    )
    body = resp.json()
    assert body["meta"]["request_id"] == "custom-req-id-123"


@pytest.mark.asyncio
async def test_request_id_generated_when_missing(error_client: AsyncClient) -> None:
    """Request ID is auto-generated when X-Request-ID header is absent."""
    resp = await error_client.get("/trigger/not-found")
    body = resp.json()
    # Should be a UUID-like string, not empty
    assert len(body["meta"]["request_id"]) > 10

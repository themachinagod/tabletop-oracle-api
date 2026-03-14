"""Unit tests for correlation and logging middleware."""

import json
import logging
from collections.abc import AsyncGenerator
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import structlog
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.logging import configure_logging
from tabletop_oracle.main import app


async def _override_db_healthy() -> AsyncGenerator[AsyncMock, None]:
    """Dependency override yielding a healthy mock DB session for middleware tests."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise


@pytest.fixture(autouse=True)
def _mock_db_for_health() -> Any:
    """Override get_db so health endpoint works without a real database."""
    from tabletop_oracle.api.deps import get_db

    app.dependency_overrides[get_db] = _override_db_healthy
    yield
    app.dependency_overrides.clear()


def _make_test_app(*, inject_user: bool = False) -> FastAPI:
    """Build a minimal FastAPI app with middleware for edge-case tests.

    Args:
        inject_user: If True, add a middleware that sets current_user
            before LoggingMiddleware runs (simulating auth middleware).
    """
    from collections.abc import Awaitable, Callable

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    from tabletop_oracle.middleware.correlation import CorrelationMiddleware
    from tabletop_oracle.middleware.logging import LoggingMiddleware

    test_app = FastAPI()

    class FakeUser:
        """Minimal user object for testing user_id binding."""

        def __init__(self, user_id: str) -> None:
            self.id = user_id

    class FakeAuthMiddleware(BaseHTTPMiddleware):
        """Injects a fake current_user onto request.state."""

        async def dispatch(
            self,
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            request.state.current_user = FakeUser("usr-42")
            return await call_next(request)

    # Middleware order: Correlation (outermost) -> Logging -> FakeAuth (innermost).
    if inject_user:
        test_app.add_middleware(FakeAuthMiddleware)
    test_app.add_middleware(LoggingMiddleware)
    test_app.add_middleware(CorrelationMiddleware)

    @test_app.get("/with-user")
    async def with_user() -> dict[str, str]:
        return {"status": "ok"}

    @test_app.get("/error")
    async def error_route() -> None:
        raise RuntimeError("deliberate test error")

    return test_app


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog state between tests."""
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _setup_capture(log_level: str = "DEBUG") -> StringIO:
    """Configure structlog and capture all output to a StringIO buffer.

    Must be called before making requests so the processor chain is active.
    Returns the buffer for assertion.
    """
    configure_logging(log_level=log_level)
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

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    return buf


def _parse_request_lines(output: str) -> list[dict[str, object]]:
    """Extract JSON log lines that are request or slow_request messages."""
    results: list[dict[str, object]] = []
    for ln in output.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            parsed = json.loads(ln)
        except json.JSONDecodeError:
            continue
        event = parsed.get("message", "")
        if event in ("request", "slow_request"):
            results.append(parsed)
    return results


def _parse_json_lines(output: str) -> list[dict[str, object]]:
    """Parse all valid JSON lines from output."""
    results: list[dict[str, object]] = []
    for ln in output.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            results.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return results


class TestCorrelationMiddleware:
    """Tests for CorrelationMiddleware."""

    @pytest.mark.asyncio
    async def test_correlation_id_generated(self) -> None:
        """Requests without X-Request-ID get one generated."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    @pytest.mark.asyncio
    async def test_correlation_id_forwarded(self) -> None:
        """Requests with X-Request-ID get the same ID echoed back."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "test-id-123"},
            )
        assert response.headers["X-Request-ID"] == "test-id-123"

    @pytest.mark.asyncio
    async def test_structlog_context_cleared_between_requests(self) -> None:
        """Structlog context must not leak between requests."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "first-request"},
            )
            # After the request, context should be clean.
            ctx = structlog.contextvars.get_contextvars()
            assert "request_id" not in ctx


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_request_logged_with_required_fields(self) -> None:
        """A successful request produces a JSON log with required fields."""
        buf = _setup_capture()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "log-test-001"},
            )

        entries = _parse_request_lines(buf.getvalue())
        assert len(entries) >= 1, f"No request log. Output: {buf.getvalue()}"

        entry = entries[0]
        assert entry["request_id"] == "log-test-001"
        assert entry["method"] == "GET"
        assert entry["endpoint"] == "/api/v1/health"
        assert "status_code" in entry
        assert "duration_ms" in entry
        assert "timestamp" in entry
        assert "level" in entry

    @pytest.mark.asyncio
    async def test_slow_request_logged_at_warning(self) -> None:
        """Requests taking >100ms are logged at WARNING level."""
        buf = _setup_capture()

        # Lower the threshold to 0ms so any request is "slow".
        with patch(
            "tabletop_oracle.middleware.logging._SLOW_REQUEST_THRESHOLD_MS",
            -1.0,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.get(
                    "/api/v1/health",
                    headers={"X-Request-ID": "slow-req"},
                )

        entries = _parse_request_lines(buf.getvalue())
        slow = [e for e in entries if e["message"] == "slow_request"]
        assert len(slow) >= 1, f"Expected slow_request log. Output: {buf.getvalue()}"
        assert slow[0]["level"] == "warning"

    @pytest.mark.asyncio
    async def test_request_id_included_in_log(self) -> None:
        """The request_id from correlation middleware appears in logs."""
        buf = _setup_capture()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "corr-id-xyz"},
            )

        entries = _parse_request_lines(buf.getvalue())
        assert any(e.get("request_id") == "corr-id-xyz" for e in entries)


class TestLoggingIntegration:
    """Integration-style tests verifying the full middleware chain."""

    @pytest.mark.asyncio
    async def test_full_request_produces_valid_json_log(self) -> None:
        """End-to-end: a request through the app produces valid JSON logs."""
        buf = _setup_capture()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")

        assert response.status_code == 200

        entries = _parse_json_lines(buf.getvalue())
        # At minimum, the request log should be present.
        request_entries = [e for e in entries if e.get("message") in ("request", "slow_request")]
        assert len(request_entries) >= 1

        # All structlog-formatted lines have timestamp and level.
        for entry in request_entries:
            assert "timestamp" in entry
            assert "level" in entry

    @pytest.mark.asyncio
    async def test_404_request_logged(self) -> None:
        """Requests to non-existent endpoints are logged with 404 status."""
        buf = _setup_capture()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/api/v1/nonexistent")

        entries = _parse_request_lines(buf.getvalue())
        assert any(e.get("status_code") == 404 for e in entries)


class TestLoggingMiddlewareEdgeCases:
    """Tests for user_id binding and exception logging paths."""

    @pytest.mark.asyncio
    async def test_user_id_bound_when_current_user_present(self) -> None:
        """When request.state.current_user is set, user_id appears in logs."""
        buf = _setup_capture()
        test_app = _make_test_app(inject_user=True)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/with-user")

        entries = _parse_request_lines(buf.getvalue())
        assert len(entries) >= 1, f"No request log. Output: {buf.getvalue()}"
        assert entries[0].get("user_id") == "usr-42"

    @pytest.mark.asyncio
    async def test_exception_logged_at_error_with_traceback(self) -> None:
        """Unhandled exceptions produce ERROR logs with traceback."""
        buf = _setup_capture()
        test_app = _make_test_app()

        transport = ASGITransport(app=test_app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/error")

        all_lines = _parse_json_lines(buf.getvalue())
        error_lines = [e for e in all_lines if e.get("message") == "request_error"]
        assert len(error_lines) >= 1, f"No error log. Output: {buf.getvalue()}"
        entry = error_lines[0]
        assert entry["level"] == "error"
        assert "traceback" in entry
        assert "deliberate test error" in str(entry["traceback"])
        assert "duration_ms" in entry

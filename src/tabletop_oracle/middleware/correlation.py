"""Correlation ID middleware — generates and propagates X-Request-ID.

Binds the request_id to structlog's context variables so every log call
within a request automatically includes it. Context is cleared at the
end of each request to prevent cross-request leakage.
"""

import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request/response cycle.

    The ID is stored on ``request.state.request_id`` and bound to
    structlog's context variables for automatic inclusion in all log
    entries produced during the request.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Generate or forward X-Request-ID header and bind to structlog context."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Bind to structlog context so all downstream log calls include it.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Prevent context leaking to the next request.
            structlog.contextvars.clear_contextvars()

"""Structured JSON request logging middleware.

Logs every request at INFO level with method, path, status_code, duration_ms,
and request_id. Slow requests (>100ms) are logged at WARNING. Unhandled
exceptions are logged at ERROR with traceback and request context.

If ``request.state.current_user`` is available (set by auth middleware),
the ``user_id`` is bound to the structlog context for the request.
"""

import time
import traceback
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_SLOW_REQUEST_THRESHOLD_MS = 100.0

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with structured JSON output."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log request method, path, status, and duration.

        Binds contextual fields (endpoint, method, user_id) to structlog
        context. Logs at WARNING for slow requests and ERROR for exceptions.
        """
        # Bind request-scoped context.
        structlog.contextvars.bind_contextvars(
            endpoint=request.url.path,
            method=request.method,
        )

        start = time.monotonic()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            structlog.contextvars.bind_contextvars(duration_ms=duration_ms)
            logger.error(
                "request_error",
                traceback=traceback.format_exc(),
            )
            raise

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        status_code = response.status_code

        # Bind user_id if auth middleware has populated it during call_next.
        current_user = getattr(request.state, "current_user", None)
        if current_user is not None:
            user_id = getattr(current_user, "id", None) or str(current_user)
            structlog.contextvars.bind_contextvars(user_id=str(user_id))

        structlog.contextvars.bind_contextvars(
            status_code=status_code,
            duration_ms=duration_ms,
        )

        if duration_ms > _SLOW_REQUEST_THRESHOLD_MS:
            logger.warning("slow_request")
        else:
            logger.info("request")

        return response

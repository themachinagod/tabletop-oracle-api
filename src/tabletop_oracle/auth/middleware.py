"""Session validation middleware.

ASGI middleware that runs on every request, validates the auth session,
and injects a :class:`CurrentUser` into ``request.state.current_user``.

Public endpoints (auth, health, docs, OpenAPI) are skipped. For all
other endpoints, a valid session cookie (or Bearer token) is required.

When ``settings.bypass_auth`` is ``True``, a deterministic mock
``CurrentUser`` is injected without any session or database lookup —
intended for local development only.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from tabletop_oracle.auth.constants import (
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_PATH,
    SESSION_COOKIE_SAMESITE,
)
from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.auth.session_store import SessionStore
from tabletop_oracle.config import settings
from tabletop_oracle.database import async_session_factory
from tabletop_oracle.models.enums import UserStatus
from tabletop_oracle.models.user import User

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Literal

    from starlette.requests import Request
    from starlette.responses import Response

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

#: Regex patterns for public endpoints that skip authentication.
_PUBLIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^/api/v1/auth(/|$)"),
    re.compile(r"^/api/v1/health$"),
    re.compile(r"^/api/v1/docs$"),
    re.compile(r"^/api/v1/openapi\.json$"),
)

#: Deterministic mock user for ``bypass_auth`` mode.
_BYPASS_USER = CurrentUser(
    user_id=UUID("00000000-0000-0000-0000-000000000000"),
    role="curator",
    email="dev@localhost",
    display_name="Dev Bypass",
    session_id="bypass",
)

_AUTH_REQUIRED_CODE = "AUTHENTICATION_REQUIRED"

#: SameSite cookie attribute typed for Starlette's delete_cookie signature.
_SAMESITE = cast("Literal['lax', 'strict', 'none']", SESSION_COOKIE_SAMESITE)

#: Session IDs are 43-char URL-safe base64 tokens (secrets.token_urlsafe(32)).
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


def _is_public_path(path: str) -> bool:
    """Return True if the request path matches a public endpoint pattern.

    Args:
        path: The URL path to check (e.g. ``/api/v1/auth/login/google``).
    """
    return any(pattern.search(path) for pattern in _PUBLIC_PATTERNS)


def _extract_session_id(request: Request) -> str | None:
    """Extract and validate the session ID from cookie or Authorization header.

    Cookie takes precedence over Bearer header per F002 design. The
    extracted value is validated against the expected 43-char URL-safe
    base64 format before being returned.

    Args:
        request: The incoming HTTP request.

    Returns:
        The session ID string, or None if not found or invalid format.
    """
    cookie_sid = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_sid:
        if _SESSION_ID_PATTERN.match(cookie_sid):
            return cookie_sid
        logger.warning("session_id_invalid_format", source="cookie")
        return None

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            if _SESSION_ID_PATTERN.match(token):
                return token
            logger.warning("session_id_invalid_format", source="bearer")
            return None

    return None


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    *,
    clear_cookie: bool = False,
) -> JSONResponse:
    """Build a structured JSON error response matching the ErrorEnvelope format.

    Args:
        status_code: HTTP status code.
        code: Machine-readable error code.
        message: Human-readable error message.
        request_id: Correlation request ID.
        clear_cookie: If True, set an expired sid cookie to clear it.
    """
    body = {
        "error": {"code": code, "message": message, "details": []},
        "meta": {"request_id": request_id},
    }
    response = JSONResponse(status_code=status_code, content=body)
    if clear_cookie:
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            path=SESSION_COOKIE_PATH,
            samesite=_SAMESITE,
            httponly=SESSION_COOKIE_HTTPONLY,
            secure=settings.session_cookie_secure,
        )
    return response


class SessionMiddleware(BaseHTTPMiddleware):
    """Validate auth sessions and inject CurrentUser into request state.

    Runs on every request. Public endpoints are skipped. For protected
    endpoints, the middleware extracts the session ID, loads the session
    and user from the database, and injects ``CurrentUser`` into
    ``request.state.current_user``.

    Sliding expiry is maintained via ``SessionStore.touch()``, which
    internally throttles writes to once per hour to avoid a DB write
    on every request (see ``SessionStore.touch`` for details).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Validate session and inject CurrentUser, or return 401."""
        path = request.url.path

        if _is_public_path(path):
            return await call_next(request)

        if settings.bypass_auth:
            request.state.current_user = _BYPASS_USER
            structlog.contextvars.bind_contextvars(
                user_id=str(_BYPASS_USER.user_id),
            )
            return await call_next(request)

        request_id = str(getattr(request.state, "request_id", "unknown"))

        session_id = _extract_session_id(request)
        if session_id is None:
            return _error_response(401, _AUTH_REQUIRED_CODE, "Authentication required", request_id)

        async with async_session_factory() as db:
            auth_session = await SessionStore.get(db, session_id)

            if auth_session is None or auth_session.expires_at < datetime.now(UTC):
                return _error_response(
                    401,
                    _AUTH_REQUIRED_CODE,
                    "Authentication required",
                    request_id,
                    clear_cookie=True,
                )

            user: User | None = await db.get(User, auth_session.user_id)
            if user is None or user.status != UserStatus.ACTIVE:
                return _error_response(
                    401,
                    _AUTH_REQUIRED_CODE,
                    "Authentication required",
                    request_id,
                    clear_cookie=True,
                )

            current_user = CurrentUser(
                user_id=user.id,
                role=user.role.value,
                email=user.email,
                display_name=user.display_name,
                session_id=session_id,
            )
            request.state.current_user = current_user

            structlog.contextvars.bind_contextvars(user_id=str(user.id))

            # Sliding expiry — throttled inside SessionStore.touch()
            await SessionStore.touch(db, session_id)
            await db.commit()

        return await call_next(request)

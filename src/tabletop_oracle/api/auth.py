"""Auth router — OAuth 2.0 login, callback, logout, and user profile.

All endpoints live under ``/api/v1/auth`` (prefix applied in router.py).
Login and callback are unauthenticated. Logout and me require a valid session.

OAuth state and PKCE code_verifier are stored in a short-lived httpOnly
cookie (``oauth_state``, 10-minute max-age, SameSite=Lax) to survive the
redirect round-trip. The callback validates state against the cookie for
CSRF protection before exchanging the authorization code.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any, Literal

from authlib.integrations.starlette_client import (  # type: ignore[import-untyped]
    OAuthError,
)
from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.auth.constants import (
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_PATH,
    SESSION_COOKIE_SAMESITE,
)
from tabletop_oracle.auth.providers import VALID_PROVIDERS, oauth
from tabletop_oracle.auth.session_store import SessionStore
from tabletop_oracle.auth.user_service import find_or_create_user
from tabletop_oracle.config import settings
from tabletop_oracle.errors.exceptions import AuthenticationError, NotFoundError
from tabletop_oracle.models.enums import OAuthProvider
from tabletop_oracle.schemas.common import DataEnvelope, ResponseMeta

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

# Type alias for Starlette's SameSite parameter.
_SameSite = Literal["lax", "strict", "none"]

#: Name of the short-lived cookie carrying OAuth state + PKCE verifier.
_OAUTH_STATE_COOKIE = "oauth_state"
#: Max age for the OAuth state cookie (10 minutes).
_OAUTH_STATE_MAX_AGE = 600


def _provider_enum(name: str) -> OAuthProvider:
    """Map a provider name string to the OAuthProvider enum.

    Args:
        name: Lowercase provider name (e.g. "google").

    Returns:
        Corresponding OAuthProvider enum value.
    """
    return OAuthProvider(name)


def _build_redirect_uri(provider: str, request: Request) -> str:
    """Build the OAuth callback redirect URI.

    Uses ``oauth_redirect_base_url`` from settings to ensure the
    redirect URI matches what is registered with the provider, rather
    than relying on the request host (which may differ behind proxies).

    Args:
        provider: Provider name (e.g. "google").
        request: The incoming request (unused but kept for signature compat).

    Returns:
        Fully-qualified callback URL.
    """
    base = settings.oauth_redirect_base_url.rstrip("/")
    return f"{base}/api/v1/auth/callback/{provider}"


@router.get("/login/{provider}")
async def login(provider: str, request: Request) -> Response:
    """Initiate OAuth 2.0 login flow with the specified provider.

    Generates a cryptographic state parameter and PKCE code_verifier,
    stores both in a short-lived httpOnly cookie, and redirects the user
    to the provider's authorization endpoint.

    Args:
        provider: OAuth provider name ("google" or "microsoft").
        request: The incoming HTTP request.

    Returns:
        RedirectResponse to the provider's authorization URL.

    Raises:
        NotFoundError: If the provider name is invalid.
    """
    if provider not in VALID_PROVIDERS:
        raise NotFoundError("Provider", provider)

    oauth_client = oauth.create_client(provider)
    if oauth_client is None:
        raise NotFoundError("Provider", provider)

    redirect_uri = _build_redirect_uri(provider, request)

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)

    # Authlib handles PKCE code_verifier/code_challenge internally
    # when code_challenge_method is set. We store state in a cookie.
    response: Response = await oauth_client.authorize_redirect(
        request,
        redirect_uri,
        state=state,
    )

    # Store state in a short-lived httpOnly cookie
    cookie_value = json.dumps({"state": state})
    response.set_cookie(
        key=_OAUTH_STATE_COOKIE,
        value=cookie_value,
        max_age=_OAUTH_STATE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )

    return response


@router.get("/callback/{provider}")
async def callback(provider: str, request: Request, db: DbSession) -> Response:
    """Handle OAuth 2.0 callback after provider authentication.

    Validates the state parameter against the cookie for CSRF protection,
    exchanges the authorization code for tokens, validates the ID token,
    extracts user claims, finds or creates the user, creates a server-side
    session, sets the session cookie, and redirects to the frontend.

    Args:
        provider: OAuth provider name ("google" or "microsoft").
        request: The incoming HTTP request (contains auth code and state).
        db: Async database session for user and session operations.

    Returns:
        RedirectResponse to the frontend with session cookie set.

    Raises:
        NotFoundError: If the provider name is invalid.
        AuthenticationError: If state validation fails, provider returns
            an error, or token exchange fails.
    """
    if provider not in VALID_PROVIDERS:
        raise NotFoundError("Provider", provider)

    # Check for error from provider
    error = request.query_params.get("error")
    if error:
        error_description = request.query_params.get("error_description", error)
        logger.warning("OAuth error from %s: %s", provider, error_description)
        raise AuthenticationError(f"Authentication failed: {error_description}")

    # Validate state from cookie
    state_cookie = request.cookies.get(_OAUTH_STATE_COOKIE)
    if not state_cookie:
        raise AuthenticationError("Missing OAuth state cookie")

    try:
        cookie_data = json.loads(state_cookie)
        expected_state = cookie_data.get("state")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise AuthenticationError("Invalid OAuth state cookie") from exc

    actual_state = request.query_params.get("state")
    if not expected_state or not actual_state or expected_state != actual_state:
        raise AuthenticationError("OAuth state mismatch — possible CSRF")

    # Exchange code for tokens
    oauth_client = oauth.create_client(provider)
    if oauth_client is None:
        raise NotFoundError("Provider", provider)

    try:
        token: dict[str, Any] = await oauth_client.authorize_access_token(request)
    except OAuthError as exc:
        logger.warning("OAuth token exchange failed for %s: %s", provider, exc)
        raise AuthenticationError("Token exchange failed") from exc

    # Extract user info from ID token claims
    user_info = _extract_user_info(token, provider)

    # Find or create user
    provider_enum = _provider_enum(provider)
    user = await find_or_create_user(
        db=db,
        provider=provider_enum,
        subject_id=user_info["sub"],
        email=user_info["email"],
        display_name=user_info["name"],
    )

    # Create server-side session
    user_agent = request.headers.get("user-agent")
    client_ip = request.client.host if request.client else None
    session_id = await SessionStore.create(
        db=db,
        user_id=user.id,
        user_agent=user_agent,
        ip_address=client_ip,
    )

    # Build redirect response to frontend
    redirect_url = settings.frontend_origin
    response = RedirectResponse(url=redirect_url, status_code=302)

    # Set session cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=settings.auth_session_timeout_days * 86400,
        httponly=SESSION_COOKIE_HTTPONLY,
        samesite=SESSION_COOKIE_SAMESITE,  # type: ignore[arg-type]
        secure=settings.session_cookie_secure,
        path=SESSION_COOKIE_PATH,
    )

    # Clear the OAuth state cookie
    response.delete_cookie(
        key=_OAUTH_STATE_COOKIE,
        path="/",
        samesite="lax",
        secure=settings.session_cookie_secure,
        httponly=True,
    )

    return response


@router.post("/logout", status_code=204)
async def logout(request: Request, db: DbSession) -> Response:
    """Log out the current user by destroying their session.

    Requires a valid session cookie. Deletes the server-side session
    and clears the session cookie.

    Args:
        request: The incoming HTTP request (carries session cookie).
        db: Async database session for session deletion.

    Returns:
        204 No Content response with cleared session cookie.

    Raises:
        AuthenticationError: If no valid session cookie is present.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise AuthenticationError()

    # Verify session exists
    session = await SessionStore.get(db, session_id)
    if session is None:
        raise AuthenticationError()

    # Delete session
    await SessionStore.delete(db, session_id)

    # Build response and clear cookie
    response = Response(status_code=204)
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path=SESSION_COOKIE_PATH,
        samesite=SESSION_COOKIE_SAMESITE,  # type: ignore[arg-type]
        secure=settings.session_cookie_secure,
        httponly=SESSION_COOKIE_HTTPONLY,
    )
    return response


@router.get("/me")
async def me(request: Request, db: DbSession) -> JSONResponse:
    """Return the current user's profile.

    Requires a valid, non-expired session. Returns user profile data
    in the standard DataEnvelope format.

    Args:
        request: The incoming HTTP request (carries session cookie).
        db: Async database session for session and user lookup.

    Returns:
        JSONResponse with DataEnvelope containing user profile.

    Raises:
        AuthenticationError: If session is missing, invalid, or expired.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise AuthenticationError()

    session = await SessionStore.get(db, session_id)
    if session is None:
        raise AuthenticationError()

    # Touch session for sliding expiry
    await SessionStore.touch(db, session_id)

    # Eagerly load user — session.user should be loaded via relationship
    user = session.user

    request_id: str = getattr(request.state, "request_id", "unknown")
    profile = {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role.value,
        "status": user.status.value,
    }
    envelope = DataEnvelope[dict[str, str]](
        data=profile,
        meta=ResponseMeta(request_id=request_id),
    )
    return JSONResponse(content=envelope.model_dump())


def _extract_user_info(token: dict[str, Any], provider: str) -> dict[str, str]:
    """Extract standardised user claims from an OAuth token response.

    Both Google and Microsoft return ``sub``, ``email``, and ``name``
    in the ID token userinfo. Falls back to reasonable defaults if
    ``name`` is missing.

    Args:
        token: The token response dict from Authlib (contains 'userinfo').
        provider: Provider name for error messages.

    Returns:
        Dict with keys: sub, email, name.

    Raises:
        AuthenticationError: If required claims are missing.
    """
    userinfo = token.get("userinfo", {})
    if not userinfo:
        raise AuthenticationError("No user info in token response")

    sub = userinfo.get("sub")
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("preferred_username") or email

    if not sub or not email:
        logger.warning("Missing required claims from %s: sub=%s, email=%s", provider, sub, email)
        raise AuthenticationError("Missing required identity claims")

    return {"sub": sub, "email": email, "name": name or ""}

"""OAuth 2.0 / OIDC provider registry using Authlib.

Configures Authlib OAuth clients for Google and Microsoft with PKCE (S256)
and OIDC auto-discovery. Provider client IDs and secrets are sourced from
application settings (environment variables).
"""

from __future__ import annotations

from authlib.integrations.starlette_client import OAuth  # type: ignore[import-untyped]

from tabletop_oracle.config import settings

# Authlib OAuth registry — shared application-level instance.
oauth = OAuth()

# Google OIDC provider
oauth.register(
    name="google",
    client_id=settings.oauth_google_client_id,
    client_secret=settings.oauth_google_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
        "code_challenge_method": "S256",
    },
)

# Microsoft OIDC provider (common tenant — multi-tenant)
oauth.register(
    name="microsoft",
    client_id=settings.oauth_microsoft_client_id,
    client_secret=settings.oauth_microsoft_client_secret,
    server_metadata_url=(
        "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration"
    ),
    client_kwargs={
        "scope": "openid email profile",
        "code_challenge_method": "S256",
    },
)

#: Valid provider names accepted by the auth router.
VALID_PROVIDERS: frozenset[str] = frozenset(("google", "microsoft"))

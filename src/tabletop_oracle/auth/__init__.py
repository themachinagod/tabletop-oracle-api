"""Authentication and session management package."""

from tabletop_oracle.auth.bootstrap import is_bootstrap_curator
from tabletop_oracle.auth.constants import (
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_PATH,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
)
from tabletop_oracle.auth.dependencies import CurrentUserDep, get_current_user, require_role
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.auth.ownership import check_ownership
from tabletop_oracle.auth.session_store import SessionStore

__all__ = [
    "SESSION_COOKIE_HTTPONLY",
    "SESSION_COOKIE_NAME",
    "SESSION_COOKIE_PATH",
    "SESSION_COOKIE_SAMESITE",
    "SESSION_COOKIE_SECURE",
    "CurrentUser",
    "CurrentUserDep",
    "SessionMiddleware",
    "SessionStore",
    "check_ownership",
    "get_current_user",
    "is_bootstrap_curator",
    "require_role",
]

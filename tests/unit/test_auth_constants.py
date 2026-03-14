"""Unit tests for auth cookie constants."""

from tabletop_oracle.auth.constants import (
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_PATH,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
)


class TestSessionCookieConstants:
    """Cookie constants match the F002 design spec."""

    def test_cookie_name(self) -> None:
        assert SESSION_COOKIE_NAME == "sid"

    def test_cookie_httponly(self) -> None:
        assert SESSION_COOKIE_HTTPONLY is True

    def test_cookie_secure_default(self) -> None:
        assert SESSION_COOKIE_SECURE is True

    def test_cookie_samesite(self) -> None:
        assert SESSION_COOKIE_SAMESITE == "lax"

    def test_cookie_path(self) -> None:
        assert SESSION_COOKIE_PATH == "/"

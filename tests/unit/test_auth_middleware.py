"""Unit tests for session middleware path matching and session extraction."""

from tabletop_oracle.auth.middleware import _extract_session_id, _is_public_path

#: Valid 43-char URL-safe base64 tokens for testing format validation.
_TOKEN_A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop0"
_TOKEN_B = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg"


class TestIsPublicPath:
    """Public path matching for auth bypass."""

    def test_auth_login_is_public(self) -> None:
        assert _is_public_path("/api/v1/auth/login/google") is True

    def test_auth_callback_is_public(self) -> None:
        assert _is_public_path("/api/v1/auth/callback/microsoft") is True

    def test_auth_logout_is_public(self) -> None:
        assert _is_public_path("/api/v1/auth/logout") is True

    def test_auth_me_is_public(self) -> None:
        """Auth endpoints are public at the middleware level (route-level auth)."""
        assert _is_public_path("/api/v1/auth/me") is True

    def test_auth_root_is_public(self) -> None:
        assert _is_public_path("/api/v1/auth") is True

    def test_health_is_public(self) -> None:
        assert _is_public_path("/api/v1/health") is True

    def test_docs_is_public(self) -> None:
        assert _is_public_path("/api/v1/docs") is True

    def test_openapi_json_is_public(self) -> None:
        assert _is_public_path("/api/v1/openapi.json") is True

    def test_games_is_protected(self) -> None:
        assert _is_public_path("/api/v1/games") is False

    def test_sessions_is_protected(self) -> None:
        assert _is_public_path("/api/v1/sessions") is False

    def test_admin_is_protected(self) -> None:
        assert _is_public_path("/api/v1/admin/users") is False

    def test_root_is_protected(self) -> None:
        assert _is_public_path("/") is False

    def test_health_with_suffix_is_protected(self) -> None:
        """Only exact /health matches, not /healthcheck."""
        assert _is_public_path("/api/v1/healthcheck") is False

    def test_docs_with_suffix_is_protected(self) -> None:
        assert _is_public_path("/api/v1/docs/extra") is False


class TestExtractSessionId:
    """Session ID extraction from cookie and Bearer header."""

    def _make_request(
        self,
        cookies: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> "FakeRequest":
        """Build a minimal request-like object for testing."""
        return FakeRequest(cookies=cookies or {}, headers=headers or {})

    def test_cookie_extraction(self) -> None:
        request = self._make_request(cookies={"sid": _TOKEN_A})
        assert _extract_session_id(request) == _TOKEN_A  # type: ignore[arg-type]

    def test_bearer_extraction(self) -> None:
        request = self._make_request(headers={"authorization": f"Bearer {_TOKEN_A}"})
        assert _extract_session_id(request) == _TOKEN_A  # type: ignore[arg-type]

    def test_cookie_takes_precedence_over_bearer(self) -> None:
        request = self._make_request(
            cookies={"sid": _TOKEN_A},
            headers={"authorization": f"Bearer {_TOKEN_B}"},
        )
        assert _extract_session_id(request) == _TOKEN_A  # type: ignore[arg-type]

    def test_no_session_returns_none(self) -> None:
        request = self._make_request()
        assert _extract_session_id(request) is None  # type: ignore[arg-type]

    def test_empty_bearer_returns_none(self) -> None:
        request = self._make_request(headers={"authorization": "Bearer "})
        assert _extract_session_id(request) is None  # type: ignore[arg-type]

    def test_non_bearer_auth_returns_none(self) -> None:
        request = self._make_request(headers={"authorization": "Basic abc123"})
        assert _extract_session_id(request) is None  # type: ignore[arg-type]

    def test_bearer_case_insensitive(self) -> None:
        request = self._make_request(headers={"authorization": f"BEARER {_TOKEN_A}"})
        assert _extract_session_id(request) == _TOKEN_A  # type: ignore[arg-type]

    def test_malformed_cookie_returns_none(self) -> None:
        """Session IDs that don't match the 43-char base64 format are rejected."""
        request = self._make_request(cookies={"sid": "too-short"})
        assert _extract_session_id(request) is None  # type: ignore[arg-type]

    def test_malformed_bearer_returns_none(self) -> None:
        """Bearer tokens that don't match the expected format are rejected."""
        request = self._make_request(headers={"authorization": "Bearer not-valid"})
        assert _extract_session_id(request) is None  # type: ignore[arg-type]


class FakeRequest:
    """Minimal request stub for unit testing session extraction."""

    def __init__(self, cookies: dict[str, str], headers: dict[str, str]) -> None:
        self.cookies = cookies
        self._headers = {k.lower(): v for k, v in headers.items()}

    @property
    def headers(self) -> "FakeHeaders":
        return FakeHeaders(self._headers)


class FakeHeaders:
    """Dict-like headers stub."""

    def __init__(self, data: dict[str, str]) -> None:
        self._data = data

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key.lower(), default)

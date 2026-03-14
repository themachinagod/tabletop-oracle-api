"""Unit tests for SessionStore.

Covers session ID generation properties and touch throttle logic
without requiring a database connection.
"""

import re
import string

from tabletop_oracle.auth.session_store import _TOUCH_THROTTLE, SessionStore


class TestGenerateSessionId:
    """Session ID generation meets cryptographic and format requirements."""

    def test_length_is_43_chars(self) -> None:
        """secrets.token_urlsafe(32) produces exactly 43 characters."""
        sid = SessionStore._generate_session_id()
        assert len(sid) == 43

    def test_url_safe_characters_only(self) -> None:
        """Session IDs contain only URL-safe base64 characters."""
        allowed = set(string.ascii_letters + string.digits + "-_")
        sid = SessionStore._generate_session_id()
        assert all(c in allowed for c in sid), f"Non-URL-safe char in {sid}"

    def test_url_safe_regex(self) -> None:
        """Session ID matches the URL-safe base64 pattern."""
        sid = SessionStore._generate_session_id()
        assert re.fullmatch(r"[A-Za-z0-9_-]+", sid) is not None

    def test_ids_are_unique(self) -> None:
        """Consecutive IDs are distinct (statistical near-certainty)."""
        ids = {SessionStore._generate_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_no_padding(self) -> None:
        """token_urlsafe does not include '=' padding characters."""
        sid = SessionStore._generate_session_id()
        assert "=" not in sid


class TestTouchThrottle:
    """Touch throttle constant is correctly configured."""

    def test_throttle_is_one_hour(self) -> None:
        assert _TOUCH_THROTTLE.total_seconds() == 3600

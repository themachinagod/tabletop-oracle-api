"""Unit tests for curator bootstrap logic.

Covers case sensitivity, empty list, whitespace handling, and
multiple email parsing.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from tabletop_oracle.auth.bootstrap import is_bootstrap_curator
from tabletop_oracle.config import Settings


def _make_settings(curator_emails: str) -> Settings:
    """Create a Settings instance with the given curator emails.

    Temporarily sets the environment variable to avoid pollution.
    """
    with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": curator_emails}):
        return Settings()


class TestIsBootstrapCurator:
    """Tests for the is_bootstrap_curator function."""

    def test_match_exact_email(self) -> None:
        """Exact email match returns True."""
        s = _make_settings("admin@example.com")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("admin@example.com") is True

    def test_case_insensitive_match(self) -> None:
        """Email matching is case-insensitive."""
        s = _make_settings("Admin@Example.COM")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("admin@example.com") is True

    def test_case_insensitive_input(self) -> None:
        """Input email casing does not affect match."""
        s = _make_settings("admin@example.com")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("ADMIN@EXAMPLE.COM") is True

    def test_no_match(self) -> None:
        """Non-matching email returns False."""
        s = _make_settings("admin@example.com")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("other@example.com") is False

    def test_empty_list(self) -> None:
        """Empty curator list returns False for any email."""
        s = _make_settings("")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("admin@example.com") is False

    def test_whitespace_only_list(self) -> None:
        """Whitespace-only curator list returns False."""
        s = _make_settings("   ")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("admin@example.com") is False

    def test_multiple_emails(self) -> None:
        """Multiple comma-separated emails are all recognised."""
        s = _make_settings("one@example.com, two@example.com, three@example.com")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("one@example.com") is True
            assert is_bootstrap_curator("two@example.com") is True
            assert is_bootstrap_curator("three@example.com") is True
            assert is_bootstrap_curator("four@example.com") is False

    def test_whitespace_around_emails(self) -> None:
        """Leading/trailing whitespace around emails is stripped."""
        s = _make_settings("  admin@example.com  ,  other@example.com  ")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("admin@example.com") is True
            assert is_bootstrap_curator("other@example.com") is True

    def test_trailing_comma(self) -> None:
        """Trailing comma does not produce a false empty entry."""
        s = _make_settings("admin@example.com,")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("admin@example.com") is True
            assert is_bootstrap_curator("") is False

    def test_whitespace_in_input(self) -> None:
        """Whitespace in the input email is stripped before comparison."""
        s = _make_settings("admin@example.com")
        with patch("tabletop_oracle.auth.bootstrap.settings", s):
            assert is_bootstrap_curator("  admin@example.com  ") is True

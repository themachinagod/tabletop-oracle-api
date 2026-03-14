"""Unit tests for CurrentUser dataclass."""

from uuid import UUID

import pytest

from tabletop_oracle.auth.models import CurrentUser


class TestCurrentUser:
    """CurrentUser dataclass has correct fields and behavior."""

    def test_create_with_all_fields(self) -> None:
        """CurrentUser can be instantiated with all required fields."""
        user = CurrentUser(
            user_id=UUID("12345678-1234-1234-1234-123456789abc"),
            role="player",
            email="test@example.com",
            display_name="Test User",
            session_id="abc123",
        )
        assert user.user_id == UUID("12345678-1234-1234-1234-123456789abc")
        assert user.role == "player"
        assert user.email == "test@example.com"
        assert user.display_name == "Test User"
        assert user.session_id == "abc123"

    def test_frozen_immutable(self) -> None:
        """CurrentUser instances are frozen (immutable)."""
        user = CurrentUser(
            user_id=UUID("12345678-1234-1234-1234-123456789abc"),
            role="curator",
            email="curator@example.com",
            display_name="Curator",
            session_id="xyz789",
        )
        with pytest.raises(AttributeError):
            user.role = "player"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two CurrentUser instances with same fields are equal."""
        kwargs = {
            "user_id": UUID("12345678-1234-1234-1234-123456789abc"),
            "role": "player",
            "email": "test@example.com",
            "display_name": "Test",
            "session_id": "sid",
        }
        assert CurrentUser(**kwargs) == CurrentUser(**kwargs)

    def test_inequality_different_role(self) -> None:
        """CurrentUser instances with different roles are not equal."""
        base = {
            "user_id": UUID("12345678-1234-1234-1234-123456789abc"),
            "email": "test@example.com",
            "display_name": "Test",
            "session_id": "sid",
        }
        assert CurrentUser(role="player", **base) != CurrentUser(role="curator", **base)

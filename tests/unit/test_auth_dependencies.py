"""Unit tests for role enforcement dependencies."""

from uuid import UUID

import pytest

from tabletop_oracle.auth.dependencies import (
    _ROLE_HIERARCHY,
    get_current_user,
    require_role,
)
from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.errors.exceptions import AuthenticationError, ForbiddenError


def _make_user(role: str = "player") -> CurrentUser:
    """Create a CurrentUser with the given role."""
    return CurrentUser(
        user_id=UUID("12345678-1234-1234-1234-123456789abc"),
        role=role,
        email="test@example.com",
        display_name="Test User",
        session_id="test-session",
    )


class FakeState:
    """Minimal request.state stub."""

    def __init__(self, *, current_user: CurrentUser | None = None) -> None:
        if current_user is not None:
            self.current_user = current_user


class FakeRequest:
    """Minimal Request stub for dependency testing."""

    def __init__(self, state: FakeState) -> None:
        self.state = state


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    def test_returns_user_when_present(self) -> None:
        user = _make_user()
        request = FakeRequest(FakeState(current_user=user))
        result = get_current_user(request)  # type: ignore[arg-type]
        assert result is user

    def test_raises_401_when_missing(self) -> None:
        request = FakeRequest(FakeState())
        with pytest.raises(AuthenticationError):
            get_current_user(request)  # type: ignore[arg-type]


class TestRequireRole:
    """Tests for require_role dependency factory."""

    def test_player_role_accepts_player(self) -> None:
        checker = require_role("player")
        user = _make_user("player")
        result = checker(current_user=user)  # type: ignore[operator]
        assert result is user

    def test_player_role_accepts_curator(self) -> None:
        checker = require_role("player")
        user = _make_user("curator")
        result = checker(current_user=user)  # type: ignore[operator]
        assert result is user

    def test_curator_role_accepts_curator(self) -> None:
        checker = require_role("curator")
        user = _make_user("curator")
        result = checker(current_user=user)  # type: ignore[operator]
        assert result is user

    def test_curator_role_rejects_player(self) -> None:
        checker = require_role("curator")
        user = _make_user("player")
        with pytest.raises(ForbiddenError):
            checker(current_user=user)  # type: ignore[operator]

    def test_unknown_role_in_user_rejected(self) -> None:
        """A user with an unknown role is rejected by any role check."""
        checker = require_role("player")
        user = _make_user("unknown")
        with pytest.raises(ForbiddenError):
            checker(current_user=user)  # type: ignore[operator]

    def test_unknown_role_in_factory_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown role"):
            require_role("admin")


class TestRoleHierarchy:
    """Tests for role hierarchy structure."""

    def test_curator_outranks_player(self) -> None:
        assert _ROLE_HIERARCHY["curator"] > _ROLE_HIERARCHY["player"]

    def test_two_roles_defined(self) -> None:
        assert set(_ROLE_HIERARCHY.keys()) == {"player", "curator"}

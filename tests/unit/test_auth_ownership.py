"""Unit tests for ownership enforcement utility."""

from uuid import UUID

import pytest

from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.auth.ownership import check_ownership
from tabletop_oracle.errors.exceptions import NotFoundError

_OWNER_ID = UUID("11111111-1111-1111-1111-111111111111")
_OTHER_ID = UUID("22222222-2222-2222-2222-222222222222")


def _make_user(user_id: UUID, role: str = "player") -> CurrentUser:
    """Create a CurrentUser with given user_id and role."""
    return CurrentUser(
        user_id=user_id,
        role=role,
        email="test@example.com",
        display_name="Test",
        session_id="sid",
    )


class TestCheckOwnership:
    """Tests for check_ownership utility."""

    def test_owner_access_allowed(self) -> None:
        """Owner of the resource passes the check."""
        user = _make_user(_OWNER_ID, "player")
        check_ownership(_OWNER_ID, user)

    def test_curator_bypass_allowed(self) -> None:
        """Curator can access any resource regardless of ownership."""
        user = _make_user(_OTHER_ID, "curator")
        check_ownership(_OWNER_ID, user)

    def test_non_owner_player_rejected_with_404(self) -> None:
        """Non-owner player gets NotFoundError (not ForbiddenError)."""
        user = _make_user(_OTHER_ID, "player")
        with pytest.raises(NotFoundError):
            check_ownership(_OWNER_ID, user, "Session", str(_OWNER_ID))

    def test_error_is_not_found_not_forbidden(self) -> None:
        """The error code is NOT_FOUND to avoid leaking resource existence."""
        user = _make_user(_OTHER_ID, "player")
        with pytest.raises(NotFoundError) as exc_info:
            check_ownership(_OWNER_ID, user, "Session", "abc")
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "NOT_FOUND"

    def test_curator_owner_access_allowed(self) -> None:
        """Curator who also happens to be the owner passes."""
        user = _make_user(_OWNER_ID, "curator")
        check_ownership(_OWNER_ID, user)

    def test_default_resource_type_and_id(self) -> None:
        """Default resource_type and resource_id used in error message."""
        user = _make_user(_OTHER_ID, "player")
        with pytest.raises(NotFoundError, match="Resource"):
            check_ownership(_OWNER_ID, user)

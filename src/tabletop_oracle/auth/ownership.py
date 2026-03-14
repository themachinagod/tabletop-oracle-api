"""Ownership enforcement utility.

Provides a single function to verify that the current user owns (or has
curator access to) a resource. Returns 404 (not 403) for non-owners to
avoid leaking resource existence — per F002 design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabletop_oracle.errors.exceptions import NotFoundError

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.auth.models import CurrentUser


def check_ownership(
    resource_user_id: UUID,
    current_user: CurrentUser,
    resource_type: str = "Resource",
    resource_id: str = "",
) -> None:
    """Verify the current user owns the resource or is a curator.

    Curators bypass the ownership check entirely. Non-curator users
    must match the resource's user_id. On failure, a NotFoundError is
    raised (not ForbiddenError) to avoid leaking resource existence.

    Args:
        resource_user_id: The UUID of the user who owns the resource.
        current_user: The authenticated user from request state.
        resource_type: Resource type name for the error message.
        resource_id: Resource identifier for the error message.

    Raises:
        NotFoundError: If the user is not a curator and does not own
            the resource.
    """
    if current_user.role == "curator":
        return

    if resource_user_id == current_user.user_id:
        return

    raise NotFoundError(resource_type, resource_id)

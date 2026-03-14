"""FastAPI dependencies for authentication and role enforcement.

Provides:
- ``get_current_user``: extracts ``CurrentUser`` from ``request.state``
- ``require_role``: dependency factory that enforces a minimum role level
- ``CurrentUserDep``: annotated type alias for convenience injection

Usage in route definitions::

    # Any authenticated user:
    @router.get("/items", dependencies=[Depends(get_current_user)])

    # Curator only:
    @router.get("/admin/items", dependencies=[Depends(require_role("curator"))])

    # Inject CurrentUser into handler:
    @router.get("/me")
    async def me(user: CurrentUserDep) -> ...:
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.errors.exceptions import AuthenticationError, ForbiddenError

#: Role hierarchy — higher index means more privilege.
_ROLE_HIERARCHY: dict[str, int] = {
    "player": 0,
    "curator": 1,
}


def get_current_user(request: Request) -> CurrentUser:
    """Extract the authenticated ``CurrentUser`` from request state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The CurrentUser instance.

    Raises:
        AuthenticationError: If no CurrentUser is present on the request.
    """
    current_user: CurrentUser | None = getattr(request.state, "current_user", None)
    if current_user is None:
        raise AuthenticationError()
    return current_user


def require_role(role: str) -> object:
    """Create a FastAPI dependency that enforces a minimum role level.

    Curator satisfies all role checks. Player role check is satisfied
    by both player and curator.

    Args:
        role: The minimum required role (``"player"`` or ``"curator"``).

    Returns:
        A FastAPI-compatible dependency callable.

    Raises:
        ValueError: If the role is not in the known hierarchy.
    """
    required_level = _ROLE_HIERARCHY.get(role)
    if required_level is None:
        msg = f"Unknown role: {role!r}. Valid roles: {sorted(_ROLE_HIERARCHY)}"
        raise ValueError(msg)

    def _check_role(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        """Verify the authenticated user has sufficient role privileges.

        Args:
            current_user: The authenticated user from request state.

        Returns:
            The CurrentUser if role check passes.

        Raises:
            ForbiddenError: If the user's role is insufficient.
        """
        user_level = _ROLE_HIERARCHY.get(current_user.role, -1)
        if user_level < required_level:
            raise ForbiddenError()
        return current_user

    _check_role.__qualname__ = f"require_role({role!r})"
    return _check_role


#: Annotated type alias for injecting CurrentUser into route handlers.
CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]

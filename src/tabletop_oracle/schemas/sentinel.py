"""UNSET sentinel for PATCH request semantics.

Provides a sentinel value to distinguish "field not provided" (omitted from
request body) from "field explicitly set to null" (present with value null).
Used as the default for optional fields in update schemas (GameUpdate,
ExpansionUpdate).

Usage in service layer:
    if game_update.publisher is not UNSET:
        game.publisher = game_update.publisher  # could be str or None
"""

from __future__ import annotations

from typing import Any


class _UnsetType:
    """Sentinel type representing an unset/omitted field.

    A single instance (UNSET) is used as the default for optional update
    schema fields. Identity comparison (``is UNSET`` / ``is not UNSET``)
    distinguishes omitted fields from explicitly null ones.
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        """Ensure only one UNSET instance exists (singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return a readable representation for debugging."""
        return "UNSET"

    def __bool__(self) -> bool:
        """UNSET is falsy, consistent with 'no value provided' semantics."""
        return False

    def __eq__(self, other: object) -> bool:
        """Equality check — only equal to itself."""
        return other is UNSET

    def __hash__(self) -> int:
        """Hashable for use in sets/dicts if needed."""
        return hash(type(self))

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> dict[str, Any]:
        """Pydantic V2 schema hook: accept any value but default to UNSET.

        The sentinel is never expected in incoming JSON — it is only used
        as a Python-side default. This schema tells Pydantic to pass the
        value through unchanged when encountered.
        """
        return {"type": "any"}


UNSET: _UnsetType = _UnsetType()
"""Singleton sentinel indicating a field was not provided in a PATCH request."""

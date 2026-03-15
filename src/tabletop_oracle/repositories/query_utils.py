"""Shared query utilities for repository layer.

Provides reusable helpers for safe query construction, including
wildcard escaping for ILIKE patterns.
"""


def escape_like(term: str) -> str:
    """Escape SQL ILIKE wildcard characters to prevent wildcard injection.

    Escapes ``\\``, ``%``, and ``_`` before interpolation into ILIKE
    patterns. The escape order matters: backslash must be escaped first
    to avoid double-escaping the backslashes introduced for ``%`` and ``_``.

    Args:
        term: Raw user-provided search string.

    Returns:
        Escaped string safe for ILIKE pattern interpolation.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

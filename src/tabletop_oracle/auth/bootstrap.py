"""Curator bootstrap — first-login role elevation for initial curators.

Parses ``INITIAL_CURATOR_EMAILS`` from settings (comma-separated) and
provides a case-insensitive check for whether a given email qualifies
for the curator role on first registration.
"""

from __future__ import annotations

from tabletop_oracle.config import settings


def _parse_curator_emails() -> frozenset[str]:
    """Parse and normalise the curator email list from settings.

    Returns:
        Frozenset of lowercased, whitespace-stripped email addresses.
        Empty frozenset if the setting is blank.
    """
    raw = settings.initial_curator_emails.strip()
    if not raw:
        return frozenset()
    return frozenset(email.strip().lower() for email in raw.split(",") if email.strip())


def is_bootstrap_curator(email: str) -> bool:
    """Check whether an email qualifies for curator role on first login.

    Comparison is case-insensitive. Leading/trailing whitespace in both
    the setting value and the argument are ignored.

    Args:
        email: The user's email address to check.

    Returns:
        True if the email is in the INITIAL_CURATOR_EMAILS list.
    """
    curator_emails = _parse_curator_emails()
    return email.strip().lower() in curator_emails

"""Session cookie constants per F002 design spec.

``SESSION_COOKIE_SECURE`` is the spec default (``True``). At runtime the
session middleware reads ``settings.session_cookie_secure`` to allow
``False`` in local development. The constant here documents the
production-intended value.
"""

SESSION_COOKIE_NAME: str = "sid"
SESSION_COOKIE_HTTPONLY: bool = True
SESSION_COOKIE_SECURE: bool = True
SESSION_COOKIE_SAMESITE: str = "lax"
SESSION_COOKIE_PATH: str = "/"

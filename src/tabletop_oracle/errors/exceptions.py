"""Application exception hierarchy."""


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class NotFoundError(AppError):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} '{identifier}' not found",
            code="NOT_FOUND",
        )


class ValidationError(AppError):
    """Input validation failed."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        super().__init__(message=message, code="VALIDATION_ERROR")


class AuthenticationError(AppError):
    """Authentication required or failed."""

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message=message, code="AUTHENTICATION_ERROR")


class ForbiddenError(AppError):
    """Insufficient permissions."""

    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(message=message, code="FORBIDDEN")

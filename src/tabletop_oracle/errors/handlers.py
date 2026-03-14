"""FastAPI exception handlers for structured error responses.

Three handlers are registered:
1. AppError — maps application exceptions to structured responses
2. RequestValidationError — maps Pydantic validation to 400 with field details
3. Exception — catch-all that logs the traceback and returns a generic 500

All responses use the ErrorEnvelope schema with a request_id from
the correlation middleware.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from tabletop_oracle.errors.exceptions import AppError, ValidationError
from tabletop_oracle.schemas.common import (
    ErrorDetail,
    ErrorEnvelope,
    ErrorMeta,
    FieldErrorDetail,
)

logger = logging.getLogger(__name__)

_GENERIC_500_MESSAGE = "An internal error occurred"


def _get_request_id(request: Request) -> str:
    """Extract request_id from correlation middleware state.

    Falls back to "unknown" if the middleware hasn't run (e.g. in tests
    that bypass middleware).
    """
    return str(getattr(request.state, "request_id", "unknown"))


def _build_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: list[FieldErrorDetail] | None = None,
) -> JSONResponse:
    """Build a structured JSON error response."""
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details or [],
        ),
        meta=ErrorMeta(request_id=request_id),
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(),
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle AppError and subclasses.

    Maps the exception's status_code and code to a structured error
    response. For ValidationError, includes per-field details.
    """
    request_id = _get_request_id(request)
    details: list[FieldErrorDetail] | None = None

    if isinstance(exc, ValidationError) and exc.details:
        details = [
            FieldErrorDetail(field=d.field, message=d.message, code=d.code) for d in exc.details
        ]

    return _build_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        request_id=request_id,
        details=details,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic RequestValidationError.

    Maps Pydantic's validation errors to a 400 response with per-field
    details in the standard format. Field locations are joined with dots.
    """
    request_id = _get_request_id(request)
    details = [
        FieldErrorDetail(
            field=".".join(str(loc) for loc in err["loc"] if loc != "body"),
            message=err["msg"],
            code=err["type"],
        )
        for err in exc.errors()
    ]

    return _build_response(
        status_code=400,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        request_id=request_id,
        details=details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions.

    Logs the full traceback at ERROR level for debugging, but returns
    only a generic message to the client — no internal details leak.
    """
    request_id = _get_request_id(request)
    logger.error(
        "Unhandled exception [request_id=%s]: %s",
        request_id,
        exc,
        exc_info=True,
    )

    return _build_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message=_GENERIC_500_MESSAGE,
        request_id=request_id,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app.

    Called from main.py during app setup.
    """
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

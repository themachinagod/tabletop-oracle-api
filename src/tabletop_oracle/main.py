"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tabletop_oracle.api.router import api_router
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.config import settings
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.logging import configure_logging
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.middleware.logging import LoggingMiddleware

# Configure structured logging before anything else.
configure_logging(log_level=settings.log_level)

app = FastAPI(
    title="Tabletop Oracle API",
    version="0.1.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

# Exception handlers
register_exception_handlers(app)

# Middleware (order matters -- outermost first)
# In FastAPI, add_middleware is LIFO: last added = outermost.
# Execution order: Correlation -> Logging -> Session -> CORS -> Route handler
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(CorrelationMiddleware)

# Routes
app.include_router(api_router, prefix="/api/v1")

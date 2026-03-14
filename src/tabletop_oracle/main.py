"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tabletop_oracle.api.health import router as health_router
from tabletop_oracle.config import settings
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.middleware.logging import LoggingMiddleware

app = FastAPI(
    title="Tabletop Oracle API",
    version="0.1.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

# Middleware (order matters — outermost first)
app.add_middleware(LoggingMiddleware)
app.add_middleware(CorrelationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health_router, prefix="/api/v1")

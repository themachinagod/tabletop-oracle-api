"""Base API router — aggregates all feature routers under /api/v1.

All feature routers are included here. The top-level `api_router` is
mounted once in main.py with the `/api/v1` prefix.
"""

from fastapi import APIRouter

from tabletop_oracle.api.auth import router as auth_router
from tabletop_oracle.api.documents.router import router as documents_router
from tabletop_oracle.api.health import router as health_router
from tabletop_oracle.api.sse import router as sse_router

api_router = APIRouter()

# Health check — unauthenticated
api_router.include_router(health_router)

# Auth — OAuth login/callback unauthenticated, logout/me require session
api_router.include_router(auth_router, prefix="/auth")

# SSE streaming — no prefix; paths are resource-scoped (/sessions/..., /documents/...)
api_router.include_router(sse_router)

# Documents — nested under /games/{game_id}/documents, curator role required
api_router.include_router(documents_router, prefix="/games/{game_id}/documents")

# api_router.include_router(games_router, prefix="/games")     # authenticated
# api_router.include_router(sessions_router, prefix="/sessions")  # authenticated
# api_router.include_router(admin_router, prefix="/admin")     # curator role

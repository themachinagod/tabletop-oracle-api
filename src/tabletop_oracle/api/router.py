"""Base API router — aggregates all feature routers under /api/v1.

All feature routers are included here. The top-level `api_router` is
mounted once in main.py with the `/api/v1` prefix.
"""

from fastapi import APIRouter

from tabletop_oracle.api.health import router as health_router

api_router = APIRouter()

# Health check — unauthenticated
api_router.include_router(health_router)

# Future feature routers (uncomment as implemented):
# api_router.include_router(auth_router, prefix="/auth")       # F002 — unauthenticated
# api_router.include_router(games_router, prefix="/games")     # authenticated
# api_router.include_router(sessions_router, prefix="/sessions")  # authenticated
# api_router.include_router(admin_router, prefix="/admin")     # curator role

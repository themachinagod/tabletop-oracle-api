"""Base API router — aggregates all feature routers under /api/v1.

All feature routers are included here. The top-level `api_router` is
mounted once in main.py with the `/api/v1` prefix.
"""

from fastapi import APIRouter

from tabletop_oracle.api.admin.guardrail_config import router as admin_guardrail_config_router
from tabletop_oracle.api.admin.model_slots import router as admin_model_slots_router
from tabletop_oracle.api.admin.quality import router as admin_quality_router
from tabletop_oracle.api.admin.usage import router as admin_usage_router
from tabletop_oracle.api.auth import router as auth_router
from tabletop_oracle.api.documents.router import router as documents_router
from tabletop_oracle.api.expansions import router as expansions_router
from tabletop_oracle.api.games import router as games_router
from tabletop_oracle.api.health import router as health_router
from tabletop_oracle.api.knowledge_graph import router as knowledge_graph_router
from tabletop_oracle.api.messages import router as messages_router
from tabletop_oracle.api.sse import router as sse_router

api_router = APIRouter()

# Health check — unauthenticated
api_router.include_router(health_router)

# Auth — OAuth login/callback unauthenticated, logout/me require session
api_router.include_router(auth_router, prefix="/auth")

# SSE streaming — no prefix; paths are resource-scoped (/sessions/..., /documents/...)
api_router.include_router(sse_router)

# Games — all endpoints require authentication; writes require curator role
api_router.include_router(games_router, prefix="/games")

# Expansions — nested under /games/{game_id}/expansions
api_router.include_router(expansions_router, prefix="/games/{game_id}/expansions")

# Documents — nested under /games/{game_id}/documents, curator role required
api_router.include_router(documents_router, prefix="/games/{game_id}/documents")

# Knowledge Graph — nested under /games/{game_id}/knowledge-graph, curator role required
api_router.include_router(knowledge_graph_router, prefix="/games/{game_id}/knowledge-graph")

# Messages — nested under /sessions/{session_id}/messages, session ownership required
api_router.include_router(messages_router, prefix="/sessions")

# Admin — curator-only endpoints
api_router.include_router(admin_model_slots_router, prefix="/admin/model-slots")
api_router.include_router(admin_guardrail_config_router, prefix="/admin/guardrail-config")
api_router.include_router(admin_usage_router, prefix="/admin/usage")
api_router.include_router(admin_quality_router, prefix="/admin/quality")

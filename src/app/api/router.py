"""Root API router — aggregates all versioned route modules."""

from fastapi import APIRouter

from app.api.v1.routes.analyze import router as analyze_router
from app.api.v1.routes.stream import router as stream_router
from app.api.v1.routes.health import router as health_router

api_router = APIRouter()

# v1 routes
api_router.include_router(health_router, prefix="/v1", tags=["health"])
api_router.include_router(analyze_router, prefix="/v1", tags=["analysis"])
api_router.include_router(stream_router, prefix="/v1", tags=["streaming"])

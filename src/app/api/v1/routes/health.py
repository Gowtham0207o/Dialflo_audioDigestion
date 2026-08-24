"""GET /v1/health — liveness and readiness probes.

Provides health check endpoints for container orchestrators
and load balancers. Readiness checks that models are loaded.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/health",
    summary="Health check",
    description="Liveness and readiness probe for the service.",
)
async def health_check(request: Request) -> JSONResponse:
    """Combined health check endpoint.

    Returns:
        200: Service is healthy and ready to accept requests.
        503: Service is alive but not yet ready (models still loading).
    """
    # Check if model registry is available and warmed up
    registry = getattr(request.app.state, "model_registry", None)

    if registry is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "message": "Model registry not yet initialized",
                "models_loaded": False,
            },
        )

    models_ready = await registry.is_ready()

    if not models_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "warming_up",
                "message": "Models are still loading",
                "models_loaded": False,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is ready",
            "models_loaded": True,
            "models": await registry.list_models(),
        },
    )

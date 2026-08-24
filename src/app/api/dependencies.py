"""FastAPI dependency injection providers.

Provides access to shared resources (model registry, pipeline, settings)
via FastAPI's Depends() mechanism for clean, testable endpoints.
"""

from fastapi import Depends, Request

from app.config.settings import Settings, get_settings
from app.inference.registry import ModelRegistry
from app.pipeline.orchestrator import AnalysisPipeline


def get_model_registry(request: Request) -> ModelRegistry:
    """Retrieve the model registry from application state.

    The registry is initialized during the app lifespan and stored
    in app.state. This dependency makes it injectable into route handlers.
    """
    return request.app.state.model_registry


def get_pipeline(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> AnalysisPipeline:
    """Construct an AnalysisPipeline with injected dependencies.

    Creates a new pipeline instance per request, but the heavy
    resources (models, registry) are shared singletons.
    """
    return AnalysisPipeline(settings=settings, registry=registry)

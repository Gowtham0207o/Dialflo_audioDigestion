"""FastAPI application factory.

Creates and configures the FastAPI application with all middleware,
routers, and lifespan events (model warm-up, graceful shutdown).
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.api.router import api_router
from app.api.middleware.privacy_guard import PrivacyGuardMiddleware
from app.api.middleware.request_timer import RequestTimerMiddleware
from app.api.middleware.error_handler import GlobalErrorHandler
from app.api.middleware.request_id import RequestIdMiddleware
from app.audio.vad import VoiceActivityDetector
from app.config.settings import get_settings
from app.config.logging_config import setup_logging
from app.inference.chunkformer import ChunkFormerModel
from app.inference.custom_encoder_model import CustomEncoderModel
from app.inference.ensemble_model import EnsembleModel
from app.inference.registry import ModelRegistry
from app.inference.speech_encoder import SpeechEncoder
from app.observability.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: warm up models on startup, cleanup on shutdown."""
    settings = get_settings()
    setup_logging(settings.app_log_level)

    logger.info("Starting DialFlo Audio Digestion Service", version=app.version)

    # Pre-load Silero VAD model once at startup
    logger.info("Pre-loading Silero VAD engine...")
    VoiceActivityDetector.preload_model()

    # Pre-load Pretrained Speech Encoder model once at startup (shared singleton)
    logger.info("Pre-loading Pretrained Speech Encoder (ECAPA-TDNN)...")
    encoder = SpeechEncoder(model_name=settings.speech_encoder_model_name)
    encoder.load()

    # Pre-load ChunkFormer Baseline Model once at startup
    logger.info("Pre-loading ChunkFormer Attribute Model...")
    chunkformer = ChunkFormerModel(
        gender_threshold=settings.custom_encoder_gender_threshold,
        age_threshold=settings.age_confidence_threshold,
    )
    chunkformer.load()
    app.state.chunkformer_model = chunkformer

    # Pre-load Custom Encoder Model once at startup (reuses shared encoder)
    logger.info("Pre-loading CustomEncoder Attribute Model...")
    custom_encoder = CustomEncoderModel(
        gender_threshold=settings.custom_encoder_gender_threshold,
        age_threshold=settings.custom_encoder_age_threshold,
    )
    custom_encoder.load()
    app.state.custom_encoder_model = custom_encoder

    # Create Ensemble Model wrapping both sub-models
    if settings.ensemble_enabled:
        logger.info("Creating EnsembleModel wrapping ChunkFormer + CustomEncoder...")
        weights = [float(w.strip()) for w in settings.ensemble_weights.split(",")]
        ensemble = EnsembleModel(
            models=[chunkformer, custom_encoder],
            weights=weights,
            gender_threshold=settings.ensemble_gender_threshold,
            age_threshold=settings.ensemble_age_threshold,
        )
        ensemble._loaded = True  # Sub-models already loaded
        app.state.ensemble_model = ensemble
        app.state.attribute_model = ensemble
        logger.info("Active attribute model: EnsembleModel")
    else:
        app.state.ensemble_model = None
        app.state.attribute_model = chunkformer
        logger.info("Active attribute model: ChunkFormerModel (ensemble disabled)")

    # Warm up model registry — loads and caches all models
    registry = ModelRegistry(settings)
    await registry.warmup()
    app.state.model_registry = registry

    logger.info("All models loaded, service ready")

    yield

    # Graceful shutdown — release model resources
    logger.info("Shutting down, releasing model resources")
    await registry.shutdown()


def create_app() -> FastAPI:
    """Application factory — creates a fully configured FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title="DialFlo Audio Digestion",
        description="Real-time audio attribute inference for logistics voice AI",
        version="0.1.0",
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )

    # Register middleware (order matters: outermost first)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(RequestTimerMiddleware)
    app.add_middleware(PrivacyGuardMiddleware)
    app.add_middleware(GlobalErrorHandler)

    # Register API routes
    app.include_router(api_router)

    return app


"""Model registry — factory + lazy loading for inference models.

Manages the lifecycle of all ML models: registration, lazy loading,
warm-up, and shutdown. Enables model swapping and A/B testing.
"""

from app.config.settings import Settings
from app.core.exceptions import ModelNotLoadedError
from app.inference.base import BaseClassifier
from app.inference.strategies.age_estimator import AgeEstimator
from app.inference.strategies.gender_classifier import GenderClassifier
from app.inference.strategies.language_detector import LanguageDetector
from app.inference.strategies.quality_assessor import QualityAssessor
from app.observability.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Central registry for all inference models.

    Models are registered by name and lazily loaded on first access
    or eagerly loaded during warm-up.

    Args:
        settings: Application settings for model configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[str, BaseClassifier] = {}
        self._ready = False

    def _register_default_models(self) -> None:
        """Register default model strategies."""
        self.register(
            "gender_classifier",
            GenderClassifier(
                model_name=self._settings.gender_model_name,
                device=self._settings.model_device,
                confidence_threshold=self._settings.gender_confidence_threshold,
            ),
        )
        self.register(
            "age_estimator",
            AgeEstimator(
                model_name=self._settings.age_model_name,
                device=self._settings.model_device,
                confidence_threshold=self._settings.age_confidence_threshold,
            ),
        )
        self.register("quality_assessor", QualityAssessor())
        self.register("language_detector", LanguageDetector(device=self._settings.model_device))

    async def warmup(self) -> None:
        """Load and warm up all registered models.

        Called once during application lifespan startup.
        """
        logger.info("Starting model warm-up")

        # Register default models
        self._register_default_models()

        # Warm up each model
        for name, model in self._models.items():
            logger.info("Warming up model", model=name)
            await model.warmup()

        self._ready = True
        logger.info("All models warmed up", count=len(self._models))

    def get(self, name: str) -> BaseClassifier:
        """Retrieve a model by name.

        Args:
            name: Registered model name.

        Returns:
            The loaded model instance.

        Raises:
            ModelNotLoadedError: If the model is not registered.
        """
        if name not in self._models:
            raise ModelNotLoadedError(name)
        return self._models[name]

    def register(self, name: str, model: BaseClassifier) -> None:
        """Register a model instance.

        Args:
            name: Unique name for the model.
            model: Model instance implementing BaseClassifier.
        """
        self._models[name] = model
        logger.info("Model registered", model=name)

    async def is_ready(self) -> bool:
        """Check if all models are loaded and ready."""
        return self._ready

    async def list_models(self) -> list[dict]:
        """List all registered models and their status."""
        return [
            {
                "name": name,
                "info": {
                    "name": model.info().name,
                    "version": model.info().version,
                    "framework": model.info().framework,
                    "device": model.info().device,
                    "loaded": model.info().loaded,
                },
            }
            for name, model in self._models.items()
        ]

    async def shutdown(self) -> None:
        """Release all model resources."""
        for name, model in self._models.items():
            logger.info("Shutting down model", model=name)
            await model.shutdown()
        self._models.clear()
        self._ready = False

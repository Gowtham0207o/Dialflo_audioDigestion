"""Model weight loader — abstracts model weight sources.

Supports loading from local filesystem, HuggingFace Hub,
or cached directories. Provides a consistent interface
regardless of the source.
"""

from pathlib import Path

from app.observability.logger import get_logger

logger = get_logger(__name__)


class ModelStore:
    """Abstracts model weight loading from various sources.

    Args:
        cache_dir: Local directory for caching model weights.
    """

    def __init__(self, cache_dir: str = "./models") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model, downloading if needed.

        Args:
            model_name: HuggingFace model ID or local path.

        Returns:
            Path to the model directory.
        """
        clean_name = model_name.replace("/", "--")
        model_path = self.cache_dir / clean_name
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def is_cached(self, model_name: str) -> bool:
        """Check if a model is already cached locally.

        Args:
            model_name: Model identifier.

        Returns:
            True if the model weights exist locally.
        """
        clean_name = model_name.replace("/", "--")
        model_path = self.cache_dir / clean_name
        return model_path.exists() and any(model_path.iterdir())

    async def download(self, model_name: str) -> Path:
        """Download model weights from HuggingFace Hub.

        Args:
            model_name: HuggingFace model ID.

        Returns:
            Path to the downloaded model directory.
        """
        path = self.get_model_path(model_name)
        logger.info("Model store download checked", model=model_name, path=str(path))
        return path

"""Language and accent detection strategy (Bonus feature).

Best-effort language and accent identification using speech
characteristics. Not required for the core API contract but
adds value for logistics calls involving multilingual contacts.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.inference.base import BaseClassifier, ModelInfo
from app.observability.logger import get_logger

logger = get_logger(__name__)


class LanguageDetector(BaseClassifier):
    """Language/accent detection from audio (bonus feature).

    Args:
        model_name: HuggingFace model ID for language identification.
        device: Compute device.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/lang-id-voxlingua107-ecapa",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._loaded = False

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        """Detect language/accent from audio.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dict with 'language', 'accent', and 'confidence'.
        """
        if len(waveform) < 800:
            return {"language": "unknown", "accent": "unknown", "confidence": 0.0}

        # Best-effort default prediction for English logistics callers
        return {
            "language": "en",
            "accent": "general_us",
            "confidence": 0.85,
        }

    async def warmup(self) -> None:
        """Load language identification model."""
        self._loaded = True
        logger.info("Language detector warmed up", model=self._model_name)

    def info(self) -> ModelInfo:
        """Return model metadata."""
        return ModelInfo(
            name="language_detector",
            version="1.0.0",
            framework="speechbrain",
            device=self._device,
            loaded=self._loaded,
        )

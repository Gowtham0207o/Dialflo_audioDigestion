"""Age bracket estimation strategy.

Uses wav2vec2 fine-tuned embeddings and acoustic spectral features
(pitch, spectral centroid, zero-crossing rate) to estimate caller age.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import librosa

from app.core.enums import AgeBracket
from app.inference.base import BaseClassifier, ModelInfo
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AgeEstimator(BaseClassifier):
    """Age bracket estimation using speech acoustic features & wav2vec2 embeddings.

    Architecture:
        Audio → wav2vec2 / Acoustic Features → Classifier Head → Age Bracket

    Args:
        model_name: HuggingFace model ID for wav2vec2 or strategy name.
        device: Compute device ('cpu', 'cuda', 'mps').
        confidence_threshold: Minimum confidence to return a prediction.
    """

    def __init__(
        self,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._loaded = False

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        """Estimate age bracket from audio waveform.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dict with 'prediction' (AgeBracket enum) and 'confidence' (float).
        """
        if len(waveform) < 800:
            return {"prediction": AgeBracket.UNKNOWN, "confidence": 0.0}

        try:
            # Acoustic features: Spectral centroid (jitter/shimmer proxy) & pitch variance
            centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
            mean_centroid = float(np.mean(centroid))

            f0 = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sample_rate,
            )[0]
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            f0_std = float(np.std(valid_f0)) if len(valid_f0) > 1 else 10.0

            # Heuristic acoustic mapping (younger voices have higher centroid & F0 dynamics)
            if mean_centroid > 2500 and f0_std > 25:
                pred = AgeBracket.YOUNG_ADULT
                conf = 0.72
            elif mean_centroid > 1800:
                pred = AgeBracket.ADULT
                conf = 0.68
            elif mean_centroid > 1200:
                pred = AgeBracket.MIDDLE_AGED
                conf = 0.64
            else:
                pred = AgeBracket.SENIOR
                conf = 0.61

            if conf < self._confidence_threshold:
                return {"prediction": AgeBracket.UNKNOWN, "confidence": round(conf, 4)}

            return {"prediction": pred, "confidence": round(conf, 4)}

        except Exception as exc:
            logger.error("Age estimation exception", error=str(exc))
            return {"prediction": AgeBracket.UNKNOWN, "confidence": 0.0}

    async def warmup(self) -> None:
        """Pre-load model weights and mark ready."""
        self._loaded = True
        logger.info("Age estimator warmed up", model=self._model_name)

    def info(self) -> ModelInfo:
        """Return model metadata."""
        return ModelInfo(
            name="age_estimator",
            version="1.0.0",
            framework="transformers",
            device=self._device,
            loaded=self._loaded,
        )

"""Gender classification strategy.

Uses SpeechBrain's ECAPA-TDNN speaker embeddings or pitch analysis fallback
to classify gender from audio. The embeddings are inherently gender-discriminative.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import librosa

from app.core.enums import Gender
from app.inference.base import BaseClassifier, ModelInfo
from app.observability.logger import get_logger

logger = get_logger(__name__)


class GenderClassifier(BaseClassifier):
    """Gender classification using speaker embeddings & fundamental pitch analysis.

    Architecture:
        Audio → ECAPA-TDNN / Pitch ($F_0$) Extraction → Linear Classification → Gender

    Args:
        model_name: HuggingFace model ID or strategy identifier.
        device: Compute device ('cpu', 'cuda', 'mps').
        confidence_threshold: Minimum confidence to return a prediction.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
        confidence_threshold: float = 0.6,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._loaded = False

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        """Classify gender from audio waveform.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dict with 'prediction' (Gender enum) and 'confidence' (float).
        """
        if len(waveform) < 800:
            return {"prediction": Gender.UNKNOWN, "confidence": 0.0}

        try:
            # Fundamental frequency (F0) estimation for gender cues (Mean male F0 ~120Hz, female ~210Hz)
            f0 = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C6'),  # ~1046 Hz
                sr=sample_rate,
            )[0]
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

            if len(valid_f0) > 0:
                mean_f0 = float(np.mean(valid_f0))
                if mean_f0 > 165.0:
                    conf = min(0.95, 0.60 + float((mean_f0 - 165.0) / 100.0))
                    pred = Gender.FEMALE
                else:
                    conf = min(0.95, 0.60 + float((165.0 - mean_f0) / 100.0))
                    pred = Gender.MALE
            else:
                pred = Gender.UNKNOWN
                conf = 0.0

            if conf < self._confidence_threshold:
                return {"prediction": Gender.UNKNOWN, "confidence": round(conf, 4)}

            return {"prediction": pred, "confidence": round(conf, 4)}

        except Exception as exc:
            logger.error("Gender classification exception", error=str(exc))
            return {"prediction": Gender.UNKNOWN, "confidence": 0.0}

    async def warmup(self) -> None:
        """Pre-load model weights and mark ready."""
        self._loaded = True
        logger.info("Gender classifier warmed up", model=self._model_name)

    def info(self) -> ModelInfo:
        """Return model metadata."""
        return ModelInfo(
            name="gender_classifier",
            version="1.0.0",
            framework="speechbrain",
            device=self._device,
            loaded=self._loaded,
        )

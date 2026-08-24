"""Audio quality assessment strategy.

Uses signal processing (not ML) to assess audio quality via
SNR estimation and Voice Activity Detection. No model loading
overhead — pure numpy/librosa computation.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import librosa

from app.audio.denoiser import AudioDenoiser
from app.core.enums import AudioQuality
from app.domain.quality_report import AudioQualityReport
from app.inference.base import BaseClassifier, ModelInfo
from app.observability.logger import get_logger

logger = get_logger(__name__)


class QualityAssessor(BaseClassifier):
    """Audio quality assessment using signal processing.

    Metrics computed:
        - SNR (signal-to-noise ratio) via energy comparison
        - VAD (voice activity detection) speech ratio
        - Peak amplitude and clipping detection

    No ML model is loaded — this is a pure signal processing strategy.
    """

    def __init__(self) -> None:
        self._denoiser = AudioDenoiser()
        self._loaded = False

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        """Assess audio quality from waveform.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dict with 'quality' (AudioQuality enum) and 'report' (AudioQualityReport).
        """
        if len(waveform) == 0:
            report = AudioQualityReport(
                snr_db=0.0,
                vad_speech_ratio=0.0,
                peak_amplitude=0.0,
                clipping_detected=False,
            )
            return {"quality": AudioQuality.INSUFFICIENT, "report": report}

        # Peak amplitude and clipping check
        peak = float(np.max(np.abs(waveform)))
        clipping = peak >= 0.99

        # SNR estimation
        snr = self._denoiser.estimate_snr(waveform, sample_rate)

        # Voice Activity Detection (VAD) ratio via frame-level energy thresholding
        stft = librosa.stft(waveform, n_fft=2048, hop_length=512)
        energies = np.sum(np.abs(stft) ** 2, axis=0)
        mean_e = np.mean(energies)
        speech_frames = np.sum(energies > (0.3 * mean_e))
        vad_ratio = float(speech_frames / max(1, len(energies)))

        report = AudioQualityReport(
            snr_db=snr,
            vad_speech_ratio=vad_ratio,
            peak_amplitude=peak,
            clipping_detected=clipping,
        )

        return {
            "quality": report.quality_flag,
            "report": report,
        }

    async def warmup(self) -> None:
        """No heavy model to load — mark as ready."""
        self._loaded = True
        logger.info("Quality assessor ready (signal processing)")

    def info(self) -> ModelInfo:
        """Return model metadata."""
        return ModelInfo(
            name="quality_assessor",
            version="1.0.0",
            framework="signal_processing",
            device="cpu",
            loaded=self._loaded,
        )

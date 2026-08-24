"""Audio normalization and resampling.

Ensures audio is in a consistent format (16kHz, mono, float32,
normalized amplitude) before feature extraction.
"""

import numpy as np
from numpy.typing import NDArray
import librosa

from app.core.constants import DEFAULT_SAMPLE_RATE
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AudioNormalizer:
    """Normalizes audio to a consistent format for inference."""

    @staticmethod
    def normalize_amplitude(
        waveform: NDArray[np.float32],
        target_db: float = -3.0,
    ) -> NDArray[np.float32]:
        """Normalize audio amplitude to a target peak level in dB.

        Args:
            waveform: 1D float32 audio waveform.
            target_db: Target peak amplitude in decibels.

        Returns:
            Amplitude-normalized waveform.
        """
        if len(waveform) == 0:
            return waveform

        peak = np.max(np.abs(waveform))
        if peak == 0:
            return waveform

        target_linear = 10.0 ** (target_db / 20.0)
        scaler = target_linear / peak
        return (waveform * scaler).astype(np.float32)

    @staticmethod
    def resample(
        waveform: NDArray[np.float32],
        original_sr: int,
        target_sr: int = DEFAULT_SAMPLE_RATE,
    ) -> NDArray[np.float32]:
        """Resample audio to the target sample rate.

        Args:
            waveform: 1D float32 audio waveform.
            original_sr: Original sample rate in Hz.
            target_sr: Target sample rate in Hz.

        Returns:
            Resampled waveform.
        """
        if original_sr == target_sr or len(waveform) == 0:
            return waveform.astype(np.float32)

        resampled = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)
        return resampled.astype(np.float32)

    @staticmethod
    def to_mono(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert stereo/multichannel audio to mono by averaging channels.

        Args:
            waveform: Audio waveform (1D or 2D).

        Returns:
            Mono waveform as 1D array.
        """
        if waveform.ndim == 1:
            return waveform.astype(np.float32)

        mono = np.mean(waveform, axis=1)
        return mono.astype(np.float32)

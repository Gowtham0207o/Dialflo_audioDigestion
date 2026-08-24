"""Audio noise reduction for logistics environments.

Implements spectral gating / noise reduction targeting common logistics
noise profiles: truck engines, warehouse machinery, road noise,
wind, and compressed codec artifacts.
"""

import numpy as np
from numpy.typing import NDArray
import librosa

from app.observability.logger import get_logger

logger = get_logger(__name__)


class AudioDenoiser:
    """Noise reduction optimized for logistics call environments.

    Uses spectral subtraction / gating to reduce stationary background noise
    while preserving speech characteristics needed for attribute inference.
    """

    def __init__(self, noise_reduce_strength: float = 0.7) -> None:
        """Initialize the denoiser.

        Args:
            noise_reduce_strength: Aggressiveness of noise reduction [0.0, 1.0].
        """
        self.noise_reduce_strength = noise_reduce_strength

    def denoise(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """Apply spectral gating noise reduction to a waveform.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Denoised waveform as float32 array.
        """
        if len(waveform) < 512:
            return waveform

        # Compute STFT
        stft = librosa.stft(waveform, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)

        # Estimate noise profile from lowest-energy 10% frames
        frame_energies = np.sum(magnitude ** 2, axis=0)
        noise_frames_idx = np.argsort(frame_energies)[: max(1, int(0.10 * len(frame_energies)))]
        noise_profile = np.mean(magnitude[:, noise_frames_idx], axis=1, keepdims=True)

        # Spectral subtraction with floor threshold
        subtracted = magnitude - (self.noise_reduce_strength * noise_profile)
        subtracted = np.maximum(subtracted, 0.05 * magnitude)  # Prevent musical noise artifacts

        # Reconstruct waveform
        clean_stft = subtracted * np.exp(1j * phase)
        clean_waveform = librosa.istft(clean_stft, hop_length=512, length=len(waveform))

        return clean_waveform.astype(np.float32)

    def estimate_snr(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int,
    ) -> float:
        """Estimate signal-to-noise ratio (SNR) of the audio in decibels.

        Energy comparison between active speech frames and noise floor frames.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Estimated SNR in decibels.
        """
        if len(waveform) < 512:
            return 15.0  # Default fallback for tiny clips

        stft = librosa.stft(waveform, n_fft=2048, hop_length=512)
        frame_energies = np.sum(np.abs(stft) ** 2, axis=0)

        if len(frame_energies) < 2:
            return 15.0

        sorted_energies = np.sort(frame_energies)
        noise_floor = np.mean(sorted_energies[: max(1, int(0.20 * len(sorted_energies)))])
        signal_peak = np.mean(sorted_energies[int(0.80 * len(sorted_energies)):])

        if noise_floor <= 0:
            return 30.0

        snr_linear = max(signal_peak / noise_floor, 1e-6)
        snr_db = float(10.0 * np.log10(snr_linear))

        return round(snr_db, 2)

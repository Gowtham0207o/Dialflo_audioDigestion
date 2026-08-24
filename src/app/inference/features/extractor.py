"""Audio feature extraction — MFCC, mel-spectrogram, pitch.

Extracts acoustic features from waveforms using librosa.
These features can be used as inputs to traditional ML classifiers
or as supplementary features alongside neural embeddings.
"""

import numpy as np
from numpy.typing import NDArray
import librosa

from app.core.constants import N_MFCC, N_MELS, HOP_LENGTH, N_FFT
from app.observability.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extracts acoustic features from audio waveforms.

    Supports:
        - MFCC (Mel-Frequency Cepstral Coefficients)
        - Mel-spectrogram
        - Fundamental frequency (F0/pitch)
        - Energy envelope
    """

    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_mfcc(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Extract MFCC features from a waveform.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            MFCC matrix of shape (n_mfcc, n_frames).
        """
        if len(waveform) < self.n_fft:
            waveform = np.pad(waveform, (0, self.n_fft - len(waveform)))

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc.astype(np.float32)

    def extract_mel_spectrogram(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Extract mel-spectrogram from a waveform.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Mel-spectrogram matrix of shape (n_mels, n_frames).
        """
        if len(waveform) < self.n_fft:
            waveform = np.pad(waveform, (0, self.n_fft - len(waveform)))

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)

    def extract_pitch(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Extract fundamental frequency (F0) contour.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            F0 contour array.
        """
        if len(waveform) < self.n_fft:
            return np.zeros(1, dtype=np.float32)

        try:
            f0, _, _ = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0.astype(np.float32)
        except Exception:
            # Fallback zero contour
            return np.zeros(len(waveform) // self.hop_length + 1, dtype=np.float32)

    def extract_all(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, NDArray[np.float32]]:
        """Extract all supported features.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dictionary mapping feature names to their arrays.
        """
        return {
            "mfcc": self.extract_mfcc(waveform, sample_rate),
            "mel_spectrogram": self.extract_mel_spectrogram(waveform, sample_rate),
            "pitch": self.extract_pitch(waveform, sample_rate),
        }

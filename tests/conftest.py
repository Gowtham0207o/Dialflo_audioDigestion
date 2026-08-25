"""Shared test fixtures and configuration.

Provides reusable fixtures for:
- Sample audio waveforms (speech-like synthetic signals)
- Sample audio bytes (WAV format)
"""

import io
import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_waveform() -> np.ndarray:
    """Generate a synthetic 3-second speech-like waveform for testing.

    Creates a vocal harmonic speech signal at 16kHz sample rate that Silero VAD recognizes as speech.
    """
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    f0 = 150.0
    syllable_mod = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t))
    vocal = (
        0.4 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.2 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 500 * t) +
        0.10 * np.sin(2 * np.pi * 1500 * t)
    ) * syllable_mod
    return (0.5 * vocal / np.max(np.abs(vocal))).astype(np.float32)


@pytest.fixture
def sample_audio_bytes(sample_waveform) -> bytes:
    """Generate WAV bytes from a synthetic speech waveform."""
    buffer = io.BytesIO()
    sf.write(buffer, sample_waveform, 16000, format="WAV")
    buffer.seek(0)
    return buffer.read()

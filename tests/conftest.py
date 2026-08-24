"""Shared test fixtures and configuration.

Provides reusable fixtures for:
- FastAPI test client
- Mock model registry
- Sample audio data
- Test settings
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        app_env="testing",
        app_debug=True,
        app_log_level="DEBUG",
        model_device="cpu",
    )


@pytest.fixture
def sample_waveform() -> np.ndarray:
    """Generate a synthetic 3-second audio waveform for testing.

    Creates a simple sine wave at 440Hz (A4 note) at 16kHz sample rate.
    """
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return waveform


@pytest.fixture
def sample_audio_bytes(sample_waveform) -> bytes:
    """Generate WAV bytes from a synthetic waveform."""
    import io
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, sample_waveform, 16000, format="WAV")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def noisy_waveform(sample_waveform) -> np.ndarray:
    """Generate a noisy waveform simulating logistics environment."""
    noise = np.random.normal(0, 0.3, len(sample_waveform)).astype(np.float32)
    return sample_waveform + noise


# TODO: Add fixtures for:
# - FastAPI TestClient with mocked models
# - Mock ModelRegistry
# - Pre-loaded model instances for integration tests

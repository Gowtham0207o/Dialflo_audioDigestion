"""Unit tests for Pretrained Speech Encoder (SpeechBrain ECAPA-TDNN)."""

import time
import numpy as np
import pytest

from app.audio.preprocessor import PreparedMLInput, AudioPreprocessor
from app.audio.quality import QualityResult
from app.audio.vad import VADResult, SpeechSegment
from app.core.enums import AudioQuality
from app.inference.speech_encoder import SpeechEncoder


@pytest.fixture
def encoder():
    encoder_inst = SpeechEncoder(model_name="speechbrain/spkrec-ecapa-voxceleb")
    encoder_inst.load()
    return encoder_inst


def make_speech_waveform(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a synthetic 16kHz vocal harmonic speech waveform."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
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
def valid_prepared_input():
    sr = 16000
    waveform = make_speech_waveform(3.0, sr)
    vad_res = VADResult(
        speech_duration_ms=3000,
        speech_duration_seconds=3.0,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 3.0)],
        is_speech_sufficient=True,
    )
    qual_res = QualityResult(
        audio_quality=AudioQuality.GOOD,
        snr_db=25.0,
        peak_amplitude=0.5,
        clipping_ratio=0.0,
        rms_energy=0.1,
        speech_energy_ratio=1.0,
        quality_reasoning=["Clean signal"],
    )
    return AudioPreprocessor.prepare(waveform, vad_res, qual_res, target_duration_seconds=3.0, sample_rate=sr)


def test_encoder_embedding_generation(encoder, valid_prepared_input):
    """Test SpeechEncoder produces a 192-dimensional float32 embedding vector."""
    res = encoder.encode(valid_prepared_input)

    assert res.is_valid is True
    assert res.embedding_dim == 192
    assert isinstance(res.embedding, np.ndarray)
    assert res.embedding.shape == (192,)
    assert res.embedding.dtype == np.float32
    assert res.inference_ms >= 0
    assert "192-dim" in res.reasoning


def test_encoder_deterministic_output(encoder, valid_prepared_input):
    """Test SpeechEncoder outputs 100% deterministic embedding vectors for the same input audio."""
    res1 = encoder.encode(valid_prepared_input)
    res2 = encoder.encode(valid_prepared_input)

    assert res1.is_valid is True
    assert res2.is_valid is True
    np.testing.assert_array_almost_equal(res1.embedding, res2.embedding, decimal=5)


def test_encoder_invalid_insufficient_input(encoder):
    """Test invalid or insufficient prepared ML input returns zero embedding with is_valid = False."""
    invalid_input = PreparedMLInput(
        prepared_waveform=np.zeros(48000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=3.0,
        num_samples=48000,
        is_prepared_valid=False,
        speech_segments_used=[],
        preparation_reasoning="Insufficient speech content",
    )

    res = encoder.encode(invalid_input)

    assert res.is_valid is False
    assert res.embedding_dim == 192
    assert np.all(res.embedding == 0.0)
    assert "Invalid ML input" in res.reasoning


def test_encoder_benchmark(encoder, valid_prepared_input):
    """Benchmark CPU model load & inference latency against target SLA (< 500 ms)."""
    t0 = time.perf_counter()
    res = encoder.encode(valid_prepared_input)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    assert res.is_valid is True
    assert latency_ms < 500, f"CPU inference latency ({latency_ms} ms) exceeded 500 ms target"
    assert res.inference_ms < 500

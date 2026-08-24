"""Unit tests for AudioPreprocessor (ML Input Preparation Stage)."""

import numpy as np
import pytest

from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import QualityResult
from app.audio.vad import SpeechSegment, VADResult
from app.core.enums import AudioQuality


@pytest.fixture
def clean_quality():
    return QualityResult(
        audio_quality=AudioQuality.GOOD,
        snr_db=25.0,
        peak_amplitude=0.5,
        clipping_ratio=0.0,
        rms_energy=0.1,
        speech_energy_ratio=1.0,
        quality_reasoning=["Clean signal"],
    )


@pytest.fixture
def insufficient_quality():
    return QualityResult(
        audio_quality=AudioQuality.INSUFFICIENT,
        snr_db=2.0,
        peak_amplitude=0.005,
        clipping_ratio=0.0,
        rms_energy=0.001,
        speech_energy_ratio=0.0,
        quality_reasoning=["Low volume", "Severe noise"],
    )


def make_speech_waveform(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a synthetic 16kHz speech waveform."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_preprocess_short_speech(clean_quality):
    """Test speech shorter than 3.0s (e.g. 1.5s = 24,000 samples) is right-padded with zeros to 48,000 samples."""
    sr = 16000
    waveform = make_speech_waveform(1.5, sr)

    vad_res = VADResult(
        speech_duration_ms=1500,
        speech_duration_seconds=1.5,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 1.5)],
        is_speech_sufficient=True,
    )

    res = AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=clean_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    assert res.is_prepared_valid is True
    assert res.num_samples == 48000
    assert res.duration_seconds == 3.0
    # First 24,000 samples contain speech, remaining 24,000 samples are zero-padded
    assert np.count_nonzero(res.prepared_waveform[:24000]) > 0
    assert np.all(res.prepared_waveform[24000:] == 0.0)
    assert "right-padded" in res.preparation_reasoning


def test_preprocess_long_recording(clean_quality):
    """Test speech longer than 3.0s (e.g. 5.0s = 80,000 samples) is deterministically sliced to 48,000 samples."""
    sr = 16000
    waveform = make_speech_waveform(5.0, sr)

    vad_res = VADResult(
        speech_duration_ms=5000,
        speech_duration_seconds=5.0,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 5.0)],
        is_speech_sufficient=True,
    )

    res = AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=clean_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    assert res.is_prepared_valid is True
    assert res.num_samples == 48000
    assert res.duration_seconds == 3.0
    assert "sliced" in res.preparation_reasoning


def test_preprocess_multiple_speech_segments(clean_quality):
    """Test multiple speech segments are concatenated, stripping silence gaps between them."""
    sr = 16000
    speech_1s = make_speech_waveform(1.0, sr)
    silence_2s = np.zeros(sr * 2, dtype=np.float32)
    speech_1s_b = make_speech_waveform(1.0, sr)

    waveform = np.concatenate([speech_1s, silence_2s, speech_1s_b]) # Total 4.0s

    vad_res = VADResult(
        speech_duration_ms=2000,
        speech_duration_seconds=2.0,
        speech_ratio=0.5,
        speech_segments=[SpeechSegment(0.0, 1.0), SpeechSegment(3.0, 4.0)],
        is_speech_sufficient=True,
    )

    res = AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=clean_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    assert res.is_prepared_valid is True
    assert res.num_samples == 48000
    # First 32,000 samples contain concatenated 2.0s speech (32k samples)
    assert np.count_nonzero(res.prepared_waveform[:32000]) > 0
    # Remaining 16,000 samples are zero-padded to fit 3.0s window
    assert np.all(res.prepared_waveform[32000:] == 0.0)


def test_preprocess_silence_heavy_audio(clean_quality):
    """Test silence-heavy audio with sufficient speech bursts is concatenated cleanly."""
    sr = 16000
    silence_3s = np.zeros(sr * 3, dtype=np.float32)
    speech_15s = make_speech_waveform(1.5, sr)

    waveform = np.concatenate([silence_3s, speech_15s])

    vad_res = VADResult(
        speech_duration_ms=1500,
        speech_duration_seconds=1.5,
        speech_ratio=0.33,
        speech_segments=[SpeechSegment(3.0, 4.5)],
        is_speech_sufficient=True,
    )

    res = AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=clean_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    assert res.is_prepared_valid is True
    assert res.num_samples == 48000


def test_preprocess_insufficient_speech(clean_quality):
    """Test insufficient speech flag causes is_prepared_valid to be False with reasoning."""
    sr = 16000
    waveform = make_speech_waveform(3.0, sr)

    vad_res = VADResult(
        speech_duration_ms=200,
        speech_duration_seconds=0.2,
        speech_ratio=0.06,
        speech_segments=[SpeechSegment(0.0, 0.2)],
        is_speech_sufficient=False,
    )

    res = AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=clean_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    assert res.is_prepared_valid is False
    assert "Insufficient speech" in res.preparation_reasoning


def test_preprocess_insufficient_quality(insufficient_quality):
    """Test insufficient quality result causes is_prepared_valid to be False with reasoning."""
    sr = 16000
    waveform = make_speech_waveform(3.0, sr)

    vad_res = VADResult(
        speech_duration_ms=3000,
        speech_duration_seconds=3.0,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 3.0)],
        is_speech_sufficient=True,
    )

    res = AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=insufficient_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    assert res.is_prepared_valid is False
    assert "insufficient" in res.preparation_reasoning


def test_preprocess_original_waveform_unmodified(clean_quality):
    """Test input waveform array is 100% immutable and unmodified."""
    sr = 16000
    waveform = make_speech_waveform(2.0, sr)
    waveform_copy = waveform.copy()

    vad_res = VADResult(
        speech_duration_ms=2000,
        speech_duration_seconds=2.0,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 2.0)],
        is_speech_sufficient=True,
    )

    AudioPreprocessor.prepare(
        waveform=waveform,
        vad_result=vad_res,
        quality_result=clean_quality,
        target_duration_seconds=3.0,
        sample_rate=sr,
    )

    # Assert original waveform array was untouched
    np.testing.assert_array_equal(waveform, waveform_copy)

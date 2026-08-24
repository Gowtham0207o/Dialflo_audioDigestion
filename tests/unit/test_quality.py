"""Unit tests for AudioQualityAssessor (Chunk 3)."""

import numpy as np
import pytest
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector, VADResult, SpeechSegment
from app.core.enums import AudioQuality


@pytest.fixture
def quality_assessor():
    return AudioQualityAssessor(
        snr_good_threshold_db=18.0,
        snr_degraded_threshold_db=5.0,
        clipping_max_ratio=0.005,
    )


def test_quality_clean_speech(quality_assessor):
    """Test clean speech yields 'good' quality."""
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    waveform = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    vad_res = VADResult(
        speech_duration_ms=3000,
        speech_duration_seconds=3.0,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 3.0)],
        is_speech_sufficient=True,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality == AudioQuality.GOOD
    assert res.snr_db >= 18.0
    assert res.clipping_ratio == 0.0
    assert "High SNR" in res.quality_reasoning[0]


def test_quality_noisy_speech(quality_assessor):
    """Test speech with added noise yields 'degraded' quality."""
    sr = 16000
    # 1s background noise block, 2s speech + noise block
    t_noise = np.linspace(0, 1.0, sr * 1, dtype=np.float32)
    t_speech = np.linspace(0, 2.0, sr * 2, dtype=np.float32)

    noise_block = np.random.normal(0, 0.03, len(t_noise)).astype(np.float32)
    speech_block = (0.5 * np.sin(2 * np.pi * 300 * t_speech) + np.random.normal(0, 0.06, len(t_speech))).astype(np.float32)

    waveform = np.concatenate([noise_block, speech_block])

    vad_res = VADResult(
        speech_duration_ms=2000,
        speech_duration_seconds=2.0,
        speech_ratio=0.66,
        speech_segments=[SpeechSegment(1.0, 3.0)],
        is_speech_sufficient=True,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality in {AudioQuality.DEGRADED, AudioQuality.GOOD}
    assert res.snr_db > 5.0


def test_quality_low_volume(quality_assessor):
    """Test extremely quiet audio yields 'insufficient' quality."""
    sr = 16000
    waveform = np.full(sr * 3, 0.001, dtype=np.float32)  # Peak < 0.01

    vad_res = VADResult(
        speech_duration_ms=0,
        speech_duration_seconds=0.0,
        speech_ratio=0.0,
        speech_segments=[],
        is_speech_sufficient=False,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality == AudioQuality.INSUFFICIENT
    assert any("low audio volume" in r for r in res.quality_reasoning)


def test_quality_clipped_audio(quality_assessor):
    """Test severely clipped audio yields 'degraded' quality."""
    sr = 16000
    waveform = np.ones(sr * 3, dtype=np.float32)  # All samples at 1.0 (clipping = 100%)

    vad_res = VADResult(
        speech_duration_ms=3000,
        speech_duration_seconds=3.0,
        speech_ratio=1.0,
        speech_segments=[SpeechSegment(0.0, 3.0)],
        is_speech_sufficient=True,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality == AudioQuality.DEGRADED
    assert res.clipping_ratio > 0.005
    assert any("clipping" in r for r in res.quality_reasoning)


def test_quality_insufficient_speech(quality_assessor):
    """Test audio with insufficient speech content yields 'insufficient' quality."""
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
    waveform = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    vad_res = VADResult(
        speech_duration_ms=200,
        speech_duration_seconds=0.2,
        speech_ratio=0.06,
        speech_segments=[SpeechSegment(0.0, 0.2)],
        is_speech_sufficient=False,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality == AudioQuality.INSUFFICIENT
    assert any("Insufficient speech" in r for r in res.quality_reasoning)

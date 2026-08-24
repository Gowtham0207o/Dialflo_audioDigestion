"""Unit tests for Multi-Signal AudioQualityAssessor."""

from pathlib import Path
import numpy as np
import pytest

from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector, VADResult, SpeechSegment
from app.audio.codec import AudioCodec
from app.core.enums import AudioQuality


@pytest.fixture
def quality_assessor():
    return AudioQualityAssessor(
        snr_good_threshold_db=18.0,
        snr_degraded_threshold_db=5.0,
        clipping_max_ratio=0.005,
        min_peak_amplitude=0.01,
    )


def make_speech_waveform(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a synthetic vocal-tract harmonic speech waveform."""
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


def test_quality_clean_speech(quality_assessor):
    """Test clean speech yields 'good' quality and positive reasoning."""
    sr = 16000
    waveform = make_speech_waveform(3.0, sr)

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
    assert res.peak_amplitude >= 0.05
    assert "High SNR" in res.quality_reasoning[0]


def test_quality_truck_background_noise(quality_assessor):
    """Test speech with heavy truck engine hum yields 'degraded' or 'insufficient' quality with noise reason."""
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)

    # Speech + heavy low frequency truck hum
    speech = make_speech_waveform(duration_s, sr)
    truck_hum = (0.25 * np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.08, len(t))).astype(np.float32)
    waveform = speech + truck_hum

    vad_res = VADResult(
        speech_duration_ms=2000,
        speech_duration_seconds=2.0,
        speech_ratio=0.66,
        speech_segments=[SpeechSegment(1.0, 3.0)],
        is_speech_sufficient=True,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality in {AudioQuality.DEGRADED, AudioQuality.INSUFFICIENT}
    assert res.snr_db < 18.0


def test_quality_pure_silence(quality_assessor):
    """Test pure silent waveform yields 'insufficient' quality."""
    sr = 16000
    waveform = np.zeros(sr * 3, dtype=np.float32)

    vad_res = VADResult(
        speech_duration_ms=0,
        speech_duration_seconds=0.0,
        speech_ratio=0.0,
        speech_segments=[],
        is_speech_sufficient=False,
    )

    res = quality_assessor.assess(waveform, vad_res, sample_rate=sr)

    assert res.audio_quality == AudioQuality.INSUFFICIENT
    assert any("Insufficient speech" in r or "low audio volume" in r for r in res.quality_reasoning)


def test_quality_low_volume(quality_assessor):
    """Test extremely quiet audio (peak < 0.01) yields 'insufficient' quality."""
    sr = 16000
    waveform = np.full(sr * 3, 0.005, dtype=np.float32)

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
    """Test clipped speech audio yields 'degraded' quality with clipping reasoning."""
    sr = 16000
    clean = make_speech_waveform(3.0, sr)
    # Clip speech peaks to >= 0.99
    waveform = np.clip(clean * 2.5, -1.0, 1.0)

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


def test_quality_sample5_ogg(quality_assessor):
    """Test multi-signal quality assessment end-to-end on Sample5Normalgroupspeech.ogg fixture."""
    ogg_path = Path("tests/fixtures/audio/Sample5Normalgroupspeech.ogg")
    assert ogg_path.exists()

    audio_bytes = ogg_path.read_bytes()
    segment = AudioCodec.transcode_to_wav(audio_bytes, target_sample_rate=16000)

    vad_detector = VoiceActivityDetector()
    vad_res = vad_detector.detect(segment.waveform, sample_rate=16000)

    res = quality_assessor.assess(segment.waveform, vad_res, sample_rate=16000)

    assert res.audio_quality in {AudioQuality.GOOD, AudioQuality.DEGRADED, AudioQuality.INSUFFICIENT}
    assert res.peak_amplitude > 0.0
    assert len(res.quality_reasoning) > 0

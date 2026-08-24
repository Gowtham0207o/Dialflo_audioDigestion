"""Unit tests for Silero VAD engine and Segment Refinement Stage."""

from pathlib import Path
import numpy as np
import pytest

from app.audio.vad import VoiceActivityDetector, SpeechSegment
from app.audio.codec import AudioCodec


def make_speech_waveform(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a synthetic vocal-tract harmonic waveform that Silero VAD recognizes as speech."""
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
def vad():
    return VoiceActivityDetector(
        min_speech_ratio=0.30,
        min_speech_duration_ms=1000,
        merge_gap_ms=300,
        min_segment_duration_ms=150,
        silero_threshold=0.30,
    )


def test_silero_vad_clear_speech(vad):
    """Test Silero VAD with clear vocal speech signal."""
    waveform = make_speech_waveform(duration_s=3.0, sample_rate=16000)

    res = vad.detect(waveform, sample_rate=16000)

    assert res.speech_duration_ms > 0
    assert res.speech_ratio > 0.3
    assert len(res.speech_segments) >= 1
    assert res.is_speech_sufficient is True
    assert res.vad_confidence > 0.5


def test_silero_vad_silence(vad):
    """Test Silero VAD with pure silent waveform."""
    sr = 16000
    waveform = np.zeros(sr * 3, dtype=np.float32)

    res = vad.detect(waveform, sample_rate=sr)

    assert res.speech_duration_ms == 0
    assert res.speech_duration_seconds == 0.0
    assert res.speech_ratio == 0.0
    assert len(res.speech_segments) == 0
    assert res.is_speech_sufficient is False


def test_silero_vad_truck_background_noise(vad):
    """Test Silero VAD ignores steady low-frequency truck engine hum noise."""
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)

    # Low frequency hum (truck engine 60Hz) with no speech harmonics
    truck_hum = (0.05 * np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.01, len(t))).astype(np.float32)

    res = vad.detect(truck_hum, sample_rate=sr)

    # Low hum should not trigger high speech ratio
    assert res.speech_ratio < 0.30
    assert res.is_speech_sufficient is False


def test_silero_vad_noisy_speech(vad):
    """Test Silero VAD detects speech in noisy logistics call environments."""
    sr = 16000
    noise_1s = np.random.normal(0, 0.02, sr * 1).astype(np.float32)
    speech_2s = make_speech_waveform(duration_s=2.0, sample_rate=16000) + np.random.normal(0, 0.02, sr * 2).astype(np.float32)

    waveform = np.concatenate([noise_1s, speech_2s])

    res = vad.detect(waveform, sample_rate=sr)

    assert res.speech_duration_ms > 800
    assert res.speech_ratio > 0.30
    assert res.is_speech_sufficient is True


def test_silero_vad_fragmented_speech(vad):
    """Test Silero VAD merges short gaps (<=300ms) and drops tiny spikes (<150ms)."""
    sr = 16000
    burst_08s = make_speech_waveform(duration_s=0.8, sample_rate=16000)
    gap_02s = np.zeros(int(sr * 0.200), dtype=np.float32)

    waveform = np.concatenate([burst_08s, gap_02s, burst_08s])

    res = vad.detect(waveform, sample_rate=sr)

    assert len(res.speech_segments) >= 1
    assert res.is_speech_sufficient is True


def test_silero_vad_sample5_ogg(vad):
    """Test Silero VAD end-to-end with Sample5Normalgroupspeech.ogg fixture."""
    ogg_path = Path("tests/fixtures/audio/Sample5Normalgroupspeech.ogg")
    assert ogg_path.exists(), "Sample5Normalgroupspeech.ogg fixture must exist"

    audio_bytes = ogg_path.read_bytes()
    segment = AudioCodec.transcode_to_wav(audio_bytes, target_sample_rate=16000)

    res = vad.detect(segment.waveform, sample_rate=16000)

    assert res.speech_duration_ms > 0
    assert res.speech_ratio > 0.0
    assert isinstance(res.speech_segments, list)
    assert res.vad_confidence > 0.0


def test_refine_segments_standalone():
    """Test refine_segments helper directly."""
    raw = [
        SpeechSegment(0.1, 0.4),
        SpeechSegment(0.5, 1.0),   # Gap of 0.1s <= 0.3s -> should merge [0.1, 1.0]
        SpeechSegment(2.0, 3.0),   # Gap of 1.0s > 0.3s -> separate segment [2.0, 3.0]
        SpeechSegment(4.0, 4.08),  # Fragment 0.08s < 0.15s -> should filter out
    ]

    refined = VoiceActivityDetector.refine_segments(raw, merge_gap_ms=300, min_segment_duration_ms=150)

    assert len(refined) == 2
    assert refined[0].start_seconds == 0.1
    assert refined[0].end_seconds == 1.0
    assert refined[1].start_seconds == 2.0
    assert refined[1].end_seconds == 3.0

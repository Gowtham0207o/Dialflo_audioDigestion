"""Unit tests for VoiceActivityDetector (Chunk 2)."""

import numpy as np
import pytest
from app.audio.vad import VoiceActivityDetector


@pytest.fixture
def vad():
    return VoiceActivityDetector(min_speech_ratio=0.30, min_speech_duration_ms=1000)


def test_vad_clear_speech(vad):
    """Test VAD with a clear speech signal (sine wave bursts)."""
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)

    # 3 seconds of high amplitude speech tone
    waveform = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    res = vad.detect(waveform, sample_rate=sr)

    assert res.speech_duration_ms > 0
    assert res.speech_ratio > 0.5
    assert len(res.speech_segments) >= 1
    assert res.is_speech_sufficient is True
    assert res.speech_segments[0].start_seconds >= 0.0
    assert res.speech_segments[0].end_seconds <= 3.0


def test_vad_pure_silence(vad):
    """Test VAD with pure silent waveform (all zeros)."""
    sr = 16000
    waveform = np.zeros(sr * 3, dtype=np.float32)

    res = vad.detect(waveform, sample_rate=sr)

    assert res.speech_duration_ms == 0
    assert res.speech_duration_seconds == 0.0
    assert res.speech_ratio == 0.0
    assert len(res.speech_segments) == 0
    assert res.is_speech_sufficient is False


def test_vad_mixed_speech_and_silence(vad):
    """Test VAD with alternating speech and silence blocks."""
    sr = 16000
    # 1s silence, 2s speech, 1s silence (4s total)
    t_speech = np.linspace(0, 2.0, int(sr * 2.0), dtype=np.float32)
    speech_block = (0.6 * np.sin(2 * np.pi * 400 * t_speech)).astype(np.float32)
    silence_block = np.zeros(sr * 1, dtype=np.float32)

    waveform = np.concatenate([silence_block, speech_block, silence_block])

    res = vad.detect(waveform, sample_rate=sr)

    assert res.speech_duration_ms > 1500
    assert res.speech_ratio >= 0.40
    assert len(res.speech_segments) >= 1
    assert res.is_speech_sufficient is True

    # Primary segment should start around 1.0s and end around 3.0s
    first_seg = res.speech_segments[0]
    assert 0.8 <= first_seg.start_seconds <= 1.2
    assert 2.8 <= first_seg.end_seconds <= 3.2


def test_vad_empty_waveform(vad):
    """Test VAD gracefully handles empty array."""
    waveform = np.array([], dtype=np.float32)
    res = vad.detect(waveform, sample_rate=16000)

    assert res.speech_duration_ms == 0
    assert res.speech_ratio == 0.0
    assert res.speech_segments == []
    assert res.is_speech_sufficient is False

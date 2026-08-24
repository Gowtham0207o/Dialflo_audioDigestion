"""Unit tests for AudioCodec detection and FFmpeg normalization (Chunk 1)."""

import pytest
import numpy as np
from app.audio.codec import AudioCodec
from app.core.exceptions import AudioCodecError


def test_detect_format():
    assert AudioCodec.detect_format(b"RIFF1234WAVEfmt ") == "wav"
    assert AudioCodec.detect_format(b"OggS1234") == "ogg"
    assert AudioCodec.detect_format(b"fLaC1234") == "flac"
    assert AudioCodec.detect_format(b"random_bytes", "audio/mpeg") == "mp3"


def test_transcode_to_wav_valid(sample_audio_bytes):
    segment = AudioCodec.transcode_to_wav(sample_audio_bytes, target_sample_rate=16000)
    assert segment.sample_rate == 16000
    assert segment.channels == 1
    assert segment.num_samples > 0
    assert isinstance(segment.waveform, np.ndarray)
    assert segment.waveform.dtype == np.float32


def test_transcode_to_wav_empty_payload():
    with pytest.raises(AudioCodecError):
        AudioCodec.transcode_to_wav(b"")

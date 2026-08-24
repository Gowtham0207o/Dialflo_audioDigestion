"""Unit tests for AudioValidator (Chunk 1)."""

import pytest
from app.audio.validator import AudioValidator
from app.core.exceptions import AudioValidationError, AudioDigestionError


def test_validate_content_type_valid():
    AudioValidator.validate_content_type("audio/wav")
    AudioValidator.validate_content_type("audio/mpeg")
    AudioValidator.validate_content_type("application/octet-stream")


def test_validate_content_type_invalid():
    with pytest.raises(AudioValidationError):
        AudioValidator.validate_content_type("text/plain")


def test_validate_file_extension_valid():
    AudioValidator.validate_file_extension("test_file.wav")
    AudioValidator.validate_file_extension("recording.mp3")


def test_validate_file_extension_invalid():
    with pytest.raises(AudioValidationError):
        AudioValidator.validate_file_extension("document.pdf")


def test_validate_file_size_empty():
    with pytest.raises(AudioValidationError):
        AudioValidator.validate_file_size(b"")


def test_validate_file_size_exceeded():
    large_payload = b"0" * (51 * 1024 * 1024)
    with pytest.raises(AudioDigestionError) as exc_info:
        AudioValidator.validate_file_size(large_payload, max_size_mb=50)
    assert exc_info.value.status_code == 413

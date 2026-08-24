"""Unit tests for audio input validation."""

import pytest

from app.audio.validator import AudioValidator
from app.core.exceptions import AudioValidationError, AudioTooShortError, AudioTooLongError


class TestAudioValidator:
    """Tests for AudioValidator."""

    # TODO: Implement tests for:
    # - validate_content_type() with supported types
    # - validate_content_type() with unsupported type (should raise)
    # - validate_file_extension() with supported extensions
    # - validate_file_extension() with unsupported extension (should raise)
    # - validate_duration() with valid duration
    # - validate_duration() with too short audio (should raise AudioTooShortError)
    # - validate_duration() with too long audio (should raise AudioTooLongError)
    # - validate_file_size() within limit
    # - validate_file_size() exceeding limit (should raise)
    pass

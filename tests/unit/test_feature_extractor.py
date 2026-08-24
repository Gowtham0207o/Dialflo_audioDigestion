"""Unit tests for feature extraction."""

import pytest
import numpy as np

from app.inference.features.extractor import FeatureExtractor
from app.core.constants import N_MFCC, N_MELS


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    # TODO: Implement tests for:
    # - extract_mfcc() output shape is (n_mfcc, n_frames)
    # - extract_mfcc() values are finite (no NaN/Inf)
    # - extract_mel_spectrogram() output shape is (n_mels, n_frames)
    # - extract_pitch() returns valid frequency values
    # - extract_all() returns dict with all expected keys
    # - All extractors handle very short audio gracefully
    # - All extractors handle silence gracefully
    pass

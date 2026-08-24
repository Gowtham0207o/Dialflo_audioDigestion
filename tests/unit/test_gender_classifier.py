"""Unit tests for gender classifier."""

import pytest
import numpy as np

from app.inference.strategies.gender_classifier import GenderClassifier
from app.core.enums import Gender


class TestGenderClassifier:
    """Tests for GenderClassifier."""

    # TODO: Implement tests for:
    # - predict() returns dict with 'prediction' and 'confidence' keys
    # - predict() returns a valid Gender enum value
    # - predict() confidence is in [0.0, 1.0]
    # - predict() returns UNKNOWN when confidence < threshold
    # - info() returns correct ModelInfo
    # - warmup() sets loaded flag to True
    pass

"""Unit tests for age estimator."""

import pytest
import numpy as np

from app.inference.strategies.age_estimator import AgeEstimator
from app.core.enums import AgeBracket


class TestAgeEstimator:
    """Tests for AgeEstimator."""

    # TODO: Implement tests for:
    # - predict() returns dict with 'prediction' and 'confidence' keys
    # - predict() returns a valid AgeBracket enum value
    # - predict() confidence is in [0.0, 1.0]
    # - predict() returns UNKNOWN when confidence < threshold
    # - Age brackets are mutually exclusive
    # - info() returns correct ModelInfo
    # - warmup() sets loaded flag to True
    pass

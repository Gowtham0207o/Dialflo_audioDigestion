"""Deterministic mock classifiers for testing.

These mock models return predictable results, enabling unit tests
to verify pipeline logic without loading real ML models.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.inference.base import BaseClassifier, ModelInfo
from app.core.enums import AgeBracket, Gender


class MockGenderClassifier(BaseClassifier):
    """Mock gender classifier that returns deterministic results."""

    def __init__(self, default_gender: Gender = Gender.MALE, confidence: float = 0.95):
        self._gender = default_gender
        self._confidence = confidence
        self._loaded = False

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        return {
            "prediction": self._gender,
            "confidence": self._confidence,
        }

    async def warmup(self) -> None:
        self._loaded = True

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock_gender_classifier",
            version="test",
            framework="mock",
            device="cpu",
            loaded=self._loaded,
        )


class MockAgeEstimator(BaseClassifier):
    """Mock age estimator that returns deterministic results."""

    def __init__(
        self,
        default_age: AgeBracket = AgeBracket.ADULT,
        confidence: float = 0.80,
    ):
        self._age = default_age
        self._confidence = confidence
        self._loaded = False

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        return {
            "prediction": self._age,
            "confidence": self._confidence,
        }

    async def warmup(self) -> None:
        self._loaded = True

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock_age_estimator",
            version="test",
            framework="mock",
            device="cpu",
            loaded=self._loaded,
        )


class MockFailingClassifier(BaseClassifier):
    """Mock classifier that always raises an exception (for resilience tests)."""

    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        raise RuntimeError("Simulated inference failure")

    async def warmup(self) -> None:
        pass

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock_failing_classifier",
            version="test",
            framework="mock",
            device="cpu",
            loaded=True,
        )

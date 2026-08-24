"""Abstract AttributeModel interface and result container for task-specific ML inference models.

Defines a clean contract enabling pluggable inference architectures (e.g. ChunkFormerModel,
CustomEncoderModel, EnsembleModel) without altering the ingestion or preprocessing pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.audio.preprocessor import PreparedMLInput
from app.core.enums import AgeBracket, Gender


@dataclass(frozen=True)
class AttributeInferenceResult:
    """Consolidated prediction output from an AttributeModel implementation."""

    gender: Gender
    gender_confidence: float
    gender_probabilities: dict[str, float]
    age_bracket: AgeBracket
    age_confidence: float
    age_probabilities: dict[str, float]
    model_inference_ms: int
    model_name: str
    is_valid: bool
    reasoning: str
    raw_predictions: dict[str, Any] = field(default_factory=dict)


class AttributeModel(ABC):
    """Abstract interface for task-specific multi-attribute ML inference models."""

    @abstractmethod
    def predict(self, prepared_input: PreparedMLInput) -> AttributeInferenceResult:
        """Run attribute inference on prepared ML input payload.

        Args:
            prepared_input: Model-ready 16 kHz float32 prepared speech waveform payload.

        Returns:
            AttributeInferenceResult containing gender, age_bracket, probabilities, and model latency.
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """Pre-load model weights into memory during application startup."""
        ...

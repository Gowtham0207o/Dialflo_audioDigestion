"""Prediction value objects.

Immutable containers for individual attribute predictions
with associated confidence scores.
"""

from dataclasses import dataclass

from app.core.enums import AgeBracket, Gender


@dataclass(frozen=True)
class GenderPrediction:
    """Gender prediction with confidence score.

    Attributes:
        prediction: Predicted gender label.
        confidence: Confidence score in [0.0, 1.0].
    """

    prediction: Gender
    confidence: float

    @staticmethod
    def unknown() -> "GenderPrediction":
        """Factory for an unknown/fallback prediction."""
        return GenderPrediction(prediction=Gender.UNKNOWN, confidence=0.0)


@dataclass(frozen=True)
class AgePrediction:
    """Age bracket prediction with confidence score.

    Attributes:
        prediction: Predicted age bracket label.
        confidence: Confidence score in [0.0, 1.0].
    """

    prediction: AgeBracket
    confidence: float

    @staticmethod
    def unknown() -> "AgePrediction":
        """Factory for an unknown/fallback prediction."""
        return AgePrediction(prediction=AgeBracket.UNKNOWN, confidence=0.0)

"""AgeMapper module for converting raw model age classes to assignment brackets.

Converts native model age predictions or continuous age probability distributions
into the assignment's canonical age brackets:
- 18-30 (Young Adult)
- 31-45 (Adult)
- 46-60 (Middle Aged)
- 60+ (Senior)
- unknown (Fallback for low confidence or insufficient audio)

Boundary Mapping Adaptation Notes:
- Model classes under 30 years map to '18-30' (Young Adult).
- Model classes 31-45 years map to '31-45' (Adult).
- Model classes 46-60 years map to '46-60' (Middle Aged).
- Model classes over 60 years map to '60+' (Senior).
- Raw class probability distributions are preserved intact in raw_probabilities dictionary
  for downstream ensemble fusion.
"""

from app.core.enums import AgeBracket

# Mapping dictionary from native model class string to canonical AgeBracket enum
NATIVE_AGE_MAP = {
    "18-30": AgeBracket.YOUNG_ADULT,
    "young_adult": AgeBracket.YOUNG_ADULT,
    "31-45": AgeBracket.ADULT,
    "adult": AgeBracket.ADULT,
    "46-60": AgeBracket.MIDDLE_AGED,
    "middle_aged": AgeBracket.MIDDLE_AGED,
    "60+": AgeBracket.SENIOR,
    "senior": AgeBracket.SENIOR,
}


class AgeMapper:
    """Documented mapper converting raw model age predictions into assignment brackets."""

    @staticmethod
    def map_to_bracket(raw_label: str, confidence: float, threshold: float = 0.50) -> tuple[AgeBracket, str]:
        """Map raw age class label to assignment AgeBracket enum with confidence thresholding.

        Args:
            raw_label: Raw output label string from model.
            confidence: Highest class prediction confidence score [0.0, 1.0].
            threshold: Minimum required confidence threshold (default 0.50).

        Returns:
            Tuple of (AgeBracket enum, reasoning string).
        """
        if confidence < threshold:
            return AgeBracket.UNKNOWN, f"Low confidence ({confidence:.4f} < {threshold}) -> UNKNOWN"

        normalized_label = raw_label.lower().strip()
        bracket = NATIVE_AGE_MAP.get(normalized_label, AgeBracket.UNKNOWN)
        if bracket == AgeBracket.UNKNOWN:
            reasoning = f"Unmapped raw age label '{raw_label}' -> UNKNOWN"
        else:
            reasoning = f"Mapped raw label '{raw_label}' -> {bracket.value} (confidence {confidence:.4f})"

        return bracket, reasoning

    @staticmethod
    def map_probabilities(raw_probabilities: dict[str, float]) -> dict[str, float]:
        """Normalize raw probabilities dictionary into canonical bracket keys ('18-30', '31-45', '46-60', '60+').

        Args:
            raw_probabilities: Raw dictionary mapping model class names to probabilities.

        Returns:
            Normalized dictionary mapping assignment bracket strings to float probabilities.
        """
        canonical_probs = {"18-30": 0.0, "31-45": 0.0, "46-60": 0.0, "60+": 0.0}

        for raw_k, prob in raw_probabilities.items():
            bracket, _ = AgeMapper.map_to_bracket(raw_k, confidence=1.0, threshold=0.0)
            if bracket != AgeBracket.UNKNOWN and bracket.value in canonical_probs:
                canonical_probs[bracket.value] += prob

        # Normalize sum to 1.0 if non-zero
        total = sum(canonical_probs.values())
        if total > 0:
            canonical_probs = {k: round(v / total, 4) for k, v in canonical_probs.items()}

        return canonical_probs

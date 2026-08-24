"""Domain enumerations for gender, age bracket, and audio quality.

These enums are the canonical source of truth for prediction labels.
They serialize to JSON-friendly string values matching the API contract.
"""

from enum import StrEnum


class Gender(StrEnum):
    """Predicted gender of the caller."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeBracket(StrEnum):
    """Predicted age bracket of the caller."""

    YOUNG_ADULT = "18-30"
    ADULT = "31-45"
    MIDDLE_AGED = "46-60"
    SENIOR = "60+"
    UNKNOWN = "unknown"


class AudioQuality(StrEnum):
    """Assessed quality of the input audio.

    - good: Clean signal, high SNR, reliable predictions expected
    - degraded: Background noise detected, predictions may be less accurate
    - insufficient: Audio too noisy or too short for reliable inference
    """

    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"

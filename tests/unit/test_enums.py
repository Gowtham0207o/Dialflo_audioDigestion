"""Unit tests for domain enums serialization."""

from app.core.enums import AgeBracket, AudioQuality, Gender


def test_gender_enum():
    assert Gender.MALE == "male"
    assert Gender.FEMALE == "female"
    assert Gender.UNKNOWN == "unknown"


def test_age_bracket_enum():
    assert AgeBracket.YOUNG_ADULT == "18-30"
    assert AgeBracket.ADULT == "31-45"
    assert AgeBracket.MIDDLE_AGED == "46-60"
    assert AgeBracket.SENIOR == "60+"
    assert AgeBracket.UNKNOWN == "unknown"


def test_audio_quality_enum():
    assert AudioQuality.GOOD == "good"
    assert AudioQuality.DEGRADED == "degraded"
    assert AudioQuality.INSUFFICIENT == "insufficient"

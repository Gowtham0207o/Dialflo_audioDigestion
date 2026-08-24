"""Response schemas matching the expected API contract.

These Pydantic models define the exact JSON structure returned by
the API, including the nested gender/age prediction objects.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.enums import AgeBracket, AudioQuality, Gender
from app.domain.analysis_result import AnalysisResult


class GenderResponse(BaseModel):
    """Gender prediction response object."""

    prediction: Gender = Field(..., description="Predicted gender")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]"
    )


class AgeBracketResponse(BaseModel):
    """Age bracket prediction response object."""

    prediction: AgeBracket = Field(..., description="Predicted age bracket")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]"
    )


class AnalyzeResponse(BaseModel):
    """Full analysis response — matches the expected API contract.

    Example:
        {
            "contact_id": "uuid",
            "gender": {"prediction": "male", "confidence": 0.87},
            "age_bracket": {"prediction": "31-45", "confidence": 0.63},
            "processing_ms": 142,
            "audio_quality": "good"
        }
    """

    contact_id: str = Field(..., description="Unique request identifier")
    gender: GenderResponse
    age_bracket: AgeBracketResponse
    processing_ms: int = Field(..., description="Processing time in milliseconds")
    audio_quality: AudioQuality = Field(..., description="Audio quality assessment")

    @classmethod
    def from_domain(cls, result: AnalysisResult) -> AnalyzeResponse:
        """Map a domain AnalysisResult to the API response schema."""
        return cls(
            contact_id=result.contact_id,
            gender=GenderResponse(
                prediction=result.gender.prediction,
                confidence=round(result.gender.confidence, 4),
            ),
            age_bracket=AgeBracketResponse(
                prediction=result.age_bracket.prediction,
                confidence=round(result.age_bracket.confidence, 4),
            ),
            processing_ms=result.processing_ms,
            audio_quality=result.audio_quality,
        )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "contact_id": "550e8400-e29b-41d4-a716-446655440000",
                "gender": {"prediction": "male", "confidence": 0.87},
                "age_bracket": {"prediction": "31-45", "confidence": 0.63},
                "processing_ms": 142,
                "audio_quality": "good",
            }
        ]
    }}


class StreamEvent(BaseModel):
    """Progressive prediction event emitted over WebSocket.

    Sent after each audio chunk is processed. Confidence scores
    may improve as more audio data is received.
    """

    chunk_index: int = Field(..., description="Index of the processed chunk")
    is_final: bool = Field(default=False, description="Whether this is the final event")
    gender: GenderResponse
    age_bracket: AgeBracketResponse
    audio_quality: AudioQuality
    cumulative_duration_ms: int = Field(
        ..., description="Total audio duration processed so far"
    )

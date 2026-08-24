"""Response schemas matching API contracts.

Defines Pydantic models for audio metadata response with Silero VAD & Multi-Signal Quality Metrics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.enums import AgeBracket, AudioQuality, Gender
from app.domain.analysis_result import AnalysisResult


class SpeechSegmentSchema(BaseModel):
    """Timestamped speech segment interval."""

    start_seconds: float = Field(..., description="Start timestamp in seconds")
    end_seconds: float = Field(..., description="End timestamp in seconds")
    confidence: float = Field(default=1.0, description="Segment VAD confidence [0.0, 1.0]")


class AudioMetadataResponse(BaseModel):
    """Audio metadata returned after FFmpeg normalization, Silero VAD, and Multi-Signal Quality Assessment."""

    duration: float = Field(..., description="Duration in seconds")
    duration_ms: int = Field(..., description="Duration in milliseconds")
    duration_seconds: float = Field(..., description="Duration in seconds")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz (16000)")
    channels: int = Field(default=1, description="Audio channels (1 = mono)")
    samples: int = Field(..., description="Total PCM float32 samples")
    total_samples: int = Field(..., description="Total PCM float32 samples")
    original_format: str = Field(..., description="Detected original format (e.g. wav, mp3, ogg)")
    processing_ms: int = Field(..., description="End-to-end processing time in milliseconds")

    # ── Silero VAD Metrics ───────────────────
    speech_duration_seconds: float = Field(..., description="Total speech duration in seconds")
    speech_duration_ms: int = Field(..., description="Total speech duration in milliseconds")
    speech_ratio: float = Field(..., description="Ratio of speech duration to total audio duration [0.0, 1.0]")
    speech_segments: list[SpeechSegmentSchema] = Field(
        default_factory=list,
        description="Timestamped speech intervals for debugging",
    )
    is_speech_sufficient: bool = Field(
        ..., description="True if speech ratio and duration exceed minimum thresholds"
    )
    vad_confidence: float = Field(
        default=1.0, description="Average VAD model confidence score [0.0, 1.0]"
    )

    # ── Multi-Signal Quality Assessment Metrics ──
    audio_quality: AudioQuality = Field(..., description="Audio quality classification flag (good, degraded, insufficient)")
    snr_db: float = Field(..., description="Estimated Signal-to-Noise Ratio in dB")
    peak_amplitude: float = Field(..., description="Peak amplitude of the audio waveform [0.0, 1.0]")
    clipping_ratio: float = Field(..., description="Ratio of clipped samples (amplitude >= 0.99)")
    rms_energy: float = Field(default=0.0, description="Root Mean Square (RMS) signal energy")
    speech_energy_ratio: float = Field(default=1.0, description="Ratio of energy in speech frames to total audio energy [0.0, 1.0]")
    quality_reasoning: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the audio_quality classification",
    )


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
    """Full analysis response schema."""

    contact_id: str = Field(..., description="Unique request identifier")
    gender: GenderResponse
    age_bracket: AgeBracketResponse
    processing_ms: int = Field(..., description="Processing time in milliseconds")
    audio_quality: AudioQuality = Field(..., description="Audio quality assessment")
    metadata: AudioMetadataResponse | None = Field(default=None, description="Normalized audio metadata")

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


class StreamEvent(BaseModel):
    """Progressive prediction event emitted over WebSocket."""

    chunk_index: int = Field(..., description="Index of the processed chunk")
    is_final: bool = Field(default=False, description="Whether this is the final event")
    gender: GenderResponse
    age_bracket: AgeBracketResponse
    audio_quality: AudioQuality
    cumulative_duration_ms: int = Field(
        ..., description="Total audio duration processed so far"
    )

"""AnalysisResult aggregate.

The root aggregate returned by the pipeline, combining all
predictions, quality assessment, and processing metadata.
"""

from dataclasses import dataclass

from app.core.enums import AudioQuality
from app.domain.prediction import AgePrediction, GenderPrediction
from app.domain.quality_report import AudioQualityReport


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis result for an audio input.

    This is the domain-level aggregate that maps 1:1 to the API response.

    Attributes:
        contact_id: Unique identifier for this analysis request.
        gender: Gender prediction with confidence.
        age_bracket: Age bracket prediction with confidence.
        audio_quality: Assessed audio quality flag.
        processing_ms: End-to-end processing time in milliseconds.
        quality_report: Detailed audio quality metrics (internal use).
    """

    contact_id: str
    gender: GenderPrediction
    age_bracket: AgePrediction
    audio_quality: AudioQuality
    processing_ms: int
    quality_report: AudioQualityReport | None = None

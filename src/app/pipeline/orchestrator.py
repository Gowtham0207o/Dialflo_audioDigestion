"""Analysis pipeline orchestrator.

Composes all processing stages into a single pipeline:
    audio bytes → validate → decode → quality → denoise → infer → result

This is the primary entry point for the /analyze endpoint.
"""

import time
import uuid

from app.config.settings import Settings
from app.core.enums import AudioQuality
from app.domain.analysis_result import AnalysisResult
from app.domain.prediction import AgePrediction, GenderPrediction
from app.inference.registry import ModelRegistry
from app.observability.logger import get_logger
from app.pipeline.stages import (
    DecodingStage,
    DenoisingStage,
    InferenceStage,
    QualityAssessmentStage,
    ValidationStage,
)

logger = get_logger(__name__)


class AnalysisPipeline:
    """End-to-end audio analysis pipeline.

    Orchestrates all processing stages and assembles the final result.
    Created per-request but uses shared model instances from the registry.

    Args:
        settings: Application settings.
        registry: Model registry with loaded models.
    """

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self._settings = settings
        self._registry = registry

        self._val_stage = ValidationStage()
        self._dec_stage = DecodingStage()
        self._qual_stage = QualityAssessmentStage()
        self._denoise_stage = DenoisingStage()
        self._infer_stage = InferenceStage()

    async def analyze(
        self,
        audio_bytes: bytes,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> AnalysisResult:
        """Run the full analysis pipeline on audio bytes.

        Args:
            audio_bytes: Raw audio bytes from the client.
            content_type: MIME type of the audio.
            filename: Original filename (for format detection).

        Returns:
            Complete AnalysisResult with all predictions.
        """
        start_time = time.perf_counter()
        contact_id = str(uuid.uuid4())

        logger.info(
            "Starting analysis pipeline",
            contact_id=contact_id,
            size_bytes=len(audio_bytes),
            content_type=content_type,
        )

        # 1. Validation Stage
        val_res = await self._val_stage.execute(audio_bytes, content_type, filename)
        if not val_res.success:
            from app.core.exceptions import AudioValidationError
            raise AudioValidationError(val_res.error or "Validation failed")

        # 2. Decoding Stage
        dec_res = await self._dec_stage.execute(audio_bytes)
        if not dec_res.success:
            from app.core.exceptions import AudioCodecError
            raise AudioCodecError(dec_res.error or "Decoding failed")
        segment = dec_res.data

        # 3. Quality Assessment Stage
        qual_res = await self._qual_stage.execute(segment, self._registry)
        audio_quality = qual_res.data["quality"] if qual_res.success else AudioQuality.DEGRADED
        quality_report = qual_res.data.get("report") if qual_res.success else None
        snr_db = quality_report.snr_db if quality_report else 10.0

        # 4. Denoising Stage
        denoise_res = await self._denoise_stage.execute(segment, snr_db)
        clean_segment = denoise_res.data if denoise_res.success else segment

        # 5. Inference Stage: Gender
        gender_res = await self._infer_stage.execute(clean_segment, "gender_classifier", self._registry)
        if gender_res.success:
            gender_pred = GenderPrediction(
                prediction=gender_res.data["prediction"],
                confidence=gender_res.data["confidence"],
            )
        else:
            gender_pred = GenderPrediction.unknown()

        # 6. Inference Stage: Age
        age_res = await self._infer_stage.execute(clean_segment, "age_estimator", self._registry)
        if age_res.success:
            age_pred = AgePrediction(
                prediction=age_res.data["prediction"],
                confidence=age_res.data["confidence"],
            )
        else:
            age_pred = AgePrediction.unknown()

        processing_ms = int((time.perf_counter() - start_time) * 1000)

        result = AnalysisResult(
            contact_id=contact_id,
            gender=gender_pred,
            age_bracket=age_pred,
            audio_quality=audio_quality,
            processing_ms=processing_ms,
            quality_report=quality_report,
        )

        logger.info(
            "Analysis complete",
            contact_id=contact_id,
            processing_ms=processing_ms,
            gender=result.gender.prediction.value,
            age_bracket=result.age_bracket.prediction.value,
            audio_quality=result.audio_quality.value,
        )

        return result

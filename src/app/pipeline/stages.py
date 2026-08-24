"""Individual pipeline stages.

Each stage is a composable unit that can be tested independently.
Stages follow a consistent interface: input → process → output.
"""

from dataclasses import dataclass
import time
from typing import Any

from app.audio.codec import AudioCodec
from app.audio.denoiser import AudioDenoiser
from app.audio.validator import AudioValidator
from app.domain.audio_segment import AudioSegment
from app.inference.registry import ModelRegistry
from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Result from a pipeline stage.

    Attributes:
        success: Whether the stage completed successfully.
        data: Output data from the stage.
        duration_ms: Stage processing time in milliseconds.
        error: Error message if stage failed.
    """

    success: bool
    data: Any = None
    duration_ms: int = 0
    error: str | None = None


class ValidationStage:
    """Stage 1: Validate audio input."""

    async def execute(
        self, audio_bytes: bytes, content_type: str | None, filename: str | None
    ) -> StageResult:
        """Validate audio format, size, and basic properties."""
        t0 = time.perf_counter()
        try:
            AudioValidator.validate_file_size(audio_bytes)
            AudioValidator.validate_content_type(content_type)
            AudioValidator.validate_file_extension(filename)
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=True, data=True, duration_ms=dur)
        except Exception as exc:
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=False, error=str(exc), duration_ms=dur)


class DecodingStage:
    """Stage 2: Decode and transcode audio to canonical format."""

    async def execute(self, audio_bytes: bytes) -> StageResult:
        """Decode audio bytes to 16kHz mono float32 waveform."""
        t0 = time.perf_counter()
        try:
            segment = AudioCodec.transcode_to_wav(audio_bytes)
            AudioValidator.validate_duration(segment)
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=True, data=segment, duration_ms=dur)
        except Exception as exc:
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=False, error=str(exc), duration_ms=dur)


class QualityAssessmentStage:
    """Stage 3: Assess audio quality before inference."""

    async def execute(self, segment: AudioSegment, registry: ModelRegistry) -> StageResult:
        """Assess SNR, VAD ratio, and overall quality."""
        t0 = time.perf_counter()
        try:
            assessor = registry.get("quality_assessor")
            res = await assessor.predict(segment.waveform, segment.sample_rate)
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=True, data=res, duration_ms=dur)
        except Exception as exc:
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=False, error=str(exc), duration_ms=dur)


class DenoisingStage:
    """Stage 4: Apply noise reduction if needed."""

    def __init__(self) -> None:
        self._denoiser = AudioDenoiser()

    async def execute(self, segment: AudioSegment, snr_db: float) -> StageResult:
        """Apply noise reduction based on SNR threshold."""
        t0 = time.perf_counter()
        try:
            if snr_db < 15.0:
                clean_wf = self._denoiser.denoise(segment.waveform, segment.sample_rate)
                clean_segment = AudioSegment(
                    waveform=clean_wf,
                    sample_rate=segment.sample_rate,
                    duration_ms=segment.duration_ms,
                    channels=1,
                    original_format=segment.original_format,
                )
            else:
                clean_segment = segment

            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=True, data=clean_segment, duration_ms=dur)
        except Exception as exc:
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=False, error=str(exc), duration_ms=dur)


class InferenceStage:
    """Stage 5: Run ML model inference."""

    async def execute(
        self, segment: AudioSegment, model_name: str, registry: ModelRegistry
    ) -> StageResult:
        """Run a specific model on the audio segment."""
        t0 = time.perf_counter()
        try:
            model = registry.get(model_name)
            res = await model.predict(segment.waveform, segment.sample_rate)
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=True, data=res, duration_ms=dur)
        except Exception as exc:
            dur = int((time.perf_counter() - t0) * 1000)
            return StageResult(success=False, error=str(exc), duration_ms=dur)

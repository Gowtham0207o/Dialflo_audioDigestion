"""WebSocket streaming analysis handler.

Manages progressive inference for the WebSocket /v1/stream endpoint.
Accumulates audio chunks and emits updated predictions as more
data arrives, with confidence scores that improve over time.
"""

import numpy as np
from numpy.typing import NDArray

from app.api.v1.schemas.responses import (
    AgeBracketResponse,
    GenderResponse,
    StreamEvent,
)
from app.audio.codec import AudioCodec
from app.core.enums import AgeBracket, AudioQuality, Gender
from app.domain.audio_segment import AudioSegment
from app.inference.strategies.age_estimator import AgeEstimator
from app.inference.strategies.gender_classifier import GenderClassifier
from app.inference.strategies.quality_assessor import QualityAssessor
from app.observability.logger import get_logger

logger = get_logger(__name__)


class StreamingAnalyzer:
    """Manages state for a single WebSocket streaming session.

    Accumulates audio chunks and produces progressive predictions.
    All audio data is held in memory and released on cleanup.
    """

    def __init__(self) -> None:
        self._audio_buffers: list[NDArray[np.float32]] = []
        self._chunk_index = 0
        self._cumulative_duration_ms = 0

        from app.inference.wav2vec2_model import Wav2Vec2AttributeModel
        self._attribute_model = Wav2Vec2AttributeModel()
        self._attribute_model.load()
        self._quality_assessor = QualityAssessor()

    async def process_chunk(self, chunk_bytes: bytes) -> StreamEvent | None:
        """Process an incoming raw audio chunk.

        Args:
            chunk_bytes: Raw audio bytes for this chunk.

        Returns:
            StreamEvent with progressive prediction, or None if invalid.
        """
        self._chunk_index += 1

        try:
            segment = AudioCodec.transcode_to_wav(chunk_bytes)
            self._audio_buffers.append(segment.waveform)
            self._cumulative_duration_ms += segment.duration_ms
        except Exception as exc:
            logger.warning("Failed to decode chunk", chunk_index=self._chunk_index, error=str(exc))
            return None

        # Concatenate accumulated waveform for progressive inference
        full_waveform = np.concatenate(self._audio_buffers)

        qual_res = await self._quality_assessor.predict(full_waveform, 16000)
        audio_quality = qual_res["quality"]

        from app.audio.preprocessor import PreparedMLInput
        prepared_input = PreparedMLInput(
            is_prepared_valid=True,
            audio_array=full_waveform,
            preparation_reasoning="stream_chunk"
        )
        attr_result = self._attribute_model.predict(prepared_input)

        return StreamEvent(
            chunk_index=self._chunk_index,
            is_final=False,
            gender=GenderResponse(
                prediction=attr_result.gender,
                confidence=round(attr_result.gender_confidence, 4),
            ),
            age_bracket=AgeBracketResponse(
                prediction=attr_result.age_bracket,
                confidence=round(attr_result.age_confidence, 4),
            ),
            audio_quality=audio_quality,
            cumulative_duration_ms=self._cumulative_duration_ms,
        )

    async def finalize(self) -> StreamEvent:
        """Produce final prediction from all accumulated audio.

        Returns:
            Final StreamEvent with is_final=True.
        """
        if not self._audio_buffers:
            return StreamEvent(
                chunk_index=self._chunk_index,
                is_final=True,
                gender=GenderResponse(prediction=Gender.UNKNOWN, confidence=0.0),
                age_bracket=AgeBracketResponse(prediction=AgeBracket.UNKNOWN, confidence=0.0),
                audio_quality=AudioQuality.INSUFFICIENT,
                cumulative_duration_ms=0,
            )

        full_waveform = np.concatenate(self._audio_buffers)

        qual_res = await self._quality_assessor.predict(full_waveform, 16000)
        
        from app.audio.preprocessor import PreparedMLInput
        prepared_input = PreparedMLInput(
            is_prepared_valid=True,
            audio_array=full_waveform,
            preparation_reasoning="stream_final"
        )
        attr_result = self._attribute_model.predict(prepared_input)

        return StreamEvent(
            chunk_index=self._chunk_index,
            is_final=True,
            gender=GenderResponse(
                prediction=attr_result.gender,
                confidence=round(attr_result.gender_confidence, 4),
            ),
            age_bracket=AgeBracketResponse(
                prediction=attr_result.age_bracket,
                confidence=round(attr_result.age_confidence, 4),
            ),
            audio_quality=qual_res["quality"],
            cumulative_duration_ms=self._cumulative_duration_ms,
        )

    async def cleanup(self) -> None:
        """Release all audio buffers (PII safety)."""
        self._audio_buffers.clear()
        self._chunk_index = 0
        self._cumulative_duration_ms = 0
        logger.debug("Streaming analyzer buffers cleared")

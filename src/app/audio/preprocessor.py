"""ML Input Preparation module.

Selects active speech regions from refined Silero VAD results, strips background silence,
and constructs a deterministic, model-ready 16 kHz mono float32 waveform fitted to the
target inference window (e.g. 3.0s / 48,000 samples) without duplicating or fabricating audio.
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from app.audio.quality import QualityResult
from app.audio.vad import SpeechSegment, VADResult
from app.core.enums import AudioQuality
from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedMLInput:
    """Model-ready input payload prepared for downstream attribute inference."""

    prepared_waveform: NDArray[np.float32]
    sample_rate: int
    duration_seconds: float
    num_samples: int
    is_prepared_valid: bool
    speech_segments_used: list[SpeechSegment] = field(default_factory=list)
    preparation_reasoning: str = ""


class AudioPreprocessor:
    """Preprocesses audio waveforms using VAD and Quality metrics for ML attribute inference."""

    @staticmethod
    def prepare(
        waveform: NDArray[np.float32],
        vad_result: VADResult,
        quality_result: QualityResult,
        target_duration_seconds: float = 3.0,
        sample_rate: int = 16000,
    ) -> PreparedMLInput:
        """Extract active speech regions, fit to target inference window, and construct ML payload.

        Args:
            waveform: Original 1D float32 PCM audio array.
            vad_result: Refined Silero VAD result.
            quality_result: Multi-signal audio quality assessment result.
            target_duration_seconds: Target inference window in seconds (default 3.0s).
            sample_rate: Audio sample rate in Hz (16000).

        Returns:
            PreparedMLInput containing deterministic float32 prepared waveform and validity flag.
        """
        # Ensure original input waveform is never mutated
        original_waveform = np.array(waveform, dtype=np.float32, copy=True)
        target_samples = int(target_duration_seconds * sample_rate)

        # 1. Validation & Sufficiency Guard
        if len(original_waveform) == 0:
            return PreparedMLInput(
                prepared_waveform=np.zeros(target_samples, dtype=np.float32),
                sample_rate=sample_rate,
                duration_seconds=0.0,
                num_samples=0,
                is_prepared_valid=False,
                speech_segments_used=[],
                preparation_reasoning="Empty input audio waveform",
            )

        if quality_result.audio_quality == AudioQuality.INSUFFICIENT:
            reason = f"Audio quality is insufficient ({', '.join(quality_result.quality_reasoning)})"
            logger.warning("ML input preparation skipped due to insufficient quality", reasoning=reason)
            return PreparedMLInput(
                prepared_waveform=np.zeros(target_samples, dtype=np.float32),
                sample_rate=sample_rate,
                duration_seconds=round(len(original_waveform) / float(sample_rate), 3),
                num_samples=len(original_waveform),
                is_prepared_valid=False,
                speech_segments_used=[],
                preparation_reasoning=reason,
            )

        if not vad_result.is_speech_sufficient or not vad_result.speech_segments:
            reason = f"Insufficient speech duration or ratio ({vad_result.speech_ratio * 100:.1f}%)"
            logger.warning("ML input preparation skipped due to insufficient speech", reasoning=reason)
            return PreparedMLInput(
                prepared_waveform=np.zeros(target_samples, dtype=np.float32),
                sample_rate=sample_rate,
                duration_seconds=round(len(original_waveform) / float(sample_rate), 3),
                num_samples=len(original_waveform),
                is_prepared_valid=False,
                speech_segments_used=[],
                preparation_reasoning=reason,
            )

        # 2. Extract Active Speech Regions
        speech_chunks: list[NDArray[np.float32]] = []
        segments_used: list[SpeechSegment] = []

        for seg in vad_result.speech_segments:
            start_idx = max(0, int(seg.start_seconds * sample_rate))
            end_idx = min(len(original_waveform), int(seg.end_seconds * sample_rate))

            if start_idx < end_idx:
                speech_chunks.append(original_waveform[start_idx:end_idx])
                segments_used.append(seg)

        if speech_chunks:
            concatenated_speech = np.concatenate(speech_chunks)
        else:
            concatenated_speech = original_waveform

        # 3. Fit to Target Inference Window (e.g. 48,000 samples)
        curr_samples = len(concatenated_speech)

        if curr_samples == target_samples:
            prepared_waveform = concatenated_speech.copy()
            reasoning = f"Speech duration exactly matches target window ({target_duration_seconds}s)"
        elif curr_samples < target_samples:
            # Shorter than target window -> right-pad with zeros without duplicating audio
            pad_len = target_samples - curr_samples
            prepared_waveform = np.pad(concatenated_speech, (0, pad_len), mode="constant", constant_values=0.0)
            reasoning = f"Speech duration ({curr_samples / sample_rate:.2f}s) right-padded to target window ({target_duration_seconds}s)"
        else:
            # Longer than target window -> deterministically slice first target_samples of continuous speech
            prepared_waveform = concatenated_speech[:target_samples].copy()
            reasoning = f"Speech duration ({curr_samples / sample_rate:.2f}s) deterministically sliced to first {target_duration_seconds}s window"

        logger.debug(
            "ML Input Preparation completed successfully",
            num_samples=len(prepared_waveform),
            target_samples=target_samples,
            segments_used_count=len(segments_used),
            reasoning=reasoning,
        )

        return PreparedMLInput(
            prepared_waveform=prepared_waveform,
            sample_rate=sample_rate,
            duration_seconds=round(len(prepared_waveform) / float(sample_rate), 3),
            num_samples=len(prepared_waveform),
            is_prepared_valid=True,
            speech_segments_used=segments_used,
            preparation_reasoning=reasoning,
        )

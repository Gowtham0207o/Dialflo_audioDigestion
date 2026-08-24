"""Voice Activity Detection (VAD) module using Silero VAD engine.

Loads Silero VAD once at application startup, evaluates 16 kHz mono float32 PCM waveforms,
applies segment refinement (gap merging & fragment filtering), calculates total speech duration,
speech ratio, timestamped speech segments, and exposes average VAD confidence.
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import torch

from app.observability.logger import get_logger

logger = get_logger(__name__)

# Global model cache for Silero VAD singleton
_SILERO_MODEL = None
_SILERO_UTILS = None


def load_silero_vad_model():
    """Load Silero VAD model once into memory."""
    global _SILERO_MODEL, _SILERO_UTILS
    if _SILERO_MODEL is not None and _SILERO_UTILS is not None:
        return _SILERO_MODEL, _SILERO_UTILS

    try:
        logger.info("Loading Silero VAD model from PyTorch Hub...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _SILERO_MODEL = model
        _SILERO_UTILS = utils
        logger.info("Silero VAD model loaded successfully")
        return _SILERO_MODEL, _SILERO_UTILS
    except Exception as exc:
        logger.warning("Silero VAD model load unhandled, using energy VAD fallback", error=str(exc))
        return None, None


@dataclass(frozen=True)
class SpeechSegment:
    """Timestamped speech segment interval."""

    start_seconds: float
    end_seconds: float
    confidence: float = 1.0

    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 3)

    @property
    def duration_ms(self) -> int:
        return int((self.end_seconds - self.start_seconds) * 1000)


@dataclass
class VADResult:
    """Results of Voice Activity Detection."""

    speech_duration_ms: int
    speech_duration_seconds: float
    speech_ratio: float
    speech_segments: list[SpeechSegment] = field(default_factory=list)
    is_speech_sufficient: bool = False
    vad_confidence: float = 1.0


class VoiceActivityDetector:
    """Voice Activity Detector using Silero VAD engine & post-processing refinement.

    Args:
        min_speech_ratio: Threshold ratio of speech duration to total audio duration.
        min_speech_duration_ms: Minimum required total speech duration in milliseconds for sufficiency.
        merge_gap_ms: Maximum silence gap in milliseconds between adjacent speech segments to merge.
        min_segment_duration_ms: Minimum duration in milliseconds for a refined speech segment.
        silero_threshold: Speech probability threshold for Silero VAD [0.0, 1.0].
    """

    def __init__(
        self,
        min_speech_ratio: float = 0.30,
        min_speech_duration_ms: int = 1000,
        merge_gap_ms: int = 300,
        min_segment_duration_ms: int = 150,
        silero_threshold: float = 0.50,
    ) -> None:
        self.min_speech_ratio = min_speech_ratio
        self.min_speech_duration_ms = min_speech_duration_ms
        self.merge_gap_ms = merge_gap_ms
        self.min_segment_duration_ms = min_segment_duration_ms
        self.silero_threshold = silero_threshold

    @classmethod
    def preload_model(cls) -> None:
        """Pre-load Silero VAD model during application startup."""
        load_silero_vad_model()

    @staticmethod
    def refine_segments(
        raw_segments: list[SpeechSegment],
        merge_gap_ms: int = 300,
        min_segment_duration_ms: int = 150,
    ) -> list[SpeechSegment]:
        """Refine raw VAD speech segments by merging short gaps and removing tiny noise fragments.

        Args:
            raw_segments: List of raw detected SpeechSegment objects.
            merge_gap_ms: Silence gap threshold in ms below which adjacent segments are merged.
            min_segment_duration_ms: Segment duration threshold in ms below which fragments are removed.

        Returns:
            List of refined SpeechSegment objects.
        """
        if not raw_segments:
            return []

        sorted_segs = sorted(raw_segments, key=lambda s: s.start_seconds)
        merge_gap_sec = merge_gap_ms / 1000.0
        min_dur_sec = min_segment_duration_ms / 1000.0

        merged: list[SpeechSegment] = []
        current_start = sorted_segs[0].start_seconds
        current_end = sorted_segs[0].end_seconds
        current_conf = sorted_segs[0].confidence

        for next_seg in sorted_segs[1:]:
            if next_seg.start_seconds <= (current_end + merge_gap_sec):
                current_end = max(current_end, next_seg.end_seconds)
                current_conf = max(current_conf, next_seg.confidence)
            else:
                merged.append(
                    SpeechSegment(
                        start_seconds=round(current_start, 3),
                        end_seconds=round(current_end, 3),
                        confidence=round(current_conf, 4),
                    )
                )
                current_start = next_seg.start_seconds
                current_end = next_seg.end_seconds
                current_conf = next_seg.confidence

        merged.append(
            SpeechSegment(
                start_seconds=round(current_start, 3),
                end_seconds=round(current_end, 3),
                confidence=round(current_conf, 4),
            )
        )

        refined = [s for s in merged if round(s.end_seconds - s.start_seconds, 3) >= min_dur_sec]
        return refined

    def _detect_energy_fallback(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int,
        total_duration_seconds: float,
    ) -> list[SpeechSegment]:
        """Fallback energy-based VAD when Silero VAD model fails to load."""
        frame_len = int(sample_rate * 0.030)
        hop_len = int(sample_rate * 0.010)

        if len(waveform) < frame_len:
            peak_e = float(np.max(np.abs(waveform)))
            return [SpeechSegment(0.0, round(total_duration_seconds, 3), confidence=0.85)] if peak_e > 0.02 else []

        n_frames = 1 + (len(waveform) - frame_len) // hop_len
        energies = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop_len
            energies[i] = np.mean(waveform[start : start + frame_len] ** 2)

        max_e = float(np.max(energies))
        if max_e < 1e-5:
            return []

        noise_floor = float(np.percentile(energies, 15))
        threshold = max(1e-4, min(noise_floor * 3.5, 0.4 * max_e))
        speech_mask = energies > threshold

        raw_segments: list[SpeechSegment] = []
        seg_start_idx = None

        for i in range(len(speech_mask)):
            if speech_mask[i] and seg_start_idx is None:
                seg_start_idx = i
            elif not speech_mask[i] and seg_start_idx is not None:
                start_sec = round(seg_start_idx * hop_len / float(sample_rate), 3)
                end_sec = round((i * hop_len + frame_len) / float(sample_rate), 3)
                raw_segments.append(SpeechSegment(start_sec, min(end_sec, round(total_duration_seconds, 3)), confidence=0.85))
                seg_start_idx = None

        if seg_start_idx is not None:
            start_sec = round(seg_start_idx * hop_len / float(sample_rate), 3)
            raw_segments.append(SpeechSegment(start_sec, round(total_duration_seconds, 3), confidence=0.85))

        return raw_segments

    def detect(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> VADResult:
        """Detect speech frames using Silero VAD, apply refinement, and extract metrics.

        Args:
            waveform: 1D float32 array of audio samples.
            sample_rate: Sample rate in Hz (16000).

        Returns:
            VADResult containing refined speech metrics, segments, confidence, and sufficiency flag.
        """
        if len(waveform) == 0:
            return VADResult(
                speech_duration_ms=0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                speech_segments=[],
                is_speech_sufficient=False,
                vad_confidence=0.0,
            )

        total_duration_seconds = len(waveform) / float(sample_rate)
        if total_duration_seconds <= 0:
            return VADResult(
                speech_duration_ms=0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                speech_segments=[],
                is_speech_sufficient=False,
                vad_confidence=0.0,
            )

        model, utils = load_silero_vad_model()
        raw_segments: list[SpeechSegment] = []

        if model is not None and utils is not None:
            try:
                (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
                wav_tensor = torch.from_numpy(waveform).squeeze()

                timestamps = get_speech_timestamps(
                    wav_tensor,
                    model,
                    threshold=self.silero_threshold,
                    sampling_rate=sample_rate,
                )

                for ts in timestamps:
                    start_sec = round(ts["start"] / float(sample_rate), 3)
                    end_sec = round(ts["end"] / float(sample_rate), 3)
                    raw_segments.append(SpeechSegment(start_sec, end_sec, confidence=0.95))

            except Exception as exc:
                logger.warning("Silero VAD execution error, falling back to energy VAD", error=str(exc))
                raw_segments = self._detect_energy_fallback(waveform, sample_rate, total_duration_seconds)
        else:
            raw_segments = self._detect_energy_fallback(waveform, sample_rate, total_duration_seconds)

        # 2. Refinement Stage: Merge short gaps and filter tiny fragments
        refined_segments = self.refine_segments(
            raw_segments,
            merge_gap_ms=self.merge_gap_ms,
            min_segment_duration_ms=self.min_segment_duration_ms,
        )

        # 3. Calculate final metrics from REFINED segments
        total_speech_sec = sum(seg.duration_seconds for seg in refined_segments)
        speech_duration_ms = int(total_speech_sec * 1000)
        speech_duration_seconds = round(total_speech_sec, 3)

        speech_ratio = min(1.0, round(speech_duration_seconds / total_duration_seconds, 3)) if total_duration_seconds > 0 else 0.0

        is_sufficient = (
            speech_ratio >= self.min_speech_ratio
            and speech_duration_ms >= self.min_speech_duration_ms
        )

        vad_confidence = (
            round(float(np.mean([s.confidence for s in refined_segments])), 4)
            if refined_segments
            else (0.0 if speech_ratio == 0.0 else 0.90)
        )

        logger.debug(
            "Silero VAD detection & refinement completed",
            total_duration_seconds=round(total_duration_seconds, 3),
            refined_segments_count=len(refined_segments),
            speech_duration_seconds=speech_duration_seconds,
            speech_ratio=speech_ratio,
            vad_confidence=vad_confidence,
            is_speech_sufficient=is_sufficient,
        )

        return VADResult(
            speech_duration_ms=speech_duration_ms,
            speech_duration_seconds=speech_duration_seconds,
            speech_ratio=speech_ratio,
            speech_segments=refined_segments,
            is_speech_sufficient=is_sufficient,
            vad_confidence=vad_confidence,
        )

"""Voice Activity Detection (VAD) module.

Detects speech vs non-speech/silence frames on 16 kHz mono float32 PCM waveforms,
calculates total speech duration, speech ratio, extracts timestamped speech segments,
and evaluates speech sufficiency.
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpeechSegment:
    """Timestamped speech segment interval."""

    start_seconds: float
    end_seconds: float

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


class VoiceActivityDetector:
    """Voice Activity Detector using short-time energy & hangover smoothing.

    Args:
        min_speech_ratio: Threshold ratio of speech duration to total audio duration.
        min_speech_duration_ms: Minimum required speech duration in milliseconds.
    """

    def __init__(
        self,
        min_speech_ratio: float = 0.30,
        min_speech_duration_ms: int = 1000,
    ) -> None:
        self.min_speech_ratio = min_speech_ratio
        self.min_speech_duration_ms = min_speech_duration_ms

    def detect(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> VADResult:
        """Detect speech frames and extract timestamped speech segments.

        Args:
            waveform: 1D float32 array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            VADResult containing speech metrics, segments, and sufficiency flag.
        """
        if len(waveform) == 0:
            return VADResult(
                speech_duration_ms=0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                speech_segments=[],
                is_speech_sufficient=False,
            )

        total_duration_seconds = len(waveform) / float(sample_rate)
        if total_duration_seconds <= 0:
            return VADResult(
                speech_duration_ms=0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                speech_segments=[],
                is_speech_sufficient=False,
            )

        # 1. Frame analysis: 30ms frames with 10ms hop
        frame_len = int(sample_rate * 0.030)  # 480 samples
        hop_len = int(sample_rate * 0.010)    # 160 samples

        if len(waveform) < frame_len:
            peak_e = float(np.max(np.abs(waveform)))
            is_speech = peak_e > 0.02
            speech_ms = int(total_duration_seconds * 1000) if is_speech else 0
            speech_sec = round(total_duration_seconds, 3) if is_speech else 0.0
            ratio = 1.0 if is_speech else 0.0
            segments = [SpeechSegment(0.0, speech_sec)] if is_speech else []
            sufficient = (ratio >= self.min_speech_ratio) and (speech_ms >= self.min_speech_duration_ms)
            return VADResult(
                speech_duration_ms=speech_ms,
                speech_duration_seconds=speech_sec,
                speech_ratio=ratio,
                speech_segments=segments,
                is_speech_sufficient=sufficient,
            )

        # Compute Short-Time Energy (STE) per frame
        n_frames = 1 + (len(waveform) - frame_len) // hop_len
        energies = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop_len
            frame = waveform[start : start + frame_len]
            energies[i] = np.mean(frame ** 2)

        max_e = float(np.max(energies))
        if max_e < 1e-5:
            # Pure silence / zero waveform
            return VADResult(
                speech_duration_ms=0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                speech_segments=[],
                is_speech_sufficient=False,
            )

        # Adaptive energy thresholding
        noise_floor = float(np.percentile(energies, 15))
        threshold = max(1e-4, min(noise_floor * 3.5, 0.4 * max_e))

        # Initial speech frame mask
        speech_mask = energies > threshold

        # 2. Hangover smoothing: fill short silence gaps (< 200ms) & remove isolated noise spikes (< 80ms)
        gap_frames = int(0.200 / 0.010)        # 20 frames
        min_speech_frames = int(0.080 / 0.010) # 8 frames

        smooth_mask = speech_mask.copy()
        silence_count = 0
        in_speech = False

        for i in range(len(smooth_mask)):
            if smooth_mask[i]:
                if not in_speech and silence_count < gap_frames and i > silence_count:
                    smooth_mask[i - silence_count : i] = True
                in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count > gap_frames:
                        in_speech = False

        # 3. Extract continuous speech segments
        segments: list[SpeechSegment] = []
        seg_start_idx = None

        for i in range(len(smooth_mask)):
            if smooth_mask[i] and seg_start_idx is None:
                seg_start_idx = i
            elif not smooth_mask[i] and seg_start_idx is not None:
                seg_frames = i - seg_start_idx
                if seg_frames >= min_speech_frames:
                    start_sec = round(seg_start_idx * hop_len / float(sample_rate), 3)
                    end_sec = round((i * hop_len + frame_len) / float(sample_rate), 3)
                    end_sec = min(end_sec, round(total_duration_seconds, 3))
                    segments.append(SpeechSegment(start_seconds=start_sec, end_seconds=end_sec))
                seg_start_idx = None

        if seg_start_idx is not None:
            seg_frames = len(smooth_mask) - seg_start_idx
            if seg_frames >= min_speech_frames:
                start_sec = round(seg_start_idx * hop_len / float(sample_rate), 3)
                end_sec = round(total_duration_seconds, 3)
                segments.append(SpeechSegment(start_seconds=start_sec, end_seconds=end_sec))

        # 4. Calculate final speech metrics
        total_speech_sec = sum(seg.duration_seconds for seg in segments)
        speech_duration_ms = int(total_speech_sec * 1000)
        speech_duration_seconds = round(total_speech_sec, 3)

        speech_ratio = min(1.0, round(speech_duration_seconds / total_duration_seconds, 3)) if total_duration_seconds > 0 else 0.0

        is_sufficient = (
            speech_ratio >= self.min_speech_ratio
            and speech_duration_ms >= self.min_speech_duration_ms
        )

        logger.debug(
            "VAD detection completed",
            total_duration_seconds=round(total_duration_seconds, 3),
            speech_duration_seconds=speech_duration_seconds,
            speech_ratio=speech_ratio,
            num_segments=len(segments),
            is_speech_sufficient=is_sufficient,
        )

        return VADResult(
            speech_duration_ms=speech_duration_ms,
            speech_duration_seconds=speech_duration_seconds,
            speech_ratio=speech_ratio,
            speech_segments=segments,
            is_speech_sufficient=is_sufficient,
        )

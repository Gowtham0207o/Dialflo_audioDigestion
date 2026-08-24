"""Audio Quality Assessment module.

Evaluates peak amplitude, clipping, signal RMS energy, and estimates Signal-to-Noise Ratio (SNR in dB)
to classify audio quality as good, degraded, or insufficient with clear reasoning.
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from app.audio.vad import VADResult
from app.core.enums import AudioQuality
from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityResult:
    """Results of Audio Quality Assessment."""

    audio_quality: AudioQuality
    snr_db: float
    peak_amplitude: float
    clipping_ratio: float
    rms_energy: float
    quality_reasoning: list[str] = field(default_factory=list)


class AudioQualityAssessor:
    """Assesses audio quality using peak amplitude, clipping, and frame-level SNR.

    Args:
        snr_good_threshold_db: Threshold above which SNR is considered good (e.g. 18.0 dB).
        snr_degraded_threshold_db: Threshold below which SNR is considered insufficient (e.g. 5.0 dB).
        clipping_max_ratio: Maximum allowed ratio of clipped samples before declaring degraded quality (e.g. 0.005).
    """

    def __init__(
        self,
        snr_good_threshold_db: float = 18.0,
        snr_degraded_threshold_db: float = 5.0,
        clipping_max_ratio: float = 0.005,
    ) -> None:
        self.snr_good_threshold_db = snr_good_threshold_db
        self.snr_degraded_threshold_db = snr_degraded_threshold_db
        self.clipping_max_ratio = clipping_max_ratio

    def assess(
        self,
        waveform: NDArray[np.float32],
        vad_result: VADResult,
        sample_rate: int = 16000,
    ) -> QualityResult:
        """Assess quality metrics and classify audio into good, degraded, or insufficient.

        Args:
            waveform: 1D float32 PCM audio array.
            vad_result: VAD result from Chunk 2.
            sample_rate: Audio sample rate in Hz.

        Returns:
            QualityResult containing flag, SNR, peak, clipping ratio, and reasoning.
        """
        if len(waveform) == 0:
            return QualityResult(
                audio_quality=AudioQuality.INSUFFICIENT,
                snr_db=0.0,
                peak_amplitude=0.0,
                clipping_ratio=0.0,
                rms_energy=0.0,
                quality_reasoning=["Empty audio payload"],
            )

        # 1. Peak amplitude & clipping calculation
        peak_amplitude = round(float(np.max(np.abs(waveform))), 4)
        num_clipped = int(np.sum(np.abs(waveform) >= 0.99))
        clipping_ratio = round(float(num_clipped / len(waveform)), 4)
        rms_energy = round(float(np.sqrt(np.mean(waveform ** 2))), 6)

        # 2. Frame-level SNR calculation
        frame_len = int(sample_rate * 0.030)
        hop_len = int(sample_rate * 0.010)

        n_frames = 1 + max(0, (len(waveform) - frame_len) // hop_len)
        if n_frames < 2 or rms_energy < 1e-4:
            snr_db = 0.0 if rms_energy < 1e-4 else 25.0
        else:
            frame_energies = np.zeros(n_frames, dtype=np.float32)
            for i in range(n_frames):
                start = i * hop_len
                frame_energies[i] = np.mean(waveform[start : start + frame_len] ** 2)

            speech_mask = np.zeros(n_frames, dtype=bool)
            for seg in vad_result.speech_segments:
                start_frame = int(seg.start_seconds * sample_rate / hop_len)
                end_frame = int(seg.end_seconds * sample_rate / hop_len)
                speech_mask[start_frame : min(end_frame, n_frames)] = True

            speech_energies = frame_energies[speech_mask]
            noise_energies = frame_energies[~speech_mask]

            speech_power = float(np.mean(speech_energies)) if len(speech_energies) > 0 else float(np.mean(frame_energies))

            if len(noise_energies) > 0 and float(np.mean(noise_energies)) > 1e-7:
                noise_power = float(np.mean(noise_energies))
            else:
                # No separate noise frames -> assume clean floor (1e-5)
                noise_power = 1e-5

            snr_linear = max(speech_power / max(noise_power, 1e-8), 1e-6)
            snr_db = round(float(10.0 * np.log10(snr_linear)), 2)

        # 3. Quality Classification & Reasoning
        reasoning: list[str] = []
        is_insufficient = False
        is_degraded = False

        # Checks for INSUFFICIENT quality
        if not vad_result.is_speech_sufficient:
            is_insufficient = True
            reasoning.append("Insufficient speech content detected by VAD")

        if peak_amplitude < 0.01:
            is_insufficient = True
            reasoning.append(f"Extremely low audio volume (peak amplitude {peak_amplitude} < 0.01)")

        if snr_db < self.snr_degraded_threshold_db:
            is_insufficient = True
            reasoning.append(f"Severe background noise (SNR {snr_db} dB < {self.snr_degraded_threshold_db} dB)")

        # Checks for DEGRADED quality if not already insufficient
        if not is_insufficient:
            if clipping_ratio > self.clipping_max_ratio:
                is_degraded = True
                reasoning.append(f"Audio clipping detected ({clipping_ratio * 100:.2f}% samples clipped)")

            if snr_db < self.snr_good_threshold_db:
                is_degraded = True
                reasoning.append(f"Moderate background noise (SNR {snr_db} dB < {self.snr_good_threshold_db} dB)")

            if vad_result.speech_ratio < 0.50:
                is_degraded = True
                reasoning.append(f"Moderate speech ratio ({vad_result.speech_ratio * 100:.1f}% < 50%)")

        if is_insufficient:
            final_quality = AudioQuality.INSUFFICIENT
        elif is_degraded:
            final_quality = AudioQuality.DEGRADED
        else:
            final_quality = AudioQuality.GOOD
            reasoning.append("High SNR, clean signal, and sufficient speech content")

        logger.debug(
            "Audio quality assessment completed",
            audio_quality=final_quality.value,
            snr_db=snr_db,
            peak_amplitude=peak_amplitude,
            clipping_ratio=clipping_ratio,
            reasoning=reasoning,
        )

        return QualityResult(
            audio_quality=final_quality,
            snr_db=snr_db,
            peak_amplitude=peak_amplitude,
            clipping_ratio=clipping_ratio,
            rms_energy=rms_energy,
            quality_reasoning=reasoning,
        )

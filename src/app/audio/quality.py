"""Multi-Signal Audio Quality Assessment module.

Evaluates 5 independent acoustic signals using refined Silero VAD speech region isolation:
1. Signal-to-Noise Ratio (SNR in dB) comparing speech regions vs background noise regions.
2. Refined Silero VAD speech ratio (speech duration / total audio duration).
3. Peak amplitude & RMS signal energy.
4. Clipping ratio (samples >= 0.99).
5. Speech energy ratio (proportion of total energy contained within active speech regions).

Classifies audio quality into good, degraded, or insufficient with explicit reasoning.
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
    """Results of Multi-Signal Audio Quality Assessment."""

    audio_quality: AudioQuality
    snr_db: float
    peak_amplitude: float
    clipping_ratio: float
    rms_energy: float
    speech_energy_ratio: float
    quality_reasoning: list[str] = field(default_factory=list)


class AudioQualityAssessor:
    """Multi-Signal Audio Quality Assessor using Silero VAD speech region isolation.

    Args:
        snr_good_threshold_db: Threshold above which SNR is considered good (e.g. 18.0 dB).
        snr_degraded_threshold_db: Threshold below which SNR is considered insufficient (e.g. 5.0 dB).
        clipping_max_ratio: Maximum allowed ratio of clipped samples before declaring degraded quality (e.g. 0.005).
        min_peak_amplitude: Minimum peak amplitude threshold before declaring insufficient quality (e.g. 0.01).
    """

    def __init__(
        self,
        snr_good_threshold_db: float = 18.0,
        snr_degraded_threshold_db: float = 5.0,
        clipping_max_ratio: float = 0.005,
        min_peak_amplitude: float = 0.01,
    ) -> None:
        self.snr_good_threshold_db = snr_good_threshold_db
        self.snr_degraded_threshold_db = snr_degraded_threshold_db
        self.clipping_max_ratio = clipping_max_ratio
        self.min_peak_amplitude = min_peak_amplitude

    def assess(
        self,
        waveform: NDArray[np.float32],
        vad_result: VADResult,
        sample_rate: int = 16000,
    ) -> QualityResult:
        """Assess quality across multiple independent acoustic signals.

        Args:
            waveform: 1D float32 PCM audio array.
            vad_result: Refined Silero VAD result from Chunk 2.
            sample_rate: Audio sample rate in Hz.

        Returns:
            QualityResult containing flag, SNR, peak, clipping, RMS, speech energy ratio, and reasoning.
        """
        if len(waveform) == 0:
            return QualityResult(
                audio_quality=AudioQuality.INSUFFICIENT,
                snr_db=0.0,
                peak_amplitude=0.0,
                clipping_ratio=0.0,
                rms_energy=0.0,
                speech_energy_ratio=0.0,
                quality_reasoning=["Empty audio payload"],
            )

        # 1. Signal 1 & 2: Peak amplitude, Clipping ratio, RMS Energy
        peak_amplitude = round(float(np.max(np.abs(waveform))), 4)
        num_clipped = int(np.sum(np.abs(waveform) >= 0.99))
        clipping_ratio = round(float(num_clipped / len(waveform)), 4)
        rms_energy = round(float(np.sqrt(np.mean(waveform ** 2))), 6)

        # 2. Silero VAD Speech Region Isolation & SNR Estimation
        frame_len = int(sample_rate * 0.030) # 30ms frame
        hop_len = int(sample_rate * 0.010)   # 10ms hop

        n_frames = 1 + max(0, (len(waveform) - frame_len) // hop_len)
        if n_frames < 2 or rms_energy < 1e-4:
            snr_db = 0.0 if rms_energy < 1e-4 else 25.0
            speech_energy_ratio = 0.0 if rms_energy < 1e-4 else 1.0
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

            total_energy = float(np.sum(frame_energies))
            speech_total_energy = float(np.sum(speech_energies)) if len(speech_energies) > 0 else 0.0
            speech_energy_ratio = round(speech_total_energy / max(total_energy, 1e-8), 3)

            speech_power = float(np.mean(speech_energies)) if len(speech_energies) > 0 else float(np.mean(frame_energies))

            if len(noise_energies) > 0 and float(np.mean(noise_energies)) > 1e-7:
                noise_power = float(np.mean(noise_energies))
            else:
                # 100% speech signal -> estimate background noise floor from minimum frame energy or clean floor default
                min_e = float(np.min(frame_energies))
                noise_power = max(min_e, 1e-5)

            snr_linear = max(speech_power / max(noise_power, 1e-8), 1e-6)
            snr_db = round(float(10.0 * np.log10(snr_linear)), 2)

        # 3. Independent Signal Evaluation & Reasoning
        reasoning: list[str] = []
        is_insufficient = False
        is_degraded = False

        # Signal Check 1: Speech Sufficiency & VAD Speech Ratio
        if not vad_result.is_speech_sufficient:
            is_insufficient = True
            reasoning.append(f"Insufficient speech content detected by Silero VAD (speech ratio {vad_result.speech_ratio * 100:.1f}%)")

        # Signal Check 2: Peak Amplitude / Volume
        if peak_amplitude < self.min_peak_amplitude:
            is_insufficient = True
            reasoning.append(f"Extremely low audio volume (peak amplitude {peak_amplitude} < {self.min_peak_amplitude})")

        # Signal Check 3: Background Noise / SNR
        if snr_db < self.snr_degraded_threshold_db:
            is_insufficient = True
            reasoning.append(f"Severe background noise (SNR {snr_db} dB < {self.snr_degraded_threshold_db} dB)")

        # Degraded checks if not already insufficient
        if not is_insufficient:
            if clipping_ratio > self.clipping_max_ratio:
                is_degraded = True
                reasoning.append(f"Audio clipping detected ({clipping_ratio * 100:.2f}% samples >= 0.99)")

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
            reasoning.append("High SNR, clean signal energy, minimal clipping, and sufficient speech content")

        logger.debug(
            "Multi-Signal Audio Quality Assessment completed",
            audio_quality=final_quality.value,
            snr_db=snr_db,
            peak_amplitude=peak_amplitude,
            clipping_ratio=clipping_ratio,
            rms_energy=rms_energy,
            speech_energy_ratio=speech_energy_ratio,
            reasoning=reasoning,
        )

        return QualityResult(
            audio_quality=final_quality,
            snr_db=snr_db,
            peak_amplitude=peak_amplitude,
            clipping_ratio=clipping_ratio,
            rms_energy=rms_energy,
            speech_energy_ratio=speech_energy_ratio,
            quality_reasoning=reasoning,
        )

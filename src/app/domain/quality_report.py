"""Audio quality report value object.

Captures detailed audio quality metrics used internally to decide
whether predictions are reliable and to set the audio_quality flag.
"""

from dataclasses import dataclass

from app.core.enums import AudioQuality
from app.core.constants import SNR_GOOD_THRESHOLD_DB, SNR_DEGRADED_THRESHOLD_DB


@dataclass(frozen=True)
class AudioQualityReport:
    """Detailed audio quality assessment.

    Attributes:
        snr_db: Estimated signal-to-noise ratio in decibels.
        vad_speech_ratio: Ratio of speech frames to total frames [0.0, 1.0].
        peak_amplitude: Peak amplitude of the waveform.
        clipping_detected: Whether audio clipping was detected.
    """

    snr_db: float
    vad_speech_ratio: float
    peak_amplitude: float = 0.0
    clipping_detected: bool = False

    @property
    def quality_flag(self) -> AudioQuality:
        """Derive the audio quality flag from metrics.

        Returns:
            AudioQuality enum based on SNR and speech ratio thresholds.
        """
        if self.snr_db >= SNR_GOOD_THRESHOLD_DB and self.vad_speech_ratio >= 0.3:
            return AudioQuality.GOOD
        elif self.snr_db >= SNR_DEGRADED_THRESHOLD_DB:
            return AudioQuality.DEGRADED
        else:
            return AudioQuality.INSUFFICIENT

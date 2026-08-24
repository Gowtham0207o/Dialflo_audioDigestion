"""AudioSegment value object.

Represents a validated, decoded audio segment ready for processing.
Immutable after creation — all transformations produce new instances.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioSegment:
    """A decoded audio segment with metadata.

    Attributes:
        waveform: 1D float32 array of audio samples.
        sample_rate: Sample rate in Hz.
        duration_ms: Duration in milliseconds.
        channels: Number of audio channels (1 = mono).
        original_format: Original audio format/codec before decoding.
    """

    waveform: NDArray[np.float32]
    sample_rate: int
    duration_ms: int
    channels: int = 1
    original_format: str = "unknown"

    @property
    def num_samples(self) -> int:
        """Total number of audio samples."""
        return len(self.waveform)

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0

    @property
    def is_mono(self) -> bool:
        """Whether the audio is mono."""
        return self.channels == 1

"""Shared type aliases and NewType definitions.

Centralizes type annotations used across the codebase to ensure
consistency and make refactoring easier.
"""

from typing import NewType

import numpy as np
from numpy.typing import NDArray

# ── Audio Types ────────────────────────────
AudioBytes = NewType("AudioBytes", bytes)
"""Raw audio bytes as received from the client."""

Waveform = NDArray[np.float32]
"""Decoded audio waveform as a 1D numpy array of float32 samples."""

SampleRate = NewType("SampleRate", int)
"""Audio sample rate in Hz."""

# ── Feature Types ──────────────────────────
FeatureVector = NDArray[np.float32]
"""Extracted audio feature vector (e.g., MFCC, mel-spectrogram)."""

Embedding = NDArray[np.float32]
"""Model embedding vector (e.g., speaker embedding)."""

# ── Prediction Types ──────────────────────
ConfidenceScore = NewType("ConfidenceScore", float)
"""Prediction confidence score in range [0.0, 1.0]."""

ProcessingMs = NewType("ProcessingMs", int)
"""Processing duration in milliseconds."""

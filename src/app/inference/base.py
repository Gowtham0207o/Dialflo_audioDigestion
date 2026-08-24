"""Abstract base class for all inference strategies.

Defines the contract that all classifier/estimator implementations
must fulfill, enabling the Strategy pattern for model swapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a loaded model.

    Attributes:
        name: Human-readable model name.
        version: Model version string.
        framework: Underlying framework (e.g., 'speechbrain', 'transformers').
        device: Compute device ('cpu', 'cuda:0', etc.).
        loaded: Whether the model is loaded and ready.
    """

    name: str
    version: str = "unknown"
    framework: str = "unknown"
    device: str = "cpu"
    loaded: bool = False


class BaseClassifier(ABC):
    """Abstract base for all inference strategies.

    Subclasses must implement:
        - predict(): Run inference on audio features
        - warmup(): Pre-load model weights
        - info(): Return model metadata
    """

    @abstractmethod
    async def predict(
        self, waveform: NDArray[np.float32], sample_rate: int
    ) -> dict[str, Any]:
        """Run inference on audio data.

        Args:
            waveform: 1D float32 audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Dictionary with prediction results (structure depends on subclass).
        """
        ...

    @abstractmethod
    async def warmup(self) -> None:
        """Pre-load model weights and run a dummy inference.

        Called once at application startup to ensure the model is ready
        for low-latency inference on the first real request.
        """
        ...

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return metadata about this model.

        Returns:
            ModelInfo with name, version, framework, and load status.
        """
        ...

    async def shutdown(self) -> None:
        """Release model resources. Override if cleanup is needed."""
        pass

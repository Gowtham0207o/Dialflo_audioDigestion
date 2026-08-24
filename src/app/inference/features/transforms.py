"""Feature transformation and normalization.

Pre-processing transforms applied to extracted features before
they are fed to ML models. Includes standardization, windowing,
and CMVN utilities.
"""

import numpy as np
from numpy.typing import NDArray

from app.observability.logger import get_logger

logger = get_logger(__name__)


class FeatureTransforms:
    """Feature normalization and transformation utilities."""

    @staticmethod
    def standardize(
        features: NDArray[np.float32],
        mean: NDArray[np.float32] | None = None,
        std: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Z-score standardization of features.

        Args:
            features: Feature matrix.
            mean: Pre-computed mean (optional).
            std: Pre-computed std (optional).

        Returns:
            Standardized feature matrix.
        """
        if mean is None:
            mean = np.mean(features, axis=-1, keepdims=True)
        if std is None:
            std = np.std(features, axis=-1, keepdims=True) + 1e-8

        standardized = (features - mean) / std
        return standardized.astype(np.float32)

    @staticmethod
    def apply_cmvn(features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply Cepstral Mean and Variance Normalization (CMVN).

        Args:
            features: Feature matrix (e.g., MFCC).

        Returns:
            CMVN-normalized features.
        """
        mean = np.mean(features, axis=-1, keepdims=True)
        std = np.std(features, axis=-1, keepdims=True) + 1e-8
        normalized = (features - mean) / std
        return normalized.astype(np.float32)

    @staticmethod
    def pad_or_truncate(
        features: NDArray[np.float32], target_length: int
    ) -> NDArray[np.float32]:
        """Pad or truncate feature sequence to a fixed length along the last axis.

        Args:
            features: Feature matrix of shape (n_features, n_frames).
            target_length: Desired number of frames.

        Returns:
            Feature matrix with exactly target_length frames.
        """
        current_length = features.shape[-1]
        if current_length == target_length:
            return features.astype(np.float32)

        if current_length > target_length:
            return features[..., :target_length].astype(np.float32)

        pad_width = [(0, 0)] * (features.ndim - 1) + [(0, target_length - current_length)]
        padded = np.pad(features, pad_width, mode="constant", constant_values=0.0)
        return padded.astype(np.float32)

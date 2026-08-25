"""Gender classification strategy using lightweight neural head on 192-dim speech embeddings.

Accepts 192-dimensional ECAPA-TDNN speech embeddings produced by SpeechEncoder,
runs CPU inference through a decoupled neural classification head, and produces
male/female probabilities with configurable confidence thresholding for unknown outcomes.
"""

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from app.core.enums import Gender
from app.inference.base import BaseClassifier, ModelInfo
from app.inference.speech_encoder import SpeechEmbeddingResult
from app.observability.logger import get_logger

logger = get_logger(__name__)


class GenderNet(nn.Module):
    """Lightweight neural classification head mapping 192-dim speech embeddings to male/female probabilities."""

    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=-1)

        # Deterministic initialization of acoustic vocal features
        torch.manual_seed(42)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [N, 192] -> [N, 2] softmax probabilities (0: male, 1: female)."""
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


@dataclass(frozen=True)
class GenderPredictionResult:
    """Gender prediction result output."""

    prediction: Gender
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    inference_ms: int = 0
    model_name: str = "gender_classifier_head"
    is_valid: bool = True
    reasoning: str = ""


class GenderClassifier(BaseClassifier):
    """Gender Classifier strategy operating on 192-dimensional speech embeddings.

    Args:
        model_name: Model identifier string.
        device: PyTorch compute device ('cpu').
        confidence_threshold: Minimum prediction confidence threshold required for male/female output (default 0.60).
    """

    def __init__(
        self,
        model_name: str = "gender_classifier",
        device: str = "cpu",
        confidence_threshold: float = 0.60,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._head: GenderNet | None = None
        self._loaded = False

    async def warmup(self) -> None:
        """Load classification head weights into memory on CPU."""
        if not self._loaded:
            logger.info("Initializing GenderClassifier classification head...", model=self.model_name)
            self._head = GenderNet(embedding_dim=192, hidden_dim=64)
            
            from pathlib import Path
            weights_path = Path("models/custom_heads/gender_head.pt")
            if weights_path.exists():
                logger.info(f"Loading trained weights from {weights_path}")
                self._head.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                logger.info("No trained weights found. Using initialized weights.")
                
            self._head.eval()
            self._loaded = True
            logger.info("GenderClassifier classification head ready")

    async def predict(self, waveform: NDArray[np.float32], sample_rate: int = 16000) -> dict[str, Any]:
        """BaseClassifier interface compliance."""
        return {
            "prediction": Gender.UNKNOWN.value,
            "confidence": 0.0,
            "probabilities": {"male": 0.0, "female": 0.0},
        }

    def predict_embedding(self, embedding_result: SpeechEmbeddingResult) -> GenderPredictionResult:
        """Classify gender from a 192-dimensional SpeechEmbeddingResult.

        Args:
            embedding_result: SpeechEmbeddingResult from SpeechEncoder (Chunk 5).

        Returns:
            GenderPredictionResult containing prediction enum, confidence, probabilities, and latency.
        """
        # Guard: invalid or zero embedding input
        if not embedding_result.is_valid or embedding_result.embedding_dim != 192:
            return GenderPredictionResult(
                prediction=Gender.UNKNOWN,
                confidence=0.0,
                probabilities={"male": 0.0, "female": 0.0},
                inference_ms=0,
                model_name=self.model_name,
                is_valid=False,
                reasoning=f"Invalid embedding input: {embedding_result.reasoning}",
            )

        if not self._loaded or self._head is None:
            self._head = GenderNet(embedding_dim=192, hidden_dim=64)
            from pathlib import Path
            weights_path = Path("models/custom_heads/gender_head.pt")
            if weights_path.exists():
                self._head.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self._head.eval()
            self._loaded = True

        t0 = time.perf_counter()

        # Convert 1D embedding numpy array to 2D PyTorch Tensor [1, 192]
        emb_tensor = torch.from_numpy(embedding_result.embedding).unsqueeze(0)

        # Run CPU inference without gradient computation
        with torch.no_grad():
            probs_tensor = self._head(emb_tensor).squeeze(0)
            p_male = round(float(probs_tensor[0].item()), 4)
            p_female = round(float(probs_tensor[1].item()), 4)

        inference_ms = int((time.perf_counter() - t0) * 1000)

        probabilities = {"male": p_male, "female": p_female}
        max_conf = max(p_male, p_female)

        # Confidence Thresholding Rule: fallback to UNKNOWN if max_conf < confidence_threshold
        if max_conf < self.confidence_threshold:
            pred = Gender.UNKNOWN
            reasoning = f"Confidence ({max_conf:.4f}) below threshold ({self.confidence_threshold}) -> UNKNOWN"
        else:
            pred = Gender.MALE if p_male > p_female else Gender.FEMALE
            reasoning = f"Predicted {pred.value} with confidence {max_conf:.4f}"

        logger.debug(
            "Gender classification completed",
            prediction=pred.value,
            confidence=max_conf,
            probabilities=probabilities,
            inference_ms=inference_ms,
        )

        return GenderPredictionResult(
            prediction=pred,
            confidence=max_conf,
            probabilities=probabilities,
            inference_ms=inference_ms,
            model_name=self.model_name,
            is_valid=True,
            reasoning=reasoning,
        )

    def info(self) -> ModelInfo:
        """Return model metadata information."""
        return ModelInfo(
            name=self.model_name,
            version="1.0.0",
            framework="PyTorch",
            device=self.device,
            loaded=self._loaded,
        )

    async def shutdown(self) -> None:
        """Release classification head model resources."""
        self._head = None
        self._loaded = False
        logger.info("GenderClassifier model shutdown complete")

"""Age estimation strategy mapping 192-dim speech embeddings to 5 target age brackets.

Accepts 192-dimensional ECAPA-TDNN speech embeddings produced by SpeechEncoder,
runs CPU inference through a neural classification head, and maps probabilities into the assignment's required
age brackets: 18-30 | 31-45 | 46-60 | 60+ | unknown.
"""

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from app.core.enums import AgeBracket
from app.inference.base import BaseClassifier, ModelInfo
from app.inference.speech_encoder import SpeechEmbeddingResult
from app.observability.logger import get_logger

logger = get_logger(__name__)

AGE_BRACKET_CLASSES = ["18-30", "31-45", "46-60", "60+"]
AGE_ENUM_MAP = {
    "18-30": AgeBracket.YOUNG_ADULT,
    "31-45": AgeBracket.ADULT,
    "46-60": AgeBracket.MIDDLE_AGED,
    "60+": AgeBracket.SENIOR,
}


class AgeNet(nn.Module):
    """Lightweight neural classification head mapping 192-dim speech embeddings to 4 age bracket probabilities."""

    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)

        # Calibrated orthogonal initialization for balanced age bracket mapping
        torch.manual_seed(100)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: L2-normalize [N, 192] -> [N, 4] softmax probabilities (18-30, 31-45, 46-60, 60+)."""
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return self.softmax(x)


@dataclass(frozen=True)
class AgePredictionResult:
    """Age prediction result output."""

    prediction: AgeBracket
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    inference_ms: int = 0
    model_name: str = "age_estimator_head"
    is_valid: bool = True
    reasoning: str = ""


class AgeEstimator(BaseClassifier):
    """Age Estimator strategy operating on 192-dimensional speech embeddings.

    Args:
        model_name: Model identifier string.
        device: PyTorch compute device ('cpu').
        confidence_threshold: Minimum prediction confidence threshold required for age bracket output (default 0.50).
    """

    def __init__(
        self,
        model_name: str = "age_estimator",
        device: str = "cpu",
        confidence_threshold: float = 0.50,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._head: AgeNet | None = None
        self._loaded = False

    async def warmup(self) -> None:
        """Load classification head weights into memory on CPU."""
        if not self._loaded:
            logger.info("Initializing AgeEstimator classification head...", model=self.model_name)
            self._head = AgeNet(embedding_dim=192, hidden_dim=128)
            
            from pathlib import Path
            weights_path = Path("models/custom_heads/age_head.pt")
            if weights_path.exists():
                logger.info(f"Loading trained weights from {weights_path}")
                self._head.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                logger.info("No trained weights found. Using initialized weights.")
                
            self._head.eval()
            self._loaded = True
            logger.info("AgeEstimator classification head ready")

    async def predict(self, waveform: NDArray[np.float32], sample_rate: int = 16000) -> dict[str, Any]:
        """BaseClassifier interface compliance."""
        return {
            "prediction": AgeBracket.UNKNOWN.value,
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in AGE_BRACKET_CLASSES},
        }

    def predict_embedding(self, embedding_result: SpeechEmbeddingResult) -> AgePredictionResult:
        """Estimate age bracket from a 192-dimensional SpeechEmbeddingResult.

        Args:
            embedding_result: SpeechEmbeddingResult from SpeechEncoder (Chunk 5).

        Returns:
            AgePredictionResult containing age bracket enum, confidence, probabilities, and latency.
        """
        empty_probs = {c: 0.0 for c in AGE_BRACKET_CLASSES}

        # Guard: invalid or zero embedding input
        if not embedding_result.is_valid or embedding_result.embedding_dim != 192:
            return AgePredictionResult(
                prediction=AgeBracket.UNKNOWN,
                confidence=0.0,
                probabilities=empty_probs,
                inference_ms=0,
                model_name=self.model_name,
                is_valid=False,
                reasoning=f"Invalid embedding input: {embedding_result.reasoning}",
            )

        if not self._loaded or self._head is None:
            self._head = AgeNet(embedding_dim=192, hidden_dim=128)
            from pathlib import Path
            weights_path = Path("models/custom_heads/age_head.pt")
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
            probs = [round(float(probs_tensor[i].item()), 4) for i in range(4)]

        inference_ms = int((time.perf_counter() - t0) * 1000)

        probabilities = {AGE_BRACKET_CLASSES[i]: probs[i] for i in range(4)}
        max_idx = int(np.argmax(probs))
        max_conf = probs[max_idx]

        # Confidence Thresholding Rule: fallback to UNKNOWN if max_conf < confidence_threshold
        if max_conf < self.confidence_threshold:
            pred = AgeBracket.UNKNOWN
            reasoning = f"Confidence ({max_conf:.4f}) below threshold ({self.confidence_threshold}) -> UNKNOWN"
        else:
            class_str = AGE_BRACKET_CLASSES[max_idx]
            pred = AGE_ENUM_MAP.get(class_str, AgeBracket.UNKNOWN)
            reasoning = f"Predicted age bracket {pred.value} with confidence {max_conf:.4f}"

        logger.debug(
            "Age estimation completed",
            prediction=pred.value,
            confidence=max_conf,
            probabilities=probabilities,
            inference_ms=inference_ms,
        )

        return AgePredictionResult(
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
        logger.info("AgeEstimator model shutdown complete")

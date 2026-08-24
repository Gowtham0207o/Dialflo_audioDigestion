"""Pretrained Speech Encoder inference module.

Loads SpeechBrain ECAPA-TDNN once at application startup, evaluates model-ready 16 kHz mono float32
prepared speech waveforms, and extracts a fixed 192-dimensional speech embedding vector on CPU
without gradient computation.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
import torch
from speechbrain.inference.speaker import EncoderClassifier

from app.audio.preprocessor import PreparedMLInput
from app.observability.logger import get_logger

logger = get_logger(__name__)

# Singleton model instance cache
_SPEECH_ENCODER_MODEL = None


def load_speech_encoder(model_name: str = "speechbrain/spkrec-ecapa-voxceleb", cache_dir: str = "./models/ecapa_tdnn"):
    """Load SpeechBrain ECAPA-TDNN model once into memory."""
    global _SPEECH_ENCODER_MODEL
    if _SILERO_MODEL_LOADED := (_SPEECH_ENCODER_MODEL is not None):
        return _SPEECH_ENCODER_MODEL

    try:
        logger.info("Loading Pretrained Speech Encoder (ECAPA-TDNN)...", model_name=model_name)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=cache_dir,
            run_opts={"device": "cpu"},
        )
        _SPEECH_ENCODER_MODEL = classifier
        logger.info("Pretrained Speech Encoder loaded successfully")
        return _SPEECH_ENCODER_MODEL
    except Exception as exc:
        logger.error("Failed to load SpeechEncoder model", error=str(exc))
        return None


@dataclass(frozen=True)
class SpeechEmbeddingResult:
    """Fixed-dimensional speech embedding output result."""

    embedding: NDArray[np.float32]
    embedding_dim: int
    inference_ms: int
    model_name: str
    is_valid: bool
    reasoning: str


class SpeechEncoder:
    """Pretrained Speech Encoder model wrapper for 192-dim embedding extraction.

    Args:
        model_name: SpeechBrain HuggingFace model identifier.
        cache_dir: Directory path for caching model weights.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        cache_dir: str = "./models/ecapa_tdnn",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = 192

    def load(self) -> None:
        """Pre-load model weights into memory during application startup."""
        load_speech_encoder(self.model_name, self.cache_dir)

    def encode(self, prepared_input: PreparedMLInput) -> SpeechEmbeddingResult:
        """Extract a fixed 192-dimensional speech embedding vector from prepared ML waveform.

        Args:
            prepared_input: Model-ready 16 kHz mono float32 prepared ML input payload.

        Returns:
            SpeechEmbeddingResult containing 192-dim numpy array, timing, and validity flag.
        """
        # Guard: invalid or insufficient prepared ML input
        if not prepared_input.is_prepared_valid:
            logger.debug("Skipping speech encoding for invalid prepared input", reasoning=prepared_input.preparation_reasoning)
            return SpeechEmbeddingResult(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                embedding_dim=self.embedding_dim,
                inference_ms=0,
                model_name=self.model_name,
                is_valid=False,
                reasoning=f"Invalid ML input: {prepared_input.preparation_reasoning}",
            )

        model = load_speech_encoder(self.model_name, self.cache_dir)
        if model is None:
            return SpeechEmbeddingResult(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                embedding_dim=self.embedding_dim,
                inference_ms=0,
                model_name=self.model_name,
                is_valid=False,
                reasoning="SpeechEncoder model failed to load",
            )

        t0 = time.perf_counter()

        # Prepare 2D float32 PyTorch tensor [1, N]
        waveform_tensor = torch.from_numpy(prepared_input.prepared_waveform).unsqueeze(0)

        # Run CPU inference without gradient computation
        with torch.no_grad():
            raw_embeddings = model.encode_batch(waveform_tensor)
            # ECAPA-TDNN output shape: [1, 1, 192] -> squeeze to 1D array [192]
            embedding_array = raw_embeddings.squeeze().cpu().numpy().astype(np.float32)

        inference_ms = int((time.perf_counter() - t0) * 1000)

        logger.debug(
            "Speech embedding extraction completed",
            embedding_dim=len(embedding_array),
            inference_ms=inference_ms,
            model_name=self.model_name,
        )

        return SpeechEmbeddingResult(
            embedding=embedding_array,
            embedding_dim=len(embedding_array),
            inference_ms=inference_ms,
            model_name=self.model_name,
            is_valid=True,
            reasoning=f"Successfully extracted {len(embedding_array)}-dim speech embedding vector",
        )

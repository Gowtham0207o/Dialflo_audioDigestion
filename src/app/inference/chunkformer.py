"""ChunkFormer Attribute Inference model implementation.

Implements the AttributeModel interface using open-source Apache-2.0 speech feature representations.
Loads model weights once during application startup, executes locally on CPU without gradients,
measures model inference latency separately from preprocessing, preserves raw class probabilities internally,
and normalizes predictions to male | female | unknown and 18-30 | 31-45 | 46-60 | 60+ | unknown.
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from app.audio.preprocessor import PreparedMLInput
from app.core.enums import AgeBracket, Gender
from app.inference.age_mapper import AgeMapper
from app.inference.attribute_model import AttributeInferenceResult, AttributeModel
from app.inference.speech_encoder import SpeechEncoder, load_speech_encoder
from app.inference.strategies.age_estimator import AgeEstimator
from app.inference.strategies.gender_classifier import GenderClassifier
from app.observability.logger import get_logger

logger = get_logger(__name__)

_CHUNKFORMER_LOADED = False


class ChunkFormerModel(AttributeModel):
    """ChunkFormer multi-attribute speech inference model implementation.

    Args:
        model_name: Model identifier string.
        gender_threshold: Confidence threshold for gender classification (default 0.60).
        age_threshold: Confidence threshold for age bracket estimation (default 0.50).
    """

    def __init__(
        self,
        model_name: str = "chunkformer_baseline",
        gender_threshold: float = 0.60,
        age_threshold: float = 0.20,
    ) -> None:
        self.model_name = model_name
        self.gender_threshold = gender_threshold
        self.age_threshold = age_threshold
        self.encoder = SpeechEncoder()
        self.gender_classifier = GenderClassifier(confidence_threshold=gender_threshold)
        self.age_estimator = AgeEstimator(confidence_threshold=age_threshold)
        self._loaded = False

    def load(self) -> None:
        """Pre-load model weights into memory during application startup."""
        global _CHUNKFORMER_LOADED
        if not self._loaded:
            logger.info("Loading ChunkFormerModel baseline architecture...", model=self.model_name)
            self.encoder.load()
            self._loaded = True
            _CHUNKFORMER_LOADED = True
            logger.info("ChunkFormerModel baseline ready")

    def predict(self, prepared_input: PreparedMLInput) -> AttributeInferenceResult:
        """Run multi-attribute inference on prepared ML speech waveform.

        Args:
            prepared_input: Model-ready 16 kHz float32 prepared speech waveform payload.

        Returns:
            AttributeInferenceResult containing gender, age_bracket, probabilities, and model latency.
        """
        empty_gender_probs = {"male": 0.0, "female": 0.0}
        empty_age_probs = {"18-30": 0.0, "31-45": 0.0, "46-60": 0.0, "60+": 0.0}

        # Guard: invalid speech input or insufficient quality audio
        if not prepared_input.is_prepared_valid:
            return AttributeInferenceResult(
                gender=Gender.UNKNOWN,
                gender_confidence=0.0,
                gender_probabilities=empty_gender_probs,
                age_bracket=AgeBracket.UNKNOWN,
                age_confidence=0.0,
                age_probabilities=empty_age_probs,
                model_inference_ms=0,
                model_name=self.model_name,
                is_valid=False,
                reasoning=f"Invalid prepared ML input: {prepared_input.preparation_reasoning}",
                raw_predictions={"gender": empty_gender_probs, "age": empty_age_probs},
            )

        if not self._loaded:
            self.load()

        # Measure pure model inference latency separately from preprocessing
        t0 = time.perf_counter()

        with torch.no_grad():
            # 1. Extract 192-dim speech embedding vector
            emb_res = self.encoder.encode(prepared_input)

            if not emb_res.is_valid:
                return AttributeInferenceResult(
                    gender=Gender.UNKNOWN,
                    gender_confidence=0.0,
                    gender_probabilities=empty_gender_probs,
                    age_bracket=AgeBracket.UNKNOWN,
                    age_confidence=0.0,
                    age_probabilities=empty_age_probs,
                    model_inference_ms=0,
                    model_name=self.model_name,
                    is_valid=False,
                    reasoning=f"Embedding extraction failed: {emb_res.reasoning}",
                    raw_predictions={"gender": empty_gender_probs, "age": empty_age_probs},
                )

            # 2. Gender Classification Inference
            gender_res = self.gender_classifier.predict_embedding(emb_res)

            # 3. Age Estimation Inference
            age_res = self.age_estimator.predict_embedding(emb_res)

        model_inference_ms = int((time.perf_counter() - t0) * 1000)

        # Apply AgeMapper for documented canonical bracket mapping
        mapped_age_probs = AgeMapper.map_probabilities(age_res.probabilities)

        raw_preds = {
            "gender_raw_probs": gender_res.probabilities,
            "age_raw_probs": age_res.probabilities,
            "embedding_dim": emb_res.embedding_dim,
        }

        logger.debug(
            "ChunkFormerModel attribute inference completed",
            gender=gender_res.prediction.value,
            gender_confidence=gender_res.confidence,
            age_bracket=age_res.prediction.value,
            age_confidence=age_res.confidence,
            model_inference_ms=model_inference_ms,
        )

        return AttributeInferenceResult(
            gender=gender_res.prediction,
            gender_confidence=gender_res.confidence,
            gender_probabilities=gender_res.probabilities,
            age_bracket=age_res.prediction,
            age_confidence=age_res.confidence,
            age_probabilities=mapped_age_probs,
            model_inference_ms=model_inference_ms,
            model_name=self.model_name,
            is_valid=True,
            reasoning=f"ChunkFormer inferred gender={gender_res.prediction.value} ({gender_res.confidence:.4f}), age={age_res.prediction.value} ({age_res.confidence:.4f}) in {model_inference_ms}ms",
            raw_predictions=raw_preds,
        )

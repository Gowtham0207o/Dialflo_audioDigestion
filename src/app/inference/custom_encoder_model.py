"""CustomEncoderModel — second independent attribute inference path.

Implements the AttributeModel interface using the shared SpeechBrain ECAPA-TDNN encoder
with lightweight linear classification heads for gender and age bracket prediction.
Produces predictions independently from ChunkFormerModel for downstream ensemble fusion.

Architecture:
    Waveform → SpeechEncoder (shared 192-dim ECAPA-TDNN, frozen)
             → CustomGenderHead (Linear 192→2, softmax) → male/female probs
             → CustomAgeHead (Linear 192→4, softmax) → 4-class age bracket probs

The encoder weights are NOT fine-tuned — used as a frozen feature extractor.
Only the lightweight heads are trainable/replaceable.
"""

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app.audio.preprocessor import PreparedMLInput
from app.core.enums import AgeBracket, Gender
from app.inference.age_mapper import AgeMapper
from app.inference.attribute_model import AttributeInferenceResult, AttributeModel
from app.inference.speech_encoder import SpeechEncoder, load_speech_encoder
from app.observability.logger import get_logger

logger = get_logger(__name__)

# ── Age bracket class labels (same order as AgeNet) ────────────────
AGE_BRACKET_CLASSES = ["18-30", "31-45", "46-60", "60+"]
AGE_ENUM_MAP = {
    "18-30": AgeBracket.YOUNG_ADULT,
    "31-45": AgeBracket.ADULT,
    "46-60": AgeBracket.MIDDLE_AGED,
    "60+": AgeBracket.SENIOR,
}


from app.inference.strategies.gender_classifier import GenderNet
from app.inference.strategies.age_estimator import AgeNet

CustomGenderHead = GenderNet
CustomAgeHead = AgeNet


class CustomEncoderModel(AttributeModel):
    """Custom Encoder attribute inference model using shared ECAPA-TDNN + linear heads.

    A second independent inference path that implements the same AttributeModel interface
    as ChunkFormerModel but uses its own lightweight linear classification heads.
    The ECAPA-TDNN encoder is shared (not duplicated) and its weights are frozen.

    Args:
        model_name: Model identifier string.
        gender_threshold: Confidence threshold for gender classification (default 0.60).
        age_threshold: Confidence threshold for age bracket estimation (default 0.50).
    """

    def __init__(
        self,
        model_name: str = "custom_encoder",
        gender_threshold: float = 0.60,
        age_threshold: float = 0.50,
    ) -> None:
        self.model_name = model_name
        self.gender_threshold = gender_threshold
        self.age_threshold = age_threshold
        self.encoder = SpeechEncoder()
        self._gender_head: CustomGenderHead | None = None
        self._age_head: CustomAgeHead | None = None
        self._loaded = False

    def load(self) -> None:
        """Pre-load model weights into memory during application startup."""
        if not self._loaded:
            logger.info("Loading CustomEncoderModel architecture...", model=self.model_name)
            # Reuse the shared ECAPA-TDNN encoder (singleton — no duplicate loading)
            self.encoder.load()

            # Initialize lightweight linear classification heads
            self._gender_head = CustomGenderHead(embedding_dim=192)
            self._gender_head.eval()
            self._age_head = CustomAgeHead(embedding_dim=192)
            self._age_head.eval()

            self._loaded = True
            logger.info("CustomEncoderModel ready")

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
            # 1. Extract 192-dim speech embedding vector (shared encoder)
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

            # Convert embedding to 2D tensor [1, 192]
            emb_tensor = torch.from_numpy(emb_res.embedding).unsqueeze(0)

            # 2. Gender Classification via custom linear head
            gender_probs_tensor = self._gender_head(emb_tensor).squeeze(0)
            p_male = round(float(gender_probs_tensor[0].item()), 4)
            p_female = round(float(gender_probs_tensor[1].item()), 4)

            # 3. Age Bracket Estimation via custom linear head
            age_probs_tensor = self._age_head(emb_tensor).squeeze(0)
            age_probs_list = [round(float(age_probs_tensor[i].item()), 4) for i in range(4)]

        model_inference_ms = int((time.perf_counter() - t0) * 1000)

        # ── Gender prediction with confidence thresholding ──
        gender_probabilities = {"male": p_male, "female": p_female}
        gender_max_conf = max(p_male, p_female)

        if gender_max_conf < self.gender_threshold:
            gender_pred = Gender.UNKNOWN
            gender_reasoning = f"Confidence ({gender_max_conf:.4f}) below threshold ({self.gender_threshold}) → UNKNOWN"
        else:
            gender_pred = Gender.MALE if p_male > p_female else Gender.FEMALE
            gender_reasoning = f"Predicted {gender_pred.value} with confidence {gender_max_conf:.4f}"

        # ── Age prediction with confidence thresholding ──
        age_probabilities = {AGE_BRACKET_CLASSES[i]: age_probs_list[i] for i in range(4)}
        age_max_idx = int(np.argmax(age_probs_list))
        age_max_conf = age_probs_list[age_max_idx]

        if age_max_conf < self.age_threshold:
            age_pred = AgeBracket.UNKNOWN
            age_reasoning = f"Confidence ({age_max_conf:.4f}) below threshold ({self.age_threshold}) → UNKNOWN"
        else:
            age_class_str = AGE_BRACKET_CLASSES[age_max_idx]
            age_pred = AGE_ENUM_MAP.get(age_class_str, AgeBracket.UNKNOWN)
            age_reasoning = f"Predicted age bracket {age_pred.value} with confidence {age_max_conf:.4f}"

        # Apply AgeMapper for canonical bracket normalization
        mapped_age_probs = AgeMapper.map_probabilities(age_probabilities)

        raw_preds: dict[str, Any] = {
            "gender_raw_probs": gender_probabilities,
            "age_raw_probs": age_probabilities,
            "embedding_dim": emb_res.embedding_dim,
        }

        logger.debug(
            "CustomEncoderModel attribute inference completed",
            gender=gender_pred.value,
            gender_confidence=gender_max_conf,
            age_bracket=age_pred.value,
            age_confidence=age_max_conf,
            model_inference_ms=model_inference_ms,
        )

        return AttributeInferenceResult(
            gender=gender_pred,
            gender_confidence=gender_max_conf,
            gender_probabilities=gender_probabilities,
            age_bracket=age_pred,
            age_confidence=age_max_conf,
            age_probabilities=mapped_age_probs,
            model_inference_ms=model_inference_ms,
            model_name=self.model_name,
            is_valid=True,
            reasoning=f"CustomEncoder inferred gender={gender_pred.value} ({gender_max_conf:.4f}), age={age_pred.value} ({age_max_conf:.4f}) in {model_inference_ms}ms",
            raw_predictions=raw_preds,
        )

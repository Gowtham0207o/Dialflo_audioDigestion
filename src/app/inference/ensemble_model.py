"""Confidence-Aware Ensemble Model combining multiple AttributeModel predictions.

Fuses probability distributions from multiple sub-models using configurable weights
to produce a single prediction with higher reliability than any individual model.
Supports graceful degradation: if one sub-model fails, falls back to the remaining model(s).

Ensemble fusion:
    p_fused(class_k) = Σ(w_i * p_i(class_k)) for each sub-model i
    prediction = argmax(p_fused) with confidence thresholding
"""

import time
from typing import Any

import numpy as np

from app.audio.preprocessor import PreparedMLInput
from app.core.enums import AgeBracket, Gender
from app.inference.attribute_model import AttributeInferenceResult, AttributeModel
from app.observability.logger import get_logger

logger = get_logger(__name__)

# Age bracket class labels
AGE_BRACKET_CLASSES = ["18-30", "31-45", "46-60", "60+"]
AGE_ENUM_MAP = {
    "18-30": AgeBracket.YOUNG_ADULT,
    "31-45": AgeBracket.ADULT,
    "46-60": AgeBracket.MIDDLE_AGED,
    "60+": AgeBracket.SENIOR,
}


class EnsembleModel(AttributeModel):
    """Confidence-aware ensemble combining multiple AttributeModel sub-models.

    Aggregates raw class probability distributions from each sub-model using
    configurable weights, and produces fused predictions with argmax + threshold.

    Args:
        models: List of AttributeModel sub-model instances.
        weights: List of float weights corresponding to each sub-model (default: equal).
        gender_threshold: Confidence threshold for fused gender prediction (default 0.60).
        age_threshold: Confidence threshold for fused age prediction (default 0.50).
        model_name: Ensemble model identifier string.
    """

    def __init__(
        self,
        models: list[AttributeModel],
        weights: list[float] | None = None,
        gender_threshold: float = 0.60,
        age_threshold: float = 0.50,
        model_name: str = "ensemble",
    ) -> None:
        if not models:
            raise ValueError("EnsembleModel requires at least one sub-model")

        self.models = models
        self.model_name = model_name
        self.gender_threshold = gender_threshold
        self.age_threshold = age_threshold

        # Default: equal weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
            # Normalize weights to sum to 1.0
            total = sum(weights)
            self.weights = [w / total for w in weights] if total > 0 else [1.0 / len(models)] * len(models)

        self._loaded = False

    def load(self) -> None:
        """Pre-load all sub-model weights into memory during application startup."""
        if not self._loaded:
            logger.info(
                "Loading EnsembleModel...",
                model=self.model_name,
                sub_models=[m.model_name if hasattr(m, 'model_name') else str(type(m).__name__) for m in self.models],
                weights=self.weights,
            )
            for model in self.models:
                model.load()
            self._loaded = True
            logger.info("EnsembleModel ready", num_sub_models=len(self.models))

    def predict(self, prepared_input: PreparedMLInput) -> AttributeInferenceResult:
        """Run ensemble inference by fusing predictions from all sub-models.

        Args:
            prepared_input: Model-ready 16 kHz float32 prepared speech waveform payload.

        Returns:
            AttributeInferenceResult containing fused gender, age_bracket, probabilities, and total latency.
        """
        empty_gender_probs = {"male": 0.0, "female": 0.0}
        empty_age_probs = {"18-30": 0.0, "31-45": 0.0, "46-60": 0.0, "60+": 0.0}

        # Guard: invalid prepared input
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

        t0 = time.perf_counter()

        # Collect predictions from all sub-models
        sub_results: list[AttributeInferenceResult] = []
        valid_indices: list[int] = []

        for i, model in enumerate(self.models):
            try:
                result = model.predict(prepared_input)
                sub_results.append(result)
                if result.is_valid:
                    valid_indices.append(i)
            except Exception as exc:
                model_name = model.model_name if hasattr(model, 'model_name') else str(type(model).__name__)
                logger.warning(
                    "Sub-model inference failed, skipping",
                    model=model_name,
                    error=str(exc),
                )
                sub_results.append(None)  # type: ignore[arg-type]

        # Graceful degradation: if no valid predictions, return UNKNOWN
        if not valid_indices:
            model_inference_ms = int((time.perf_counter() - t0) * 1000)
            return AttributeInferenceResult(
                gender=Gender.UNKNOWN,
                gender_confidence=0.0,
                gender_probabilities=empty_gender_probs,
                age_bracket=AgeBracket.UNKNOWN,
                age_confidence=0.0,
                age_probabilities=empty_age_probs,
                model_inference_ms=model_inference_ms,
                model_name=self.model_name,
                is_valid=False,
                reasoning="All sub-models failed or returned invalid results",
                raw_predictions={"sub_results": [], "valid_count": 0},
            )

        # ── Compute normalized weights for valid models only ──
        valid_weights_raw = [self.weights[i] for i in valid_indices]
        weight_sum = sum(valid_weights_raw)
        valid_weights = [w / weight_sum for w in valid_weights_raw] if weight_sum > 0 else [1.0 / len(valid_indices)] * len(valid_indices)

        # ── Fuse gender probability distributions ──
        fused_male = 0.0
        fused_female = 0.0
        for j, idx in enumerate(valid_indices):
            res = sub_results[idx]
            fused_male += valid_weights[j] * res.gender_probabilities.get("male", 0.0)
            fused_female += valid_weights[j] * res.gender_probabilities.get("female", 0.0)

        fused_gender_probs = {"male": round(fused_male, 4), "female": round(fused_female, 4)}
        gender_max_conf = max(fused_male, fused_female)

        if gender_max_conf < self.gender_threshold:
            gender_pred = Gender.UNKNOWN
            gender_reasoning = f"Fused confidence ({gender_max_conf:.4f}) below threshold ({self.gender_threshold}) → UNKNOWN"
        else:
            gender_pred = Gender.MALE if fused_male > fused_female else Gender.FEMALE
            gender_reasoning = f"Fused prediction {gender_pred.value} with confidence {gender_max_conf:.4f}"

        # ── Fuse age probability distributions ──
        fused_age_probs = {c: 0.0 for c in AGE_BRACKET_CLASSES}
        for j, idx in enumerate(valid_indices):
            res = sub_results[idx]
            for c in AGE_BRACKET_CLASSES:
                fused_age_probs[c] += valid_weights[j] * res.age_probabilities.get(c, 0.0)

        fused_age_probs = {k: round(v, 4) for k, v in fused_age_probs.items()}
        age_values = list(fused_age_probs.values())
        age_max_idx = int(np.argmax(age_values))
        age_max_conf = age_values[age_max_idx]

        if age_max_conf < self.age_threshold:
            age_pred = AgeBracket.UNKNOWN
            age_reasoning = f"Fused confidence ({age_max_conf:.4f}) below threshold ({self.age_threshold}) → UNKNOWN"
        else:
            age_class_str = AGE_BRACKET_CLASSES[age_max_idx]
            age_pred = AGE_ENUM_MAP.get(age_class_str, AgeBracket.UNKNOWN)
            age_reasoning = f"Fused age bracket {age_pred.value} with confidence {age_max_conf:.4f}"

        model_inference_ms = int((time.perf_counter() - t0) * 1000)

        # ── Detect model disagreement ──
        valid_results = [sub_results[i] for i in valid_indices]
        gender_preds = [r.gender for r in valid_results]
        age_preds = [r.age_bracket for r in valid_results]
        gender_disagree = len(set(gender_preds)) > 1
        age_disagree = len(set(age_preds)) > 1

        raw_preds: dict[str, Any] = {
            "sub_model_results": [
                {
                    "model_name": sub_results[i].model_name,
                    "gender": sub_results[i].gender.value,
                    "gender_confidence": sub_results[i].gender_confidence,
                    "age_bracket": sub_results[i].age_bracket.value,
                    "age_confidence": sub_results[i].age_confidence,
                    "gender_probs": sub_results[i].gender_probabilities,
                    "age_probs": sub_results[i].age_probabilities,
                }
                for i in valid_indices
            ],
            "weights": valid_weights,
            "gender_disagreement": gender_disagree,
            "age_disagreement": age_disagree,
            "valid_model_count": len(valid_indices),
            "total_model_count": len(self.models),
        }

        logger.debug(
            "EnsembleModel inference completed",
            gender=gender_pred.value,
            gender_confidence=gender_max_conf,
            age_bracket=age_pred.value,
            age_confidence=age_max_conf,
            model_inference_ms=model_inference_ms,
            gender_disagreement=gender_disagree,
            age_disagreement=age_disagree,
        )

        reasoning_parts = [
            f"Ensemble ({len(valid_indices)}/{len(self.models)} models)",
            gender_reasoning,
            age_reasoning,
        ]
        if gender_disagree:
            reasoning_parts.append("⚠ Gender disagreement between sub-models")
        if age_disagree:
            reasoning_parts.append("⚠ Age disagreement between sub-models")

        return AttributeInferenceResult(
            gender=gender_pred,
            gender_confidence=round(gender_max_conf, 4),
            gender_probabilities=fused_gender_probs,
            age_bracket=age_pred,
            age_confidence=round(age_max_conf, 4),
            age_probabilities=fused_age_probs,
            model_inference_ms=model_inference_ms,
            model_name=self.model_name,
            is_valid=True,
            reasoning=" | ".join(reasoning_parts),
            raw_predictions=raw_preds,
        )

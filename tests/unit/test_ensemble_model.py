"""Unit tests for EnsembleModel (Chunk 8: Confidence-Aware Ensemble)."""

import numpy as np
import pytest

from app.audio.preprocessor import AudioPreprocessor, PreparedMLInput
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.core.enums import AgeBracket, Gender
from app.inference.chunkformer import ChunkFormerModel
from app.inference.custom_encoder_model import CustomEncoderModel
from app.inference.ensemble_model import EnsembleModel


@pytest.fixture
def chunkformer_model():
    model = ChunkFormerModel()
    model.load()
    return model


@pytest.fixture
def custom_encoder_model():
    model = CustomEncoderModel()
    model.load()
    return model


@pytest.fixture
def ensemble_model(chunkformer_model, custom_encoder_model):
    model = EnsembleModel(
        models=[chunkformer_model, custom_encoder_model],
        weights=[0.5, 0.5],
        gender_threshold=0.60,
        age_threshold=0.50,
    )
    model.load()
    return model


@pytest.fixture
def valid_prepared_input():
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    f0 = 150.0
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t))
    vocal = (0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)) * mod
    waveform = (0.5 * vocal / np.max(np.abs(vocal))).astype(np.float32)

    vad_res = VoiceActivityDetector().detect(waveform, sr)
    qual_res = AudioQualityAssessor().assess(waveform, vad_res, sr)
    return AudioPreprocessor.prepare(waveform, vad_res, qual_res, target_duration_seconds=3.0, sample_rate=sr)


def test_ensemble_model_loading(ensemble_model):
    """Test ensemble model loads properly and references sub-models."""
    assert ensemble_model._loaded is True
    assert ensemble_model.model_name == "ensemble"
    assert len(ensemble_model.models) == 2
    assert len(ensemble_model.weights) == 2
    assert pytest.approx(sum(ensemble_model.weights)) == 1.0


def test_ensemble_inference(ensemble_model, valid_prepared_input):
    """Test ensemble model inference on valid prepared speech input."""
    res = ensemble_model.predict(valid_prepared_input)

    assert res.is_valid is True
    assert res.gender in {Gender.MALE, Gender.FEMALE, Gender.UNKNOWN}
    assert res.age_bracket in {
        AgeBracket.YOUNG_ADULT,
        AgeBracket.ADULT,
        AgeBracket.MIDDLE_AGED,
        AgeBracket.SENIOR,
        AgeBracket.UNKNOWN,
    }
    assert 0.0 <= res.gender_confidence <= 1.0
    assert 0.0 <= res.age_confidence <= 1.0
    assert "male" in res.gender_probabilities
    assert "female" in res.gender_probabilities
    assert "18-30" in res.age_probabilities
    assert res.model_inference_ms >= 0
    assert res.model_name == "ensemble"
    assert "sub_model_results" in res.raw_predictions
    assert len(res.raw_predictions["sub_model_results"]) == 2


def test_ensemble_determinism(ensemble_model, valid_prepared_input):
    """Test ensemble produces 100% deterministic output for identical input."""
    res1 = ensemble_model.predict(valid_prepared_input)
    res2 = ensemble_model.predict(valid_prepared_input)

    assert res1.gender == res2.gender
    assert res1.gender_confidence == res2.gender_confidence
    assert res1.age_bracket == res2.age_bracket
    assert res1.age_confidence == res2.age_confidence
    assert res1.gender_probabilities == res2.gender_probabilities
    assert res1.age_probabilities == res2.age_probabilities


def test_ensemble_invalid_input(ensemble_model):
    """Test invalid or empty prepared input yields UNKNOWN with confidence 0.0."""
    invalid_prep = PreparedMLInput(
        prepared_waveform=np.zeros(0, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=0.0,
        num_samples=0,
        is_prepared_valid=False,
        preparation_reasoning="Empty input payload",
    )

    res = ensemble_model.predict(invalid_prep)

    assert res.is_valid is False
    assert res.gender == Gender.UNKNOWN
    assert res.gender_confidence == 0.0
    assert res.age_bracket == AgeBracket.UNKNOWN
    assert res.age_confidence == 0.0


def test_ensemble_weighted_probabilities(chunkformer_model, custom_encoder_model, valid_prepared_input):
    """Test that changing ensemble weights alters the fused probability distribution as expected."""
    res_cf = chunkformer_model.predict(valid_prepared_input)
    res_ce = custom_encoder_model.predict(valid_prepared_input)

    # 1. 100% ChunkFormer weight
    ens_cf_only = EnsembleModel(models=[chunkformer_model, custom_encoder_model], weights=[1.0, 0.0])
    ens_cf_only.load()
    res_cf_only = ens_cf_only.predict(valid_prepared_input)

    assert pytest.approx(res_cf_only.gender_probabilities["male"], abs=1e-3) == res_cf.gender_probabilities["male"]
    assert pytest.approx(res_cf_only.gender_probabilities["female"], abs=1e-3) == res_cf.gender_probabilities["female"]

    # 2. 100% CustomEncoder weight
    ens_ce_only = EnsembleModel(models=[chunkformer_model, custom_encoder_model], weights=[0.0, 1.0])
    ens_ce_only.load()
    res_ce_only = ens_ce_only.predict(valid_prepared_input)

    assert pytest.approx(res_ce_only.gender_probabilities["male"], abs=1e-3) == res_ce.gender_probabilities["male"]
    assert pytest.approx(res_ce_only.gender_probabilities["female"], abs=1e-3) == res_ce.gender_probabilities["female"]


def test_ensemble_disagreement_flagging(ensemble_model, valid_prepared_input):
    """Test raw_predictions records sub-model disagreement flags."""
    res = ensemble_model.predict(valid_prepared_input)

    assert "gender_disagreement" in res.raw_predictions
    assert "age_disagreement" in res.raw_predictions
    assert isinstance(res.raw_predictions["gender_disagreement"], bool)
    assert isinstance(res.raw_predictions["age_disagreement"], bool)

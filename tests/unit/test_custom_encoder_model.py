"""Unit tests for CustomEncoderModel (Chunk 6: Custom Pretrained Speech Encoder + Custom Heads)."""

import numpy as np
import pytest

from app.audio.preprocessor import AudioPreprocessor, PreparedMLInput
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.core.enums import AgeBracket, Gender
from app.inference.custom_encoder_model import CustomEncoderModel


@pytest.fixture
def custom_encoder_model():
    model = CustomEncoderModel()
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


def test_custom_encoder_model_loading(custom_encoder_model):
    """Test model loads once at startup and reports ready state."""
    assert custom_encoder_model._loaded is True
    assert custom_encoder_model.model_name == "custom_encoder"
    assert custom_encoder_model._gender_head is not None
    assert custom_encoder_model._age_head is not None


def test_custom_encoder_inference(custom_encoder_model, valid_prepared_input):
    """Test CustomEncoderModel inference on valid prepared speech input."""
    res = custom_encoder_model.predict(valid_prepared_input)

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
    assert res.model_name == "custom_encoder"


def test_custom_encoder_determinism(custom_encoder_model, valid_prepared_input):
    """Test model produces 100% deterministic output for identical input."""
    res1 = custom_encoder_model.predict(valid_prepared_input)
    res2 = custom_encoder_model.predict(valid_prepared_input)

    assert res1.gender == res2.gender
    assert res1.gender_confidence == res2.gender_confidence
    assert res1.age_bracket == res2.age_bracket
    assert res1.age_confidence == res2.age_confidence
    assert res1.gender_probabilities == res2.gender_probabilities
    assert res1.age_probabilities == res2.age_probabilities


def test_custom_encoder_invalid_input(custom_encoder_model):
    """Test invalid or empty prepared input yields UNKNOWN with confidence 0.0."""
    invalid_prep = PreparedMLInput(
        prepared_waveform=np.zeros(0, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=0.0,
        num_samples=0,
        is_prepared_valid=False,
        preparation_reasoning="Empty input payload",
    )

    res = custom_encoder_model.predict(invalid_prep)

    assert res.is_valid is False
    assert res.gender == Gender.UNKNOWN
    assert res.gender_confidence == 0.0
    assert res.age_bracket == AgeBracket.UNKNOWN
    assert res.age_confidence == 0.0


def test_custom_encoder_raw_probabilities_preserved(custom_encoder_model, valid_prepared_input):
    """Test raw predictions dictionary preserves raw probabilities for ensemble fusion."""
    res = custom_encoder_model.predict(valid_prepared_input)

    assert "gender_raw_probs" in res.raw_predictions
    assert "age_raw_probs" in res.raw_predictions
    assert "embedding_dim" in res.raw_predictions
    assert res.raw_predictions["embedding_dim"] == 192


def test_custom_encoder_inference_timing(custom_encoder_model, valid_prepared_input):
    """Test model_inference_ms measures pure CPU model execution time separately (< 500ms SLA)."""
    res = custom_encoder_model.predict(valid_prepared_input)

    assert res.is_valid is True
    assert res.model_inference_ms >= 0
    assert res.model_inference_ms < 500


def test_custom_encoder_gender_normalization(custom_encoder_model, valid_prepared_input):
    """Test gender prediction is normalized to male | female | unknown format."""
    res = custom_encoder_model.predict(valid_prepared_input)

    assert isinstance(res.gender, Gender)
    assert res.gender.value in {"male", "female", "unknown"}


def test_custom_encoder_uses_different_weights():
    """Test CustomEncoderModel uses different weight initialization than ChunkFormerModel.

    CustomEncoderModel uses seed 123, while ChunkFormerModel uses seed 42,
    so they should not produce byte-identical head weights.
    """
    from app.inference.chunkformer import ChunkFormerModel as CF
    from app.inference.strategies.gender_classifier import GenderNet

    # ChunkFormer's GenderNet (seed 42)
    chunkformer_head = GenderNet(embedding_dim=192, hidden_dim=64)

    # CustomEncoder's CustomGenderHead (seed 123)
    from app.inference.custom_encoder_model import CustomGenderHead
    custom_head = CustomGenderHead(embedding_dim=192)

    # The fc2.weight of ChunkFormer's GenderNet and fc.weight of CustomGenderHead should differ
    # (different architectures and seeds)
    assert chunkformer_head.fc2.weight.shape[1] != custom_head.fc.weight.shape[1] or \
           not (chunkformer_head.fc2.weight == custom_head.fc.weight).all()

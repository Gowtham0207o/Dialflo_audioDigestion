"""Unit tests for ChunkFormerModel (Chunk 5: ChunkFormer Attribute Inference)."""

import numpy as np
import pytest

from app.audio.preprocessor import AudioPreprocessor, PreparedMLInput
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.core.enums import AgeBracket, Gender
from app.inference.age_mapper import AgeMapper
from app.inference.chunkformer import ChunkFormerModel


@pytest.fixture
def chunkformer_model():
    model = ChunkFormerModel()
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


def test_chunkformer_model_loading(chunkformer_model):
    """Test model loads once at startup and reports ready state."""
    assert chunkformer_model._loaded is True
    assert chunkformer_model.model_name == "chunkformer_baseline"


def test_chunkformer_inference(chunkformer_model, valid_prepared_input):
    """Test ChunkFormer inference on valid prepared speech input."""
    res = chunkformer_model.predict(valid_prepared_input)

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


def test_chunkformer_determinism(chunkformer_model, valid_prepared_input):
    """Test model produces 100% deterministic output for identical input."""
    res1 = chunkformer_model.predict(valid_prepared_input)
    res2 = chunkformer_model.predict(valid_prepared_input)

    assert res1.gender == res2.gender
    assert res1.gender_confidence == res2.gender_confidence
    assert res1.age_bracket == res2.age_bracket
    assert res1.age_confidence == res2.age_confidence
    assert res1.gender_probabilities == res2.gender_probabilities
    assert res1.age_probabilities == res2.age_probabilities


def test_chunkformer_invalid_input(chunkformer_model):
    """Test invalid or empty prepared input yields UNKNOWN with confidence 0.0."""
    invalid_prep = PreparedMLInput(
        prepared_waveform=np.zeros(0, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=0.0,
        num_samples=0,
        is_prepared_valid=False,
        preparation_reasoning="Empty input payload",
    )

    res = chunkformer_model.predict(invalid_prep)

    assert res.is_valid is False
    assert res.gender == Gender.UNKNOWN
    assert res.gender_confidence == 0.0
    assert res.age_bracket == AgeBracket.UNKNOWN
    assert res.age_confidence == 0.0


def test_chunkformer_gender_normalization(chunkformer_model, valid_prepared_input):
    """Test gender prediction is normalized to male | female | unknown format."""
    res = chunkformer_model.predict(valid_prepared_input)

    assert isinstance(res.gender, Gender)
    assert res.gender.value in {"male", "female", "unknown"}


def test_chunkformer_age_mapping():
    """Test AgeMapper correctly converts raw labels and normalizes probabilities."""
    bracket, reasoning = AgeMapper.map_to_bracket("18-30", confidence=0.85)
    assert bracket == AgeBracket.YOUNG_ADULT
    assert "Mapped" in reasoning

    bracket_low, reasoning_low = AgeMapper.map_to_bracket("18-30", confidence=0.30, threshold=0.50)
    assert bracket_low == AgeBracket.UNKNOWN
    assert "Low confidence" in reasoning_low

    raw_probs = {"18-30": 0.70, "31-45": 0.20, "46-60": 0.10}
    mapped_probs = AgeMapper.map_probabilities(raw_probs)
    assert mapped_probs["18-30"] == 0.70
    assert mapped_probs["31-45"] == 0.20
    assert mapped_probs["46-60"] == 0.10


def test_chunkformer_raw_probabilities_preserved(chunkformer_model, valid_prepared_input):
    """Test raw predictions dictionary preserves raw probabilities for ensemble fusion."""
    res = chunkformer_model.predict(valid_prepared_input)

    assert "gender_raw_probs" in res.raw_predictions
    assert "age_raw_probs" in res.raw_predictions
    assert "embedding_dim" in res.raw_predictions


def test_chunkformer_inference_timing(chunkformer_model, valid_prepared_input):
    """Test model_inference_ms measures pure CPU model execution time separately (< 500ms SLA)."""
    res = chunkformer_model.predict(valid_prepared_input)

    assert res.is_valid is True
    assert res.model_inference_ms >= 0
    assert res.model_inference_ms < 500

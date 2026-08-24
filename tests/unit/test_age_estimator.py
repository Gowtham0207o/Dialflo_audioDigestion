"""Unit tests for AgeEstimator (ChunkFormer Baseline Age & Gender Inference)."""

import numpy as np
import pytest

from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.core.enums import AgeBracket
from app.inference.speech_encoder import SpeechEmbeddingResult, SpeechEncoder
from app.inference.strategies.age_estimator import AgeEstimator


@pytest.fixture
def age_estimator():
    return AgeEstimator(confidence_threshold=0.50)


@pytest.fixture
def encoder():
    enc = SpeechEncoder()
    enc.load()
    return enc


@pytest.fixture
def valid_speech_embedding(encoder):
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    f0 = 150.0
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t))
    vocal = (0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)) * mod
    waveform = (0.5 * vocal / np.max(np.abs(vocal))).astype(np.float32)

    vad_res = VoiceActivityDetector().detect(waveform, sr)
    qual_res = AudioQualityAssessor().assess(waveform, vad_res, sr)
    prep_input = AudioPreprocessor.prepare(waveform, vad_res, qual_res, target_duration_seconds=3.0, sample_rate=sr)

    return encoder.encode(prep_input)


def test_age_estimator_prediction(age_estimator, valid_speech_embedding):
    """Test age estimator produces an age bracket prediction and probabilities."""
    res = age_estimator.predict_embedding(valid_speech_embedding)

    assert res.is_valid is True
    assert res.prediction in {
        AgeBracket.YOUNG_ADULT,
        AgeBracket.ADULT,
        AgeBracket.MIDDLE_AGED,
        AgeBracket.SENIOR,
        AgeBracket.UNKNOWN,
    }
    assert 0.0 <= res.confidence <= 1.0
    assert "18-30" in res.probabilities
    assert "31-45" in res.probabilities
    assert "46-60" in res.probabilities
    assert "60+" in res.probabilities


def test_age_estimator_confidence_threshold(valid_speech_embedding):
    """Test confidence thresholding falls back to UNKNOWN when max probability < threshold."""
    strict_estimator = AgeEstimator(confidence_threshold=1.01)
    res = strict_estimator.predict_embedding(valid_speech_embedding)

    assert res.prediction == AgeBracket.UNKNOWN
    assert "UNKNOWN" in res.reasoning


def test_age_estimator_invalid_input(age_estimator):
    """Test invalid speech embedding yields UNKNOWN with confidence 0.0."""
    invalid_emb = SpeechEmbeddingResult(
        embedding=np.zeros(192, dtype=np.float32),
        embedding_dim=192,
        inference_ms=0,
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        is_valid=False,
        reasoning="Invalid input",
    )

    res = age_estimator.predict_embedding(invalid_emb)

    assert res.is_valid is False
    assert res.prediction == AgeBracket.UNKNOWN
    assert res.confidence == 0.0


def test_age_estimator_determinism(age_estimator, valid_speech_embedding):
    """Test age estimator produces 100% deterministic output for the same embedding."""
    res1 = age_estimator.predict_embedding(valid_speech_embedding)
    res2 = age_estimator.predict_embedding(valid_speech_embedding)

    assert res1.prediction == res2.prediction
    assert res1.confidence == res2.confidence
    assert res1.probabilities == res2.probabilities


def test_age_estimator_latency_benchmark(age_estimator, valid_speech_embedding):
    """Test age estimator CPU inference latency is within SLA (< 50 ms)."""
    res = age_estimator.predict_embedding(valid_speech_embedding)

    assert res.is_valid is True
    assert res.inference_ms < 50

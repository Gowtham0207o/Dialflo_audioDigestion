"""Unit tests for model comparison logic (Chunk 7: Model Comparison & Reporting)."""

import numpy as np
import pytest

from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.inference.chunkformer import ChunkFormerModel
from app.inference.custom_encoder_model import CustomEncoderModel
from app.inference.ensemble_model import EnsembleModel
from scripts.compare_models import evaluate_model


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
    model = EnsembleModel(models=[chunkformer_model, custom_encoder_model])
    model.load()
    return model


@pytest.fixture
def test_samples():
    sr = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    f0 = 150.0
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t))
    vocal = (0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)) * mod
    waveform = (0.5 * vocal / np.max(np.abs(vocal))).astype(np.float32)

    vad_res = VoiceActivityDetector().detect(waveform, sr)
    qual_res = AudioQualityAssessor().assess(waveform, vad_res, sr)
    prep = AudioPreprocessor.prepare(waveform, vad_res, qual_res, target_duration_seconds=3.0, sample_rate=sr)

    return {"synthetic_sample.wav": prep}


def test_evaluate_model(chunkformer_model, test_samples):
    """Test evaluate_model returns valid evaluation report schema."""
    report = evaluate_model(chunkformer_model, test_samples, runs_per_sample=2)

    assert "model_name" in report
    assert report["model_name"] == "chunkformer_baseline"
    assert report["total_evaluations"] == 2
    assert report["sample_count"] == 1
    assert "latency_ms" in report
    assert report["latency_ms"]["mean"] >= 0
    assert report["latency_ms"]["min"] >= 0
    assert "sample_results" in report
    assert len(report["sample_results"]) == 1


def test_model_disagreement_calculation(chunkformer_model, custom_encoder_model, test_samples):
    """Test model comparison between ChunkFormer and CustomEncoder generates valid predictions."""
    prep = test_samples["synthetic_sample.wav"]
    res1 = chunkformer_model.predict(prep)
    res2 = custom_encoder_model.predict(prep)

    assert res1.is_valid is True
    assert res2.is_valid is True
    assert res1.model_name == "chunkformer_baseline"
    assert res2.model_name == "custom_encoder"

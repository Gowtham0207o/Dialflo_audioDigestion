"""Unit and Evaluation tests for GenderClassifier (Chunk 6)."""

from pathlib import Path
import numpy as np
import pytest

from app.audio.codec import AudioCodec
from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.core.enums import Gender
from app.inference.speech_encoder import SpeechEmbeddingResult, SpeechEncoder
from app.inference.strategies.gender_classifier import GenderClassifier


@pytest.fixture
def classifier():
    clf = GenderClassifier(confidence_threshold=0.60)
    return clf


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


def test_gender_valid_embedding(classifier, valid_speech_embedding):
    """Test classification on a valid 192-dim speech embedding."""
    res = classifier.predict_embedding(valid_speech_embedding)

    assert res.is_valid is True
    assert res.prediction in {Gender.MALE, Gender.FEMALE, Gender.UNKNOWN}
    assert 0.0 <= res.confidence <= 1.0
    assert "male" in res.probabilities
    assert "female" in res.probabilities
    assert res.inference_ms >= 0


def test_gender_low_confidence_unknown(classifier, valid_speech_embedding):
    """Test confidence thresholding falls back to UNKNOWN when max probability < threshold."""
    # Set impossible threshold (1.01) to force UNKNOWN fallback
    strict_clf = GenderClassifier(confidence_threshold=1.01)
    res = strict_clf.predict_embedding(valid_speech_embedding)

    assert res.prediction == Gender.UNKNOWN
    assert "UNKNOWN" in res.reasoning


def test_gender_deterministic_inference(classifier, valid_speech_embedding):
    """Test classifier produces 100% deterministic outputs for the same embedding."""
    res1 = classifier.predict_embedding(valid_speech_embedding)
    res2 = classifier.predict_embedding(valid_speech_embedding)

    assert res1.prediction == res2.prediction
    assert res1.confidence == res2.confidence
    assert res1.probabilities == res2.probabilities


def test_gender_invalid_input(classifier):
    """Test invalid speech embedding yields UNKNOWN with confidence 0.0."""
    invalid_emb = SpeechEmbeddingResult(
        embedding=np.zeros(192, dtype=np.float32),
        embedding_dim=192,
        inference_ms=0,
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        is_valid=False,
        reasoning="Invalid input",
    )

    res = classifier.predict_embedding(invalid_emb)

    assert res.is_valid is False
    assert res.prediction == Gender.UNKNOWN
    assert res.confidence == 0.0


def test_gender_labeled_evaluation_set(encoder, classifier):
    """Evaluation test on small labeled dataset containing male and female speech samples.

    Reports Accuracy, Precision, Recall, F1 score, Confusion Matrix, and Confidence Statistics.
    """
    labeled_samples = [
        ("tests/fixtures/audio/male_sample_1.wav", Gender.MALE),
        ("tests/fixtures/audio/female_sample_1.wav", Gender.FEMALE),
        ("tests/fixtures/audio/Sample5Normalgroupspeech.ogg", Gender.FEMALE),
    ]

    y_true = []
    y_pred = []
    confidences = []

    vad_detector = VoiceActivityDetector()
    quality_assessor = AudioQualityAssessor()

    for file_path, ground_truth in labeled_samples:
        p = Path(file_path)
        assert p.exists(), f"Fixture file missing: {file_path}"

        audio_bytes = p.read_bytes()
        segment = AudioCodec.transcode_to_wav(audio_bytes, target_sample_rate=16000)

        vad_res = vad_detector.detect(segment.waveform, sample_rate=16000)
        qual_res = quality_assessor.assess(segment.waveform, vad_res, sample_rate=16000)
        prep_input = AudioPreprocessor.prepare(segment.waveform, vad_res, qual_res, target_duration_seconds=3.0, sample_rate=16000)
        emb_res = encoder.encode(prep_input)
        pred_res = classifier.predict_embedding(emb_res)

        y_true.append(ground_truth.value)
        y_pred.append(pred_res.prediction.value)
        confidences.append(pred_res.confidence)

    # Compute evaluation metrics
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    accuracy = correct / float(total)

    # Simple binary metrics for male class
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "male" and p == "male")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "male" and p == "male")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "male" and p != "male")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != "male" and p != "male")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    confusion_matrix = {
        "TP (Male -> Male)": tp,
        "FP (Non-Male -> Male)": fp,
        "TN (Non-Male -> Non-Male)": tn,
        "FN (Male -> Non-Male)": fn,
    }

    conf_stats = {
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "min_confidence": round(float(np.min(confidences)), 4),
        "max_confidence": round(float(np.max(confidences)), 4),
    }

    # Print clean benchmark evaluation report
    print("\n================ GENDER CLASSIFIER EVALUATION REPORT ================")
    print(f"Total Evaluation Samples: {total}")
    print(f"Accuracy:                 {accuracy * 100:.2f}%")
    print(f"Precision (Male):         {precision * 100:.2f}%")
    print(f"Recall (Male):            {recall * 100:.2f}%")
    print(f"F1 Score (Male):          {f1:.4f}")
    print(f"Confusion Matrix:         {confusion_matrix}")
    print(f"Confidence Statistics:    {conf_stats}")
    print("=====================================================================")

    assert total > 0
    assert 0.0 <= accuracy <= 1.0
    assert len(confidences) == total

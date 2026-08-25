"""Tests for classification metrics."""

import pytest
from evaluation.metrics.classification import compute_gender_metrics, compute_age_metrics
from evaluation.pipeline.runner import EvalRecord


def _make_record(gt_g, pr_g, gt_a, pr_a):
    return EvalRecord(
        sample_id="test",
        audio_path="test.mp3",
        gt_gender=gt_g,
        gt_age=gt_a,
        model_name="TestModel",
        pred_gender=pr_g,
        pred_age=pr_a,
        gender_confidence=0.9,
        age_confidence=0.9,
        gender_probs={},
        age_probs={},
        audio_quality="good",
        snr_db=20.0,
        is_valid=True,
        preprocess_ms=10,
        inference_ms=10,
        total_ms=20,
        error=None
    )


def test_compute_gender_metrics():
    records = [
        _make_record("male", "male", "18-30", "18-30"),
        _make_record("male", "female", "18-30", "18-30"),
        _make_record("female", "female", "18-30", "18-30"),
        _make_record("female", "female", "18-30", "18-30"),
        _make_record("male", "unknown", "18-30", "18-30"),
    ]
    
    metrics = compute_gender_metrics(records)
    
    # 5 total with gt_gender. 1 predicted unknown.
    assert metrics["total_eval_samples"] == 5
    assert metrics["unknown_rate"] == 0.2
    assert metrics["coverage"] == 0.8
    
    # 4 used for classification: [male, male, female, female] vs [male, female, female, female]
    # correct = 3/4
    assert metrics["accuracy"] == 0.75
    
    cm = metrics["confusion_matrix"]
    assert cm["male_as_male"] == 1
    assert cm["male_as_female"] == 1
    assert cm["female_as_female"] == 2
    assert cm["female_as_male"] == 0


def test_compute_age_metrics():
    records = [
        _make_record("male", "male", "18-30", "18-30"),
        _make_record("male", "male", "31-45", "18-30"),
        _make_record("male", "male", "46-60", "46-60"),
        _make_record("male", "male", "60+", "unknown"),
        _make_record("male", "male", None, "18-30"), # skipped from age eval
    ]
    
    metrics = compute_age_metrics(records)
    
    # 4 have gt_age. 1 is unknown.
    assert metrics["total_eval_samples"] == 4
    assert metrics["unknown_rate"] == 0.25
    assert metrics["coverage"] == 0.75
    
    # 3 used for classification: [18-30, 31-45, 46-60] vs [18-30, 18-30, 46-60]
    # correct = 2/3
    assert pytest.approx(metrics["accuracy"]) == 2/3
    
    cm = metrics["confusion_matrix"]
    assert cm["18-30_as_18-30"] == 1
    assert cm["31-45_as_18-30"] == 1
    assert cm["46-60_as_46-60"] == 1

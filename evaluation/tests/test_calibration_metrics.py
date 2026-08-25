"""Tests for calibration metrics."""

import pytest
from evaluation.metrics.calibration import compute_brier_score, compute_ece


def test_brier_score():
    y_true = [1, 0, 1, 0]
    conf = [0.9, 0.1, 0.6, 0.4]
    
    # Expected: 
    # ( (0.9-1)^2 + (0.1-0)^2 + (0.6-1)^2 + (0.4-0)^2 ) / 4
    # ( 0.01 + 0.01 + 0.16 + 0.16 ) / 4
    # 0.34 / 4 = 0.085
    
    score = compute_brier_score(y_true, conf)
    assert pytest.approx(score) == 0.085


def test_ece():
    y_true = [1, 1, 0, 0, 1]
    conf = [0.95, 0.9, 0.9, 0.1, 0.2]
    
    # n_bins = 10, bins = [0, 0.1, 0.2, 0.3, ..., 1.0]
    # 0.95 -> bin 9 (0.9 - 1.0)
    # 0.9  -> bin 9 (0.9 - 1.0) -- digitize right=False makes 0.9 fall in bin 9
    # 0.9  -> bin 9
    # 0.1  -> bin 1 (0.1 - 0.2)
    # 0.2  -> bin 2 (0.2 - 0.3)
    
    ece, bin_data = compute_ece(y_true, conf, n_bins=10)
    
    assert len(bin_data) == 10
    
    # Verify we got some non-zero ECE without calculating exactly by hand
    assert ece > 0
    
    # Check bin counts
    counts = [b["sample_count"] for b in bin_data]
    assert counts[9] == 3
    assert counts[1] == 1
    assert counts[2] == 1
    
    assert sum(counts) == 5

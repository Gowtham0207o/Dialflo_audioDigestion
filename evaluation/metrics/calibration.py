"""Confidence Calibration metrics.

Computes Brier score and Expected Calibration Error (ECE).
"""

import numpy as np
from evaluation.pipeline.runner import EvalRecord


def compute_brier_score(y_true_binary: list[int], confidences: list[float]) -> float:
    """Compute Brier score: (1/N) * sum((conf - y)^2)."""
    if not y_true_binary:
        return 0.0
    
    y = np.array(y_true_binary)
    p = np.array(confidences)
    
    return float(np.mean((p - y) ** 2))


def compute_ece(y_true_binary: list[int], confidences: list[float], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE).
    
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    if not y_true_binary:
        return 0.0, []
        
    y = np.array(y_true_binary)
    p = np.array(confidences)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1 to n_bins (or more if out of bounds)
    # We want indices 0 to n_bins-1, so we subtract 1.
    # We also handle the edge case where p == 1.0.
    bin_indices = np.digitize(p, bins, right=True)
    # Correction: digitize(right=True) makes bins (bins[i-1], bins[i]]
    # So 0.0 falls into bin 0 (out of bounds), we need to fix it.
    bin_indices = np.digitize(p, bins, right=False) - 1
    # Handle p == 1.0 which falls into bin_indices == n_bins
    bin_indices[bin_indices == n_bins] = n_bins - 1
    
    ece = 0.0
    N = len(y)
    
    bin_data = []
    
    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        
        if bin_count > 0:
            bin_acc = np.mean(y[mask])
            bin_conf = np.mean(p[mask])
            bin_weight = bin_count / N
            
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
            bin_data.append({
                "bin_index": b,
                "bin_range": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
                "sample_count": int(bin_count),
                "average_confidence": float(bin_conf),
                "empirical_accuracy": float(bin_acc),
            })
        else:
             bin_data.append({
                "bin_index": b,
                "bin_range": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
                "sample_count": 0,
                "average_confidence": 0.0,
                "empirical_accuracy": 0.0,
            })
            
    return float(ece), bin_data


def evaluate_calibration(records: list[EvalRecord], attribute: str, n_bins: int = 10) -> dict:
    """Evaluate calibration for either 'gender' or 'age'."""
    y_true_binary = []
    confidences = []
    
    for r in records:
        if attribute == "gender":
            if r.gt_gender and r.pred_gender != "unknown":
                y_true_binary.append(1 if r.gt_gender == r.pred_gender else 0)
                confidences.append(r.gender_confidence)
        elif attribute == "age":
            if r.gt_age and r.pred_age != "unknown":
                y_true_binary.append(1 if r.gt_age == r.pred_age else 0)
                confidences.append(r.age_confidence)
                
    brier_score = compute_brier_score(y_true_binary, confidences)
    ece, bin_data = compute_ece(y_true_binary, confidences, n_bins)
    
    return {
        "brier_score": brier_score,
        "ece": ece,
        "bins": bin_data
    }

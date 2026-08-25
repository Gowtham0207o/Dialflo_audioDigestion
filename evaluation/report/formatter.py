"""Report formatter.

Aggregates metrics, prints console summaries, and writes JSON/CSV outputs.
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.metrics.calibration import evaluate_calibration
from evaluation.metrics.classification import compute_age_metrics, compute_gender_metrics
from evaluation.pipeline.runner import EvalRecord

# Attempt to import tabulate, but don't fail if not present (handled in CLI)
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

logger = logging.getLogger(__name__)


def _compute_latency_stats(records: list[EvalRecord], key: str) -> dict:
    """Compute latency stats for a specific latency key (e.g., 'inference_ms')."""
    latencies = [getattr(r, key) for r in records if r.is_valid]
    if not latencies:
        return {"mean": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}
        
    return {
        "mean": int(np.mean(latencies)),
        "p50": int(np.percentile(latencies, 50)),
        "p90": int(np.percentile(latencies, 90)),
        "p95": int(np.percentile(latencies, 95)),
        "p99": int(np.percentile(latencies, 99)),
    }


def _analyze_audio_quality(records: list[EvalRecord]) -> dict:
    """Analyze results by audio quality and SNR buckets."""
    quality_groups = defaultdict(list)
    snr_groups = defaultdict(list)
    
    for r in records:
        if not r.is_valid:
            continue
            
        quality_groups[r.audio_quality].append(r)
        
        if r.snr_db >= 18.0:
            snr_groups[">=18 dB"].append(r)
        elif r.snr_db >= 10.0:
            snr_groups["10-18 dB"].append(r)
        elif r.snr_db >= 5.0:
            snr_groups["5-10 dB"].append(r)
        else:
            snr_groups["<5 dB"].append(r)
            
    def _summarize_group(group_records):
        g_metrics = compute_gender_metrics(group_records)
        a_metrics = compute_age_metrics(group_records)
        
        avg_g_conf = np.mean([r.gender_confidence for r in group_records]) if group_records else 0.0
        avg_a_conf = np.mean([r.age_confidence for r in group_records]) if group_records else 0.0
        
        return {
            "sample_count": len(group_records),
            "gender_accuracy": g_metrics.get("accuracy", 0.0),
            "age_accuracy": a_metrics.get("accuracy", 0.0),
            "avg_gender_confidence": float(avg_g_conf),
            "avg_age_confidence": float(avg_a_conf),
        }

    return {
        "by_quality": {k: _summarize_group(v) for k, v in quality_groups.items()},
        "by_snr": {k: _summarize_group(v) for k, v in snr_groups.items()}
    }


def _compute_agreement(records1: list[EvalRecord], records2: list[EvalRecord]) -> dict:
    """Compute agreement between two sets of records (assumed to be paired)."""
    # Create dicts keyed by sample_id for safe pairing
    d1 = {r.sample_id: r for r in records1 if r.is_valid}
    d2 = {r.sample_id: r for r in records2 if r.is_valid}
    
    common_ids = set(d1.keys()).intersection(set(d2.keys()))
    if not common_ids:
        return {"gender_agreement": 0.0, "age_agreement": 0.0, "samples": 0}
        
    g_agree = 0
    a_agree = 0
    
    for sid in common_ids:
        if d1[sid].pred_gender == d2[sid].pred_gender:
            g_agree += 1
        if d1[sid].pred_age == d2[sid].pred_age:
            a_agree += 1
            
    total = len(common_ids)
    return {
        "gender_agreement": g_agree / total,
        "age_agreement": a_agree / total,
        "samples": total
    }


def generate_report(
    records: list[EvalRecord], 
    n_bins: int = 10, 
    baseline_data: dict | None = None
) -> dict:
    """Generate a full evaluation report."""
    
    # Group records by model
    records_by_model = defaultdict(list)
    for r in records:
        records_by_model[r.model_name].append(r)
        
    report = {
        "models": {},
        "comparison": {}
    }
    
    for model_name, model_records in records_by_model.items():
        g_metrics = compute_gender_metrics(model_records)
        a_metrics = compute_age_metrics(model_records)
        
        g_calib = evaluate_calibration(model_records, "gender", n_bins)
        a_calib = evaluate_calibration(model_records, "age", n_bins)
        
        latency = {
            "preprocess": _compute_latency_stats(model_records, "preprocess_ms"),
            "inference": _compute_latency_stats(model_records, "inference_ms"),
            "total": _compute_latency_stats(model_records, "total_ms"),
        }
        
        quality_analysis = _analyze_audio_quality(model_records)
        
        report["models"][model_name] = {
            "gender": {**g_metrics, "calibration": g_calib},
            "age": {**a_metrics, "calibration": a_calib},
            "latency": latency,
            "quality_analysis": quality_analysis,
            "total_samples": len(model_records),
            "valid_samples": sum(1 for r in model_records if r.is_valid)
        }
        
    # If multiple models, compute agreement
    if "ChunkFormerModel" in records_by_model and "CustomEncoderModel" in records_by_model:
        report["comparison"]["chunkformer_vs_custom_agreement"] = _compute_agreement(
            records_by_model["ChunkFormerModel"], 
            records_by_model["CustomEncoderModel"]
        )

    return report


def print_console_summary(report: dict, baseline_data: dict | None = None):
    """Print the evaluation summary to the console."""
    if not tabulate:
        logger.error("tabulate package not installed. Cannot print console summary. Use `pip install tabulate`.")
        return

    print("\n" + "=" * 80)
    print("                      EVALUATION HARNESS SUMMARY")
    print("=" * 80)
    
    headers = ["Model", "Gender F1", "Age F1", "Gender ECE", "Age ECE", "P95 Latency"]
    table_data = []
    
    for model_name, data in report["models"].items():
        g_f1 = data["gender"].get("macro_f1", 0.0)
        a_f1 = data["age"].get("macro_f1", 0.0)
        g_ece = data["gender"]["calibration"]["ece"]
        a_ece = data["age"]["calibration"]["ece"]
        p95_lat = data["latency"]["total"]["p95"]
        
        row = [
            model_name,
            f"{g_f1:.4f}",
            f"{a_f1:.4f}",
            f"{g_ece:.4f}",
            f"{a_ece:.4f}",
            f"{p95_lat} ms"
        ]
        table_data.append(row)
        
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    if "chunkformer_vs_custom_agreement" in report.get("comparison", {}):
        agreement = report["comparison"]["chunkformer_vs_custom_agreement"]
        print("\nModel Agreement (ChunkFormer vs Custom):")
        print(f"  Gender: {agreement['gender_agreement']*100:.1f}%")
        print(f"  Age:    {agreement['age_agreement']*100:.1f}%")
        
    if baseline_data:
        # Simple baseline comparison print for the first model in the report
        model_name = list(report["models"].keys())[0]
        data = report["models"][model_name]
        print("\nBaseline Comparison:")
        try:
             b_g_f1 = baseline_data["models"][model_name]["gender"]["macro_f1"]
             diff = data["gender"]["macro_f1"] - b_g_f1
             print(f"  Gender F1: {data['gender']['macro_f1']:.4f} (diff: {diff:+.4f})")
        except KeyError:
             print("  Could not find matching baseline data.")


def write_json_report(report: dict, output_path: str):
    """Write the full nested JSON report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote JSON report to {path}")


def write_csv_predictions(records: list[EvalRecord], output_path: str):
    """Write per-sample predictions to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not records:
        logger.warning("No records to write to CSV.")
        return
        
    fieldnames = [
        "sample_id", "audio_path", "model_name", 
        "gt_gender", "pred_gender", "gender_confidence",
        "gt_age", "pred_age", "age_confidence",
        "audio_quality", "snr_db", "is_valid",
        "preprocess_ms", "inference_ms", "total_ms", "error"
    ]
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sample_id": r.sample_id,
                "audio_path": r.audio_path,
                "model_name": r.model_name,
                "gt_gender": r.gt_gender,
                "pred_gender": r.pred_gender,
                "gender_confidence": r.gender_confidence,
                "gt_age": r.gt_age,
                "pred_age": r.pred_age,
                "age_confidence": r.age_confidence,
                "audio_quality": r.audio_quality,
                "snr_db": r.snr_db,
                "is_valid": r.is_valid,
                "preprocess_ms": r.preprocess_ms,
                "inference_ms": r.inference_ms,
                "total_ms": r.total_ms,
                "error": r.error
            })
            
    logger.info(f"Wrote CSV predictions to {path}")

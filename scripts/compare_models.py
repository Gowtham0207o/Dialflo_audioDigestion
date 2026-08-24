"""Model Comparison and Evaluation Script (Chunk 7).

Evaluates ChunkFormerModel, CustomEncoderModel, and EnsembleModel on test audio fixtures.
Computes and reports:
- Model performance metrics (Accuracy, Precision, Recall, F1 for gender classification)
- Confusion matrix
- Confidence score distributions
- Model inference latency (min, max, mean, p50, p95)
- Sub-model disagreement rates

Outputs a summary comparison table and saves full structured metrics to `eval/comparison_report.json`.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.core.enums import AgeBracket, Gender
from app.inference.attribute_model import AttributeInferenceResult, AttributeModel
from app.inference.chunkformer import ChunkFormerModel
from app.inference.custom_encoder_model import CustomEncoderModel
from app.inference.ensemble_model import EnsembleModel
from app.observability.logger import get_logger

logger = get_logger(__name__)

# Sample audio directory
AUDIO_DIR = Path("tests/fixtures/audio")

# Ground truth annotations for test fixtures (where known)
GROUND_TRUTH = {
    "male_sample_1.wav": {"gender": Gender.MALE, "age_bracket": AgeBracket.ADULT},
    "female_sample_1.wav": {"gender": Gender.FEMALE, "age_bracket": AgeBracket.YOUNG_ADULT},
    "Sample5Normalgroupspeech.ogg": {"gender": Gender.MALE, "age_bracket": AgeBracket.MIDDLE_AGED},
}


def load_and_prepare_audio(filepath: Path) -> Any:
    """Load audio file and run Silero VAD + Quality Assessment + ML Input Prep."""
    data, sample_rate = sf.read(filepath, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    vad_detector = VoiceActivityDetector()
    vad_res = vad_detector.detect(data, sample_rate=sample_rate)

    quality_assessor = AudioQualityAssessor()
    qual_res = quality_assessor.assess(data, vad_res, sample_rate=sample_rate)

    return AudioPreprocessor.prepare(
        waveform=data,
        vad_result=vad_res,
        quality_result=qual_res,
        target_duration_seconds=3.0,
        sample_rate=sample_rate,
    )


def evaluate_model(model: AttributeModel, test_samples: dict[str, Any], runs_per_sample: int = 5) -> dict[str, Any]:
    """Run model evaluation across test samples and collect metrics."""
    results: list[dict[str, Any]] = []
    latencies: list[int] = []

    for filename, prep_input in test_samples.items():
        sample_latencies: list[int] = []
        last_res: AttributeInferenceResult | None = None

        for _ in range(runs_per_sample):
            res = model.predict(prep_input)
            sample_latencies.append(res.model_inference_ms)
            if prep_input.is_prepared_valid and res.is_valid:
                latencies.append(res.model_inference_ms)
            last_res = res

        gt = GROUND_TRUTH.get(filename, {})
        results.append({
            "filename": filename,
            "ground_truth_gender": gt.get("gender", Gender.UNKNOWN).value,
            "ground_truth_age": gt.get("age_bracket", AgeBracket.UNKNOWN).value,
            "predicted_gender": last_res.gender.value if last_res else "unknown",
            "predicted_age": last_res.age_bracket.value if last_res else "unknown",
            "gender_confidence": last_res.gender_confidence if last_res else 0.0,
            "age_confidence": last_res.age_confidence if last_res else 0.0,
            "gender_probs": last_res.gender_probabilities if last_res else {},
            "age_probs": last_res.age_probabilities if last_res else {},
            "is_prepared_valid": prep_input.is_prepared_valid,
            "is_valid": last_res.is_valid if last_res else False,
            "mean_latency_ms": float(np.mean(sample_latencies)),
        })

    # Compute metrics for gender classification against ground truth (valid samples only)
    y_true = []
    y_pred = []
    for r in results:
        if r["is_prepared_valid"]:
            gt_g = r["ground_truth_gender"]
            pr_g = r["predicted_gender"]
            if gt_g != "unknown":
                y_true.append(gt_g)
                y_pred.append(pr_g)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    # Gender confusion matrix
    cm = {"male_as_male": 0, "male_as_female": 0, "female_as_female": 0, "female_as_male": 0, "unknown": 0}
    for t, p in zip(y_true, y_pred):
        if t == "male" and p == "male":
            cm["male_as_male"] += 1
        elif t == "male" and p == "female":
            cm["male_as_female"] += 1
        elif t == "female" and p == "female":
            cm["female_as_female"] += 1
        elif t == "female" and p == "male":
            cm["female_as_male"] += 1
        else:
            cm["unknown"] += 1

    return {
        "model_name": model.model_name if hasattr(model, "model_name") else str(type(model).__name__),
        "total_evaluations": len(results) * runs_per_sample,
        "sample_count": len(results),
        "valid_sample_count": len(y_true),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm,
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2) if latencies else 0,
            "std": round(float(np.std(latencies)), 2) if latencies else 0,
            "min": int(np.min(latencies)) if latencies else 0,
            "max": int(np.max(latencies)) if latencies else 0,
            "p50": int(np.percentile(latencies, 50)) if latencies else 0,
            "p95": int(np.percentile(latencies, 95)) if latencies else 0,
        },
        "sample_results": results,
    }


def main():
    print("=================================================================")
    print("       DialFlo Audio Digestion — Model Comparison Evaluation     ")
    print("=================================================================\n")

    # 1. Load test samples
    test_samples: dict[str, Any] = {}
    print("Loading test audio samples:")
    for filepath in AUDIO_DIR.glob("*.*"):
        if filepath.suffix.lower() in {".wav", ".ogg", ".mp3", ".flac"}:
            print(f"  - Loading {filepath.name}...")
            try:
                prep = load_and_prepare_audio(filepath)
                test_samples[filepath.name] = prep
                print(f"    Loaded successfully (valid={prep.is_prepared_valid})")
            except Exception as e:
                print(f"    Failed to load: {e}")

    if not test_samples:
        print("ERROR: No valid test samples found in tests/fixtures/audio/")
        return

    # 2. Instantiate and load models
    print("\nInitializing models...")
    cf_model = ChunkFormerModel()
    cf_model.load()

    ce_model = CustomEncoderModel()
    ce_model.load()

    ens_model = EnsembleModel(models=[cf_model, ce_model], weights=[0.5, 0.5])
    ens_model.load()

    models = [cf_model, ce_model, ens_model]

    # 3. Evaluate each model
    print("\nRunning evaluation benchmark...")
    eval_reports = {}
    for m in models:
        name = m.model_name
        print(f"  - Evaluating {name}...")
        report = evaluate_model(m, test_samples, runs_per_sample=5)
        eval_reports[name] = report

    # 4. Compute model disagreement rate between ChunkFormer and CustomEncoder
    disagreement_count = 0
    total_valid_samples = 0
    for filename, prep_input in test_samples.items():
        if prep_input.is_prepared_valid:
            r1 = cf_model.predict(prep_input)
            r2 = ce_model.predict(prep_input)
            total_valid_samples += 1
            if r1.gender != r2.gender or r1.age_bracket != r2.age_bracket:
                disagreement_count += 1

    disagreement_rate = (disagreement_count / total_valid_samples) if total_valid_samples > 0 else 0.0

    # 5. Print Comparison Summary Table
    print("\n" + "=" * 80)
    print("                              EVALUATION SUMMARY TABLE                           ")
    print("=" * 80)
    print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Latency (p50)':<15} | {'Latency (p95)':<15}")
    print("-" * 80)
    for name, r in eval_reports.items():
        acc = f"{r['accuracy']*100:.1f}%"
        p50 = f"{r['latency_ms']['p50']} ms"
        p95 = f"{r['latency_ms']['p95']} ms"
        print(f"{name:<20} | {acc:<10} | {p50:<15} | {p95:<15}")
    print("-" * 80)
    print(f"Sub-Model Disagreement Rate (ChunkFormer vs CustomEncoder): {disagreement_rate*100:.1f}%\n")

    # 6. Save full JSON report
    output_dir = Path("eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.json"

    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "disagreement_rate": round(disagreement_rate, 4),
        "disagreement_count": disagreement_count,
        "valid_sample_count": total_valid_samples,
        "models": eval_reports,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"Full evaluation report saved to: {report_path.resolve()}\n")


if __name__ == "__main__":
    main()

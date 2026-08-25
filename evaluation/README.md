# DialFlo Evaluation Harness

This package provides a comprehensive evaluation harness for the DialFlo Audio Digestion pipeline against the Mozilla Common Voice dataset. 

It explicitly **reuses the exact production inference pipeline**, ensuring that performance metrics perfectly reflect real-world server behavior.

## Features
- **Phase 1**: Dynamic TSV parsing, gender and age mapping, deterministic subsampling.
- **Phase 2**: Production pipeline integration (reusing `AudioCodec`, `AudioPreprocessor`, `VoiceActivityDetector`, `AudioQualityAssessor`).
- **Phase 3-4**: Computes Accuracy, F1, Confusion Matrix, Brier score, and Expected Calibration Error (ECE).
- **Phase 5-8**: Generates structured reports, CLI summary tables, and exports complete predictions to CSV.

## Requirements

Ensure optional dependencies are installed:
```bash
pip install -e ".[eval]"
```
*(Requires `scikit-learn` and `tabulate`)*

FFmpeg must be installed and accessible in your local environment's `PATH`.

## How to Run

1. Download a Mozilla Common Voice release (e.g., CV17)
2. Extract it to a folder, which should contain `validated.tsv` (or `test.tsv`) and a `clips/` directory.
3. Run the evaluation harness:

```bash
python -m evaluation.run \
  --dataset-path ./data/common_voice \
  --model-mode all \
  --limit 100 \
  --seed 42 \
  --output evaluation/results.json \
  --output-csv evaluation/predictions.csv \
  --calibration-bins 10
```

### Model Modes
- `--model-mode chunkformer`: Evaluates the baseline `ChunkFormerModel`.
- `--model-mode custom`: Evaluates the `CustomEncoderModel`.
- `--model-mode ensemble`: Evaluates the `EnsembleModel` (fusing the two).
- `--model-mode all`: Evaluates all three models simultaneously, loading weights once and comparing their agreement rates.

## Age Mapping Strategy

The harness maps Common Voice age strings to DialFlo's enum brackets as follows:
- `teens` -> skipped
- `twenties` -> `18-30`
- `thirties`, `forties` -> `31-45`
- `fifties` -> `46-60`
- `sixties`, `seventies`, `eighties`, `nineties` -> `60+`

Samples lacking valid age metadata are evaluated for gender only, and excluded from age metrics.

## Baseline Comparison
You can compare a new run to an existing JSON report using the `--baseline` flag:
```bash
python -m evaluation.run ... --baseline evaluation/results_previous.json
```
This will print the difference in Macro F1 score between the current run and the baseline.

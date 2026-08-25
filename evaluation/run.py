"""Evaluation harness CLI entrypoint."""

import argparse
import json
import logging
import sys
from pathlib import Path

from evaluation.adapter.common_voice import CommonVoiceAdapter
from evaluation.pipeline.runner import PipelineRunner
from evaluation.report.formatter import generate_report, print_console_summary, write_csv_predictions, write_json_report


def setup_logger():
    """Setup basic console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Eval harness for DialFlo Audio Digestion models on Common Voice")
    parser.add_argument("--dataset-path", required=True, type=str, help="Path to Common Voice dataset directory (must contain TSV and clips/)")
    parser.add_argument("--model-mode", choices=["chunkformer", "custom", "ensemble", "all"], default="all", help="Which model(s) to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default="evaluation/results.json", help="Path to save JSON report")
    parser.add_argument("--output-csv", type=str, default="evaluation/predictions.csv", help="Path to save CSV predictions")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of bins for ECE calculation")
    parser.add_argument("--baseline", type=str, default=None, help="Path to previous JSON report to use as baseline")
    
    args = parser.parse_args()
    setup_logger()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting evaluation harness for mode: {args.model_mode}")
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path {dataset_path} does not exist.")
        sys.exit(1)
        
    # Phase 1: Load samples
    try:
        adapter = CommonVoiceAdapter(dataset_path)
        samples = adapter.load_samples(limit=args.limit, seed=args.seed)
    except Exception as e:
        logger.error(f"Failed to load samples: {e}")
        sys.exit(1)
        
    if not samples:
        logger.error("No valid samples loaded.")
        sys.exit(1)
        
    # Phase 2: Run pipeline
    runner = PipelineRunner(model_mode=args.model_mode)
    try:
        records = runner.run(samples)
    except Exception as e:
        logger.error(f"Pipeline runner failed critically: {e}")
        sys.exit(1)
        
    if not records:
        logger.error("No eval records were produced.")
        sys.exit(1)
        
    # Load baseline if provided
    baseline_data = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load baseline {args.baseline}: {e}")
        else:
            logger.warning(f"Baseline file {args.baseline} not found.")

    # Phase 3-8: Generate report and output
    report = generate_report(records, n_bins=args.calibration_bins, baseline_data=baseline_data)
    
    print_console_summary(report, baseline_data=baseline_data)
    
    if args.output:
        write_json_report(report, args.output)
        
    if args.output_csv:
        write_csv_predictions(records, args.output_csv)
        
    logger.info("Evaluation harness complete.")


if __name__ == "__main__":
    main()

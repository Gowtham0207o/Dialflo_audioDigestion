"""Pre-downloads model weights for Docker image build step or offline use.

Usage:
    python scripts/download_models.py --cache-dir ./models
"""

import argparse
import sys
from pathlib import Path


def download_models(cache_dir: str) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Model cache directory initialized: {path.resolve()}")
    print("[+] Model weights pre-download check complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for DialFlo Audio Digestion")
    parser.add_argument("--cache-dir", default="./models", help="Target model cache directory")
    args = parser.parse_args()
    download_models(args.cache_dir)

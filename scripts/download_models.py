"""Pre-downloads model weights for Docker image build step or offline use.

Usage:
    python scripts/download_models.py --cache-dir ./models
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from transformers import AutoConfig, AutoProcessor
from app.inference.wav2vec2_model import AgeGenderModel

def download_models(cache_dir: str) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Model cache directory initialized: {path.resolve()}")
    
    model_id = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    print(f"[+] Downloading and caching model: {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = AgeGenderModel.from_pretrained(model_id, config=config, cache_dir=cache_dir)
        print("[+] Model weights successfully pre-downloaded and cached.")
    except Exception as e:
        print(f"[-] Failed to download models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for DialFlo Audio Digestion")
    parser.add_argument("--cache-dir", default="./models", help="Target model cache directory")
    args = parser.parse_args()
    download_models(args.cache_dir)

"""Generates synthetic test WAV files for smoke tests and unit tests.

Usage:
    python scripts/generate_sample_audio.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_audio_files(output_dir: str = "tests/fixtures/audio") -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)

    # 1. Clean synthetic speech (sine mix)
    clean_wave = (0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    clean_path = target_dir / "sample_clean.wav"
    sf.write(clean_path, clean_wave, sr)
    print(f"[+] Created synthetic clean WAV: {clean_path}")

    # 2. Noisy synthetic speech
    noise = np.random.normal(0, 0.25, len(clean_wave)).astype(np.float32)
    noisy_wave = clean_wave + noise
    noisy_path = target_dir / "sample_noisy.wav"
    sf.write(noisy_path, noisy_wave, sr)
    print(f"[+] Created synthetic noisy WAV: {noisy_path}")


if __name__ == "__main__":
    generate_audio_files()

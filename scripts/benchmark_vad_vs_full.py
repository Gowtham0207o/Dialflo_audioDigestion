"""Benchmark script comparing ChunkFormer attribute inference on Full Audio vs VAD-Selected Speech.

Evaluates:
- Audio Duration (Full vs VAD Speech)
- Inference Latency (Full vs VAD Speech)
- Prediction Stability (Gender & Age output consistency)
- Quality / Noise reduction metrics
"""

from pathlib import Path
import time

import numpy as np

from app.audio.codec import AudioCodec
from app.audio.preprocessor import AudioPreprocessor, PreparedMLInput
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.inference.chunkformer import ChunkFormerModel


def run_benchmark() -> None:
    """Execute benchmark suite across sample audio files."""
    sample_files = [
        "tests/fixtures/audio/male_sample_1.wav",
        "tests/fixtures/audio/female_sample_1.wav",
        "tests/fixtures/audio/Sample5Normalgroupspeech.ogg",
    ]

    print("\n==========================================================================")
    print("      CHUNKFORMER BENCHMARK: FULL AUDIO vs VAD-SELECTED SPEECH            ")
    print("==========================================================================")

    model = ChunkFormerModel()
    model.load()

    vad_detector = VoiceActivityDetector()
    quality_assessor = AudioQualityAssessor()

    for file_path in sample_files:
        p = Path(file_path)
        if not p.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        print(f"\n--- File: {p.name} ---")
        audio_bytes = p.read_bytes()

        # 1. Transcode / Normalize to 16 kHz Mono float32 PCM
        segment = AudioCodec.transcode_to_wav(audio_bytes, target_sample_rate=16000)

        # 2. VAD & Quality Assessment
        vad_res = vad_detector.detect(segment.waveform, sample_rate=16000)
        qual_res = quality_assessor.assess(segment.waveform, vad_res, sample_rate=16000)

        # 3. Mode A: Full Audio (Raw waveform fitted to target window)
        full_prep = PreparedMLInput(
            prepared_waveform=segment.waveform[:48000] if len(segment.waveform) >= 48000 else np.pad(segment.waveform, (0, 48000 - len(segment.waveform))),
            sample_rate=16000,
            duration_seconds=3.0,
            num_samples=48000,
            is_prepared_valid=True,
            preparation_reasoning="Full audio raw input payload",
        )

        t0 = time.perf_counter()
        full_res = model.predict(full_prep)
        full_latency_ms = int((time.perf_counter() - t0) * 1000)

        # 4. Mode B: VAD-Selected Speech (Speech regions extracted & normalized)
        vad_prep = AudioPreprocessor.prepare(segment.waveform, vad_res, qual_res, target_duration_seconds=3.0, sample_rate=16000)

        t0 = time.perf_counter()
        vad_res_inf = model.predict(vad_prep)
        vad_latency_ms = int((time.perf_counter() - t0) * 1000)

        print(f"Total File Duration:       {segment.duration_seconds:.2f}s ({segment.num_samples} samples)")
        print(f"VAD Speech Duration:       {vad_res.speech_duration_seconds:.2f}s (Ratio: {vad_res.speech_ratio:.2%})")
        print(f"Audio Quality Assessed:    {qual_res.audio_quality.value} (SNR: {qual_res.snr_db:.1f} dB)")
        print("\n[ Mode A: Full Audio Input ]")
        print(f"  Gender Prediction:      {full_res.gender.value} (Confidence: {full_res.gender_confidence:.4f})")
        print(f"  Age Prediction:         {full_res.age_bracket.value} (Confidence: {full_res.age_confidence:.4f})")
        print(f"  Inference Latency:      {full_latency_ms} ms")
        print("\n[ Mode B: VAD-Selected Speech Input ]")
        print(f"  Gender Prediction:      {vad_res_inf.gender.value} (Confidence: {vad_res_inf.gender_confidence:.4f})")
        print(f"  Age Prediction:         {vad_res_inf.age_bracket.value} (Confidence: {vad_res_inf.age_confidence:.4f})")
        print(f"  Inference Latency:      {vad_latency_ms} ms")

        # Compare prediction agreement
        gender_match = "MATCH" if full_res.gender == vad_res_inf.gender else "MISMATCH"
        age_match = "MATCH" if full_res.age_bracket == vad_res_inf.age_bracket else "MISMATCH"
        print(f"  Result Agreement:       Gender={gender_match}, Age={age_match}")

    print("\n==========================================================================\n")


if __name__ == "__main__":
    run_benchmark()

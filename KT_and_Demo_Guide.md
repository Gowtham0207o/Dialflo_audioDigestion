# DialFlo Audio Digestion — Knowledge Transfer (KT) & Demo Guide

This document provides a complete technical Knowledge Transfer (KT) of the **DialFlo Audio Digestion Service** and a step-by-step process for conducting a live demonstration for your teammates.

---

## 1. Project Executive Summary

- **Service Purpose**: Real-time audio attribute inference microservice for logistics voice AI systems.
- **Core Functionality**: Ingests raw caller audio (any format), cleanses & extracts active speech, evaluates signal quality, and infers caller attributes (**Gender** and **Age Bracket**) with **sub-second latency**.
- **Tech Stack**: Python 3.11, FastAPI, PyTorch, SpeechBrain (ECAPA-TDNN), Silero VAD, FFmpeg, Docker.
- **Privacy Guarantee**: **Zero-persistence PII design**. Audio files are processed exclusively in RAM (`bytes` / `numpy.ndarray`) and garbage-collected when the request completes. No audio touches disk or database.

---

## 2. System Architecture & 6-Stage Processing Pipeline

Every audio request (`POST /v1/analyze`) flows through 6 discrete, composable processing stages:

```
Raw Audio Payload (Multipart upload / WS)
   │
   ▼
[ 1. Ingestion & Validation ] ──► Format, extension & 50MB file size checks
   │
   ▼
[ 2. FFmpeg Codec Transcoding ] ──► Converts WAV/MP3/OGG/Opus/FLAC ──► 16 kHz Mono float32 PCM
   │
   ▼
[ 3. Silero VAD & Refinement ] ──► Speech region isolation, gap merging (<300ms), fragment filtering
   │
   ▼
[ 4. Quality Assessment ] ──────► Evaluates SNR (dB), clipping ratio, peak amplitude & VAD speech ratio
   │
   ▼
[ 5. ML Input Preparation ] ────► Concatenates speech & normalizes to 3.0s window (48,000 samples)
   │
   ▼
[ 6. Attribute Inference ] ─────► 192-dim ECAPA-TDNN embedding ──► Gender & Age prediction
   │
   ▼
JSON API Response Metadata
```

### Stage Details:

1. **Ingestion & Validation (`AudioValidator`)**
   - Validates MIME type, file extension, and file size limits (< 50MB).
2. **Codec Normalization (`AudioCodec`)**
   - Uses FFmpeg to transcode any input audio format into standardized **16,000 Hz, 1-channel mono, 32-bit float PCM waveform data**.
3. **Voice Activity Detection (`VoiceActivityDetector`)**
   - Runs **Silero VAD** neural engine to isolate human speech from silence. Merges gaps smaller than 300ms and filters out noise bursts smaller than 150ms.
4. **Quality Assessment (`AudioQualityAssessor`)**
   - Analyzes 5 acoustic metrics (SNR, Peak, Clipping, RMS, Speech Energy Ratio). Flags audio as `good`, `degraded`, or `insufficient`.
5. **ML Input Preparation (`AudioPreprocessor`)**
   - Strips background silence, concatenates active speech chunks, and fits the audio payload into a deterministic 3.0-second inference window (48,000 float32 samples).
6. **Attribute Inference (`SpeechEncoder` & `ChunkFormerModel` / `EnsembleModel`)**
   - **SpeechEncoder**: Extracts a 192-dimensional speaker embedding vector using SpeechBrain's ECAPA-TDNN (`spkrec-ecapa-voxceleb`).
   - **Gender Classifier**: Predicts `male` or `female` with probability distribution and confidence thresholding (`0.60`).
   - **Age Estimator**: Maps acoustic prosodic features into canonical age brackets: `18-30` (Young Adult), `31-45` (Adult), `46-60` (Middle Aged), `60+` (Senior).
   - **Ensemble Model**: Fuses predictions from `ChunkFormerModel` and `CustomEncoderModel` for higher prediction stability.

---

## 3. Key Design & Production Features

- **Zero-Disk PII Security**: Strict compliance with GDPR & SOC 2. `PrivacyGuardMiddleware` strips audio references from loggers.
- **Read-Only Container Hardening**: The Docker container runs as non-root with a `read_only: true` filesystem. Temporary caches use a RAM-backed `tmpfs` mount (`/tmp/cache`).
- **Resilience & Circuit Breaker (`CircuitBreaker`)**: Protects the service against cascading ML inference failures under heavy load.
- **Observability**: Structured JSON logging (`structlog`), Prometheus metrics (`prometheus-client`), and latency breakdown per stage (`decode_ms`, `vad_ms`, `quality_ms`, `inference_ms`).

---

## 4. Step-by-Step Demo Process for Teammates

Follow this 5-step sequence when demoing the project to your team:

### Step 1: Container & Health Check
Show that the service is running and healthy inside Docker:
```powershell
docker compose ps
curl http://localhost:8000/v1/health
```
- **What to highlight**: Point out the instant HTTP 200 response returning `status: ok` and service version.

---

### Step 2: Interactive Browser UI (Swagger OpenAPI)
Open your browser to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

1. Expand the **`POST /v1/analyze`** route.
2. Click **"Try it out"**.
3. Upload an audio sample file (`.wav` or `.ogg`).
4. Click **Execute** and show the response JSON structure:
   - `duration` & `speech_duration_seconds`
   - `speech_segments` (VAD start/end timestamps)
   - `audio_quality` & `snr_db`
   - `gender` prediction + confidence
   - `age_bracket` prediction + confidence
   - `timing_breakdown` (stage-by-stage latency in milliseconds)

---

### Step 3: Command-Line Curl Demo
Demonstrate live API calls directly from terminal:
```powershell
curl.exe -X POST "http://localhost:8000/v1/analyze" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@tests/fixtures/audio/Sample5Normalgroupspeech.ogg;type=audio/ogg"
```
- **What to highlight**: Show that end-to-end processing completes in **sub-second time (< 600ms)**.

---

### Step 4: Model Benchmarking & Evaluation Scripts
Demonstrate model evaluation and latency benchmarking:

1. **Model Comparison Evaluation**:
   ```powershell
   python scripts/compare_models.py
   ```
   - **What to highlight**: Shows accuracy, precision, confusion matrix, and model latency comparison between `ChunkFormer`, `CustomEncoder`, and `EnsembleModel`.

2. **Concurrent Load Latency Benchmark**:
   ```powershell
   python scripts/benchmark.py
   ```
   - **What to highlight**: Demonstrates P50, P95, and P99 latencies under concurrent request load.

---

### Step 5: Code Base Architecture Tour (Code Walkthrough)

Guide your teammate through the core source code directory (`src/app/`):

1. `src/app/api/v1/routes/analyze.py`: Ingestion controller orchestrating the 6 pipeline stages.
2. `src/app/audio/codec.py`: FFmpeg transcoding to 16kHz float32 PCM.
3. `src/app/audio/vad.py`: Silero Voice Activity Detection engine.
4. `src/app/audio/quality.py`: Multi-signal quality assessment.
5. `src/app/inference/speech_encoder.py`: 192-dim ECAPA-TDNN embedding extractor.
6. `src/app/inference/chunkformer.py` & `ensemble_model.py`: Multi-attribute gender & age inference engines.

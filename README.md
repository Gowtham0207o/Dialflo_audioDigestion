# Dialflo Audio Digestion API

A real-time, CPU-optimized audio attribute inference service designed for logistics voice AI. It ingests speech audio via HTTP or WebSockets, isolates voice activity, assesses audio quality, and predicts speaker gender and age brackets using ECAPA-TDNN embeddings.

## Architecture

We process audio through a strict, deterministic pipeline before any neural inference occurs.

```mermaid
flowchart TD
    A[Client Request] --> B{Transport}
    B -->|HTTP| C[REST API]
    B -->|WebSocket| D[Stream API]
    
    C --> E[Validation Stage]
    D --> E
    
    E --> F[Decoding Stage]
    F -->|FFmpeg/SoundFile| G[16kHz Mono Float32]
    
    G --> H[VAD Stage]
    H -->|Silero| I[Speech Segments]
    
    I --> J[Quality Stage]
    J -->|SNR, Clipping, RMS| K[Audio Quality Report]
    
    K --> L[Preprocessing Stage]
    L -->|Truncate/Pad to 3s| M[Prepared ML Input]
    
    M --> N[Inference Stage]
    N -->|SpeechEncoder| O[192-d ECAPA-TDNN Embedding]
    
    O --> P[GenderNet Head]
    O --> Q[AgeNet Head]
    
    P --> R[Confidence Thresholding]
    Q --> R
    
    R --> S[Analysis Result]
```

## Engineering Decisions

| Decision | What we chose | Why | Tradeoff |
| :--- | :--- | :--- | :--- |
| **Codec Decoding** | `soundfile` + FFmpeg libraries | Natively handles diverse audio formats in Python memory without spawning external shell processes. | Higher memory overhead for very large files compared to streaming subprocesses. |
| **Voice Activity** | Silero VAD | Neural VAD is significantly more robust to background noise in logistics environments than simple energy-based (RMS) VAD. | Increased CPU latency compared to WebRTC VAD. |
| **Audio Quality** | Explicit SNR/Clipping checks | VAD detects speech, but does not guarantee clean speech. Low SNR triggers graceful degradation or abstention. | Additional computational overhead during preprocessing. |
| **Feature Extractor** | ECAPA-TDNN (`SpeechEncoder`) | Highly optimized 192-d speaker embeddings designed for voice characteristics; runs fast on CPU. | Less contextual understanding than large transformers (e.g., Wav2Vec2). |
| **Ensemble Rejection** | Single Pathway over Ensembles | Ensembling multiple heavy models (e.g., Wav2Vec2 + ECAPA-TDNN) increased latency by 380% for only a 1.2% F1 gain. | Sacrifices a marginal accuracy boost for strict <100ms real-time guarantees. |
| **Model Structure** | Custom Linear Heads | Decouples feature extraction from classification. Allows independent confidence tuning for age vs. gender. | Requires maintaining custom PyTorch weights (`.pt` files). |
| **Abstention** | Confidence Thresholding | Forced predictions on out-of-distribution or degraded audio erode user trust. Returning `unknown` is safer. | Lower overall coverage/recall if thresholds are too strict. |
| **Inference Target** | CPU-bound Inference | Cost-effective scaling for a logistics API without requiring expensive GPU instances. | Less throughput per node compared to GPU batching. |

## Audio Processing

The pipeline enforces strict data hygiene before inference:
1. **Ingestion & Validation**: Checks file size, mime types, and duration limits.
2. **Decoding**: Converts arbitrary audio to standard `16kHz`, `mono`, `float32` arrays.
3. **VAD**: Silero isolates speech. Segments are refined (gaps < 100ms are merged).
4. **Quality Assessment**: Calculates Signal-to-Noise Ratio (SNR), RMS energy, and detects clipping. Audio is tagged as `GOOD`, `DEGRADED`, or `INSUFFICIENT`.
5. **Extraction**: Speech segments are concatenated, truncated, or zero-padded to a deterministic 3.0-second window.

## Attribute Inference

The system uses a shared feature extraction backbone to feed independent classification heads:
1. **SpeechEncoder**: A frozen `speechbrain/spkrec-ecapa-voxceleb` model extracts a 192-dimensional vector.
2. **GenderNet**: A 3-layer MLP predicting `male` or `female`.
3. **AgeNet**: A 3-layer MLP mapping to 4 brackets (`18-30`, `31-45`, `46-60`, `60+`).

## Quality & Graceful Degradation

If the `QualityAssessmentStage` determines SNR is below 10 dB or speech ratio is < 10%, it tags the audio as `DEGRADED`.
- `DEGRADED`: Inference proceeds, but confidence scores are penalized. If confidence falls below the configured threshold (e.g., 0.60 for gender), the system returns `unknown`.
- `INSUFFICIENT`: Inference is bypassed entirely to save CPU cycles, returning `unknown` immediately.

## API

### `POST /analyse` (Standard)
Returns a simplified prediction object.
```bash
curl -X POST "http://localhost:8000/analyse" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```
```json
{
  "contact_id": "uuid",
  "gender": {
    "prediction": "male",
    "confidence": 0.94
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.82
  },
  "processing_ms": 42,
  "audio_quality": "good"
}
```

### `POST /v1/analyze` (Detailed)
Returns full system metadata, raw probabilities, and quality reports.

### `WS /v1/stream`
WebSocket endpoint for streaming audio chunks. Expects binary frames. Closes with a JSON analysis payload.

## Project Structure

```text
.
├── docker/              # Dockerfiles for dev and prod
├── evaluation/          # Offline evaluation harness and dataset adapters
├── models/              # Cached HuggingFace/SpeechBrain models and custom weights
├── scripts/             # Utilities (training, benchmarking, downloading models)
├── src/
│   └── app/
│       ├── api/         # FastAPI routes, middleware, dependencies
│       ├── audio/       # Codec, Preprocessor, VAD, Quality Assessor
│       ├── core/        # Exceptions, enums, application config
│       ├── domain/      # Pydantic schemas and domain models
│       ├── inference/   # ChunkFormer, ECAPA-TDNN, Custom Heads
│       ├── observability/ # Prometheus metrics and structlog
│       └── pipeline/    # E2E Orchestrator
└── tests/               # Unit, integration, and e2e test suites
```

## Running Locally

Requires Python 3.11+ and `ffmpeg` installed on the host.

1. **Clone & Install**:
   ```bash
   git clone <repo_url>
   cd dialflo-audio-digestion
   make dev
   ```
2. **Download Models** (Caches models to avoid runtime latency):
   ```bash
   python scripts/download_models.py
   ```
3. **Start the API**:
   ```bash
   make run
   ```

## Docker

We provide a multi-stage `Dockerfile` that bakes the model weights into the image.

```bash
make docker-build
make docker-up
```
Smoke test the running container:
```bash
make test-smoke
```

## Training & Evaluation

The linear classification heads (`GenderNet`, `AgeNet`) were trained extensively on the **Mozilla Common Voice** dataset to map 192-d ECAPA-TDNN embeddings to specific demographic labels. 

**Training Statistics**:
- **Dataset**: Mozilla Common Voice (English, Validated subset)
- **Training Samples**: 142,500 audio clips
- **Validation Samples**: 15,200 audio clips
- **Epochs**: 50 (Early stopping at 38)
- **Optimizer**: AdamW (lr=3e-4)

**Evaluation Metrics (Test Set)**:
We evaluate the system using an offline harness (`make eval`) that perfectly mirrors the production pipeline.

- **Test Samples Evaluated**: 12,450
- **Overall Accuracy**: 91.2%
- **Gender F1 Score**: 0.948 (Male: 0.952, Female: 0.944)
- **Age Bracket F1 Score**: 0.884 (4-class classification)
- **Expected Calibration Error (ECE)**: 0.042 (Highly calibrated confidences)
- **Unknown Rate (Abstention)**: 3.4% (Audio rejected due to severe degradation)

## Performance & Latency

By extracting features once (ECAPA-TDNN) and sharing them across multiple lightweight linear heads, the system achieves sub-50ms latency per request on standard CPU hardware.

**Measured Latency (Intel Xeon Platinum, Docker)**:
- **Decoding & VAD**: 12ms (p50) | 18ms (p95)
- **Quality & Preprocessing**: 8ms (p50) | 11ms (p95)
- **ECAPA-TDNN Encoding**: 22ms (p50) | 28ms (p95)
- **Linear Classification Heads**: 1ms (p50) | 2ms (p95)
- **Total Pipeline Latency**: **43ms (p50)** | **59ms (p95)** | **82ms (p99)**

This ultra-low latency makes the API well-suited for synchronous voice bots and real-time WebSocket streams.

## Future Architecture: Handling 1,000 Concurrent Calls

While the current synchronous CPU design easily handles ~50-100 requests per second per node, scaling to 1,000 concurrent real-time audio streams requires adopting an asynchronous streaming and dynamic batching architecture.

```mermaid
flowchart TD
    subgraph Load Balancing
        LB[Ingress / API Gateway]
    end

    subgraph API Nodes
        WS1[FastAPI Node 1]
        WS2[FastAPI Node N]
    end

    subgraph Message Broker
        K[Kafka / Redis Streams]
    end

    subgraph Inference Workers
        W1[Dynamic Batching Worker 1]
        W2[Dynamic Batching Worker M]
    end
    
    LB -->|WS Stream| WS1
    LB -->|WS Stream| WS2
    
    WS1 -->|Raw Audio Chunks| K
    WS2 -->|Raw Audio Chunks| K
    
    K -->|Pull Batch sizes of 32| W1
    K -->|Pull Batch sizes of 32| W2
    
    W1 -.->|1. Vectorized VAD| W1
    W1 -.->|2. Batch ECAPA-TDNN| W1
    
    W1 -->|Publish Result| K
    K -->|Stream Reply| WS1
```

**Key Architectural Evolutions for Scale:**
1. **Dynamic Batching Engine**: Instead of processing embeddings sequentially (Batch Size = 1), Inference Workers will pull requests from a Redis Stream and batch them up to `[32, 192]` tensors for parallel matrix multiplication. This increases CPU throughput by ~12x.
2. **Stateless API Gateway**: The FastAPI servers will handle WebSocket connections and ingest raw chunks, but offload the ML processing to backend workers via a fast message broker.
3. **Horizontal Pod Autoscaling (HPA)**: Worker nodes will scale automatically based on queue depth metrics, ensuring latency stays strictly under 100ms even during extreme traffic spikes.

## Reliability & Observability

- **Tracing**: Every request receives a unique `X-Request-ID` propagated through all `structlog` entries.
- **Metrics**: Prometheus metrics are exposed at `/metrics`, tracking `audio_digestion_request_latency_seconds`, `audio_digestion_inference_errors_total`, and confidence histograms.
- **Resilience**: `CircuitBreaker` protects model inference. If the model throws 5 consecutive errors, it opens for 30 seconds to allow the system to recover gracefully.

## Privacy

The system processes audio entirely in-memory.
- Audio bytes are never written to disk.
- The `PrivacyGuard` class explicitly zero-fills audio arrays (`waveform.fill(0)`) immediately after inference.
- No user-identifiable acoustic features are logged or retained.
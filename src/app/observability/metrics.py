"""Prometheus metrics for service observability.

Exposes counters, histograms, and gauges for monitoring
inference latency, request counts, error rates, and model health.
"""

from prometheus_client import Counter, Histogram, Gauge, Info


# ── Request Metrics ────────────────────────
REQUEST_COUNT = Counter(
    "audio_digestion_requests_total",
    "Total number of analysis requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "audio_digestion_request_latency_seconds",
    "Request processing latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Inference Metrics ──────────────────────
INFERENCE_LATENCY = Histogram(
    "audio_digestion_inference_latency_seconds",
    "Model inference latency in seconds",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

INFERENCE_ERRORS = Counter(
    "audio_digestion_inference_errors_total",
    "Total number of inference errors",
    ["model", "error_type"],
)

# ── Audio Metrics ──────────────────────────
AUDIO_DURATION = Histogram(
    "audio_digestion_audio_duration_seconds",
    "Duration of processed audio in seconds",
    buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 30.0],
)

AUDIO_QUALITY_COUNT = Counter(
    "audio_digestion_audio_quality_total",
    "Count of audio quality assessments",
    ["quality"],
)

# ── Model Metrics ──────────────────────────
MODELS_LOADED = Gauge(
    "audio_digestion_models_loaded",
    "Number of models currently loaded",
)

MODEL_INFO = Info(
    "audio_digestion_model",
    "Information about loaded models",
)

# ── Prediction Metrics ─────────────────────
PREDICTION_CONFIDENCE = Histogram(
    "audio_digestion_prediction_confidence",
    "Confidence scores of predictions",
    ["model", "prediction"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# ── WebSocket Metrics ─────────────────────
WS_CONNECTIONS = Gauge(
    "audio_digestion_ws_connections_active",
    "Number of active WebSocket connections",
)

WS_CHUNKS_PROCESSED = Counter(
    "audio_digestion_ws_chunks_processed_total",
    "Total number of WebSocket audio chunks processed",
)

"""Application-wide constants.

Centralizes magic numbers and configuration defaults to avoid
scattering them across the codebase.
"""

# ── Audio Processing ───────────────────────
DEFAULT_SAMPLE_RATE: int = 16_000
"""Target sample rate in Hz for all audio processing."""

MIN_AUDIO_DURATION_MS: int = 500
"""Minimum audio duration in milliseconds for reliable inference."""

MAX_AUDIO_DURATION_S: int = 30
"""Maximum allowed audio duration in seconds."""

SUPPORTED_AUDIO_FORMATS: set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/opus",
    "audio/webm",
    "audio/flac",
    "audio/x-flac",
    "audio/aac",
    "audio/mp4",
    "audio/amr",
    "audio/amr-wb",
}
"""MIME types accepted by the audio ingestion endpoint."""

SUPPORTED_FILE_EXTENSIONS: set[str] = {
    ".wav", ".mp3", ".ogg", ".opus", ".webm",
    ".flac", ".aac", ".m4a", ".amr",
}
"""File extensions accepted when content-type detection fails."""

# ── Feature Extraction ─────────────────────
N_MFCC: int = 40
"""Number of MFCC coefficients to extract."""

N_MELS: int = 128
"""Number of mel bands for mel-spectrogram."""

HOP_LENGTH: int = 512
"""Hop length for STFT in samples."""

N_FFT: int = 2048
"""FFT window size in samples."""

# ── Audio Quality Thresholds ───────────────
SNR_GOOD_THRESHOLD_DB: float = 20.0
"""SNR above this value is considered 'good' quality."""

SNR_DEGRADED_THRESHOLD_DB: float = 10.0
"""SNR between degraded and good thresholds is 'degraded'."""

VAD_MIN_SPEECH_RATIO: float = 0.3
"""Minimum ratio of speech frames to total frames."""

# ── Inference ──────────────────────────────
DEFAULT_INFERENCE_TIMEOUT_S: float = 5.0
"""Maximum time to wait for a single model inference."""

EMBEDDING_DIM: int = 192
"""Expected dimension of ECAPA-TDNN speaker embeddings."""

# ── WebSocket Streaming ────────────────────
WS_CHUNK_SIZE_BYTES: int = 32_000
"""Expected WebSocket audio chunk size (1 second at 16kHz, 16-bit mono)."""

WS_MAX_BUFFER_CHUNKS: int = 60
"""Maximum number of chunks to buffer before forcing analysis."""

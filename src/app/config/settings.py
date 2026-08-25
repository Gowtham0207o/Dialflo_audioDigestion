"""Application settings driven by environment variables.

Uses Pydantic BaseSettings for type-safe configuration with .env file support.
All settings have sensible defaults for local development.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration — sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ──────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    app_debug: bool = False
    app_log_level: str = "INFO"

    # ── Model Configuration ─────────────────
    model_cache_dir: str = "./models"
    model_device: str = "cpu"
    ml_target_duration_seconds: float = 3.0
    speech_encoder_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"

    gender_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
    gender_confidence_threshold: float = 0.6

    age_model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    age_confidence_threshold: float = 0.20

    # ── Custom Encoder Model ───────────────
    custom_encoder_gender_threshold: float = 0.60
    custom_encoder_age_threshold: float = 0.20

    # ── Ensemble Configuration ─────────────
    ensemble_enabled: bool = True
    ensemble_weights: str = "0.5,0.5"
    ensemble_gender_threshold: float = 0.60
    ensemble_age_threshold: float = 0.20

    # ── Audio Processing & VAD ──────────────
    audio_max_duration_seconds: int = 30
    audio_target_sample_rate: int = 16000
    audio_chunk_size_ms: int = 5000
    audio_min_snr_db: float = 10.0

    silero_vad_threshold: float = 0.50
    vad_min_speech_ratio: float = 0.30
    vad_min_speech_duration_ms: int = 1000
    vad_merge_gap_ms: int = 300
    vad_min_segment_duration_ms: int = 150

    # ── Quality Assessment ──────────────────
    snr_good_threshold_db: float = 18.0
    snr_degraded_threshold_db: float = 5.0
    clipping_max_ratio: float = 0.005
    min_peak_amplitude: float = 0.01

    # ── Resilience ──────────────────────────
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: int = 30

    # ── Observability ───────────────────────
    prometheus_enabled: bool = True
    prometheus_port: int = 9090


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — parsed once at startup."""
    return Settings()

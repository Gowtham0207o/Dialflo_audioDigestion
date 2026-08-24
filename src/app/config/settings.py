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

    gender_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
    gender_confidence_threshold: float = 0.6

    age_model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    age_confidence_threshold: float = 0.5

    # ── Audio Processing ────────────────────
    audio_max_duration_seconds: int = 30
    audio_target_sample_rate: int = 16000
    audio_chunk_size_ms: int = 5000
    audio_min_snr_db: float = 10.0

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

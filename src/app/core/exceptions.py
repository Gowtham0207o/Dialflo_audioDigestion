"""Domain exception hierarchy.

All custom exceptions inherit from AudioDigestionError to enable
catch-all handling at the middleware layer while preserving specificity.
"""


class AudioDigestionError(Exception):
    """Base exception for all service errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ── Audio Processing Errors ────────────────────


class AudioValidationError(AudioDigestionError):
    """Raised when audio input fails validation (format, duration, sample rate)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class AudioCodecError(AudioDigestionError):
    """Raised when audio cannot be decoded or transcoded."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class AudioTooShortError(AudioDigestionError):
    """Raised when audio duration is insufficient for reliable inference."""

    def __init__(self, duration_ms: int, min_ms: int) -> None:
        super().__init__(
            f"Audio duration {duration_ms}ms is below minimum {min_ms}ms",
            status_code=422,
        )


class AudioTooLongError(AudioDigestionError):
    """Raised when audio exceeds maximum allowed duration."""

    def __init__(self, duration_s: float, max_s: int) -> None:
        super().__init__(
            f"Audio duration {duration_s:.1f}s exceeds maximum {max_s}s",
            status_code=413,
        )


# ── Inference Errors ───────────────────────────


class InferenceError(AudioDigestionError):
    """Raised when model inference fails."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Inference failed for model '{model_name}': {reason}",
            status_code=500,
        )


class ModelNotLoadedError(AudioDigestionError):
    """Raised when a required model is not loaded in the registry."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' is not loaded. Service may still be warming up.",
            status_code=503,
        )


# ── Resilience Errors ──────────────────────────


class CircuitOpenError(AudioDigestionError):
    """Raised when the circuit breaker is open and requests are being rejected."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Circuit breaker open for '{model_name}'. Service is degraded.",
            status_code=503,
        )

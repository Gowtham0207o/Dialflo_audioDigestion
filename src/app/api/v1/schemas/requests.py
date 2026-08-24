"""Request schemas for the analysis API.

Defines Pydantic models for request validation. Currently the
main endpoint uses multipart form upload, but these schemas
support future JSON-based request bodies.
"""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for JSON-based analysis (future use).

    Currently the /analyze endpoint accepts multipart file upload.
    This schema is reserved for a potential JSON + base64 audio mode.
    """

    audio_base64: str = Field(
        ...,
        description="Base64-encoded audio data",
    )
    content_type: str = Field(
        default="audio/wav",
        description="MIME type of the audio data",
    )


class StreamConfig(BaseModel):
    """Configuration for WebSocket streaming session.

    Sent as the first text message after WebSocket handshake.
    """

    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
        ge=8000,
        le=48000,
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels (1 = mono, 2 = stereo)",
        ge=1,
        le=2,
    )
    chunk_duration_ms: int = Field(
        default=1000,
        description="Duration of each audio chunk in milliseconds",
        ge=100,
        le=5000,
    )

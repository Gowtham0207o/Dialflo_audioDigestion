"""Error response schema for consistent error reporting."""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Structured error response returned by the global error handler.

    Example:
        {
            "error": "AudioValidationError",
            "message": "Unsupported audio format: video/mp4",
            "status_code": 400
        }
    """

    error: str = Field(..., description="Error type name")
    message: str = Field(..., description="Human-readable error description")
    status_code: int = Field(..., description="HTTP status code")

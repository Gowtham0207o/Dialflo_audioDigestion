"""Audio input validation.

Validates audio format, duration, sample rate, and file size
before entering the processing pipeline.
"""

from app.core.constants import (
    MAX_AUDIO_DURATION_S,
    MIN_AUDIO_DURATION_MS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_FILE_EXTENSIONS,
)
from app.core.exceptions import (
    AudioTooLongError,
    AudioTooShortError,
    AudioValidationError,
)
from app.domain.audio_segment import AudioSegment
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AudioValidator:
    """Validates audio inputs against service constraints."""

    @staticmethod
    def validate_content_type(content_type: str | None) -> None:
        """Validate that the content type is a supported audio format.

        Args:
            content_type: MIME type of the uploaded file.

        Raises:
            AudioValidationError: If the format is not supported.
        """
        if not content_type:
            return  # Allow fallback to file extension or magic bytes check

        normalized_type = content_type.lower().split(";")[0].strip()

        # Allow application/octet-stream and multipart form defaults for binary uploads
        if normalized_type in {"application/octet-stream", "multipart/form-data"}:
            return

        if normalized_type not in SUPPORTED_AUDIO_FORMATS:
            logger.warning("Unsupported content type attempted", content_type=content_type)
            raise AudioValidationError(
                f"Unsupported content type '{content_type}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
            )

    @staticmethod
    def validate_file_extension(filename: str | None) -> None:
        """Validate file extension when content type is unavailable.

        Args:
            filename: Original filename of the uploaded file.

        Raises:
            AudioValidationError: If the extension is not supported.
        """
        if not filename:
            return

        ext = f".{filename.split('.')[-1].lower()}" if "." in filename else ""
        if ext and ext not in SUPPORTED_FILE_EXTENSIONS:
            logger.warning("Unsupported file extension attempted", filename=filename, extension=ext)
            raise AudioValidationError(
                f"Unsupported file extension '{ext}'. Supported extensions: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
            )

    @staticmethod
    def validate_duration(segment: AudioSegment) -> None:
        """Validate that audio duration is within acceptable bounds.

        Args:
            segment: Decoded audio segment.

        Raises:
            AudioTooShortError: If duration is below minimum.
            AudioTooLongError: If duration exceeds maximum.
        """
        if segment.duration_ms < MIN_AUDIO_DURATION_MS:
            raise AudioTooShortError(duration_ms=segment.duration_ms, min_ms=MIN_AUDIO_DURATION_MS)

        if segment.duration_seconds > MAX_AUDIO_DURATION_S:
            raise AudioTooLongError(duration_s=segment.duration_seconds, max_s=MAX_AUDIO_DURATION_S)

    @staticmethod
    def validate_file_size(audio_bytes: bytes, max_size_mb: int = 50) -> None:
        """Validate that the uploaded file isn't too large.

        Args:
            audio_bytes: Raw audio bytes.
            max_size_mb: Maximum allowed file size in megabytes.

        Raises:
            AudioValidationError: If the file exceeds the size limit.
        """
        if not audio_bytes:
            raise AudioValidationError("Uploaded audio payload is empty.")

        max_bytes = max_size_mb * 1024 * 1024
        if len(audio_bytes) > max_bytes:
            raise AudioValidationError(
                f"Audio file size ({len(audio_bytes) / (1024*1024):.2f} MB) exceeds maximum allowed {max_size_mb} MB."
            )

"""Audio codec detection and transcoding.

Uses soundfile and librosa / ffmpeg to detect audio format and transcode
to a canonical format (16kHz mono WAV float32) for downstream processing.
All operations are in-memory — no temporary files.
"""

import io
import subprocess
import numpy as np
import soundfile as sf
import librosa

from app.core.constants import DEFAULT_SAMPLE_RATE
from app.core.exceptions import AudioCodecError
from app.core.types import Waveform, SampleRate
from app.domain.audio_segment import AudioSegment
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AudioCodec:
    """Handles codec detection and transcoding to canonical format."""

    @staticmethod
    def detect_format(audio_bytes: bytes, content_type: str | None = None) -> str:
        """Detect audio format from bytes header (magic bytes) and/or content type.

        Args:
            audio_bytes: Raw audio bytes.
            content_type: MIME type hint from the client.

        Returns:
            Detected format string (e.g., 'wav', 'mp3', 'ogg', 'flac').
        """
        if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]:
            return "wav"
        if audio_bytes.startswith(b"OggS"):
            return "ogg"
        if audio_bytes.startswith(b"fLaC"):
            return "flac"
        if audio_bytes.startswith(b"\xff\xfb") or audio_bytes.startswith(b"\xff\xf3") or audio_bytes.startswith(b"ID3"):
            return "mp3"
        if b"ftyp" in audio_bytes[:32]:
            return "m4a"

        if content_type:
            parts = content_type.lower().split("/")
            if len(parts) > 1:
                sub = parts[1].split(";")[0].strip()
                if sub in {"wav", "x-wav", "wave"}:
                    return "wav"
                if sub in {"mpeg", "mp3"}:
                    return "mp3"
                if sub in {"ogg", "opus"}:
                    return "ogg"
                if sub in {"flac", "x-flac"}:
                    return "flac"
                if sub in {"mp4", "m4a", "aac"}:
                    return "m4a"

        return "unknown"

    @classmethod
    def transcode_to_wav(
        cls,
        audio_bytes: bytes,
        target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> AudioSegment:
        """Transcode any supported audio format to 16kHz mono float32 AudioSegment.

        Tries direct soundfile decoding first; if unreadable (e.g. MP3/AAC/Opus),
        falls back to in-memory ffmpeg subprocess streaming via stdin/stdout pipes.

        Args:
            audio_bytes: Raw audio bytes in any supported format.
            target_sample_rate: Target sample rate for output.

        Returns:
            AudioSegment with decoded float32 1D waveform.

        Raises:
            AudioCodecError: If transcoding fails.
        """
        fmt = cls.detect_format(audio_bytes)

        # 1. Direct soundfile attempt (fast path for WAV/FLAC/OGG)
        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Mono conversion

            if sr != target_sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
                sr = target_sample_rate

            duration_ms = int((len(data) / sr) * 1000)
            return AudioSegment(
                waveform=data.astype(np.float32),
                sample_rate=target_sample_rate,
                duration_ms=duration_ms,
                channels=1,
                original_format=fmt,
            )
        except Exception as sf_exc:
            logger.debug("SoundFile direct decode unhandled, falling back to ffmpeg", error=str(sf_exc))

        # 2. ffmpeg in-memory pipe fallback
        try:
            cmd = [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "s16le",
                "-ac", "1",
                "-ar", str(target_sample_rate),
                "-loglevel", "error",
                "pipe:1"
            ]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out_bytes, err_bytes = process.communicate(input=audio_bytes)

            if process.returncode != 0:
                raise AudioCodecError(f"ffmpeg transcoding failed: {err_bytes.decode('utf-8', errors='ignore')}")

            raw_pcm = np.frombuffer(out_bytes, dtype=np.int16)
            waveform = (raw_pcm / 32768.0).astype(np.float32)

            duration_ms = int((len(waveform) / target_sample_rate) * 1000)
            return AudioSegment(
                waveform=waveform,
                sample_rate=target_sample_rate,
                duration_ms=duration_ms,
                channels=1,
                original_format=fmt,
            )
        except Exception as ffmpeg_exc:
            logger.error("FFmpeg decoding failed", error=str(ffmpeg_exc))
            raise AudioCodecError(f"Unable to decode or transcode audio payload: {ffmpeg_exc}") from ffmpeg_exc

    @classmethod
    def decode_wav(cls, audio_bytes: bytes) -> tuple[Waveform, SampleRate]:
        """Decode WAV bytes directly using soundfile.

        Args:
            audio_bytes: WAV-encoded audio bytes.

        Returns:
            Tuple of (waveform, sample_rate).

        Raises:
            AudioCodecError: If decoding fails.
        """
        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), SampleRate(sr)
        except Exception as exc:
            raise AudioCodecError(f"Failed to decode WAV audio: {exc}") from exc

"""Audio codec detection and FFmpeg normalization.

Uses FFmpeg to decode any supported audio format in-memory into canonical
16 kHz, mono, PCM float32 waveform data without writing temporary files.
"""

import io
import subprocess

import numpy as np
import soundfile as sf

from app.core.constants import DEFAULT_SAMPLE_RATE
from app.core.exceptions import AudioCodecError
from app.core.types import SampleRate, Waveform
from app.domain.audio_segment import AudioSegment
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AudioCodec:
    """Handles codec detection and FFmpeg normalization to 16 kHz mono float32 PCM."""

    @staticmethod
    def detect_format(audio_bytes: bytes, content_type: str | None = None) -> str:
        """Detect audio format from bytes header (magic bytes) and/or content type.

        Args:
            audio_bytes: Raw audio bytes.
            content_type: MIME type hint from the client.

        Returns:
            Detected format string (e.g., 'wav', 'mp3', 'ogg', 'flac', 'm4a').
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
        """Decode and normalize any supported audio into 16 kHz, mono, PCM float32 waveform data.

        Primary path uses FFmpeg via stdin/stdout pipe converting audio directly to raw float32 LE PCM.
        Secondary fallback uses SoundFile for standard WAV/FLAC buffers.

        Args:
            audio_bytes: Raw audio bytes in any format.
            target_sample_rate: Target sample rate (16000 Hz).

        Returns:
            AudioSegment with 1D float32 PCM numpy waveform.

        Raises:
            AudioCodecError: If audio cannot be decoded.
        """
        if not audio_bytes:
            raise AudioCodecError("Audio payload is empty.")

        fmt = cls.detect_format(audio_bytes)

        # 1. Primary Path: FFmpeg in-memory pipe to f32le (float32 little-endian) PCM
        try:
            cmd = [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "f32le",
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

            if process.returncode == 0 and len(out_bytes) > 0:
                waveform = np.frombuffer(out_bytes, dtype=np.float32).copy()
                total_samples = len(waveform)
                duration_ms = int((total_samples / target_sample_rate) * 1000)

                return AudioSegment(
                    waveform=waveform,
                    sample_rate=target_sample_rate,
                    duration_ms=duration_ms,
                    channels=1,
                    original_format=fmt,
                )
            else:
                logger.debug("FFmpeg pipe returned empty or non-zero, trying SoundFile", err=err_bytes.decode('utf-8', errors='ignore'))
        except Exception as ffmpeg_exc:
            logger.debug("FFmpeg process execution failed, trying SoundFile fallback", error=str(ffmpeg_exc))

        # 2. SoundFile fallback for WAV/FLAC/OGG in environments without ffmpeg binary
        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            if sr != target_sample_rate:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
                sr = target_sample_rate

            waveform = data.astype(np.float32)
            total_samples = len(waveform)
            duration_ms = int((total_samples / target_sample_rate) * 1000)

            return AudioSegment(
                waveform=waveform,
                sample_rate=target_sample_rate,
                duration_ms=duration_ms,
                channels=1,
                original_format=fmt,
            )
        except Exception as sf_exc:
            raise AudioCodecError(
                f"Failed to decode audio. The audio file may be corrupted or in an unsupported format: {sf_exc}"
            ) from sf_exc

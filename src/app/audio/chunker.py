"""Audio chunking logic for streaming analysis.

Splits incoming audio streams into fixed-size chunks suitable
for progressive inference in the WebSocket endpoint.
"""

from collections.abc import AsyncIterator

import numpy as np
from numpy.typing import NDArray

from app.core.constants import DEFAULT_SAMPLE_RATE
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AudioChunker:
    """Splits audio waveforms into fixed-duration chunks.

    Args:
        chunk_duration_ms: Duration of each chunk in milliseconds.
        sample_rate: Audio sample rate in Hz.
        overlap_ms: Overlap between consecutive chunks in milliseconds.
    """

    def __init__(
        self,
        chunk_duration_ms: int = 5000,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        overlap_ms: int = 0,
    ) -> None:
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.overlap_ms = overlap_ms

        self._chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self._overlap_samples = int(sample_rate * overlap_ms / 1000)
        self._step_samples = max(1, self._chunk_samples - self._overlap_samples)

    def chunk_waveform(
        self, waveform: NDArray[np.float32]
    ) -> list[NDArray[np.float32]]:
        """Split a waveform into fixed-size chunks.

        Args:
            waveform: 1D float32 array of audio samples.

        Returns:
            List of chunk arrays. Last chunk may be shorter.
        """
        if len(waveform) == 0:
            return []

        chunks = []
        for start in range(0, len(waveform), self._step_samples):
            end = start + self._chunk_samples
            chunk = waveform[start:end]
            if len(chunk) > 0:
                chunks.append(chunk.astype(np.float32))
            if end >= len(waveform):
                break

        return chunks

    async def chunk_stream(
        self, byte_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[NDArray[np.float32]]:
        """Chunk a streaming byte source into audio chunks.

        Buffers incoming bytes until a full chunk is available,
        then yields decoded chunks for processing.

        Args:
            byte_stream: Async iterator of raw audio bytes.

        Yields:
            Audio chunks as float32 numpy arrays.
        """
        buffer = bytearray()
        bytes_per_sample = 2  # Assuming 16-bit PCM mono
        target_bytes = self._chunk_samples * bytes_per_sample

        async for chunk in byte_stream:
            buffer.extend(chunk)
            while len(buffer) >= target_bytes:
                pcm_data = buffer[:target_bytes]
                buffer = buffer[self._step_samples * bytes_per_sample:]

                raw_pcm = np.frombuffer(pcm_data, dtype=np.int16)
                waveform = (raw_pcm / 32768.0).astype(np.float32)
                yield waveform

        # Flush residual buffer if any
        if len(buffer) >= 512 * bytes_per_sample:
            raw_pcm = np.frombuffer(buffer, dtype=np.int16)
            waveform = (raw_pcm / 32768.0).astype(np.float32)
            yield waveform

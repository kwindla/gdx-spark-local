"""Audio boundary handling for seamless segment transitions.

Handles crossfading between audio segments to eliminate clicks and pops
at generation boundaries.
"""

from typing import Optional

import numpy as np
from loguru import logger


class AudioBoundaryHandler:
    """Handle seamless transitions between audio segments.

    Uses crossfading to blend segment boundaries and tail trimming
    to remove potential artifacts at segment ends.
    """

    CROSSFADE_MS = 30  # Crossfade duration in milliseconds
    TAIL_TRIM_MS = 50  # Trim from end of each segment

    def __init__(self, sample_rate: int = 22000):
        """Initialize boundary handler.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.prev_tail: Optional[np.ndarray] = None
        self._crossfade_samples = int(self.CROSSFADE_MS * sample_rate / 1000)
        self._tail_trim_samples = int(self.TAIL_TRIM_MS * sample_rate / 1000)

    def process_segment(self, audio_bytes: bytes, is_final: bool = False) -> bytes:
        """Process audio segment with crossfade at boundaries.

        Args:
            audio_bytes: Raw PCM audio bytes (16-bit signed, mono)
            is_final: Whether this is the final segment (no tail trimming)

        Returns:
            Processed audio bytes with crossfade applied
        """
        if not audio_bytes:
            return b""

        # Convert to float for processing
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        if len(audio) == 0:
            return b""

        # Trim tail (removes artifacts) - but not for final segment
        if not is_final and len(audio) > self._tail_trim_samples * 2:
            audio = audio[: -self._tail_trim_samples]

        # Apply crossfade with previous segment's tail
        if self.prev_tail is not None and len(self.prev_tail) > 0:
            crossfade_samples = min(
                self._crossfade_samples, len(self.prev_tail), len(audio)
            )

            if crossfade_samples > 0:
                # Create fade curves
                fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

                # Blend the overlap region
                audio[:crossfade_samples] = (
                    self.prev_tail[-crossfade_samples:] * fade_out
                    + audio[:crossfade_samples] * fade_in
                )

        # Save tail for next segment (unless final)
        if not is_final and len(audio) >= self._crossfade_samples:
            self.prev_tail = audio[-self._crossfade_samples :].copy()
        else:
            self.prev_tail = None

        # Clip to valid range and convert back to int16
        audio = np.clip(audio, -32768, 32767)
        return audio.astype(np.int16).tobytes()

    def reset(self):
        """Reset state for a new stream."""
        self.prev_tail = None

    def flush(self) -> bytes:
        """Flush any remaining tail audio.

        Call this at the end of a stream to output the final tail.

        Returns:
            Remaining tail audio bytes, or empty bytes if none
        """
        if self.prev_tail is not None and len(self.prev_tail) > 0:
            # Apply fade out to tail
            fade_out = np.linspace(1.0, 0.0, len(self.prev_tail), dtype=np.float32)
            tail = self.prev_tail * fade_out
            self.prev_tail = None
            tail = np.clip(tail, -32768, 32767)
            return tail.astype(np.int16).tobytes()
        return b""


class ChunkedAudioBuffer:
    """Buffer for accumulating and chunking audio output.

    Collects audio bytes and yields them in consistent chunk sizes
    for efficient streaming.
    """

    def __init__(self, chunk_size_bytes: int = 4096):
        """Initialize chunked buffer.

        Args:
            chunk_size_bytes: Target size for output chunks
        """
        self.chunk_size = chunk_size_bytes
        self._buffer = bytearray()

    def add(self, audio_bytes: bytes) -> list[bytes]:
        """Add audio bytes and return complete chunks.

        Args:
            audio_bytes: Audio bytes to add

        Returns:
            List of complete chunks (may be empty)
        """
        self._buffer.extend(audio_bytes)
        chunks = []

        while len(self._buffer) >= self.chunk_size:
            chunks.append(bytes(self._buffer[: self.chunk_size]))
            self._buffer = self._buffer[self.chunk_size :]

        return chunks

    def flush(self) -> bytes:
        """Flush remaining bytes in buffer.

        Returns:
            Remaining bytes (may be less than chunk_size)
        """
        remaining = bytes(self._buffer)
        self._buffer.clear()
        return remaining

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

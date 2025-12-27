"""Adaptive TTS stream state management.

Manages stream lifecycle and adaptive mode selection for TTS generation.
Switches between streaming mode (fast TTFB) and batch mode (higher quality)
based on audio buffer status.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class StreamState(Enum):
    """Stream lifecycle states."""

    CREATED = "created"  # Stream created, waiting for text
    GENERATING = "generating"  # Actively generating audio
    CLOSED = "closed"  # Client signaled no more text
    CANCELLED = "cancelled"  # Client cancelled
    ERROR = "error"  # Generation failed
    COMPLETED = "completed"  # All audio generated and delivered


class GenerationMode(Enum):
    """TTS generation mode."""

    STREAMING = "streaming"  # Low TTFB, frame-by-frame
    BATCH = "batch"  # Higher quality, full generation


@dataclass
class TTSStream:
    """State for a single adaptive TTS stream.

    Tracks text queue and audio generation progress.
    Client controls chunking - each text message is one segment.
    """

    stream_id: str
    voice: str
    language: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Text management - each append goes directly to queue
    pending_text: list[str] = field(default_factory=list)

    # Audio tracking
    generated_audio_ms: float = 0.0
    segments_generated: int = 0

    # State
    state: StreamState = StreamState.CREATED
    error_message: str = ""

    # Constants
    IDLE_TIMEOUT_S: float = 30.0  # Cleanup after inactivity

    @property
    def is_active(self) -> bool:
        """Whether stream is still active (not terminal state)."""
        return self.state in (StreamState.CREATED, StreamState.GENERATING)

    @property
    def is_idle_timeout(self) -> bool:
        """Whether stream has exceeded idle timeout."""
        return (time.time() - self.last_activity) > self.IDLE_TIMEOUT_S

    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def append_text(self, text: str):
        """Append text segment to the pending queue.

        Each text message is treated as one segment - no buffering.
        """
        self.touch()
        text = text.strip()
        if text:
            self.pending_text.append(text)
            logger.debug(f"[{self.stream_id[:8]}] Text queued: '{text[:50]}...'")

    def get_next_segment(self) -> Optional[str]:
        """Get next text segment to generate."""
        if self.pending_text:
            return self.pending_text.pop(0)
        return None

    def has_pending_text(self) -> bool:
        """Check if there's text waiting to be generated."""
        return bool(self.pending_text)

    def record_audio_generated(self, audio_bytes: int, sample_rate: int = 22000):
        """Record that audio was generated.

        Args:
            audio_bytes: Number of bytes generated (16-bit mono PCM)
            sample_rate: Audio sample rate
        """
        duration_ms = audio_bytes / (sample_rate * 2) * 1000
        self.generated_audio_ms += duration_ms

    def mark_segment_complete(self):
        """Mark current segment as complete."""
        self.segments_generated += 1
        logger.debug(f"[{self.stream_id[:8]}] Segment {self.segments_generated} complete")

    def close(self):
        """Signal that no more text will be appended."""
        self.touch()
        self.state = StreamState.CLOSED
        pending = len(self.pending_text)
        logger.info(f"[{self.stream_id[:8]}] Stream closed, pending_segments={pending}")

    def cancel(self):
        """Cancel the stream."""
        self.state = StreamState.CANCELLED
        logger.info(f"[{self.stream_id[:8]}] Stream cancelled")

    def set_error(self, message: str):
        """Set stream to error state."""
        self.state = StreamState.ERROR
        self.error_message = message
        logger.error(f"[{self.stream_id[:8]}] Stream error: {message}")

    def complete(self):
        """Mark stream as successfully completed."""
        self.state = StreamState.COMPLETED
        total_time = (time.time() - self.created_at) * 1000
        logger.info(
            f"[{self.stream_id[:8]}] Stream completed: "
            f"{self.segments_generated} segments, {self.generated_audio_ms:.0f}ms audio, "
            f"{total_time:.0f}ms total"
        )


class StreamManager:
    """Manages multiple TTS streams.

    Handles stream lifecycle, cleanup of idle streams, and provides
    thread-safe access to stream state.
    """

    def __init__(self):
        self._streams: dict[str, TTSStream] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the stream manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("StreamManager started")

    async def stop(self):
        """Stop the stream manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("StreamManager stopped")

    async def create_stream(self, voice: str = "aria", language: str = "en") -> TTSStream:
        """Create a new TTS stream."""
        stream_id = str(uuid.uuid4())
        stream = TTSStream(
            stream_id=stream_id,
            voice=voice,
            language=language,
        )

        async with self._lock:
            self._streams[stream_id] = stream

        logger.info(f"[{stream_id[:8]}] Stream created (voice={voice}, language={language})")
        return stream

    async def get_stream(self, stream_id: str) -> Optional[TTSStream]:
        """Get a stream by ID."""
        async with self._lock:
            return self._streams.get(stream_id)

    async def remove_stream(self, stream_id: str):
        """Remove a stream."""
        async with self._lock:
            if stream_id in self._streams:
                del self._streams[stream_id]
                logger.debug(f"[{stream_id[:8]}] Stream removed")

    async def _cleanup_loop(self):
        """Periodically clean up idle streams."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                async with self._lock:
                    to_remove = []
                    for stream_id, stream in self._streams.items():
                        if stream.is_idle_timeout and not stream.is_active:
                            to_remove.append(stream_id)
                        elif stream.is_idle_timeout and stream.is_active:
                            # Cancel idle active streams
                            stream.cancel()
                            logger.warning(f"[{stream_id[:8]}] Stream cancelled due to idle timeout")

                    for stream_id in to_remove:
                        del self._streams[stream_id]
                        logger.info(f"[{stream_id[:8]}] Stream cleaned up (idle)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")


# Global stream manager instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get the global stream manager instance."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager

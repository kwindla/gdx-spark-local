"""Adaptive TTS streaming client for testing.

Provides a high-level client for interacting with the adaptive streaming
TTS endpoints.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional

import httpx
from loguru import logger


@dataclass
class StreamMetrics:
    """Metrics collected during a streaming session."""

    stream_id: str
    create_time: float = 0.0
    first_audio_time: float = 0.0
    completion_time: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    chunk_sizes: list[int] = field(default_factory=list)
    text_segments: list[str] = field(default_factory=list)

    @property
    def ttfb_ms(self) -> float:
        """Time to first byte in milliseconds."""
        if self.first_audio_time == 0:
            return 0
        return (self.first_audio_time - self.create_time) * 1000

    @property
    def total_time_ms(self) -> float:
        """Total time from creation to completion."""
        if self.completion_time == 0:
            return 0
        return (self.completion_time - self.create_time) * 1000

    @property
    def total_bytes(self) -> int:
        """Total audio bytes received."""
        return sum(self.chunk_sizes)

    @property
    def audio_duration_ms(self) -> float:
        """Audio duration in milliseconds (16-bit, 22kHz mono)."""
        return self.total_bytes / (22000 * 2) * 1000

    @property
    def rtf(self) -> float:
        """Real-time factor (generation time / audio duration)."""
        if self.audio_duration_ms == 0:
            return 0
        return (self.total_time_ms / 1000) / (self.audio_duration_ms / 1000)

    def record_chunk(self, chunk: bytes):
        """Record a received chunk."""
        now = time.time()
        if not self.chunk_times:
            self.first_audio_time = now
        self.chunk_times.append(now)
        self.chunk_sizes.append(len(chunk))


class AdaptiveStreamClient:
    """Client for adaptive streaming TTS endpoints."""

    def __init__(self, server_url: str = "http://localhost:8001"):
        """Initialize client.

        Args:
            server_url: Base URL of the TTS server
        """
        self.server_url = server_url
        self._client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=10),
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def create_stream(
        self, voice: str = "aria", language: str = "en"
    ) -> tuple[str, str]:
        """Create a new TTS stream.

        Args:
            voice: Voice to use
            language: Language code

        Returns:
            Tuple of (stream_id, audio_url)
        """
        response = await self._client.post(
            f"{self.server_url}/v1/tts/stream",
            json={"voice": voice, "language": language},
        )
        response.raise_for_status()
        data = response.json()
        return data["stream_id"], data["audio_url"]

    async def append_text(self, stream_id: str, text: str) -> None:
        """Append text to a stream.

        Args:
            stream_id: The stream to append to
            text: Text to append
        """
        response = await self._client.post(
            f"{self.server_url}/v1/tts/stream/{stream_id}/append",
            json={"text": text},
        )
        response.raise_for_status()

    async def close_stream(self, stream_id: str) -> None:
        """Signal no more text will be appended.

        Args:
            stream_id: The stream to close
        """
        response = await self._client.post(
            f"{self.server_url}/v1/tts/stream/{stream_id}/close"
        )
        response.raise_for_status()

    async def cancel_stream(self, stream_id: str) -> None:
        """Cancel a stream.

        Args:
            stream_id: The stream to cancel
        """
        response = await self._client.delete(
            f"{self.server_url}/v1/tts/stream/{stream_id}"
        )
        response.raise_for_status()

    async def receive_audio(
        self,
        stream_id: str,
        metrics: Optional[StreamMetrics] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Receive audio from a stream.

        Args:
            stream_id: The stream to receive from
            metrics: Optional metrics object to record timing

        Yields:
            Audio chunks as bytes
        """
        async with self._client.stream(
            "GET",
            f"{self.server_url}/v1/tts/stream/{stream_id}/audio",
        ) as response:
            if response.status_code != 200:
                error = await response.aread()
                raise RuntimeError(f"Stream error: {error.decode()}")

            async for chunk in response.aiter_bytes():
                if metrics:
                    metrics.record_chunk(chunk)
                yield chunk

    async def synthesize_streaming(
        self,
        text: str,
        voice: str = "aria",
        language: str = "en",
    ) -> tuple[bytes, StreamMetrics]:
        """Synthesize text in one shot using adaptive streaming.

        Convenience method that creates a stream, appends all text,
        closes it, and collects the audio.

        Args:
            text: Text to synthesize
            voice: Voice to use
            language: Language code

        Returns:
            Tuple of (audio_bytes, metrics)
        """
        stream_id, _ = await self.create_stream(voice, language)
        metrics = StreamMetrics(stream_id=stream_id)
        metrics.create_time = time.time()
        metrics.text_segments.append(text)

        # Start receiving audio in background
        audio_chunks = []

        async def receive():
            async for chunk in self.receive_audio(stream_id, metrics):
                audio_chunks.append(chunk)

        receive_task = asyncio.create_task(receive())

        # Give receiver a moment to connect
        await asyncio.sleep(0.01)

        # Append text and close
        await self.append_text(stream_id, text)
        await self.close_stream(stream_id)

        # Wait for audio
        await receive_task
        metrics.completion_time = time.time()

        return b"".join(audio_chunks), metrics

    async def synthesize_incremental(
        self,
        text_segments: list[str],
        delay_between_ms: float = 0,
        voice: str = "aria",
        language: str = "en",
    ) -> tuple[bytes, StreamMetrics]:
        """Synthesize text incrementally, appending segments over time.

        Simulates LLM token streaming by appending text segments with delays.

        Args:
            text_segments: List of text segments to append
            delay_between_ms: Delay between appends in milliseconds
            voice: Voice to use
            language: Language code

        Returns:
            Tuple of (audio_bytes, metrics)
        """
        stream_id, _ = await self.create_stream(voice, language)
        metrics = StreamMetrics(stream_id=stream_id)
        metrics.create_time = time.time()
        metrics.text_segments = text_segments.copy()

        audio_chunks = []

        async def receive():
            async for chunk in self.receive_audio(stream_id, metrics):
                audio_chunks.append(chunk)

        receive_task = asyncio.create_task(receive())
        await asyncio.sleep(0.01)

        # Append segments with delay
        for segment in text_segments:
            await self.append_text(stream_id, segment)
            if delay_between_ms > 0:
                await asyncio.sleep(delay_between_ms / 1000)

        await self.close_stream(stream_id)
        await receive_task
        metrics.completion_time = time.time()

        return b"".join(audio_chunks), metrics

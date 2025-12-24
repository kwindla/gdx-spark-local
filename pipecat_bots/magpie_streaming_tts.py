"""Streaming HTTP client for Magpie TTS server.

Connects to the Magpie TTS HTTP server's streaming endpoint for low-latency
speech synthesis. Yields audio frames as they arrive from the server.

Usage:
    tts = MagpieStreamingTTSService(server_url="http://localhost:8001")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

import time
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.services.tts_service import TTSService

# Default sample rate (Magpie outputs at 22kHz)
MAGPIE_SAMPLE_RATE = 22000


class MagpieStreamingTTSService(TTSService):
    """Streaming HTTP client for Magpie TTS server.

    Connects to a local Magpie TTS HTTP server's streaming endpoint for
    low-latency speech synthesis. Audio chunks are yielded as they arrive,
    minimizing time-to-first-byte.

    Features:
        - Streaming audio delivery with ~140ms TTFB
        - Configurable chunking parameters
        - TTFB metrics reporting
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie Streaming TTS."""

        language: str = "en"
        # Streaming parameters
        first_chunk_frames: int = 8  # ~140ms TTFB
        chunk_size_frames: int = 16  # ~740ms chunks
        overlap_frames: int = 12  # For quality at chunk boundaries

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "aria",
        language: str = "en",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Magpie Streaming TTS client.

        Args:
            server_url: TTS server URL (default: http://localhost:8001)
            voice: Speaker voice (john, sofia, aria, jason, leo)
            language: Language code (en, es, de, fr, vi, it, zh)
            sample_rate: Output sample rate (default: 22000)
            params: Additional TTS parameters including streaming config.
        """
        super().__init__(sample_rate=sample_rate or MAGPIE_SAMPLE_RATE, **kwargs)

        self._params = params or MagpieStreamingTTSService.InputParams()

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()
        self._sample_rate = sample_rate or MAGPIE_SAMPLE_RATE

        # HTTP client with connection pooling and streaming support
        self._client = httpx.AsyncClient(timeout=60.0)

        self.set_model_name("magpie-streaming")
        self.set_voice(voice)

        logger.info(
            f"MagpieStreamingTTS initialized: server={server_url}, "
            f"voice={voice}, language={language}, "
            f"first_chunk={self._params.first_chunk_frames}frames"
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Magpie TTS streaming endpoint.

        Streams audio chunks as they become available from the server,
        yielding TTSAudioRawFrame for each chunk.

        Args:
            text: The text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame (multiple), TTSStoppedFrame
        """
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Normalize unicode characters
        text = text.replace("\u2018", "'")
        text = text.replace("\u2019", "'")
        text = text.replace("\u201C", '"')
        text = text.replace("\u201D", '"')
        text = text.replace("\u2014", "-")
        text = text.replace("\u2013", "-")

        logger.debug(f"MagpieStreamingTTS: Generating [{text[:50]}...]")

        start_time = time.time()
        first_chunk_received = False
        total_bytes = 0
        chunk_count = 0

        try:
            # Make streaming HTTP request
            async with self._client.stream(
                "POST",
                f"{self._server_url}/v1/audio/speech/stream",
                json={
                    "input": text,
                    "voice": self._voice,
                    "language": self._language,
                    "first_chunk_frames": self._params.first_chunk_frames,
                    "chunk_size_frames": self._params.chunk_size_frames,
                    "overlap_frames": self._params.overlap_frames,
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"TTS server error: {response.status_code} - {error_text.decode()}"
                    logger.error(error_msg)
                    yield ErrorFrame(error=error_msg)
                    yield TTSStoppedFrame()
                    return

                # Get sample rate from headers
                sample_rate = int(
                    response.headers.get("X-Sample-Rate", self._sample_rate)
                )

                # Stream audio chunks as they arrive
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue

                    if not first_chunk_received:
                        await self.stop_ttfb_metrics()
                        ttfb_ms = (time.time() - start_time) * 1000
                        logger.info(f"MagpieStreamingTTS TTFB: {ttfb_ms:.0f}ms")
                        first_chunk_received = True

                    total_bytes += len(chunk)
                    chunk_count += 1

                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=sample_rate,
                        num_channels=1,
                    )

            # Log completion metrics
            elapsed = time.time() - start_time
            duration_ms = total_bytes / (sample_rate * 2) * 1000

            logger.info(
                f"MagpieStreamingTTS: {total_bytes} bytes in {chunk_count} chunks, "
                f"duration={duration_ms:.0f}ms, total_time={elapsed*1000:.0f}ms"
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()

        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to TTS server at {self._server_url}: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"MagpieStreamingTTS error: {e}")
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def set_voice(self, voice: str):
        """Change the speaker voice.

        Args:
            voice: Speaker name (john, sofia, aria, jason, leo).
        """
        self._voice = voice.lower()
        super().set_voice(voice)
        logger.debug(f"MagpieStreamingTTS: Voice changed to {voice}")

    def set_language(self, language: str):
        """Change the language.

        Args:
            language: Language code (en, es, de, fr, vi, it, zh).
        """
        self._language = language.lower()
        logger.debug(f"MagpieStreamingTTS: Language changed to {language}")

    def set_streaming_params(
        self,
        first_chunk_frames: Optional[int] = None,
        chunk_size_frames: Optional[int] = None,
        overlap_frames: Optional[int] = None,
    ):
        """Update streaming parameters.

        Args:
            first_chunk_frames: Frames before first audio output (~46ms/frame)
            chunk_size_frames: Frames per subsequent chunk
            overlap_frames: Overlap frames for quality at boundaries
        """
        if first_chunk_frames is not None:
            self._params.first_chunk_frames = first_chunk_frames
        if chunk_size_frames is not None:
            self._params.chunk_size_frames = chunk_size_frames
        if overlap_frames is not None:
            self._params.overlap_frames = overlap_frames

        logger.debug(
            f"MagpieStreamingTTS: Streaming params updated: "
            f"first_chunk={self._params.first_chunk_frames}, "
            f"chunk_size={self._params.chunk_size_frames}, "
            f"overlap={self._params.overlap_frames}"
        )

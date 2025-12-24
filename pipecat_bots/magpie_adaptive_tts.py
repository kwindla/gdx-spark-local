"""Adaptive streaming TTS service for Magpie TTS server.

Uses the adaptive streaming API to achieve:
- Fast TTFB (~150ms) via streaming mode for first segment
- Higher quality via batch mode when buffer is healthy (>1000ms)
- Seamless transitions between modes

Follows DeepgramTTSService/CartesiaTTSService patterns:
- pause_frame_processing=True for proper frame handling
- Override _handle_interruption() for interruption handling
- run_tts sends text to stream, yields TTSStartedFrame
- Background task receives audio and pushes frames via push_frame()

Usage:
    tts = MagpieAdaptiveTTSService(server_url="http://localhost:8001")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

# Default sample rate (Magpie outputs at 22kHz)
MAGPIE_SAMPLE_RATE = 22000


class MagpieAdaptiveTTSService(TTSService):
    """Adaptive streaming TTS service for Magpie TTS server.

    Creates a single stream per LLM response, appending text as it arrives.
    Audio is received in a background task and pushed as frames.

    Key behavior:
    - run_tts: Appends text to stream, yields TTSStartedFrame
    - Background task: Receives audio, pushes TTSAudioRawFrame
    - LLMFullResponseEndFrame: Closes stream
    - Interruption: Cancels stream via _handle_interruption
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie Adaptive TTS."""

        language: str = "en"

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
        """Initialize Magpie Adaptive TTS service.

        Args:
            server_url: TTS server URL (default: http://localhost:8001)
            voice: Speaker voice (john, sofia, aria, jason, leo)
            language: Language code (en, es, de, fr, vi, it, zh)
            sample_rate: Output sample rate (default: 22000)
            params: Additional TTS parameters.
        """
        # Key parameters:
        # - aggregate_sentences=False: Don't buffer sentences - stream text immediately
        #   (we do our own buffering on the server side)
        # - pause_frame_processing=False: Let LLM text flow through continuously
        #   (our audio comes from background task, not run_tts generator)
        # - push_stop_frames=False: We manually control TTSStoppedFrame via stream lifecycle
        super().__init__(
            sample_rate=sample_rate or MAGPIE_SAMPLE_RATE,
            aggregate_sentences=False,
            pause_frame_processing=False,
            push_stop_frames=False,
            **kwargs,
        )

        self._params = params or MagpieAdaptiveTTSService.InputParams()

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()
        self._sample_rate = sample_rate or MAGPIE_SAMPLE_RATE

        # Separate HTTP clients with separate connection pools to avoid blocking:
        # - _client: for short requests (create stream, append text, close)
        # - _stream_client: for long-running streaming (audio receiver)
        # Use limits to ensure separate connections
        self._client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._stream_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )

        # Stream state
        self._active_stream_id: Optional[str] = None
        self._audio_receiver_task: Optional[asyncio.Task] = None
        self._stream_lock = asyncio.Lock()
        self._stream_start_time: Optional[float] = None

        self.set_model_name("magpie-adaptive")
        self.set_voice(voice)

        logger.info(
            f"MagpieAdaptiveTTS initialized: server={server_url}, "
            f"voice={voice}, language={language}"
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Handle StartFrame - initialize service."""
        await super().start(frame)
        logger.debug("MagpieAdaptiveTTS started")

    async def stop(self, frame: EndFrame):
        """Handle EndFrame - cleanup."""
        await super().stop(frame)
        await self._cancel_stream()
        logger.debug("MagpieAdaptiveTTS stopped")

    async def cancel(self, frame: CancelFrame):
        """Handle CancelFrame - cleanup."""
        await super().cancel(frame)
        await self._cancel_stream()
        logger.debug("MagpieAdaptiveTTS cancelled")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        """Handle interruption by cancelling the active stream.

        This follows the Pipecat pattern where interruptions are handled
        via this method rather than in process_frame.
        """
        await super()._handle_interruption(frame, direction)
        await self._cancel_stream()
        logger.debug("MagpieAdaptiveTTS interrupted")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling LLMFullResponseEndFrame to close stream."""
        await super().process_frame(frame, direction)

        # When LLM response ends, close the stream to flush remaining audio
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._close_stream()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Append text to stream, following Deepgram pattern.

        Audio is delivered asynchronously via the background receiver task.
        We yield TTSStartedFrame on every call (like Deepgram) and end with
        yield None to complete the generator properly.
        """
        t0 = time.time()

        # Create stream on first text
        async with self._stream_lock:
            if self._active_stream_id is None:
                await self._create_stream()

        t1 = time.time()
        logger.debug(f"run_tts({text[:20]}): lock took {(t1-t0)*1000:.1f}ms")

        # Yield TTSStartedFrame on every call (like DeepgramTTSService)
        yield TTSStartedFrame()

        t2 = time.time()
        logger.debug(f"run_tts({text[:20]}): yield TTSStartedFrame took {(t2-t1)*1000:.1f}ms")

        # Append text to the stream
        try:
            response = await self._client.post(
                f"{self._server_url}/v1/tts/stream/{self._active_stream_id}/append",
                json={"text": text},
            )
            response.raise_for_status()
            t3 = time.time()
            logger.debug(f"run_tts({text[:20]}): HTTP POST took {(t3-t2)*1000:.1f}ms, total {(t3-t0)*1000:.1f}ms")
        except Exception as e:
            logger.error(f"Failed to append text: {e}")
            yield ErrorFrame(error=str(e))
            return

        # End with yield None to complete the generator (like DeepgramTTSService)
        yield None

    async def _create_stream(self):
        """Create a new TTS stream and start audio receiver."""
        try:
            self._stream_start_time = time.time()

            response = await self._client.post(
                f"{self._server_url}/v1/tts/stream",
                json={"voice": self._voice, "language": self._language},
            )
            response.raise_for_status()
            data = response.json()
            self._active_stream_id = data["stream_id"]

            logger.info(f"Created stream: {self._active_stream_id[:8]}")

            # Start background audio receiver
            self._audio_receiver_task = asyncio.create_task(
                self._receive_audio(),
                name=f"audio-receiver-{self._active_stream_id[:8]}",
            )

        except Exception as e:
            logger.error(f"Failed to create stream: {e}")
            raise

    async def _receive_audio(self):
        """Background task to receive audio chunks and push frames."""
        stream_id = self._active_stream_id
        if not stream_id:
            return

        first_audio_received = False

        try:
            # Use separate stream client to avoid blocking POST requests
            async with self._stream_client.stream(
                "GET",
                f"{self._server_url}/v1/tts/stream/{stream_id}/audio",
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise RuntimeError(f"Audio stream error: {error_text.decode()}")

                sample_rate = int(
                    response.headers.get("X-Sample-Rate", self._sample_rate)
                )

                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue

                    # First chunk - stop TTFB metrics
                    if not first_audio_received:
                        await self.stop_ttfb_metrics()
                        ttfb_ms = (time.time() - self._stream_start_time) * 1000
                        logger.info(f"MagpieAdaptiveTTS TTFB: {ttfb_ms:.0f}ms")
                        first_audio_received = True

                    # Push audio frame
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=sample_rate,
                            num_channels=1,
                        )
                    )

            # Audio stream completed normally - signal TTS done
            logger.debug(f"Audio receiver completed for stream {stream_id[:8]}")
            await self.push_frame(TTSStoppedFrame())

        except asyncio.CancelledError:
            logger.debug(f"Audio receiver cancelled for stream {stream_id[:8]}")
            raise

        except Exception as e:
            logger.error(f"Audio receiver error: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))

    async def _close_stream(self):
        """Close the stream and wait for audio to complete."""
        async with self._stream_lock:
            if self._active_stream_id is None:
                return

            stream_id = self._active_stream_id
            logger.info(f"Closing stream: {stream_id[:8]}")

            try:
                # Signal no more text
                response = await self._client.post(
                    f"{self._server_url}/v1/tts/stream/{stream_id}/close",
                )
                response.raise_for_status()

                # Wait for audio receiver to complete
                if self._audio_receiver_task:
                    try:
                        await asyncio.wait_for(self._audio_receiver_task, timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Audio receiver timeout for {stream_id[:8]}")
                        self._audio_receiver_task.cancel()
                        try:
                            await self._audio_receiver_task
                        except asyncio.CancelledError:
                            pass

            except Exception as e:
                logger.error(f"Error closing stream: {e}")

            finally:
                self._active_stream_id = None
                self._audio_receiver_task = None

    async def _cancel_stream(self):
        """Cancel the stream immediately."""
        async with self._stream_lock:
            if self._active_stream_id is None:
                return

            stream_id = self._active_stream_id
            logger.info(f"Cancelling stream: {stream_id[:8]}")

            try:
                # Cancel audio receiver first
                if self._audio_receiver_task:
                    self._audio_receiver_task.cancel()
                    try:
                        await self._audio_receiver_task
                    except asyncio.CancelledError:
                        pass

                # Cancel stream on server
                await self._client.delete(
                    f"{self._server_url}/v1/tts/stream/{stream_id}",
                )

            except Exception as e:
                logger.error(f"Error cancelling stream: {e}")

            finally:
                self._active_stream_id = None
                self._audio_receiver_task = None

    async def close(self):
        """Close HTTP clients and clean up resources."""
        await self._cancel_stream()
        await self._client.aclose()
        await self._stream_client.aclose()

    def set_voice(self, voice: str):
        """Change the speaker voice."""
        self._voice = voice.lower()
        super().set_voice(voice)
        logger.debug(f"MagpieAdaptiveTTS: Voice changed to {voice}")

    def set_language(self, language: str):
        """Change the language."""
        self._language = language.lower()
        logger.debug(f"MagpieAdaptiveTTS: Language changed to {language}")

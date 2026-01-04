#
# NVIDIA WebSocket STT Service for Pipecat
#
# Connects to NVIDIA Parakeet ASR server via WebSocket for streaming transcription.
#

"""NVIDIA Parakeet streaming speech-to-text service implementation."""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601


class NVidiaWebSocketSTTService(WebsocketSTTService):
    """NVIDIA Parakeet streaming speech-to-text service.

    Provides real-time speech recognition using NVIDIA's Parakeet ASR model
    via WebSocket. Supports interim results for responsive transcription.

    The server expects:
    - Audio: 16-bit PCM, 16kHz, mono
    - Reset signal: {"type": "reset"} to finalize current utterance

    The server sends:
    - Ready: {"type": "ready"}
    - Transcript: {"type": "transcript", "text": "...", "is_final": true/false}
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8080",
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize the NVIDIA STT service.

        Args:
            url: WebSocket URL of the NVIDIA ASR server.
            sample_rate: Audio sample rate (must be 16000 for Parakeet).
            **kwargs: Additional arguments passed to the parent WebsocketSTTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._websocket = None
        self._receive_task: Optional[asyncio.Task] = None
        self._ready = False
        # Lock to ensure any in-progress audio send completes before reset
        self._audio_send_lock = asyncio.Lock()
        # Diagnostic: track audio bytes sent since last reset
        self._audio_bytes_sent = 0

        # Frame ordering fix: hold UserStoppedSpeakingFrame until final transcript arrives
        # This prevents the 500ms aggregator timeout when transcript arrives after UserStoppedSpeaking
        self._waiting_for_final: bool = False
        self._pending_user_stopped_frame: Optional[UserStoppedSpeakingFrame] = None
        self._pending_frame_direction: FrameDirection = FrameDirection.DOWNSTREAM
        self._pending_frame_timeout_task: Optional[asyncio.Task] = None
        self._pending_frame_timeout_s: float = 0.5  # 500ms fallback timeout

        # STT processing time metric: VADUserStoppedSpeaking -> final transcript
        self._vad_stopped_time: Optional[float] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the NVIDIA STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA STT service.

        Args:
            frame: The end frame.
        """
        # Clean up pending frame state
        await self._cancel_pending_frame_timeout()
        if self._pending_user_stopped_frame:
            await self.push_frame(
                self._pending_user_stopped_frame,
                self._pending_frame_direction
            )
            self._pending_user_stopped_frame = None
        # Send final reset to ensure any buffered audio is transcribed
        await self._send_reset()
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA STT service.

        Args:
            frame: The cancel frame.
        """
        # Clean up pending frame state (discard on cancel)
        await self._cancel_pending_frame_timeout()
        self._pending_user_stopped_frame = None
        self._waiting_for_final = False
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to NVIDIA ASR server for transcription.

        Args:
            audio: Raw audio bytes (16-bit PCM, 16kHz, mono).

        Yields:
            Frame: None (transcription results come via WebSocket receive task).
        """
        if self._websocket and self._ready:
            try:
                async with self._audio_send_lock:
                    self._audio_bytes_sent += len(audio)
                    await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
                await self._report_error(ErrorFrame(f"Failed to send audio: {e}"))
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with NVIDIA-specific handling.

        Implements frame ordering fix to ensure TranscriptionFrame arrives at the
        aggregator before UserStoppedSpeakingFrame. This prevents the 500ms
        aggregation timeout that occurs when frames arrive in the wrong order.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        # Handle UserStartedSpeakingFrame - reset pending frame state
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._cancel_pending_frame_timeout()
            self._pending_user_stopped_frame = None
            self._waiting_for_final = False
            self._vad_stopped_time = None  # Reset STT metric timer
            await super().process_frame(frame, direction)
            return

        # Handle UserStoppedSpeakingFrame - hold it if waiting for final transcript
        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._waiting_for_final:
                # Hold this frame until final transcript arrives
                self._pending_user_stopped_frame = frame
                self._pending_frame_direction = direction
                self._start_pending_frame_timeout()
                logger.debug(f"{self} holding UserStoppedSpeakingFrame at {time.time():.3f}")
                return  # Don't pass through yet
            # If not waiting for final, pass through normally
            await super().process_frame(frame, direction)
            return

        # All other frames pass through normally
        await super().process_frame(frame, direction)

        # Trigger transcript finalization on VAD silence detection.
        # VAD fires after ~200ms of silence, so all speech audio has already
        # been sent. The server adds 480ms silence padding for trailing context.
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._waiting_for_final = True
            self._vad_stopped_time = time.time()  # Start STT metric timer
            await self._send_reset()

    async def _send_reset(self):
        """Send reset signal to trigger final transcription.

        Acquires audio_send_lock to ensure any in-progress audio send completes
        before the reset signal is sent.
        """
        if self._websocket and self._ready:
            try:
                async with self._audio_send_lock:
                    await self._websocket.send(json.dumps({"type": "reset"}))
                    # Log inside lock to get accurate byte count
                    samples = self._audio_bytes_sent // 2
                    duration_ms = (samples * 1000) // 16000
                    logger.debug(f"{self} sent reset (audio: {duration_ms}ms)")
                    self._audio_bytes_sent = 0
            except Exception as e:
                logger.error(f"{self} failed to send reset: {e}")

    def _start_pending_frame_timeout(self):
        """Start timeout task to release pending UserStoppedSpeakingFrame.

        If the final transcript doesn't arrive within the timeout, we release
        the held frame anyway to prevent the pipeline from getting stuck.
        """
        if self._pending_frame_timeout_task:
            self._pending_frame_timeout_task.cancel()
        self._pending_frame_timeout_task = asyncio.create_task(
            self._pending_frame_timeout_handler()
        )

    async def _pending_frame_timeout_handler(self):
        """Handle timeout for pending UserStoppedSpeakingFrame."""
        try:
            await asyncio.sleep(self._pending_frame_timeout_s)
            if self._pending_user_stopped_frame:
                logger.debug(
                    f"{self} timeout waiting for final transcript, "
                    f"releasing UserStoppedSpeakingFrame"
                )
                await self.push_frame(
                    self._pending_user_stopped_frame,
                    self._pending_frame_direction
                )
                self._pending_user_stopped_frame = None
                self._waiting_for_final = False
        except asyncio.CancelledError:
            pass

    async def _cancel_pending_frame_timeout(self):
        """Cancel the pending frame timeout task."""
        if self._pending_frame_timeout_task:
            self._pending_frame_timeout_task.cancel()
            try:
                await self._pending_frame_timeout_task
            except asyncio.CancelledError:
                pass
            self._pending_frame_timeout_task = None

    async def _release_pending_frame(self):
        """Release the pending UserStoppedSpeakingFrame after final transcript.

        Always resets _waiting_for_final since the final transcript has arrived.
        If UserStoppedSpeakingFrame arrives later, it should pass through normally.
        """
        # Always reset waiting state - the final transcript has arrived
        self._waiting_for_final = False

        if self._pending_user_stopped_frame:
            await self._cancel_pending_frame_timeout()
            logger.debug(f"{self} releasing UserStoppedSpeakingFrame at {time.time():.3f}")
            await self.push_frame(
                self._pending_user_stopped_frame,
                self._pending_frame_direction
            )
            self._pending_user_stopped_frame = None

    async def _connect(self):
        """Connect to the NVIDIA ASR service."""
        logger.debug(f"{self} connecting to {self._url}")
        await self._connect_websocket()

        # Start receive task
        self._receive_task = asyncio.create_task(
            self._receive_task_handler(self._report_error)
        )

        await self._call_event_handler("on_connected", self)

    async def _disconnect(self):
        """Disconnect from the NVIDIA ASR service."""
        logger.debug(f"{self} disconnecting")

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        await self._disconnect_websocket()
        await self._call_event_handler("on_disconnected", self)

    async def _connect_websocket(self):
        """Establish the websocket connection."""
        try:
            self._websocket = await websockets.connect(self._url)
            self._ready = False

            # Wait for ready message
            try:
                ready_msg = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                data = json.loads(ready_msg)
                if data.get("type") == "ready":
                    self._ready = True
                    logger.info(f"{self} connected and ready")
                else:
                    logger.warning(f"{self} unexpected initial message: {data}")
                    self._ready = True  # Proceed anyway
            except asyncio.TimeoutError:
                logger.warning(f"{self} timeout waiting for ready message, proceeding anyway")
                self._ready = True

        except Exception as e:
            logger.error(f"{self} connection failed: {e}")
            await self._report_error(ErrorFrame(f"Connection failed: {e}"))
            raise

    async def _disconnect_websocket(self):
        """Close the websocket connection."""
        self._ready = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} error closing websocket: {e}")
            finally:
                self._websocket = None

    async def _receive_messages(self):
        """Receive and process websocket messages from NVIDIA ASR server."""
        if not self._websocket:
            return

        async for message in self._websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "transcript":
                    await self._handle_transcript(data)
                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"{self} server error: {error_msg}")
                    await self._report_error(ErrorFrame(f"Server error: {error_msg}"))
                elif msg_type == "ready":
                    # Server might send another ready message after reset
                    self._ready = True
                    logger.debug(f"{self} server ready")
                else:
                    logger.debug(f"{self} unknown message type: {msg_type}")

            except json.JSONDecodeError as e:
                logger.error(f"{self} invalid JSON: {e}")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _handle_transcript(self, data: dict):
        """Handle a transcript message from the server.

        For final transcripts, releases any pending UserStoppedSpeakingFrame
        AFTER pushing the transcript. This ensures correct frame ordering at
        the aggregator, preventing the 500ms timeout.

        Args:
            data: The transcript message data.
        """
        text = data.get("text", "")
        is_final = data.get("is_final", False)

        if not text:
            return

        await self.stop_ttfb_metrics()

        timestamp = time_now_iso8601()

        if is_final:
            logger.debug(f"{self} final transcript at {time.time():.3f}: {text[:50]}...")
            # Push transcript first
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    timestamp,
                    language=None,
                )
            )
            await self.stop_processing_metrics()

            # Emit STT processing time metric
            if self._vad_stopped_time is not None:
                processing_time = time.time() - self._vad_stopped_time
                logger.info(f"{self} NemotronSTT TTFB: {processing_time*1000:.0f}ms")
                metrics_frame = MetricsFrame(
                    data=[
                        TTFBMetricsData(
                            processor="NemotronSTT",
                            value=processing_time,
                        )
                    ]
                )
                await self.push_frame(metrics_frame)
                self._vad_stopped_time = None

            # Then release any pending UserStoppedSpeakingFrame
            await self._release_pending_frame()
        else:
            logger.trace(f"{self} interim: {text[:30]}...")
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    timestamp,
                    language=None,
                )
            )

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

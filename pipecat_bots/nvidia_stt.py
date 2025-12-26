#
# NVIDIA WebSocket STT Service for Pipecat
#
# Connects to NVIDIA Parakeet ASR server via WebSocket for streaming transcription.
#

"""NVIDIA Parakeet streaming speech-to-text service implementation."""

import asyncio
import json
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
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
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
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA STT service.

        Args:
            frame: The cancel frame.
        """
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
                await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
                await self._report_error(ErrorFrame(f"Failed to send audio: {e}"))
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with NVIDIA-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # Trigger transcript finalization on VAD silence detection (200ms)
        # This fires BEFORE Smart Turn analysis, giving us earlier final transcripts
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_reset()

    async def _send_reset(self):
        """Send reset signal to trigger final transcription."""
        if self._websocket and self._ready:
            try:
                await self._websocket.send(json.dumps({"type": "reset"}))
                logger.debug(f"{self} sent reset signal")
            except Exception as e:
                logger.error(f"{self} failed to send reset: {e}")

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
            logger.debug(f"{self} final transcript: {text[:50]}...")
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    timestamp,
                    language=None,
                )
            )
            await self.stop_processing_metrics()
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

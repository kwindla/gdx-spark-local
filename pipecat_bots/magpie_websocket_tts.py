"""WebSocket TTS service for Magpie TTS server.

Uses WebSocket for token-by-token streaming from LLM to TTS:
- Text tokens are sent immediately as they arrive from LLM
- Audio streams back asynchronously on the same connection
- Server accumulates text and flushes when enough audio is buffered (~4 words)

Architecture follows Deepgram's WebSocket TTS pattern:
- Extends WebsocketTTSService for auto-reconnection
- Uses _receive_task_handler for robust message receiving
- Calls flush_audio() on LLMFullResponseEndFrame

Usage:
    tts = MagpieWebSocketTTSService(server_url="http://localhost:8001")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

import json
import time
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import WebsocketTTSService

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install websockets: pip install websockets")
    raise

# Magpie outputs at 22kHz
MAGPIE_SAMPLE_RATE = 22000


class MagpieWebSocketTTSService(WebsocketTTSService):
    """WebSocket TTS service for token-by-token LLM streaming.

    Key behavior:
    - On StartFrame: connect WebSocket
    - On each LLM token: send text message via WebSocket
    - Audio arrives via _receive_messages, pushed as TTSAudioRawFrame
    - On LLMFullResponseEndFrame: send close message to trigger final flush
    - On interruption: disconnect and reconnect

    Message protocol:
    - Send: {"type": "init", "voice": "...", "language": "..."}
    - Send: {"type": "text", "text": "..."}
    - Send: {"type": "close"} - triggers server to flush remaining text
    - Receive: binary audio data
    - Receive: {"type": "done", ...} - stream complete
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie WebSocket TTS."""
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
        """Initialize Magpie WebSocket TTS service.

        Args:
            server_url: TTS server URL (http:// or ws://)
            voice: Speaker voice (aria, john, sofia, jason, leo)
            language: Language code (en, es, de, fr, vi, it, zh)
            sample_rate: Output sample rate (default: 22000)
        """
        super().__init__(
            sample_rate=sample_rate or MAGPIE_SAMPLE_RATE,
            # Don't aggregate sentences - we want token-by-token streaming
            aggregate_sentences=False,
            **kwargs,
        )

        self._params = params or MagpieWebSocketTTSService.InputParams()

        # Convert http:// to ws://
        if server_url.startswith("http://"):
            server_url = "ws://" + server_url[7:]
        elif server_url.startswith("https://"):
            server_url = "wss://" + server_url[8:]

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()

        # Stream state
        self._stream_active = False
        self._stream_start_time: Optional[float] = None
        self._first_audio_received = False

        # Receive task
        self._receive_task = None

        self.set_model_name("magpie-websocket")
        self.set_voice(voice)

        logger.info(
            f"MagpieWebSocketTTS initialized: server={self._server_url}, "
            f"voice={voice}, language={language}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        """Handle StartFrame - connect WebSocket."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Handle EndFrame - disconnect."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Handle CancelFrame - disconnect."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for LLM response end."""
        await super().process_frame(frame, direction)

        # When the LLM finishes responding, send close to flush remaining text
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by disconnecting and reconnecting."""
        await super()._handle_interruption(frame, direction)
        # Disconnect and reconnect to clear server state
        await self._disconnect()
        await self._connect()

    async def _connect(self):
        """Connect to WebSocket and start receive task."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()
        self._stream_active = False

    async def _connect_websocket(self):
        """Establish WebSocket connection and send init message."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_url = f"{self._server_url}/ws/tts/stream"
            logger.debug(f"Connecting to WebSocket: {ws_url}")

            self._websocket = await websocket_connect(ws_url)

            # Send init message with voice/language
            await self._websocket.send(
                json.dumps({
                    "type": "init",
                    "voice": self._voice,
                    "language": self._language,
                })
            )

            await self._call_event_handler("on_connected")
            logger.info("Connected to Magpie TTS WebSocket")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self.push_error(ErrorFrame(error=f"WebSocket connection failed: {e}"))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Magpie TTS WebSocket")
                await self._websocket.close()
        except Exception as e:
            logger.debug(f"WebSocket close error: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Binary message = audio data
                if not self._first_audio_received:
                    await self.stop_ttfb_metrics()
                    if self._stream_start_time:
                        ttfb_ms = (time.time() - self._stream_start_time) * 1000
                        logger.info(f"MagpieWebSocketTTS TTFB: {ttfb_ms:.0f}ms")
                    self._first_audio_received = True

                await self.push_frame(
                    TTSAudioRawFrame(message, self.sample_rate, 1)
                )

            elif isinstance(message, str):
                # JSON control message
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "stream_created":
                        stream_id = msg.get("stream_id", "")[:8]
                        logger.debug(f"WS stream created: {stream_id}")

                    elif msg_type == "done":
                        total_ms = msg.get("total_audio_ms", 0)
                        segments = msg.get("segments_generated", 0)
                        logger.info(
                            f"WS stream complete: {total_ms:.0f}ms audio, "
                            f"{segments} segments"
                        )
                        self._stream_active = False
                        await self.push_frame(TTSStoppedFrame())

                    elif msg_type == "error":
                        error_msg = msg.get("message", "Unknown error")
                        is_fatal = msg.get("fatal", False)
                        logger.error(f"WS TTS error: {error_msg} (fatal={is_fatal})")
                        await self.push_frame(ErrorFrame(error=error_msg))
                        if is_fatal:
                            self._stream_active = False

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")

    async def flush_audio(self):
        """Send close message to trigger server to flush remaining text.

        Called when LLM finishes a complete response.
        """
        if self._websocket and self._stream_active:
            try:
                await self._websocket.send(json.dumps({"type": "close"}))
                logger.debug("WS close sent (flush)")
            except Exception as e:
                logger.debug(f"Failed to send close: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Send text token to TTS stream via WebSocket.

        Called for each token from LLM (aggregate_sentences=False).
        Audio arrives asynchronously via _receive_messages.

        Args:
            text: Text token to synthesize.

        Yields:
            TTSStartedFrame on first token, then None.
        """
        logger.debug(f"MagpieWebSocketTTS: run_tts [{text}]")

        if not text:
            yield None
            return

        try:
            # Reconnect if websocket is closed
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            # Start new stream if not active
            if not self._stream_active:
                self._stream_active = True
                self._stream_start_time = time.time()
                self._first_audio_received = False
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()

            await self.start_tts_usage_metrics(text)

            # Send text message
            await self._get_websocket().send(
                json.dumps({"type": "text", "text": text})
            )

            yield None

        except Exception as e:
            logger.error(f"MagpieWebSocketTTS error: {e}")
            yield ErrorFrame(error=f"TTS error: {e}")

    async def close(self):
        """Close WebSocket connection."""
        await self._disconnect()

    def set_voice(self, voice: str):
        """Change speaker voice."""
        self._voice = voice.lower()
        super().set_voice(voice)

    def set_language(self, language: str):
        """Change language."""
        self._language = language.lower()

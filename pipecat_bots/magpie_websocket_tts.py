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

import asyncio
import json
import re
import time
from collections import deque
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

# Import the continue frame for LLM/TTS synchronization
from .frames import ChunkedLLMContinueGenerationFrame

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install websockets: pip install websockets")
    raise

# Magpie outputs at 22kHz
MAGPIE_SAMPLE_RATE = 22000

# Regex pattern for emoji and other non-speakable characters
# Covers most emoji ranges including emoticons, symbols, and pictographs
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "]+",
    flags=re.UNICODE
)


def sanitize_text_for_tts(text: str) -> str:
    """Remove emojis and normalize special characters for TTS.

    Args:
        text: Raw text from LLM

    Returns:
        Sanitized text safe for TTS synthesis
    """
    # Remove emojis
    text = EMOJI_PATTERN.sub("", text)
    # Normalize curly quotes and dashes
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("\u2014", "-")  # EM DASH
    text = text.replace("\u2013", "-")  # EN DASH
    return text


# Sentence boundary pattern - matches .!? followed by optional quotes/parens and space
SENTENCE_BOUNDARY_PATTERN = re.compile(r'([.!?]["\'\)]*\s)')


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences for TTS.

    Splits at sentence boundaries (.!? followed by space) to keep TTS chunks
    small and avoid GPU OOM on long text. Each sentence retains its trailing
    whitespace for proper TTS pacing.

    Args:
        text: Text containing one or more sentences

    Returns:
        List of sentences. If no sentence boundaries found, returns [text].

    Example:
        >>> split_into_sentences("Hello! How are you? I'm fine. ")
        ["Hello! ", "How are you? ", "I'm fine. "]
    """
    if not text:
        return []

    # Split on sentence boundaries, keeping the delimiter
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)

    # Recombine: each sentence = content + delimiter
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and SENTENCE_BOUNDARY_PATTERN.match(parts[i + 1]):
            # Content followed by delimiter
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        elif parts[i]:
            # Trailing content without delimiter (incomplete sentence)
            sentences.append(parts[i])
            i += 1
        else:
            i += 1

    return sentences if sentences else [text] if text else []


class MagpieWebSocketTTSService(WebsocketTTSService):
    """WebSocket TTS service for token-by-token LLM streaming.

    Key behavior:
    - On StartFrame: connect WebSocket
    - On each LLM token: send text message via WebSocket
    - Audio arrives via _receive_messages, pushed as TTSAudioRawFrame
    - On LLMFullResponseEndFrame: send close message to trigger final flush
    - On interruption: disconnect and reconnect

    Adaptive mode switching:
    - First segment of each response uses streaming mode for fast TTFB (~370ms)
    - Subsequent segments use batch mode for higher quality

    Message protocol:
    - Send: {"type": "init", "voice": "...", "language": "...", "default_mode": "batch"}
    - Send: {"type": "text", "text": "...", "mode": "stream|batch", "preset": "..."}
    - Send: {"type": "close"} - triggers server to flush remaining text
    - Receive: binary audio data
    - Receive: {"type": "segment_complete", ...} - segment done, more may come
    - Receive: {"type": "done", ...} - stream complete
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie WebSocket TTS."""
        language: str = "en"
        # Streaming preset for first segment: aggressive (~185ms), balanced (~280ms), conservative (~370ms)
        streaming_preset: str = "conservative"
        # Whether to use streaming mode for first segment (vs batch for all)
        use_adaptive_mode: bool = True
        # Pause duration (ms) between sentences for natural speech rhythm
        sentence_pause_ms: int = 250

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
        self._is_first_segment = True  # Track if this is first segment of response
        # Queue tracking which segments need silence pause after them
        # Populated in run_tts(), consumed in _receive_messages() on segment_complete
        self._segment_sentence_boundary_queue: deque[bool] = deque()

        # Generation tracking for interruption handling
        # _gen is incremented on interruption to invalidate old audio
        # _confirmed_gen is set to _gen when stream_created is received
        # Audio is only accepted when _confirmed_gen == _gen
        self._gen = 0
        self._confirmed_gen = 0

        # Receive task
        self._receive_task = None

        self.set_model_name("magpie-websocket")
        self.set_voice(voice)

        logger.info(
            f"MagpieWebSocketTTS initialized: server={self._server_url}, "
            f"voice={voice}, language={language}, "
            f"adaptive_mode={self._params.use_adaptive_mode}, "
            f"streaming_preset={self._params.streaming_preset}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    def _ends_at_sentence_boundary(self, text: str) -> bool:
        """Check if text ends at a sentence boundary (for inter-sentence pauses)."""
        text = text.strip()
        return bool(text) and text[-1] in '.!?'

    def _generate_silence_frames(self, duration_ms: int) -> bytes:
        """Generate silence audio (zeros) for the given duration.

        Args:
            duration_ms: Duration of silence in milliseconds

        Returns:
            Bytes of silent audio (16-bit PCM, mono, at sample_rate)
        """
        # 16-bit PCM = 2 bytes per sample
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return bytes(num_samples * 2)  # zeros

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
        """Handle interruption by resetting state and cancelling server generation.

        Per design spec, WebSocket connections persist for the pipeline lifetime.
        We reset server state via the cancel message, which immediately stops
        generation (unlike close which lets pending audio finish).

        NOTE: We intentionally do NOT call super()._handle_interruption() because
        the base WebsocketTTSService disconnects/reconnects, which violates our
        design spec. We handle all interruption logic here.
        """
        # Stop any metrics (what base class would do)
        await self.stop_all_metrics()

        # Increment generation to invalidate any in-flight audio
        # Audio will be discarded until we receive stream_created for the new gen
        self._gen += 1

        # Send cancel message to immediately stop server generation
        # Unlike flush_audio() which sends "close" and lets audio finish,
        # "cancel" tells server to stop immediately
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
            except Exception as e:
                logger.debug(f"Failed to send cancel: {e}")

        # Reset local state for next response
        self._stream_active = False
        self._stream_start_time = None
        self._first_audio_received = False
        self._is_first_segment = True
        self._segment_sentence_boundary_queue.clear()

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

        # Reset all stream state for clean slate
        self._stream_active = False
        self._stream_start_time = None
        self._first_audio_received = False
        self._is_first_segment = True
        self._segment_sentence_boundary_queue.clear()

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

                # Discard stale audio from previous generation (after interruption)
                # Audio is only valid after we receive stream_created for current gen
                if self._confirmed_gen != self._gen:
                    logger.debug(
                        f"Discarding stale audio (confirmed={self._confirmed_gen}, "
                        f"current={self._gen})"
                    )
                    continue

                if not self._first_audio_received:
                    await self.stop_ttfb_metrics()
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
                        # Confirm this generation - audio after this point is valid
                        self._confirmed_gen = self._gen

                    elif msg_type == "segment_complete":
                        # Segment done, switch to batch mode for subsequent segments
                        segment = msg.get("segment", 0)
                        audio_ms = msg.get("audio_ms", 0)
                        logger.debug(f"WS segment {segment} complete: {audio_ms:.0f}ms audio")

                        # After first segment completes, subsequent segments use batch mode
                        self._is_first_segment = False
                        self._first_audio_received = False  # Reset for next segment TTFB

                        # Inject silence pause if this segment ended at sentence boundary
                        if self._segment_sentence_boundary_queue:
                            ended_with_sentence = self._segment_sentence_boundary_queue.popleft()
                            if ended_with_sentence and self._params.sentence_pause_ms > 0:
                                silence = self._generate_silence_frames(
                                    self._params.sentence_pause_ms
                                )
                                await self.push_frame(
                                    TTSAudioRawFrame(silence, self.sample_rate, 1)
                                )

                        # Signal ChunkedLLMService that this segment is complete
                        # so it can continue generating the next chunk
                        await self.push_frame(
                            ChunkedLLMContinueGenerationFrame(),
                            FrameDirection.UPSTREAM
                        )

                    elif msg_type == "done":
                        total_ms = msg.get("total_audio_ms", 0)
                        segments = msg.get("segments_generated", 0)
                        logger.info(
                            f"WS stream complete: {total_ms:.0f}ms audio, "
                            f"{segments} segments"
                        )
                        self._stream_active = False
                        self._is_first_segment = True  # Reset for next response
                        self._segment_sentence_boundary_queue.clear()
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
        Resets first_segment flag so next response starts with streaming mode.
        """
        if self._websocket and self._stream_active:
            try:
                await self._websocket.send(json.dumps({"type": "close"}))
            except Exception as e:
                logger.debug(f"Failed to send close: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Send text to TTS stream via WebSocket.

        Splits multi-sentence text into individual sentences to avoid GPU OOM
        on long chunks. Each sentence is sent separately to the TTS server.

        In adaptive mode:
        - First segment uses streaming mode for fast TTFB
        - Subsequent segments use batch mode for higher quality

        Inter-sentence pauses:
        - Silence is injected AFTER segment_complete in _receive_messages
        - This ensures the pause occurs after the audio is played, not before

        Args:
            text: Text to synthesize (may contain multiple sentences).

        Yields:
            TTSStartedFrame on first token, then None.
        """
        # Sanitize text (remove emojis, normalize quotes/dashes)
        logger.debug(f"MagpieWebSocketTTS: run_tts before sanitize_text_for_tts [{text}]")
        text = sanitize_text_for_tts(text)

        logger.debug(f"MagpieWebSocketTTS: run_tts [{text}]")

        # Skip empty or whitespace-only text (e.g., after emoji removal)
        if not text or not text.strip():
            yield None
            return

        # Split into individual sentences to keep TTS chunks small (avoid GPU OOM)
        sentences = split_into_sentences(text)

        try:
            # Reconnect if websocket is closed
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            # Start new stream if not active
            if not self._stream_active:
                self._stream_active = True
                self._stream_start_time = time.time()
                self._first_audio_received = False
                self._is_first_segment = True  # First segment of new response
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()

            # Send each sentence separately to TTS server
            for sentence in sentences:
                if not sentence or not sentence.strip():
                    continue

                # Build text message with mode selection
                msg = {"type": "text", "text": sentence}

                if self._params.use_adaptive_mode:
                    # First segment: streaming for fast TTFB
                    # Subsequent segments: batch for quality
                    if self._is_first_segment:
                        msg["mode"] = "stream"
                        msg["preset"] = self._params.streaming_preset
                    else:
                        msg["mode"] = "batch"

                await self._get_websocket().send(json.dumps(msg))

                # Track whether this segment ends at sentence boundary
                # Consumed by _receive_messages on segment_complete to inject silence
                self._segment_sentence_boundary_queue.append(
                    self._ends_at_sentence_boundary(sentence)
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

"""Synthetic input transport for testing STT services.

Provides a BaseInputTransport implementation that reads audio from
bytes/files and pumps frames into a Pipecat Pipeline with real-time pacing.

Uses real Silero VAD for speech detection, ensuring VAD frames are emitted
based on actual speech content rather than fixed timing from audio file duration.
"""

import asyncio
import time
import wave
from pathlib import Path
from typing import Optional

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    SystemFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)


class AudioPlaybackCompleteFrame(SystemFrame):
    """Emitted when audio file playback completes (before silence phase).

    This frame signals to downstream processors (like TranscriptionCollector)
    exactly when the audio content finished, enabling accurate latency measurement.
    """
    pass
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


class SyntheticInputTransport(BaseInputTransport):
    """Input transport that plays audio from bytes or file with real-time pacing.

    This transport is designed for testing STT services in a Pipeline context.
    It reads pre-recorded audio and pushes it through the pipeline at real-time
    pace, simulating a real audio source as closely as possible.

    Uses real Silero VAD for speech detection:
    1. Streams all audio at real-time pace
    2. Silero VAD detects actual speech start/end in the audio content
    3. VADUserStartedSpeakingFrame/VADUserStoppedSpeakingFrame are emitted based on speech
    4. After audio ends, sends silence for post_audio_silence_ms (keeps pipeline alive)

    The VAD frames are based on actual speech content, not audio file duration.
    This ensures accurate latency measurement from when speech actually ends.
    """

    def __init__(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        vad_stop_secs: float = 0.2,
        post_audio_silence_ms: int = 2000,
    ):
        """Initialize the synthetic input transport.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate in Hz
            chunk_ms: Duration of each audio chunk in ms
            vad_stop_secs: Silence duration for VAD to trigger stop (default 0.2s)
            post_audio_silence_ms: Silence to send after audio ends (default 2000ms)
        """
        # Create Silero VAD with configurable stop threshold
        vad_analyzer = SileroVADAnalyzer(
            params=VADParams(stop_secs=vad_stop_secs)
        )

        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=sample_rate,
            vad_analyzer=vad_analyzer,
        )
        super().__init__(params)

        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._vad_stop_secs = vad_stop_secs
        self._post_audio_silence_ms = post_audio_silence_ms

        # Calculate chunk size in bytes (16-bit audio = 2 bytes per sample)
        samples_per_chunk = int(sample_rate * chunk_ms / 1000)
        self._chunk_size = samples_per_chunk * 2

        # Total duration for logging
        total_samples = len(audio_data) // 2
        self._duration_seconds = total_samples / sample_rate

        # Pump task reference
        self._pump_task: Optional[asyncio.Task] = None

        # Event to signal when audio pumping is complete
        self._audio_complete = asyncio.Event()

        # Smart Turn tracking - timestamp of last VADUserStoppedSpeakingFrame
        self._last_vad_stopped_time: Optional[float] = None

    @property
    def vad_stop_secs(self) -> float:
        """Return the VAD stop duration for timing calculations."""
        return self._vad_stop_secs

    @classmethod
    def from_file(
        cls,
        audio_path: str | Path,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
    ) -> "SyntheticInputTransport":
        """Create a transport from an audio file.

        Args:
            audio_path: Path to PCM audio file
            sample_rate: Audio sample rate in Hz
            chunk_ms: Duration of each audio chunk in ms

        Returns:
            SyntheticInputTransport instance
        """
        audio_data = Path(audio_path).read_bytes()
        return cls(
            audio_data=audio_data,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
        )

    async def start(self, frame: StartFrame):
        """Start the transport and begin pumping audio.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        # Reset Smart Turn state for this sample
        self._last_vad_stopped_time = None

        # Wait for transport to be ready (creates audio task in parent)
        await self.set_transport_ready(frame)

        logger.debug(
            f"SyntheticInputTransport starting: {self._duration_seconds:.2f}s audio, "
            f"{len(self._audio_data)} bytes, {self._chunk_ms}ms chunks"
        )

        # Launch the audio pumping task
        self._pump_task = self.create_task(self._pump_audio())

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Override to track VADUserStoppedSpeakingFrame timing for Smart Turn."""
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_vad_stopped_time = time.time()
        await super().push_frame(frame, direction)

    async def _handle_user_interruption(self, vad_state: VADState, emulated: bool = False):
        """Override to suppress automatic UserStoppedSpeakingFrame emission.

        In production, UserStoppedSpeakingFrame is emitted by turn detection when
        the user's turn is actually complete (not just a brief pause). We simulate
        this by only emitting it 250ms after the audio sample ends.

        VADUserStoppedSpeakingFrame is still emitted normally (by _handle_vad)
        for mid-utterance soft resets.
        """
        if vad_state == VADState.SPEAKING:
            # Handle speaking start normally - emits UserStartedSpeakingFrame
            await super()._handle_user_interruption(vad_state, emulated)
        elif vad_state == VADState.QUIET:
            # DON'T emit UserStoppedSpeakingFrame - we'll do it manually later
            # Just update the internal state
            logger.debug(
                "Suppressing automatic UserStoppedSpeakingFrame (will emit after audio ends)"
            )
            self._user_speaking = False

    async def _pump_audio(self):
        """Pump audio frames into the pipeline with real-time pacing.

        Flow:
        1. Send audio at real-time pace
        2. Silero VAD automatically detects speech start/end and emits VAD frames
        3. After audio ends, send silence for post_audio_silence_ms (keeps pipeline alive)
        4. Signal completion

        VAD frames are emitted by the base class based on actual speech detection,
        NOT based on audio file duration. This ensures accurate latency measurement.
        """
        try:
            sleep_time = self._chunk_ms / 1000
            silence_data = bytes(self._chunk_size)  # Zero-filled = silence

            # Chunk and send audio at real-time pace
            # VAD frames will be emitted automatically by BaseInputTransport
            chunks_sent = 0
            for offset in range(0, len(self._audio_data), self._chunk_size):
                chunk = self._audio_data[offset : offset + self._chunk_size]

                frame = InputAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                await self.push_audio_frame(frame)
                chunks_sent += 1
                await asyncio.sleep(sleep_time)

            logger.debug(f"Sent {chunks_sent} audio chunks ({self._duration_seconds:.2f}s)")

            # Signal that audio playback is complete (for accurate latency measurement)
            await self.push_frame(AudioPlaybackCompleteFrame())

            # Smart Turn: Send UserStoppedSpeakingFrame based on VAD timing
            # This mimics production Pipecat pipelines with Smart Turn
            SMART_TURN_DELAY = 0.05  # 50ms after VAD
            VAD_RECENT_THRESHOLD = 0.2  # 200ms

            audio_end_time = time.time()
            user_stopped_sent = False

            if self._last_vad_stopped_time and (audio_end_time - self._last_vad_stopped_time) < VAD_RECENT_THRESHOLD:
                # Case 1: VAD fired recently - schedule UserStoppedSpeakingFrame at VAD+100ms
                target_time = self._last_vad_stopped_time + SMART_TURN_DELAY

                if target_time <= audio_end_time:
                    # Target already passed - send immediately
                    logger.debug("Smart Turn: VAD recent, sending UserStoppedSpeakingFrame immediately")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    user_stopped_sent = True
                else:
                    # Wait until target time while sending silence
                    while time.time() < target_time:
                        await self._send_silence_chunk(silence_data, sleep_time)
                    logger.debug(f"Smart Turn: Sending UserStoppedSpeakingFrame (VAD+{SMART_TURN_DELAY*1000:.0f}ms)")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    user_stopped_sent = True
            elif self._last_vad_stopped_time:
                # Case 2: VAD fired earlier (mid-utterance pause) - send at audio_end + 100ms
                # The audio file ended but VAD fired a while ago. Send UserStopped shortly
                # after audio ends, which is a reasonable approximation of Smart Turn behavior.
                target_time = audio_end_time + SMART_TURN_DELAY
                while time.time() < target_time:
                    await self._send_silence_chunk(silence_data, sleep_time)
                logger.debug(f"Smart Turn: VAD was {(audio_end_time - self._last_vad_stopped_time)*1000:.0f}ms ago, sending UserStoppedSpeakingFrame")
                await self.push_frame(UserStoppedSpeakingFrame())
                user_stopped_sent = True
            else:
                # Case 3: No VAD at all - wait for fresh VAD event (speech hasn't ended yet)
                logger.debug("Smart Turn: No VAD yet, waiting for fresh VAD event...")
                timeout_time = audio_end_time + 5.0  # 5s safety timeout

                while not user_stopped_sent:
                    await self._send_silence_chunk(silence_data, sleep_time)

                    if self._last_vad_stopped_time:
                        # New VAD event occurred during silence phase
                        time_since_vad = time.time() - self._last_vad_stopped_time
                        if time_since_vad >= SMART_TURN_DELAY:
                            logger.debug(f"Smart Turn: Sending UserStoppedSpeakingFrame ({time_since_vad*1000:.0f}ms after fresh VAD)")
                            await self.push_frame(UserStoppedSpeakingFrame())
                            user_stopped_sent = True
                            break

                    if time.time() > timeout_time:
                        logger.warning("Smart Turn: Timeout, forcing UserStoppedSpeakingFrame")
                        await self.push_frame(UserStoppedSpeakingFrame())
                        user_stopped_sent = True
                        break

            # Send brief additional silence for pipeline stability
            for _ in range(5):  # 100ms at 20ms chunks
                await self._send_silence_chunk(silence_data, sleep_time)

            logger.debug("Smart Turn: Silence phase complete")

            # Wait for audio queue to drain
            logger.debug("Waiting for audio queue to drain...")
            await self._audio_in_queue.join()
            logger.debug("Audio queue drained")

            # Signal completion
            logger.debug("Audio pumping complete")
            self._audio_complete.set()

        except asyncio.CancelledError:
            logger.debug("Audio pump task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in audio pump: {e}")
            self._audio_complete.set()

    async def _send_silence_chunk(self, silence_data: bytes, sleep_time: float):
        """Send a single silence chunk with real-time pacing."""
        silence_frame = InputAudioRawFrame(
            audio=silence_data,
            sample_rate=self._sample_rate,
            num_channels=1,
        )
        await self.push_audio_frame(silence_frame)
        await asyncio.sleep(sleep_time)

    async def wait_for_audio_complete(self, timeout: float = 60.0) -> bool:
        """Wait for audio pumping to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if audio completed, False if timeout.
        """
        try:
            await asyncio.wait_for(self._audio_complete.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def audio_complete(self) -> asyncio.Event:
        """Event that is set when audio pumping is complete."""
        return self._audio_complete

    async def cleanup(self):
        """Cleanup the transport."""
        if self._pump_task:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None

        await super().cleanup()

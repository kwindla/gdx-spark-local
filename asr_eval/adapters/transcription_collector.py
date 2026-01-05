"""Transcription collector for capturing STT results from a Pipeline.

Provides a FrameProcessor that captures TranscriptionFrame outputs
and makes them available for evaluation.
"""

import asyncio
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class TranscriptionTiming:
    """Timing information for a transcription.

    Tracks timing from when actual speech ends to when we receive the
    final TranscriptionFrame.

    Key timing model (with real Silero VAD):
    - T0: Actual speech ends in audio
    - T0 + stop_secs: Silero VAD detects silence, emits VADUserStoppedSpeakingFrame
    - Tf: Final TranscriptionFrame received

    Latency = (Tf - Tvad) + stop_secs
            = time from VAD fire to transcription, plus VAD detection delay
    """

    # When the LAST VADUserStoppedSpeakingFrame was received
    # (updated for each VAD stop event, so naturally captures the last one)
    vad_stopped_time: Optional[float] = None

    # Legacy: When audio streaming finished (external call, kept for compatibility)
    audio_finished_time: Optional[float] = None

    # When the first transcription result was received
    first_result_time: Optional[float] = None

    # When the final transcription was received
    final_result_time: Optional[float] = None

    # VAD stop detection delay in seconds (how long VAD waits for silence)
    vad_stop_secs: float = 0.2

    @property
    def time_to_first_result_ms(self) -> Optional[float]:
        """Time from actual end of speech to first result in ms."""
        ref_time = self.vad_stopped_time or self.audio_finished_time
        if ref_time and self.first_result_time:
            raw_latency = (self.first_result_time - ref_time) * 1000
            return raw_latency + (self.vad_stop_secs * 1000)
        return None

    @property
    def time_to_transcription_ms(self) -> Optional[float]:
        """Time from actual end of speech to final transcription in ms.

        This is the key latency metric: time from when the user actually
        stops speaking to when we receive the final TranscriptionFrame.

        Formula: (Tf - Tvad) * 1000 + stop_secs * 1000

        The stop_secs offset accounts for VAD detection delay - VAD fires
        after stop_secs of silence, so actual speech ended stop_secs earlier.
        """
        ref_time = self.vad_stopped_time or self.audio_finished_time
        if ref_time and self.final_result_time:
            raw_latency = (self.final_result_time - ref_time) * 1000
            return raw_latency + (self.vad_stop_secs * 1000)
        return None


@dataclass
class CollectedResult:
    """Collected transcription result with all metadata."""

    final_texts: list[str] = field(default_factory=list)
    interim_results: list[str] = field(default_factory=list)
    timing: TranscriptionTiming = field(default_factory=TranscriptionTiming)
    error: Optional[str] = None

    @property
    def final_text(self) -> Optional[str]:
        """Get combined final transcription text."""
        if not self.final_texts:
            return None
        return " ".join(self.final_texts)

    @property
    def is_complete(self) -> bool:
        """Whether we have received any final transcription."""
        return len(self.final_texts) > 0


class TranscriptionCollector(FrameProcessor):
    """Collects TranscriptionFrame results from a Pipeline.

    This processor captures both interim and final transcription results,
    along with timing metrics for evaluation. It accumulates ALL TranscriptionFrames
    that arrive during the post-audio silence period (typically 2s).

    The timing.final_result_time is updated for each TranscriptionFrame, so it
    always reflects the timestamp of the LAST frame received.
    """

    def __init__(
        self,
        name: str = "TranscriptionCollector",
        capture_audio_path: Optional[str] = None,
        vad_stop_secs: float = 0.2,
    ):
        """Initialize the collector.

        Args:
            name: Name for logging purposes.
            capture_audio_path: If set, save all input audio frames to this WAV file.
            vad_stop_secs: VAD stop detection delay in seconds (for latency calculation).
        """
        super().__init__(name=name)
        self._vad_stop_secs = vad_stop_secs
        self._result = CollectedResult()
        self._result.timing.vad_stop_secs = vad_stop_secs
        self._complete_event = asyncio.Event()
        self._pipeline_ended = asyncio.Event()
        self._capture_audio_path = capture_audio_path
        self._captured_audio: bytearray = bytearray()

    @property
    def result(self) -> CollectedResult:
        """Get the collected result."""
        return self._result

    def mark_audio_finished(self) -> None:
        """Mark when audio streaming finished (legacy/fallback).

        This is a fallback for timing measurement. The preferred method is
        to use VADUserStoppedSpeakingFrame timing which is captured automatically
        within the pipeline for more accurate latency measurement.
        """
        self._result.timing.audio_finished_time = time.time()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, capturing transcription results.

        Args:
            frame: The frame to process.
            direction: Frame direction (upstream/downstream).
        """
        await super().process_frame(frame, direction)

        now = time.time()
        frame_name = type(frame).__name__

        # Log all frames for debugging
        logger.debug(f"Collector received: {frame_name}")

        # Capture audio frames if enabled
        if isinstance(frame, InputAudioRawFrame) and self._capture_audio_path:
            self._captured_audio.extend(frame.audio)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # Record VAD stopped time for accurate latency measurement
            self._result.timing.vad_stopped_time = now
            logger.debug(f"VAD stopped at {now:.3f}")

        elif isinstance(frame, InterimTranscriptionFrame):
            # Record first result time
            if self._result.timing.first_result_time is None:
                self._result.timing.first_result_time = now

            self._result.interim_results.append(frame.text)
            logger.debug(f"Interim transcription: {frame.text[:50]}...")

        elif isinstance(frame, TranscriptionFrame):
            # Record timing - final_result_time is updated for EACH frame,
            # so it ends up being the timestamp of the LAST frame
            if self._result.timing.first_result_time is None:
                self._result.timing.first_result_time = now
            self._result.timing.final_result_time = now

            # Accumulate final texts (some services send multiple TranscriptionFrames)
            self._result.final_texts.append(frame.text)

            logger.debug(f"Final transcription: {frame.text[:50]}...")

        elif isinstance(frame, EndFrame):
            # Pipeline is ending
            self._pipeline_ended.set()

        # Pass frame through to any downstream processors
        await self.push_frame(frame, direction)

    async def wait_for_result(self, timeout: float = 30.0) -> CollectedResult:
        """Wait for transcription result.

        IMPORTANT: This should be called AFTER transport.wait_for_audio_complete()
        returns. At that point, 2s of post-audio silence has been sent at real-time
        pace, which gives STT services ample time to send all TranscriptionFrames.

        We simply wait for at least one TranscriptionFrame, then return. Since
        the 2s silence period has already elapsed, all frames should have arrived.
        The timing.final_result_time will be the timestamp of the LAST frame.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            CollectedResult with transcription and timing.

        Raises:
            asyncio.TimeoutError: If timeout is reached before transcription.
        """
        start_time = time.time()
        poll_interval = 0.01  # 10ms polling

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._result.error = f"Timeout after {timeout}s waiting for transcription"
                raise asyncio.TimeoutError(self._result.error)

            # If pipeline ended, return what we have
            if self._pipeline_ended.is_set():
                break

            # If we have any transcription, return it
            # The 2s post-audio silence has already given time for all frames to arrive
            if self._result.final_texts:
                logger.debug(
                    f"Returning result with {len(self._result.final_texts)} TranscriptionFrames"
                )
                break

            await asyncio.sleep(poll_interval)

        return self._result

    def reset(self) -> None:
        """Reset the collector for a new transcription."""
        self._result = CollectedResult()
        self._result.timing.vad_stop_secs = self._vad_stop_secs
        self._complete_event.clear()
        self._pipeline_ended.clear()
        self._captured_audio.clear()

    def save_captured_audio(self, sample_rate: int = 16000) -> Optional[str]:
        """Save captured audio to WAV file.

        Args:
            sample_rate: Sample rate of the captured audio.

        Returns:
            Path to saved WAV file, or None if no audio captured or path not set.
        """
        if not self._capture_audio_path or not self._captured_audio:
            return None

        path = Path(self._capture_audio_path)
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(self._captured_audio))

        logger.info(f"Saved {len(self._captured_audio)} bytes of captured audio to {path}")
        return str(path)

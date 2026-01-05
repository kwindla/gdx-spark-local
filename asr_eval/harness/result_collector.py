"""Result collector for capturing STT transcription results.

Captures TranscriptionFrame outputs and timing metrics from STT services.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class TranscriptionTiming:
    """Timing information for a transcription."""

    # When the turn-finished signal was sent (UserStoppedSpeakingFrame)
    turn_finished_time: Optional[float] = None

    # When the first transcription result was received
    first_result_time: Optional[float] = None

    # When the final transcription was received
    final_result_time: Optional[float] = None

    @property
    def time_to_first_result_ms(self) -> Optional[float]:
        """Time from turn-finished to first result in ms."""
        if self.turn_finished_time and self.first_result_time:
            return (self.first_result_time - self.turn_finished_time) * 1000
        return None

    @property
    def time_to_transcription_ms(self) -> Optional[float]:
        """Time from turn-finished to final transcription in ms.

        This is the key metric: time from when we signal "turn finished"
        to when we receive the final TranscriptionFrame.
        """
        if self.turn_finished_time and self.final_result_time:
            return (self.final_result_time - self.turn_finished_time) * 1000
        return None


@dataclass
class CollectedResult:
    """Collected transcription result with all metadata."""

    final_text: Optional[str] = None
    interim_results: list[str] = field(default_factory=list)
    timing: TranscriptionTiming = field(default_factory=TranscriptionTiming)
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Whether we have received a final transcription."""
        return self.final_text is not None


class ResultCollector(FrameProcessor):
    """Collects TranscriptionFrame results from an STT pipeline.

    This processor captures both interim and final transcription results,
    along with timing metrics for evaluation.
    """

    def __init__(self, name: str = "ResultCollector"):
        super().__init__(name=name)
        self.result = CollectedResult()
        self._result_event = asyncio.Event()

    def mark_turn_finished(self) -> None:
        """Mark when the turn-finished signal is sent.

        Call this immediately before sending UserStoppedSpeakingFrame
        to the STT service for accurate timing.
        """
        self.result.timing.turn_finished_time = time.time()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames, capturing transcription results.

        Args:
            frame: The frame to process
            direction: Frame direction (upstream/downstream)
        """
        now = time.time()

        if isinstance(frame, InterimTranscriptionFrame):
            # Record first result time
            if self.result.timing.first_result_time is None:
                self.result.timing.first_result_time = now

            self.result.interim_results.append(frame.text)

        elif isinstance(frame, TranscriptionFrame):
            # Record timing
            if self.result.timing.first_result_time is None:
                self.result.timing.first_result_time = now
            self.result.timing.final_result_time = now

            # Store final text
            self.result.final_text = frame.text

            # Signal completion
            self._result_event.set()

        # Pass frame through
        await self.push_frame(frame, direction)

    async def wait_for_result(self, timeout: float = 30.0) -> CollectedResult:
        """Wait for the final transcription result.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            CollectedResult with transcription and timing

        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        try:
            await asyncio.wait_for(self._result_event.wait(), timeout)
        except asyncio.TimeoutError:
            self.result.error = f"Timeout after {timeout}s waiting for transcription"
            raise

        return self.result

    def reset(self) -> None:
        """Reset the collector for a new transcription."""
        self.result = CollectedResult()
        self._result_event.clear()

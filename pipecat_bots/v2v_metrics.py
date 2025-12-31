"""Voice-to-voice response time metrics processor."""

import time
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    MetricsFrame,
    UserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class V2VMetricsProcessor(FrameProcessor):
    """Measures voice-to-voice response time.

    Tracks the time from VADUserStoppedSpeakingFrame to BotStartedSpeakingFrame,
    adding vad_stop_secs to account for the VAD silence detection delay.

    Emits a MetricsFrame with TTFBMetricsData labeled "ServerVoiceToVoice".

    Placement: Just before transport.output() in the pipeline.
    """

    def __init__(self, *, vad_stop_secs: float = 0.0, **kwargs):
        """Initialize the V2V metrics processor.

        Args:
            vad_stop_secs: The VAD stop_secs parameter value. This is added to
                the measured time to account for the silence detection delay.
        """
        super().__init__(**kwargs)
        self._vad_stop_secs = vad_stop_secs
        self._vad_stopped_time: Optional[float] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to measure V2V response time."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # User stopped speaking - start timing
            self._vad_stopped_time = time.time()

        elif isinstance(frame, BotStartedSpeakingFrame):
            # Bot started speaking - calculate and emit metric
            if self._vad_stopped_time is not None:
                frame_to_frame_time = time.time() - self._vad_stopped_time
                v2v_time = frame_to_frame_time + self._vad_stop_secs

                logger.info(f"V2VMetrics: ServerVoiceToVoice TTFB: {v2v_time*1000:.0f}ms")

                metrics_frame = MetricsFrame(
                    data=[
                        TTFBMetricsData(
                            processor="ServerVoiceToVoice",
                            value=v2v_time,
                        )
                    ]
                )
                await self.push_frame(metrics_frame)

                # Reset for next exchange
                self._vad_stopped_time = None

        elif isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking (possibly interrupting) - reset timer
            if self._vad_stopped_time is not None:
                logger.debug("V2VMetrics: User started speaking, resetting timer")
                self._vad_stopped_time = None

        # Always pass the frame through
        await self.push_frame(frame, direction)

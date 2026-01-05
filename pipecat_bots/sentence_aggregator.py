#
# Custom SentenceAggregator that flushes on LLM response end.
#
# Fixes a bug in pipecat's SentenceAggregator where the buffer is not flushed
# when LLMFullResponseEndFrame arrives without punctuation.
#

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import match_endofsentence


class SentenceAggregator(FrameProcessor):
    """Aggregates text frames into complete sentences.

    This is a fixed version of pipecat's SentenceAggregator that also flushes
    the buffer when LLMFullResponseEndFrame is received, ensuring that text
    without trailing punctuation is still sent to TTS.
    """

    def __init__(self):
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Ignore interim transcriptions
        if isinstance(frame, InterimTranscriptionFrame):
            return

        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
            if match_endofsentence(self._aggregation):
                await self.push_frame(TextFrame(self._aggregation))
                self._aggregation = ""
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Flush any remaining text when LLM response ends
            if self._aggregation:
                await self.push_frame(TextFrame(self._aggregation))
                self._aggregation = ""
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            # Flush on session end as well
            if self._aggregation:
                await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

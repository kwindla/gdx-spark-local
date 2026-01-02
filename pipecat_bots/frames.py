"""Custom frame types for voice agent pipeline.

Extracted to a shared module to avoid circular imports between LLM and TTS services.
"""

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame


class ChunkedLLMContinueGenerationFrame(SystemFrame):
    """Signal frame sent upstream by TTS when a segment completes.

    This frame tells the LLM service that TTS has finished processing
    the current chunk and generation can continue to the next chunk.

    Used by both LlamaCppBufferedLLMService and MagpieWebSocketTTSService
    for synchronization between LLM generation and TTS synthesis.
    """

    pass


@dataclass
class LLMCacheWarmFrame(SystemFrame):
    """Request to pre-warm the LLM's KV cache with partial user text.

    Sent when interim transcription is available. The LLM can use this
    to send a n_predict=0 request to populate the cache, so when the
    final transcription arrives, more tokens are already cached.

    Attributes:
        text: The interim/partial user transcription text.
    """

    text: str

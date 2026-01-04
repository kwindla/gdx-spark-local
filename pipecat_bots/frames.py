"""Custom frame types for voice agent pipeline.

Extracted to a shared module to avoid circular imports between LLM and TTS services.
"""

from pipecat.frames.frames import SystemFrame


class ChunkedLLMContinueGenerationFrame(SystemFrame):
    """Signal frame sent upstream by TTS when a segment completes.

    This frame tells the LLM service that TTS has finished processing
    the current chunk and generation can continue to the next chunk.

    Used by both LlamaCppBufferedLLMService and MagpieWebSocketTTSService
    for synchronization between LLM generation and TTS synthesis.
    """

    pass

"""Nemotron-Speech-ASR: Streaming STT service for Pipecat on DGX Spark."""

__version__ = "0.1.0"

from nemotron_speech.stt_service import NemotronSTTService

__all__ = ["NemotronSTTService", "__version__"]

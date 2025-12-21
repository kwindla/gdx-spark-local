"""Nemotron-Speech-ASR STT service for Pipecat."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService


@dataclass
class NemotronSTTSettings:
    """Configuration for Nemotron-Speech-ASR service."""

    model_path: str = "models/nemotron_speech_asr.nemo"
    right_context: int = 1  # 160ms latency (0=80ms, 1=160ms, 6=560ms, 13=1.12s)
    left_context: int = 70
    chunk_size_ms: int = 80
    sample_rate: int = 16000
    device: str = "cuda"


class NemotronSTTService(STTService):
    """
    Pipecat STT service using Nemotron-Speech-ASR cache-aware streaming model.

    This service provides real-time speech-to-text transcription using NVIDIA's
    Nemotron-Speech-ASR model with cache-aware streaming inference.

    Example::

        from nemotron_speech import NemotronSTTService, NemotronSTTSettings

        settings = NemotronSTTSettings(
            model_path="models/nemotron_speech_asr.nemo",
            right_context=1,  # 160ms latency
        )
        stt = NemotronSTTService(settings=settings)

        # Use in Pipecat pipeline
        pipeline = Pipeline([transport.input(), stt, ...])
    """

    def __init__(
        self,
        settings: Optional[NemotronSTTSettings] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Nemotron STT service.

        Args:
            settings: Configuration for the STT service.
            **kwargs: Additional arguments passed to STTService.
        """
        super().__init__(sample_rate=16000, **kwargs)
        self._settings = settings or NemotronSTTSettings()
        self._model: Any = None
        self._cache: Any = None
        self._audio_buffer = bytearray()
        self._chunk_samples = int(
            self._settings.chunk_size_ms * self._settings.sample_rate / 1000
        )
        self._initialized = False

    async def start(self, frame: Any) -> None:
        """Start the STT service and load the model.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if not self._initialized:
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(None, self._load_model)
            self._initialized = True
            await self._call_event_handler("on_connected")
            logger.info(f"Nemotron STT service started with model: {self._settings.model_path}")

    def _load_model(self) -> Any:
        """Load the NeMo ASR model (runs in executor to avoid blocking)."""
        try:
            import nemo.collections.asr as nemo_asr
            import torch

            model = nemo_asr.models.ASRModel.restore_from(self._settings.model_path)

            if self._settings.device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            model.eval()

            logger.info(f"Model loaded: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio bytes through cache-aware streaming inference.

        This implementation accumulates audio until we have enough for a chunk,
        then processes it through the model.

        Args:
            audio: Raw audio bytes (PCM 16-bit, 16kHz).

        Yields:
            TranscriptionFrame with transcription results.
        """
        if self._model is None:
            logger.warning("Model not loaded, skipping audio processing")
            return

        # Accumulate audio
        self._audio_buffer.extend(audio)

        # Calculate bytes per chunk (16-bit = 2 bytes per sample)
        bytes_per_chunk = self._chunk_samples * 2

        # Process complete chunks
        while len(self._audio_buffer) >= bytes_per_chunk:
            chunk_bytes = bytes(self._audio_buffer[:bytes_per_chunk])
            self._audio_buffer = self._audio_buffer[bytes_per_chunk:]

            # Convert to float32
            audio_np = np.frombuffer(chunk_bytes, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Process through model in executor
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None, self._process_chunk, audio_float
            )

            if transcription and transcription.strip():
                from datetime import datetime

                yield TranscriptionFrame(
                    text=transcription,
                    user_id=self._user_id,
                    timestamp=datetime.now().isoformat(),
                    language=None,
                    result=None,
                )

    def _process_chunk(self, audio: np.ndarray) -> Optional[str]:
        """Process a single audio chunk through the cache-aware model.

        Args:
            audio: Audio samples as float32 numpy array.

        Returns:
            Transcribed text or None if no speech detected.

        Note:
            This is a simplified implementation. The full implementation
            should use NeMo's cache-aware streaming API with proper
            cache management between chunks.
        """
        try:
            import torch

            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)

                if self._settings.device == "cuda":
                    audio_tensor = audio_tensor.cuda()

                audio_len = torch.tensor([len(audio)])
                if self._settings.device == "cuda":
                    audio_len = audio_len.cuda()

                # Forward pass
                # Note: This is simplified - actual cache-aware inference
                # requires using NeMo's streaming buffer classes
                logits, _, _ = self._model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len,
                )

                # Decode
                hypotheses = self._model.decoding.ctc_decoder_predictions_tensor(
                    logits, decoder_lengths=None
                )

                if hypotheses and hypotheses[0]:
                    return hypotheses[0]
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

        return None

    async def stop(self) -> None:
        """Stop the STT service and release resources."""
        await self._call_event_handler("on_disconnected")
        self._model = None
        self._cache = None
        self._audio_buffer.clear()
        self._initialized = False
        logger.info("Nemotron STT service stopped")

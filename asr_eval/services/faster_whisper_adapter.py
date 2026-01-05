"""Adapter for local Faster-Whisper STT.

This adapter uses the faster-whisper library directly (not through Pipecat)
for high-quality local transcription.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from asr_eval.config import get_config
from asr_eval.models import AudioSample, ServiceName, TranscriptionResult
from asr_eval.services.base import BaseSTTAdapter


class FasterWhisperAdapter(BaseSTTAdapter):
    """Adapter for local Faster-Whisper transcription.

    Unlike other adapters, this one doesn't use Pipecat's streaming
    interface. Instead, it processes complete audio files directly
    with the faster-whisper library.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ):
        """Initialize the adapter.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3-turbo)
            device: Device to run on (cuda, cpu)
            compute_type: Compute type (float16, int8, etc.)
            language: Language code for transcription
        """
        self.config = get_config()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.FASTER_WHISPER

    async def create_service(self):
        """Load the Whisper model.

        Note: This adapter doesn't return a Pipecat service.
        The model is used directly for transcription.
        """
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading Faster-Whisper model: {self.model_size} "
                f"(device={self.device}, compute_type={self.compute_type})"
            )

            # Load model in thread pool to avoid blocking
            self._model = await asyncio.to_thread(
                WhisperModel,
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

            logger.info("Faster-Whisper model loaded")

        return self._model

    async def cleanup(self) -> None:
        """Clean up the model."""
        # Keep model loaded for efficiency (it's expensive to reload)
        pass

    async def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        """Transcribe an audio sample directly.

        This method bypasses the Pipecat harness and processes
        the audio file directly with faster-whisper.

        Args:
            sample: AudioSample to transcribe

        Returns:
            TranscriptionResult with transcription and metrics
        """
        import time

        # Ensure model is loaded
        await self.create_service()

        # Load audio
        audio_path = Path(sample.audio_path)
        if not audio_path.exists():
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=sample.duration_seconds * 1000,
                rtf=0,
                error=f"Audio file not found: {audio_path}",
            )

        try:
            # Load PCM audio and convert to float32
            audio_bytes = audio_path.read_bytes()
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Time the transcription
            start_time = time.time()

            # Run transcription in thread pool
            segments, info = await asyncio.to_thread(
                self._model.transcribe,
                audio_float,
                language=self.language,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True,
            )

            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            transcribed_text = " ".join(text_parts)

            end_time = time.time()
            transcription_time_ms = (end_time - start_time) * 1000
            audio_duration_ms = sample.duration_seconds * 1000
            rtf = transcription_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text=transcribed_text,
                time_to_transcription_ms=transcription_time_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=rtf,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error transcribing with Faster-Whisper: {e}")
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=sample.duration_seconds * 1000,
                rtf=0,
                error=str(e),
            )

    async def transcribe_batch(
        self,
        samples: list[AudioSample],
        progress_callback: Optional[callable] = None,
    ) -> list[TranscriptionResult]:
        """Transcribe a batch of samples.

        Args:
            samples: List of AudioSample to transcribe
            progress_callback: Optional callback(current, total, sample_id)

        Returns:
            List of TranscriptionResult objects
        """
        results = []

        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, len(samples), sample.sample_id)

            result = await self.transcribe(sample)
            results.append(result)

            if result.error:
                logger.warning(f"[{i+1}/{len(samples)}] Error: {result.error}")
            else:
                text_preview = (
                    f"{result.transcribed_text[:50]}..."
                    if len(result.transcribed_text) > 50
                    else result.transcribed_text
                )
                logger.info(
                    f"[{i+1}/{len(samples)}] {text_preview} "
                    f"({result.time_to_transcription_ms:.0f}ms)"
                )

        return results

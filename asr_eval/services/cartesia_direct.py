"""Direct Cartesia adapter using REST API for batch transcription."""

import asyncio
import io
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger

from asr_eval.config import get_config
from asr_eval.models import AudioSample, ServiceName, TranscriptionResult
from asr_eval.services.base import BaseSTTAdapter


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert raw PCM audio to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    wav_buffer.seek(0)
    return wav_buffer.read()


class CartesiaDirectAdapter(BaseSTTAdapter):
    """Direct Cartesia adapter using the REST Speech-to-Text API."""

    API_URL = "https://api.cartesia.ai/stt"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ink-whisper",
        language: str = "en",
    ):
        self.config = get_config()
        self.api_key = api_key or self.config.cartesia_api_key
        self.model = model
        self.language = language
        self._session = None

        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not set in environment")

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.CARTESIA

    async def create_service(self):
        """Initialize the HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Cartesia-Version": "2025-04-16",
                }
            )
        return self._session

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        """Transcribe an audio sample using Cartesia Speech-to-Text API."""
        await self.create_service()

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
            # Read and convert audio
            pcm_bytes = audio_path.read_bytes()
            wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1)

            # Prepare multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                wav_bytes,
                filename='audio.wav',
                content_type='audio/wav'
            )
            form_data.add_field('model', self.model)
            form_data.add_field('language', self.language)

            # Time the transcription
            start_time = time.time()

            async with self._session.post(self.API_URL, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return TranscriptionResult(
                        sample_id=sample.sample_id,
                        service_name=self.service_name,
                        transcribed_text="",
                        time_to_transcription_ms=0,
                        audio_duration_ms=sample.duration_seconds * 1000,
                        rtf=0,
                        error=f"API error {response.status}: {error_text}",
                    )

                result = await response.json()

            end_time = time.time()
            transcription_time_ms = (end_time - start_time) * 1000
            audio_duration_ms = sample.duration_seconds * 1000
            rtf = transcription_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            # Extract transcript from response
            transcript = result.get("text", "")

            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text=transcript,
                time_to_transcription_ms=transcription_time_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=rtf,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error transcribing with Cartesia: {e}")
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
        """Transcribe a batch of samples."""
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

        # Clean up after batch
        await self.cleanup()

        return results

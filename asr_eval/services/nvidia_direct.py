"""Direct NVIDIA Parakeet adapter using WebSocket for batch transcription."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import websockets
from loguru import logger

from asr_eval.config import get_config
from asr_eval.models import AudioSample, ServiceName, TranscriptionResult
from asr_eval.services.base import BaseSTTAdapter


class NvidiaDirectAdapter(BaseSTTAdapter):
    """Direct NVIDIA Parakeet adapter using WebSocket.

    Bypasses Pipecat framework and communicates directly with the
    NVIDIA ASR WebSocket server.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        self.config = get_config()
        self.url = url or self.config.nvidia_asr_url
        self.sample_rate = sample_rate

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.NVIDIA_PARAKEET

    async def create_service(self):
        """Not used for direct adapter."""
        return None

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    async def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        """Transcribe an audio sample using NVIDIA Parakeet WebSocket API."""
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
            # Read audio data (already 16-bit PCM at 16kHz)
            audio_data = audio_path.read_bytes()

            # Time the transcription
            start_time = time.time()

            # Connect to WebSocket
            async with websockets.connect(self.url) as ws:
                # Wait for ready message
                ready_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                ready_data = json.loads(ready_msg)
                if ready_data.get("type") != "ready":
                    logger.warning(f"Unexpected initial message: {ready_data}")

                # Send audio in chunks (similar to real-time streaming)
                chunk_size = 1600  # 50ms at 16kHz (800 samples * 2 bytes)
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await ws.send(chunk)

                # Send reset to finalize transcription
                await ws.send(json.dumps({"type": "reset"}))

                # Collect transcript
                final_text = ""
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(msg)

                        if data.get("type") == "transcript":
                            text = data.get("text", "")
                            is_final = data.get("is_final", False)

                            if is_final:
                                final_text = text
                                break
                        elif data.get("type") == "error":
                            error_msg = data.get("message", "Unknown error")
                            return TranscriptionResult(
                                sample_id=sample.sample_id,
                                service_name=self.service_name,
                                transcribed_text="",
                                time_to_transcription_ms=0,
                                audio_duration_ms=sample.duration_seconds * 1000,
                                rtf=0,
                                error=f"Server error: {error_msg}",
                            )
                    except asyncio.TimeoutError:
                        # No more messages, use whatever we got
                        break

            end_time = time.time()
            transcription_time_ms = (end_time - start_time) * 1000
            audio_duration_ms = sample.duration_seconds * 1000
            rtf = transcription_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text=final_text,
                time_to_transcription_ms=transcription_time_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=rtf,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error transcribing with NVIDIA Parakeet: {e}")
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

        return results

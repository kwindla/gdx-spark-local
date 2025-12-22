"""Local NVIDIA Riva TTS service for Pipecat.

Connects to local Riva/Magpie TTS server without SSL.

NOTE: The Magpie TTS NIM container cannot handle concurrent requests.
We use an asyncio.Lock to serialize TTS calls.
"""

import asyncio
import threading
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

import riva.client

# Magpie TTS outputs audio at 44100 Hz
MAGPIE_SAMPLE_RATE = 44100


class RivaLocalTTSService(TTSService):
    """Local NVIDIA Riva TTS service (no SSL).

    Connects to a local Riva or Magpie TTS server via insecure gRPC.
    """

    class InputParams(BaseModel):
        """Input parameters for local Riva TTS."""

        language: Optional[Language] = Language.EN_US

    def __init__(
        self,
        *,
        server: str = "localhost:50051",
        voice_id: str = "Magpie-Multilingual.EN-US.Aria",
        sample_rate: int = MAGPIE_SAMPLE_RATE,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize local Riva TTS service.

        Args:
            server: Local gRPC server address (default: localhost:50051)
            voice_id: Voice name (default: Magpie-Multilingual.EN-US.Aria)
            sample_rate: Audio sample rate (default: 22050 Hz for Magpie TTS).
            params: Additional TTS parameters.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or RivaLocalTTSService.InputParams()

        self._server = server
        self._voice_id = voice_id
        self._language_code = params.language
        self._output_sample_rate = sample_rate

        self.set_model_name("riva-local")
        self.set_voice(voice_id)

        # Connect WITHOUT SSL for local server
        logger.info(f"Connecting to local Riva TTS at {server}")
        auth = riva.client.Auth(
            uri=server,
            use_ssl=False,
        )
        self._service = riva.client.SpeechSynthesisService(auth)

        # Lock to serialize TTS requests - Magpie NIM can't handle concurrent requests
        self._tts_lock = asyncio.Lock()

        logger.info(f"Connected to Riva TTS, voice: {voice_id}, sample_rate: {sample_rate}Hz")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports TTFB metrics generation.
        """
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using local Riva TTS.

        Uses streaming synthesis (synthesize_online) to get complete audio.

        Args:
            text: The text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame(s), TTSStoppedFrame
        """
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Normalize unicode characters that Magpie TTS can't handle
        # Curly quotes/apostrophes cause character_mapping failures and truncated audio
        # Use explicit Unicode codepoints to ensure correct replacement
        text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
        text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
        text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
        text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
        text = text.replace("\u2014", "-")  # EM DASH
        text = text.replace("\u2013", "-")  # EN DASH
        text = text.replace("\u2032", "'")  # PRIME
        text = text.replace("\u02BC", "'")  # MODIFIER LETTER APOSTROPHE

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Use streaming synthesis to get complete audio
            # Run in thread to avoid blocking event loop
            def synthesize_streaming(attempt: int = 1):
                # Create fresh Riva client for each request to avoid gRPC
                # thread-safety issues when running in asyncio.to_thread()
                auth = riva.client.Auth(uri=self._server, use_ssl=False)
                service = riva.client.SpeechSynthesisService(auth)

                audio_chunks = []
                chunk_count = 0
                total_bytes = 0
                try:
                    stream = service.synthesize_online(
                        text,
                        self._voice_id,
                        self._language_code,
                    )
                    logger.debug(f"synthesize_streaming: attempt {attempt}, started for [{text[:50]}...]")
                    for resp in stream:
                        if resp.audio:
                            chunk_bytes = len(resp.audio)
                            total_bytes += chunk_bytes
                            chunk_count += 1
                            audio_chunks.append(resp.audio)
                        else:
                            logger.warning(f"synthesize_streaming: chunk {chunk_count + 1} had no audio")
                    logger.debug(
                        f"synthesize_streaming: attempt {attempt} completed with {chunk_count} chunks, "
                        f"{total_bytes} bytes total"
                    )
                except StopIteration:
                    logger.info(f"synthesize_streaming: StopIteration after {chunk_count} chunks")
                except Exception as e:
                    logger.error(
                        f"synthesize_streaming: error after {chunk_count} chunks, "
                        f"{total_bytes} bytes: {type(e).__name__}: {e}"
                    )

                return b"".join(audio_chunks)

            def is_truncated(audio_bytes: bytes, text_len: int) -> bool:
                """Check if audio is truncated based on expected duration.

                Heuristic: expect at least 30ms of audio per character.
                At 44100Hz, 16-bit mono, that's 2646 bytes per character minimum.
                """
                if not audio_bytes or text_len == 0:
                    return True
                bytes_per_char = len(audio_bytes) / text_len
                # 30ms * 44100Hz * 2 bytes = 2646 bytes minimum per char
                min_bytes_per_char = 1500  # More lenient threshold
                return bytes_per_char < min_bytes_per_char

            # Serialize TTS requests - Magpie NIM crashes with concurrent requests
            # CRITICAL: Keep lock held until ALL frames are yielded to prevent
            # interleaved frames from concurrent run_tts calls
            async with self._tts_lock:
                max_retries = 3
                audio = None

                for attempt in range(1, max_retries + 1):
                    audio = await asyncio.to_thread(synthesize_streaming, attempt)

                    if not is_truncated(audio, len(text)):
                        break

                    duration_ms = len(audio) / (self._output_sample_rate * 2) * 1000 if audio else 0
                    logger.warning(
                        f"{self}: Truncation detected on attempt {attempt}/{max_retries} - "
                        f"got {duration_ms:.0f}ms for {len(text)} chars, retrying..."
                    )

                    if attempt < max_retries:
                        # Wait before retry with exponential backoff
                        await asyncio.sleep(0.2 * attempt)

                await self.stop_ttfb_metrics()

                if audio:
                    audio_bytes = len(audio)
                    duration_ms = audio_bytes / (self._output_sample_rate * 2) * 1000
                    logger.info(
                        f"{self}: Synthesized {audio_bytes} bytes, "
                        f"duration={duration_ms:.0f}ms at {self._output_sample_rate}Hz"
                    )
                    yield TTSAudioRawFrame(
                        audio=audio,
                        sample_rate=self._output_sample_rate,
                        num_channels=1,
                    )
                else:
                    logger.warning(f"{self}: No audio in response!")

                await self.start_tts_usage_metrics(text)
                yield TTSStoppedFrame()

                await asyncio.sleep(0.1)  # Small cooldown between requests

        except Exception as e:
            logger.error(f"{self} TTS error: {e}")
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()

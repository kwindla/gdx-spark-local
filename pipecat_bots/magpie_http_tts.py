"""HTTP client for Magpie TTS server.

Connects to the Magpie TTS HTTP server for speech synthesis.
Runs on the host - no NeMo/PyTorch dependencies required.

Usage:
    tts = MagpieHTTPTTSService(server_url="http://localhost:8001")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

import asyncio
from typing import AsyncGenerator, Optional

import httpx
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

# Default sample rate (will be fetched from server)
DEFAULT_SAMPLE_RATE = 22000


class MagpieHTTPTTSService(TTSService):
    """HTTP client for Magpie TTS server.

    Connects to a local Magpie TTS HTTP server for speech synthesis.
    No NeMo/PyTorch dependencies - runs on host.
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie HTTP TTS."""

        language: str = "en"

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "aria",
        language: str = "en",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Magpie HTTP TTS client.

        Args:
            server_url: TTS server URL (default: http://localhost:8001)
            voice: Speaker voice (john, sofia, aria, jason, leo)
            language: Language code (en, es, de, fr, vi, it, zh)
            sample_rate: Output sample rate (default: fetched from server)
            params: Additional TTS parameters.
        """
        # Will update sample_rate after fetching from server
        super().__init__(sample_rate=sample_rate or DEFAULT_SAMPLE_RATE, **kwargs)

        params = params or MagpieHTTPTTSService.InputParams()

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()
        self._sample_rate = sample_rate

        # HTTP client with connection pooling
        self._client = httpx.AsyncClient(timeout=30.0)
        self._config_fetched = False

        self.set_model_name("magpie-http")
        self.set_voice(voice)

        logger.info(
            f"MagpieHTTPTTS initialized: server={server_url}, "
            f"voice={voice}, language={language}"
        )

    async def _ensure_config(self):
        """Fetch server config if not already done."""
        if self._config_fetched:
            return

        try:
            resp = await self._client.get(f"{self._server_url}/v1/audio/config")
            if resp.status_code == 200:
                config = resp.json()
                server_sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)

                # Update sample rate if not explicitly set
                if self._sample_rate is None:
                    self._sample_rate = server_sample_rate
                    # Update parent class sample rate
                    self._settings.sample_rate = server_sample_rate

                logger.info(
                    f"MagpieHTTPTTS config: sample_rate={server_sample_rate}Hz, "
                    f"voices={config.get('voices', [])}"
                )
            self._config_fetched = True
        except Exception as e:
            logger.warning(f"Failed to fetch TTS config: {e}")
            self._config_fetched = True  # Don't retry on every request

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Magpie TTS HTTP server.

        Args:
            text: The text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame
        """
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Fetch config on first request
        await self._ensure_config()

        # Normalize unicode characters
        text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
        text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
        text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
        text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
        text = text.replace("\u2014", "-")  # EM DASH
        text = text.replace("\u2013", "-")  # EN DASH

        logger.debug(f"MagpieHTTPTTS: Generating [{text[:50]}...]")

        try:
            # Make HTTP request to TTS server
            resp = await self._client.post(
                f"{self._server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    "language": self._language,
                    "response_format": "pcm",
                },
            )

            if resp.status_code != 200:
                error_msg = f"TTS server error: {resp.status_code} - {resp.text}"
                logger.error(error_msg)
                yield ErrorFrame(error=error_msg)
                yield TTSStoppedFrame()
                return

            await self.stop_ttfb_metrics()

            audio_bytes = resp.content

            # Get sample rate from response headers or use default
            sample_rate = int(resp.headers.get("X-Sample-Rate", self._sample_rate or DEFAULT_SAMPLE_RATE))
            duration_ms = float(resp.headers.get("X-Duration-Ms", 0))

            logger.info(
                f"MagpieHTTPTTS: Received {len(audio_bytes)} bytes, "
                f"duration={duration_ms:.0f}ms at {sample_rate}Hz"
            )

            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=sample_rate,
                num_channels=1,
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()

        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to TTS server at {self._server_url}: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"MagpieHTTPTTS error: {e}")
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def set_voice(self, voice: str):
        """Change the speaker voice.

        Args:
            voice: Speaker name (john, sofia, aria, jason, leo).
        """
        self._voice = voice.lower()
        super().set_voice(voice)
        logger.info(f"MagpieHTTPTTS: Voice changed to {voice}")

    def set_language(self, language: str):
        """Change the language.

        Args:
            language: Language code (en, es, de, fr, vi, it, zh).
        """
        self._language = language.lower()
        logger.info(f"MagpieHTTPTTS: Language changed to {language}")

"""Adapter for Deepgram STT service."""

from typing import Optional

from asr_eval.config import get_config
from asr_eval.models import ServiceName
from asr_eval.services.base import BaseSTTAdapter


class DeepgramAdapter(BaseSTTAdapter):
    """Adapter for Deepgram STT service.

    Uses Pipecat's DeepgramSTTService for real-time streaming transcription.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-2",
        language: str = "en",
    ):
        """Initialize the adapter.

        Args:
            api_key: Deepgram API key (default from config)
            model: Deepgram model to use
            language: Language code
        """
        self.config = get_config()
        self.api_key = api_key or self.config.deepgram_api_key
        self.model = model
        self.language = language
        self._service = None

        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set in environment")

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.DEEPGRAM

    async def create_service(self):
        """Create the DeepgramSTTService instance."""
        from pipecat.services.deepgram import DeepgramSTTService

        self._service = DeepgramSTTService(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
        )
        return self._service

    async def cleanup(self) -> None:
        """Clean up the service."""
        if self._service:
            try:
                await self._service._disconnect()
            except Exception:
                pass
            self._service = None

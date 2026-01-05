"""Adapter for Cartesia STT service."""

from typing import Optional

from asr_eval.config import get_config
from asr_eval.models import ServiceName
from asr_eval.services.base import BaseSTTAdapter


class CartesiaAdapter(BaseSTTAdapter):
    """Adapter for Cartesia STT service.

    Uses Pipecat's CartesiaSTTService for real-time streaming transcription.
    Cartesia uses the "ink-whisper" model by default.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ink-whisper",
        language: str = "en",
    ):
        """Initialize the adapter.

        Args:
            api_key: Cartesia API key (default from config)
            model: Cartesia model to use
            language: Language code
        """
        self.config = get_config()
        self.api_key = api_key or self.config.cartesia_api_key
        self.model = model
        self.language = language
        self._service = None

        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not set in environment")

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.CARTESIA

    async def create_service(self):
        """Create the CartesiaSTTService instance."""
        from pipecat.services.cartesia import CartesiaSTTService

        self._service = CartesiaSTTService(
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

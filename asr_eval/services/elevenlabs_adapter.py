"""Adapter for ElevenLabs STT service."""

from typing import Optional

from asr_eval.config import get_config
from asr_eval.models import ServiceName
from asr_eval.services.base import BaseSTTAdapter


class ElevenLabsAdapter(BaseSTTAdapter):
    """Adapter for ElevenLabs STT service.

    Uses Pipecat's ElevenLabsSTTService for transcription.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "scribe_v1",
        language: str = "en",
    ):
        """Initialize the adapter.

        Args:
            api_key: ElevenLabs API key (default from config)
            model: ElevenLabs model to use
            language: Language code
        """
        self.config = get_config()
        self.api_key = api_key or self.config.elevenlabs_api_key
        self.model = model
        self.language = language
        self._service = None

        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set in environment")

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.ELEVENLABS

    async def create_service(self):
        """Create the ElevenLabsSTTService instance."""
        from pipecat.services.elevenlabs import ElevenLabsSTTService

        self._service = ElevenLabsSTTService(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
        )
        return self._service

    async def cleanup(self) -> None:
        """Clean up the service."""
        if self._service:
            try:
                # ElevenLabs STT doesn't have a persistent connection
                pass
            except Exception:
                pass
            self._service = None

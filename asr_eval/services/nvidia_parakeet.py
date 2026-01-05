"""Adapter for NVIDIA Parakeet STT service."""

from typing import Optional

from asr_eval.config import get_config
from asr_eval.models import ServiceName
from asr_eval.services.base import BaseSTTAdapter


class NvidiaParakeetAdapter(BaseSTTAdapter):
    """Adapter for the local NVIDIA Parakeet STT service.

    Uses the existing NVidiaWebSocketSTTService from pipecat_bots.
    Requires the ASR server to be running locally.
    """

    def __init__(self, url: Optional[str] = None):
        """Initialize the adapter.

        Args:
            url: WebSocket URL for the ASR server (default from config)
        """
        self.config = get_config()
        self.url = url or self.config.nvidia_asr_url
        self._service = None

    @property
    def service_name(self) -> ServiceName:
        return ServiceName.NVIDIA_PARAKEET

    async def create_service(self):
        """Create the NVidiaWebSocketSTTService instance."""
        # Import here to avoid circular imports and missing dependencies
        from pipecat_bots.nvidia_stt import NVidiaWebSocketSTTService

        self._service = NVidiaWebSocketSTTService(
            url=self.url,
            sample_rate=self.config.sample_rate,
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

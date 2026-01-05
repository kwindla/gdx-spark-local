"""Base class for STT service adapters."""

from abc import ABC, abstractmethod
from typing import Any

from asr_eval.models import ServiceName


class BaseSTTAdapter(ABC):
    """Abstract base class for STT service adapters.

    Each adapter wraps a specific STT service (Pipecat or direct)
    and provides a common interface for the test harness.
    """

    @property
    @abstractmethod
    def service_name(self) -> ServiceName:
        """Return the service name enum value."""
        pass

    @abstractmethod
    async def create_service(self) -> Any:
        """Create and return the STT service instance.

        Returns:
            A Pipecat STT service or compatible object
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources held by the adapter."""
        pass

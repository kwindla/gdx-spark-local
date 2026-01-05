"""Registry of Pipecat STT services with configurations."""

from typing import Dict, Optional

from asr_eval.adapters.universal_pipecat import (
    EvaluationMode,
    PipecatServiceConfig,
    UniversalPipecatAdapter,
)
from asr_eval.config import get_config
from asr_eval.models import ServiceName


def create_deepgram_config(
    api_key: Optional[str] = None,
    model: str = "nova-2",
    language: str = "en",
) -> PipecatServiceConfig:
    """Create Deepgram STT service configuration.

    Args:
        model: Deepgram model name. Options include:
            - "nova-2" (default, best accuracy)
            - "nova-2-general"
            - "whisper-medium", "whisper-large", etc. (OpenAI Whisper via Deepgram)
    """
    from pipecat.services.deepgram import DeepgramSTTService
    from pipecat.transcriptions.language import Language

    config = get_config()
    return PipecatServiceConfig(
        service_class=DeepgramSTTService,
        service_kwargs={
            "api_key": api_key or config.deepgram_api_key,
            "model": model,
            "language": Language.EN,
        },
        service_name=ServiceName.DEEPGRAM,
        needs_aiohttp_session=False,
    )


def create_cartesia_config(
    api_key: Optional[str] = None,
    model: str = "ink-whisper",
) -> PipecatServiceConfig:
    """Create Cartesia STT service configuration."""
    from pipecat.services.cartesia import CartesiaSTTService

    config = get_config()
    return PipecatServiceConfig(
        service_class=CartesiaSTTService,
        service_kwargs={
            "api_key": api_key or config.cartesia_api_key,
        },
        service_name=ServiceName.CARTESIA,
        needs_aiohttp_session=False,
    )


def create_elevenlabs_config(
    api_key: Optional[str] = None,
    model: str = "scribe_v1",
) -> PipecatServiceConfig:
    """Create ElevenLabs STT service configuration.

    Note: ElevenLabsSTTService requires an aiohttp session, which will
    be automatically created and managed by the UniversalPipecatAdapter.
    """
    from pipecat.services.elevenlabs import ElevenLabsSTTService

    config = get_config()
    return PipecatServiceConfig(
        service_class=ElevenLabsSTTService,
        service_kwargs={
            "api_key": api_key or config.elevenlabs_api_key,
            "model": model,
        },
        service_name=ServiceName.ELEVENLABS,
        needs_aiohttp_session=True,
        aiohttp_session_key="aiohttp_session",
    )


def create_elevenlabs_realtime_config(
    api_key: Optional[str] = None,
    model: str = "scribe_v2_realtime",
) -> PipecatServiceConfig:
    """Create ElevenLabs Realtime STT service configuration."""
    from pipecat.services.elevenlabs import ElevenLabsRealtimeSTTService

    config = get_config()
    return PipecatServiceConfig(
        service_class=ElevenLabsRealtimeSTTService,
        service_kwargs={
            "api_key": api_key or config.elevenlabs_api_key,
            "model": model,
        },
        service_name=ServiceName.ELEVENLABS,
        needs_aiohttp_session=False,
    )


def create_nvidia_parakeet_config(
    url: Optional[str] = None,
) -> PipecatServiceConfig:
    """Create NVIDIA Parakeet STT service configuration.

    Uses the local NVidiaWebSocketSTTService that connects to the
    NVIDIA Parakeet ASR server via WebSocket.
    """
    from pipecat_bots.nvidia_stt import NVidiaWebSocketSTTService

    config = get_config()
    return PipecatServiceConfig(
        service_class=NVidiaWebSocketSTTService,
        service_kwargs={
            "url": url or config.nvidia_asr_url,
            "sample_rate": config.sample_rate,
        },
        service_name=ServiceName.NVIDIA_PARAKEET,
        needs_aiohttp_session=False,
    )


def create_openai_config(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-transcribe",
    language: str = "en",
) -> PipecatServiceConfig:
    """Create OpenAI STT service configuration."""
    from pipecat.services.openai import OpenAISTTService
    from pipecat.transcriptions.language import Language

    config = get_config()
    return PipecatServiceConfig(
        service_class=OpenAISTTService,
        service_kwargs={
            "api_key": api_key or config.openai_api_key,
            "model": model,
            "language": Language.EN,
        },
        service_name=ServiceName.DEEPGRAM,  # Placeholder - would need new enum
        needs_aiohttp_session=False,
    )


def create_groq_config(
    api_key: Optional[str] = None,
    model: str = "whisper-large-v3-turbo",
    language: str = "en",
) -> PipecatServiceConfig:
    """Create Groq STT service configuration."""
    from pipecat.services.groq import GroqSTTService
    from pipecat.transcriptions.language import Language

    config = get_config()
    return PipecatServiceConfig(
        service_class=GroqSTTService,
        service_kwargs={
            "api_key": api_key or config.groq_api_key,
            "model": model,
            "language": Language.EN,
        },
        service_name=ServiceName.DEEPGRAM,  # Placeholder - would need new enum
        needs_aiohttp_session=False,
    )


# Registry of factory functions for known services
SERVICE_CONFIG_FACTORIES: Dict[ServiceName, callable] = {
    ServiceName.DEEPGRAM: create_deepgram_config,
    ServiceName.CARTESIA: create_cartesia_config,
    ServiceName.ELEVENLABS: create_elevenlabs_config,
    ServiceName.NVIDIA_PARAKEET: create_nvidia_parakeet_config,
}


def create_pipecat_adapter(
    service_name: ServiceName,
    mode: EvaluationMode = EvaluationMode.AUTO,
    simulate_realtime: bool = True,
    **kwargs,
) -> UniversalPipecatAdapter:
    """Create a universal Pipecat adapter for a known service.

    Args:
        service_name: The service to create an adapter for
        mode: Evaluation mode (streaming, batch, or auto)
        simulate_realtime: Whether to simulate real-time streaming
        **kwargs: Service-specific configuration overrides

    Returns:
        Configured UniversalPipecatAdapter
    """
    if service_name not in SERVICE_CONFIG_FACTORIES:
        raise ValueError(f"Unknown service: {service_name}")

    config = SERVICE_CONFIG_FACTORIES[service_name](**kwargs)
    return UniversalPipecatAdapter(
        config=config,
        mode=mode,
        simulate_realtime=simulate_realtime,
    )


def get_available_services() -> list[ServiceName]:
    """Get list of services with Pipecat adapter support."""
    return list(SERVICE_CONFIG_FACTORIES.keys())

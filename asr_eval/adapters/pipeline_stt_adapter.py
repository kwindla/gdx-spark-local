"""Pipeline-based STT adapter for Pipecat services.

Uses a proper Pipeline context with PipelineTask and PipelineRunner
to run STT services, ensuring all Pipecat lifecycle requirements are met.
"""

import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor

from asr_eval.adapters.synthetic_input_transport import SyntheticInputTransport
from asr_eval.adapters.transcription_collector import TranscriptionCollector
from asr_eval.config import get_config
from asr_eval.models import AdapterType, AudioSample, ServiceName, TranscriptionResult


async def create_stt_service(
    service_name: ServiceName,
    aiohttp_session: Optional[aiohttp.ClientSession] = None,
    model: Optional[str] = None,
) -> FrameProcessor:
    """Create an STT service instance for the given service name.

    Args:
        service_name: The STT service to create.
        aiohttp_session: Optional aiohttp session for services that need it.
        model: Optional model name override (e.g., "nova-3", "whisper-large").

    Returns:
        Configured STT service instance.

    Raises:
        ValueError: If service_name is not supported.
    """
    config = get_config()

    if service_name == ServiceName.DEEPGRAM:
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.transcriptions.language import Language

        return DeepgramSTTService(
            api_key=config.deepgram_api_key,
            model=model or "nova-3",
            language=Language.EN,
        )

    elif service_name == ServiceName.CARTESIA:
        from pipecat.services.cartesia.stt import CartesiaSTTService

        return CartesiaSTTService(api_key=config.cartesia_api_key)

    elif service_name == ServiceName.ELEVENLABS:
        from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

        if aiohttp_session is None:
            raise ValueError("ElevenLabs requires an aiohttp session")
        return ElevenLabsSTTService(
            api_key=config.elevenlabs_api_key,
            aiohttp_session=aiohttp_session,
        )

    elif service_name == ServiceName.NVIDIA_PARAKEET:
        from pipecat_bots.nvidia_stt import NVidiaWebSocketSTTService

        return NVidiaWebSocketSTTService(
            url=config.nvidia_asr_url,
            sample_rate=config.sample_rate,
        )

    elif service_name == ServiceName.SPEECHMATICS:
        from pipecat.services.speechmatics.stt import SpeechmaticsSTTService

        return SpeechmaticsSTTService(
            api_key=config.speechmatics_api_key,
            sample_rate=config.sample_rate,
            params=SpeechmaticsSTTService.InputParams(
                # Reduce silence trigger for faster server-side VAD response
                # Default is 0.5s, using 0.2s to match our Silero VAD
                end_of_utterance_silence_trigger=0.2,
                # Also reduce max_delay for faster partial->final promotion
                max_delay=0.5,
            ),
        )

    elif service_name == ServiceName.SONIOX:
        from pipecat.services.soniox.stt import SonioxSTTService

        return SonioxSTTService(
            api_key=config.soniox_api_key,
            sample_rate=config.sample_rate,
            # Use client-side VAD (Silero) instead of server-side endpoint detection
            # This sends FINALIZE_MESSAGE on UserStoppedSpeakingFrame for faster response
            vad_force_turn_endpoint=True,
        )

    else:
        raise ValueError(f"Unsupported service: {service_name}")


class PipelineSTTAdapter:
    """Adapter that runs STT services via a proper Pipecat Pipeline.

    This adapter creates a full Pipeline with:
    - SyntheticInputTransport: Pumps audio from file into the pipeline
    - STT Service: The Pipecat STT service to evaluate
    - TranscriptionCollector: Captures the final TranscriptionFrame

    The Pipeline is run with PipelineTask and PipelineRunner, ensuring
    all Pipecat lifecycle requirements (TaskManager, etc.) are met.
    """

    def __init__(
        self,
        service_name: ServiceName,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        model: Optional[str] = None,
        vad_stop_secs: float = 0.2,
        post_audio_silence_ms: int = 2000,
    ):
        """Initialize the adapter.

        Args:
            service_name: The STT service to use.
            sample_rate: Audio sample rate in Hz.
            chunk_ms: Duration of each audio chunk in ms.
            model: Optional model name (e.g., "nova-3", "whisper-large").
            vad_stop_secs: Silence for Silero VAD to trigger stop (default 0.2s).
            post_audio_silence_ms: Silence after audio ends (default 2000ms).
        """
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.model = model
        self.vad_stop_secs = vad_stop_secs
        self.post_audio_silence_ms = post_audio_silence_ms

    async def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        """Transcribe an audio sample using the configured STT service.

        Args:
            sample: The audio sample to transcribe.

        Returns:
            TranscriptionResult with the transcription and timing metrics.
        """
        # Load audio data
        audio_path = Path(sample.audio_path)
        if not audio_path.exists():
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=sample.duration_seconds * 1000,
                error=f"Audio file not found: {audio_path}",
                adapter_type=AdapterType.PIPECAT_STREAMING,
            )

        audio_data = audio_path.read_bytes()
        audio_duration_ms = sample.duration_seconds * 1000

        # Create components
        transport = SyntheticInputTransport(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            vad_stop_secs=self.vad_stop_secs,
            post_audio_silence_ms=self.post_audio_silence_ms,
        )
        # Pass vad_stop_secs for accurate latency calculation
        # (latency = time from VAD fire to transcription + vad detection delay)
        collector = TranscriptionCollector(vad_stop_secs=self.vad_stop_secs)

        # Create aiohttp session for services that need it
        session: Optional[aiohttp.ClientSession] = None
        if self.service_name == ServiceName.ELEVENLABS:
            session = aiohttp.ClientSession()

        try:
            # Create STT service
            stt_service = await create_stt_service(
                self.service_name,
                aiohttp_session=session,
                model=self.model,
            )

            # Build pipeline
            pipeline = Pipeline([transport, stt_service, collector])

            # Create task with audio parameters
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    audio_in_sample_rate=self.sample_rate,
                    audio_in_channels=1,
                ),
            )

            # Run the pipeline in background
            runner = PipelineRunner(handle_sigint=False)
            pipeline_coro = runner.run(task)
            pipeline_task = asyncio.create_task(pipeline_coro)

            try:
                # Wait for audio to be pumped and queue drained
                # The transport now waits for queue.join() before signaling complete
                await transport.wait_for_audio_complete(timeout=60.0)

                # Mark when audio finished being sent
                collector.mark_audio_finished()

                # Wait for transcription result with timeout
                # This waits for TranscriptionFrame to arrive
                result = await collector.wait_for_result(timeout=30.0)

            finally:
                # Cancel the pipeline task (this triggers cleanup)
                await task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

            # Get result
            result = collector.result
            text = result.final_text or ""
            time_ms = result.timing.time_to_transcription_ms or 0

            logger.debug(
                f"[{self.service_name.value}] Transcribed {len(text)} chars in {time_ms:.1f}ms"
            )

            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text=text,
                time_to_transcription_ms=time_ms,
                audio_duration_ms=audio_duration_ms,
                error=result.error,
                adapter_type=AdapterType.PIPECAT_STREAMING,
            )

        except asyncio.TimeoutError as e:
            logger.error(f"[{self.service_name.value}] Timeout: {e}")
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=audio_duration_ms,
                error=f"Timeout: {e}",
                adapter_type=AdapterType.PIPECAT_STREAMING,
            )

        except Exception as e:
            logger.error(f"[{self.service_name.value}] Error: {e}")
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=audio_duration_ms,
                error=str(e),
                adapter_type=AdapterType.PIPECAT_STREAMING,
            )

        finally:
            # Cleanup aiohttp session
            if session:
                await session.close()


async def transcribe_with_pipeline(
    sample: AudioSample,
    service_name: ServiceName,
) -> TranscriptionResult:
    """Convenience function to transcribe a sample with a service.

    Args:
        sample: The audio sample to transcribe.
        service_name: The STT service to use.

    Returns:
        TranscriptionResult with the transcription and timing metrics.
    """
    adapter = PipelineSTTAdapter(service_name=service_name)
    return await adapter.transcribe(sample)

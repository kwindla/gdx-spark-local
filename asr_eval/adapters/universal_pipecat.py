"""Universal Pipecat STT adapter for evaluating any Pipecat STT service."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import (
    SegmentedSTTService,
    STTService,
)

from asr_eval.harness.result_collector import ResultCollector
from asr_eval.harness.synthetic_transport import SyntheticAudioSource
from asr_eval.models import AudioSample, ServiceName, TranscriptionResult


class EvaluationMode(str, Enum):
    """Evaluation mode for STT testing."""

    STREAMING = "streaming"  # Real-time audio streaming (20ms chunks with delays)
    BATCH = "batch"  # Complete audio with VAD frames (no delays)
    AUTO = "auto"  # Auto-detect based on service type


class AdapterType(str, Enum):
    """Type of adapter used for transcription."""

    PIPECAT_STREAMING = "pipecat_streaming"
    PIPECAT_BATCH = "pipecat_batch"
    DIRECT = "direct"


@dataclass
class PipecatServiceConfig:
    """Configuration for a Pipecat STT service."""

    service_class: Type[STTService]
    service_kwargs: Dict[str, Any]
    service_name: ServiceName
    needs_aiohttp_session: bool = False
    aiohttp_session_key: str = "aiohttp_session"


class UniversalPipecatAdapter:
    """Universal adapter for testing any Pipecat STT service.

    Handles:
    - Service lifecycle (start, process, stop) using public API
    - Streaming vs batch evaluation modes
    - aiohttp session management for services that need it
    - Proper cleanup without private API calls
    """

    def __init__(
        self,
        config: PipecatServiceConfig,
        mode: EvaluationMode = EvaluationMode.AUTO,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        simulate_realtime: bool = True,
        timeout: float = 30.0,
    ):
        """Initialize the universal adapter.

        Args:
            config: Service configuration including class and kwargs
            mode: Evaluation mode (streaming, batch, or auto-detect)
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk in ms
            simulate_realtime: Whether to sleep between chunks in streaming mode
            timeout: Timeout for transcription in seconds
        """
        self.config = config
        self.mode = mode
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.simulate_realtime = simulate_realtime
        self.timeout = timeout

        self._service: Optional[STTService] = None
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._collector: Optional[ResultCollector] = None

    @property
    def service_name(self) -> ServiceName:
        """Return the service name."""
        return self.config.service_name

    @property
    def adapter_type(self) -> AdapterType:
        """Return the adapter type based on mode and service class."""
        effective_mode = self._resolve_mode()
        if effective_mode == EvaluationMode.STREAMING:
            return AdapterType.PIPECAT_STREAMING
        return AdapterType.PIPECAT_BATCH

    def _resolve_mode(self) -> EvaluationMode:
        """Resolve AUTO mode to concrete mode based on service type."""
        if self.mode != EvaluationMode.AUTO:
            return self.mode

        # Auto-detect based on service class
        if issubclass(self.config.service_class, SegmentedSTTService):
            return EvaluationMode.BATCH
        return EvaluationMode.STREAMING

    async def _create_service(self) -> STTService:
        """Create the Pipecat STT service instance."""
        kwargs = self.config.service_kwargs.copy()

        # Handle aiohttp session if needed
        if self.config.needs_aiohttp_session:
            if self._aiohttp_session is None:
                self._aiohttp_session = aiohttp.ClientSession()
            kwargs[self.config.aiohttp_session_key] = self._aiohttp_session

        self._service = self.config.service_class(**kwargs)
        return self._service

    async def _cleanup(self) -> None:
        """Clean up resources using public API only."""
        # Send EndFrame to trigger proper shutdown
        if self._service:
            try:
                await self._service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
            except Exception as e:
                logger.debug(f"Error sending EndFrame: {e}")

        # Close aiohttp session if we created it
        if self._aiohttp_session:
            try:
                await self._aiohttp_session.close()
            except Exception:
                pass
            self._aiohttp_session = None

        self._service = None

    async def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        """Transcribe an audio sample using Pipecat service.

        Args:
            sample: AudioSample to transcribe

        Returns:
            TranscriptionResult with transcription and metrics
        """
        audio_path = Path(sample.audio_path)
        if not audio_path.exists():
            return self._error_result(sample, f"Audio file not found: {audio_path}")

        audio_data = audio_path.read_bytes()
        effective_mode = self._resolve_mode()

        try:
            # Create service and collector
            service = await self._create_service()
            self._collector = ResultCollector()
            service.link(self._collector)

            # Initialize service with StartFrame
            start_frame = StartFrame(
                allow_interruptions=False,
                enable_metrics=False,
                enable_usage_metrics=False,
                report_only_initial_ttfb=True,
                audio_in_sample_rate=self.sample_rate,
            )
            await service.process_frame(start_frame, FrameDirection.DOWNSTREAM)

            # Process based on mode
            if effective_mode == EvaluationMode.STREAMING:
                result = await self._process_streaming(service, audio_data, sample)
            else:
                result = await self._process_batch(service, audio_data, sample)

            return result

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return self._error_result(sample, str(e))

        finally:
            await self._cleanup()

    async def _process_streaming(
        self,
        service: STTService,
        audio_data: bytes,
        sample: AudioSample,
    ) -> TranscriptionResult:
        """Process audio in streaming mode (20ms chunks with delays)."""
        source = SyntheticAudioSource(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
            simulate_realtime=self.simulate_realtime,
        )

        # Signal user started speaking
        await service.process_frame(
            UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
        )

        # Stream audio frames
        async for frame in source.stream_frames():
            await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Mark turn finished time BEFORE sending the frame
        self._collector.mark_turn_finished()

        # Signal user stopped speaking
        await service.process_frame(
            UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
        )

        return await self._wait_for_result(sample)

    async def _process_batch(
        self,
        service: STTService,
        audio_data: bytes,
        sample: AudioSample,
    ) -> TranscriptionResult:
        """Process audio in batch mode (send all audio, then VAD frames)."""
        source = SyntheticAudioSource(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
            simulate_realtime=False,  # No delays in batch mode
        )

        # Signal user started speaking
        await service.process_frame(
            UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
        )

        # Send all audio frames without delay
        async for frame in source.stream_frames():
            await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Mark turn finished time
        self._collector.mark_turn_finished()

        # Signal user stopped speaking - triggers transcription
        await service.process_frame(
            UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
        )

        return await self._wait_for_result(sample)

    async def _wait_for_result(self, sample: AudioSample) -> TranscriptionResult:
        """Wait for transcription result from collector."""
        try:
            result = await self._collector.wait_for_result(timeout=self.timeout)
        except asyncio.TimeoutError:
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.service_name,
                transcribed_text="",
                time_to_transcription_ms=self.timeout * 1000,
                audio_duration_ms=sample.duration_seconds * 1000,
                rtf=self.timeout / sample.duration_seconds,
                error="Transcription timeout",
            )

        time_to_transcription_ms = result.timing.time_to_transcription_ms or 0
        audio_duration_ms = sample.duration_seconds * 1000
        rtf = (
            time_to_transcription_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        )

        return TranscriptionResult(
            sample_id=sample.sample_id,
            service_name=self.service_name,
            transcribed_text=result.final_text or "",
            time_to_transcription_ms=time_to_transcription_ms,
            audio_duration_ms=audio_duration_ms,
            rtf=rtf,
            timestamp=datetime.utcnow(),
            error=result.error,
        )

    def _error_result(self, sample: AudioSample, error: str) -> TranscriptionResult:
        """Create error result."""
        return TranscriptionResult(
            sample_id=sample.sample_id,
            service_name=self.service_name,
            transcribed_text="",
            time_to_transcription_ms=0,
            audio_duration_ms=sample.duration_seconds * 1000,
            rtf=0,
            error=error,
        )

    async def transcribe_batch(
        self,
        samples: list[AudioSample],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[TranscriptionResult]:
        """Transcribe a batch of samples.

        Args:
            samples: List of AudioSample to transcribe
            progress_callback: Optional callback(current, total, sample_id)

        Returns:
            List of TranscriptionResult objects
        """
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

"""Unified wrapper for testing STT services.

Provides a common interface for running transcription tests across
different STT service implementations.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from asr_eval.harness.result_collector import CollectedResult, ResultCollector
from asr_eval.harness.synthetic_transport import SyntheticAudioSource
from asr_eval.models import AudioSample, ServiceName, TranscriptionResult


class STTServiceAdapter(ABC):
    """Abstract base class for STT service adapters.

    Each STT service (Deepgram, Cartesia, etc.) needs an adapter that
    implements this interface to work with the test harness.
    """

    @property
    @abstractmethod
    def service_name(self) -> ServiceName:
        """Return the service name enum value."""
        pass

    @abstractmethod
    async def create_service(self):
        """Create and return the Pipecat STT service instance."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources."""
        pass


class STTServiceHarness:
    """Unified harness for testing STT services.

    Handles the complete lifecycle of running a transcription test:
    1. Initialize the STT service
    2. Stream audio frames
    3. Send VAD signals
    4. Collect results with timing
    5. Clean up
    """

    def __init__(
        self,
        adapter: STTServiceAdapter,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        simulate_realtime: bool = True,
        timeout: float = 30.0,
    ):
        """Initialize the harness.

        Args:
            adapter: STT service adapter
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk
            simulate_realtime: Whether to simulate real-time streaming
            timeout: Timeout for transcription in seconds
        """
        self.adapter = adapter
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.simulate_realtime = simulate_realtime
        self.timeout = timeout

    async def run_transcription(
        self,
        sample: AudioSample,
    ) -> TranscriptionResult:
        """Run transcription on an audio sample.

        Args:
            sample: AudioSample to transcribe

        Returns:
            TranscriptionResult with transcription and metrics
        """
        # Load audio
        audio_path = Path(sample.audio_path)
        if not audio_path.exists():
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.adapter.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=sample.duration_seconds * 1000,
                rtf=0,
                error=f"Audio file not found: {audio_path}",
            )

        audio_data = audio_path.read_bytes()

        # Create audio source
        source = SyntheticAudioSource(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
            simulate_realtime=self.simulate_realtime,
        )

        # Create result collector
        collector = ResultCollector()

        # Create STT service
        try:
            service = await self.adapter.create_service()
        except Exception as e:
            logger.error(f"Failed to create STT service: {e}")
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.adapter.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=sample.duration_seconds * 1000,
                rtf=0,
                error=f"Service creation failed: {e}",
            )

        try:
            # Link collector to receive service output
            service.link(collector)

            # Initialize service
            start_frame = StartFrame(
                allow_interruptions=False,
                enable_metrics=False,
                enable_usage_metrics=False,
                report_only_initial_ttfb=True,
                audio_in_sample_rate=self.sample_rate,
            )
            await service.process_frame(start_frame, FrameDirection.DOWNSTREAM)

            # Signal user started speaking
            await service.process_frame(
                UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )

            # Stream audio frames
            async for frame in source.stream_frames():
                await service.process_frame(frame, FrameDirection.DOWNSTREAM)

            # Mark turn finished time BEFORE sending the frame
            collector.mark_turn_finished()

            # Signal user stopped speaking - this triggers finalization
            await service.process_frame(
                UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )

            # Wait for final transcription
            try:
                result = await collector.wait_for_result(timeout=self.timeout)
            except asyncio.TimeoutError:
                return TranscriptionResult(
                    sample_id=sample.sample_id,
                    service_name=self.adapter.service_name,
                    transcribed_text="",
                    time_to_transcription_ms=self.timeout * 1000,
                    audio_duration_ms=sample.duration_seconds * 1000,
                    rtf=self.timeout / sample.duration_seconds,
                    error="Transcription timeout",
                )

            # Calculate metrics
            time_to_transcription_ms = result.timing.time_to_transcription_ms or 0
            audio_duration_ms = sample.duration_seconds * 1000
            rtf = time_to_transcription_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.adapter.service_name,
                transcribed_text=result.final_text or "",
                time_to_transcription_ms=time_to_transcription_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=rtf,
                timestamp=datetime.utcnow(),
                error=result.error,
            )

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return TranscriptionResult(
                sample_id=sample.sample_id,
                service_name=self.adapter.service_name,
                transcribed_text="",
                time_to_transcription_ms=0,
                audio_duration_ms=sample.duration_seconds * 1000,
                rtf=0,
                error=str(e),
            )

        finally:
            # Clean up
            try:
                await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
            except Exception:
                pass

            try:
                await self.adapter.cleanup()
            except Exception:
                pass

    async def run_batch(
        self,
        samples: list[AudioSample],
        progress_callback: Optional[callable] = None,
    ) -> list[TranscriptionResult]:
        """Run transcription on a batch of samples.

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

            result = await self.run_transcription(sample)
            results.append(result)

            if result.error:
                logger.warning(
                    f"[{i+1}/{len(samples)}] Error: {result.error}"
                )
            else:
                logger.info(
                    f"[{i+1}/{len(samples)}] {result.transcribed_text[:50]}... "
                    f"({result.time_to_transcription_ms:.0f}ms)"
                )

        return results

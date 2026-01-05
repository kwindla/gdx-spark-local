"""Synthetic audio transport for testing STT services.

Simulates real-time audio streaming from pre-recorded audio files
by yielding AudioRawFrame objects at realistic intervals.
"""

import asyncio
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import AudioRawFrame

from asr_eval.config import get_config


class SyntheticAudioSource:
    """Simulates real-time audio streaming from audio bytes.

    This class takes raw PCM audio data and yields AudioRawFrame objects
    at a rate that simulates real-time streaming, matching how audio
    would arrive from a microphone or WebRTC connection.
    """

    def __init__(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        simulate_realtime: bool = True,
        config: Optional[get_config] = None,
    ):
        """Initialize the synthetic audio source.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk in ms
            simulate_realtime: Whether to sleep between chunks
            config: Optional configuration override
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.simulate_realtime = simulate_realtime
        self.config = config or get_config()

        # Calculate chunk size in bytes (16-bit audio = 2 bytes per sample)
        samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        self.chunk_size = samples_per_chunk * 2

        # Calculate total duration
        self.total_samples = len(audio_data) // 2
        self.duration_seconds = self.total_samples / sample_rate

    @property
    def num_chunks(self) -> int:
        """Number of chunks in the audio."""
        return (len(self.audio_data) + self.chunk_size - 1) // self.chunk_size

    async def stream_frames(self) -> AsyncGenerator[AudioRawFrame, None]:
        """Generate AudioRawFrame objects simulating real-time streaming.

        Yields:
            AudioRawFrame objects containing audio chunks
        """
        sleep_time = self.chunk_duration_ms / 1000 if self.simulate_realtime else 0

        for offset in range(0, len(self.audio_data), self.chunk_size):
            chunk = self.audio_data[offset : offset + self.chunk_size]

            # Create AudioRawFrame
            frame = AudioRawFrame(
                audio=chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def get_all_audio(self) -> bytes:
        """Get all audio data at once (for non-streaming services).

        Returns:
            Complete audio data as bytes
        """
        return self.audio_data


class AudioFileSource(SyntheticAudioSource):
    """Synthetic audio source that loads from a file."""

    def __init__(
        self,
        audio_path: str,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        simulate_realtime: bool = True,
    ):
        """Initialize from an audio file.

        Args:
            audio_path: Path to PCM audio file
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk in ms
            simulate_realtime: Whether to sleep between chunks
        """
        from pathlib import Path

        audio_data = Path(audio_path).read_bytes()
        super().__init__(
            audio_data=audio_data,
            sample_rate=sample_rate,
            chunk_duration_ms=chunk_duration_ms,
            simulate_realtime=simulate_realtime,
        )
        self.audio_path = audio_path

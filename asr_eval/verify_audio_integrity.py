#!/usr/bin/env python3
"""Verify audio integrity through the pipeline.

Tests:
1. Are bytes captured byte-for-byte identical to input (excluding silence)?
2. Is the pipeline configured for 16kHz correctly?
3. Are 20ms chunks sent with accurate timing?
4. Is silence sent with correct timing?
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from asr_eval.adapters.synthetic_input_transport import SyntheticInputTransport


@dataclass
class TimingRecord:
    """Record of when a chunk was received."""
    chunk_index: int
    timestamp: float
    chunk_size: int
    is_silence: bool


@dataclass
class AudioVerificationResult:
    """Results of audio verification."""
    # Byte comparison
    input_audio_size: int = 0
    captured_audio_size: int = 0
    bytes_match: bool = False
    first_mismatch_offset: int = -1

    # Timing analysis
    chunk_timings: list[TimingRecord] = field(default_factory=list)
    expected_interval_ms: float = 20.0

    # Sample rate verification
    expected_sample_rate: int = 16000
    actual_sample_rates: list[int] = field(default_factory=list)

    def timing_stats(self):
        """Calculate timing statistics."""
        if len(self.chunk_timings) < 2:
            return None

        intervals = []
        audio_intervals = []
        silence_intervals = []

        for i in range(1, len(self.chunk_timings)):
            interval_ms = (self.chunk_timings[i].timestamp - self.chunk_timings[i-1].timestamp) * 1000
            intervals.append(interval_ms)

            if self.chunk_timings[i].is_silence:
                silence_intervals.append(interval_ms)
            else:
                audio_intervals.append(interval_ms)

        def stats(data):
            if not data:
                return None
            return {
                "count": len(data),
                "min_ms": min(data),
                "max_ms": max(data),
                "avg_ms": sum(data) / len(data),
                "jitter_ms": max(data) - min(data),
            }

        return {
            "all": stats(intervals),
            "audio": stats(audio_intervals),
            "silence": stats(silence_intervals),
        }


class AudioVerifier(FrameProcessor):
    """Captures audio frames and records timing for verification."""

    def __init__(self, input_audio: bytes, expected_sample_rate: int = 16000):
        super().__init__(name="AudioVerifier")
        self._input_audio = input_audio
        self._expected_sample_rate = expected_sample_rate
        self._captured_audio = bytearray()
        self._chunk_timings: list[TimingRecord] = []
        self._chunk_index = 0
        self._audio_chunks_expected = 0
        self._complete = asyncio.Event()

    @property
    def result(self) -> AudioVerificationResult:
        """Get verification result."""
        result = AudioVerificationResult(
            input_audio_size=len(self._input_audio),
            captured_audio_size=len(self._captured_audio),
            expected_sample_rate=self._expected_sample_rate,
            chunk_timings=self._chunk_timings,
        )

        # Compare bytes
        min_len = min(len(self._input_audio), len(self._captured_audio))
        result.bytes_match = (
            len(self._input_audio) == len(self._captured_audio) and
            self._input_audio == bytes(self._captured_audio)
        )

        if not result.bytes_match and min_len > 0:
            # Find first mismatch
            for i in range(min_len):
                if self._input_audio[i] != self._captured_audio[i]:
                    result.first_mismatch_offset = i
                    break
            else:
                # Lengths differ but content matches up to min_len
                result.first_mismatch_offset = min_len

        # Collect sample rates
        result.actual_sample_rates = list(set(
            t.chunk_size for t in self._chunk_timings if not t.is_silence
        ))

        return result

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            now = time.time()
            chunk = frame.audio

            # Determine if this is silence (all zeros)
            is_silence = all(b == 0 for b in chunk)

            # Only capture non-silence audio for byte comparison
            if not is_silence:
                self._captured_audio.extend(chunk)

            # Record timing
            self._chunk_timings.append(TimingRecord(
                chunk_index=self._chunk_index,
                timestamp=now,
                chunk_size=len(chunk),
                is_silence=is_silence,
            ))
            self._chunk_index += 1

            # Log sample rate from frame
            if frame.sample_rate != self._expected_sample_rate:
                logger.warning(
                    f"Chunk {self._chunk_index}: sample_rate={frame.sample_rate}, "
                    f"expected={self._expected_sample_rate}"
                )

        elif isinstance(frame, EndFrame):
            self._complete.set()

        await self.push_frame(frame, direction)

    async def wait_complete(self, timeout: float = 60.0):
        try:
            await asyncio.wait_for(self._complete.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for EndFrame")


async def verify_audio(audio_path: str, sample_rate: int = 16000, chunk_ms: int = 20):
    """Run audio verification test."""

    # Load input audio
    input_audio = Path(audio_path).read_bytes()
    logger.info(f"Input audio: {len(input_audio)} bytes ({len(input_audio) / 2 / sample_rate:.3f}s)")

    # Calculate expected chunk count
    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    chunk_size = samples_per_chunk * 2  # 16-bit
    expected_audio_chunks = (len(input_audio) + chunk_size - 1) // chunk_size
    expected_silence_chunks = 1000 // chunk_ms  # 1s of silence

    logger.info(f"Expected: {expected_audio_chunks} audio chunks + {expected_silence_chunks} silence chunks")
    logger.info(f"Chunk size: {chunk_size} bytes ({chunk_ms}ms at {sample_rate}Hz)")

    # Create components
    transport = SyntheticInputTransport(
        audio_data=input_audio,
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
    )
    verifier = AudioVerifier(input_audio, expected_sample_rate=sample_rate)

    # Build pipeline
    pipeline = Pipeline([transport, verifier])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_in_channels=1,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)
    pipeline_coro = runner.run(task)
    pipeline_task = asyncio.create_task(pipeline_coro)

    try:
        # Wait for audio pumping to complete
        await transport.wait_for_audio_complete(timeout=60.0)

        # Give a moment for frames to propagate
        await asyncio.sleep(0.5)

    finally:
        await task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass

    # Get results
    result = verifier.result

    # Report results
    print("\n" + "=" * 60)
    print("AUDIO INTEGRITY VERIFICATION RESULTS")
    print("=" * 60)

    # Question 1: Byte-for-byte match
    print("\n1. BYTE COMPARISON (excluding silence)")
    print(f"   Input audio:    {result.input_audio_size} bytes")
    print(f"   Captured audio: {result.captured_audio_size} bytes")
    print(f"   Match: {'YES ✓' if result.bytes_match else 'NO ✗'}")
    if not result.bytes_match:
        if result.first_mismatch_offset >= 0:
            print(f"   First mismatch at byte offset: {result.first_mismatch_offset}")
        if result.input_audio_size != result.captured_audio_size:
            diff = result.captured_audio_size - result.input_audio_size
            print(f"   Size difference: {diff:+d} bytes")

    # Question 2: Sample rate
    print("\n2. SAMPLE RATE CONFIGURATION")
    print(f"   Expected: {result.expected_sample_rate} Hz")

    # Check frame sample rates
    non_silence_chunks = [t for t in result.chunk_timings if not t.is_silence]
    if non_silence_chunks:
        print(f"   Chunk sizes seen: {set(t.chunk_size for t in non_silence_chunks)}")
        expected_chunk_size = int(result.expected_sample_rate * 20 / 1000) * 2
        print(f"   Expected chunk size for 20ms @ 16kHz: {expected_chunk_size} bytes")

    # Question 3 & 4: Timing
    stats = result.timing_stats()
    if stats:
        print("\n3. AUDIO CHUNK TIMING (20ms expected)")
        if stats["audio"]:
            s = stats["audio"]
            print(f"   Count:  {s['count']} chunks")
            print(f"   Min:    {s['min_ms']:.2f} ms")
            print(f"   Max:    {s['max_ms']:.2f} ms")
            print(f"   Avg:    {s['avg_ms']:.2f} ms")
            print(f"   Jitter: {s['jitter_ms']:.2f} ms")

            # Check for significant deviation
            if s['avg_ms'] < 19 or s['avg_ms'] > 21:
                print(f"   WARNING: Average interval deviates from 20ms by {abs(s['avg_ms'] - 20):.2f}ms")

        print("\n4. SILENCE CHUNK TIMING (20ms expected)")
        if stats["silence"]:
            s = stats["silence"]
            print(f"   Count:  {s['count']} chunks")
            print(f"   Min:    {s['min_ms']:.2f} ms")
            print(f"   Max:    {s['max_ms']:.2f} ms")
            print(f"   Avg:    {s['avg_ms']:.2f} ms")
            print(f"   Jitter: {s['jitter_ms']:.2f} ms")

    # Overall timing analysis
    print("\n5. TOTAL TIMING")
    if result.chunk_timings:
        total_chunks = len(result.chunk_timings)
        audio_chunks = len([t for t in result.chunk_timings if not t.is_silence])
        silence_chunks = len([t for t in result.chunk_timings if t.is_silence])

        first_time = result.chunk_timings[0].timestamp
        last_time = result.chunk_timings[-1].timestamp
        actual_duration = last_time - first_time
        expected_duration = (total_chunks - 1) * 0.020  # -1 because intervals

        print(f"   Total chunks:    {total_chunks} (audio={audio_chunks}, silence={silence_chunks})")
        print(f"   Actual duration: {actual_duration:.3f}s")
        print(f"   Expected:        {expected_duration:.3f}s")
        print(f"   Drift:           {(actual_duration - expected_duration) * 1000:.1f}ms")

    print("\n" + "=" * 60)

    return result


async def main():
    import argparse
    import sys

    from asr_eval.storage.database import Database

    parser = argparse.ArgumentParser(description="Verify audio integrity through pipeline")
    parser.add_argument("--sample-id", type=str, help="Specific sample ID to test")
    parser.add_argument("--audio-path", type=str, help="Direct path to PCM audio file")
    args = parser.parse_args()

    if args.audio_path:
        audio_path = args.audio_path
    else:
        # Get sample from database
        db = Database()

        if args.sample_id:
            sample = await db.get_sample(args.sample_id)
            if not sample:
                logger.error(f"Sample not found: {args.sample_id}")
                sys.exit(1)
            audio_path = sample.audio_path
        else:
            # Get first sample with ground truth
            samples = await db.get_all_samples()
            sample = None
            for s in samples[:10]:
                gt = await db.get_ground_truth(s.sample_id)
                if gt:
                    sample = s
                    break

            if not sample:
                logger.error("No samples found")
                sys.exit(1)
            audio_path = sample.audio_path

    logger.info(f"Testing audio: {audio_path}")
    await verify_audio(audio_path)


if __name__ == "__main__":
    asyncio.run(main())

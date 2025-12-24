#!/usr/bin/env python3
"""Experiment 1: Parallel Generation Testing.

Tests concurrent TTS generation to determine:
- Whether parallel requests cause quality degradation
- TTFB regression under load
- RTF changes with concurrency

Usage:
    python -m tests.adaptive_streaming.test_parallel
"""

import asyncio
import json
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx

# Test configuration
SERVER_URL = "http://localhost:8001"
SAMPLE_RATE = 22000
OUTPUT_DIR = Path("samples_parallel")

# Test sentences - different text for each parallel request
TEST_SENTENCES = [
    "Hello! How can I help you today?",
    "The weather is beautiful this morning.",
    "I'd be happy to explain that concept.",
    "Let me think about the best approach.",
]


@dataclass
class GenerationMetrics:
    """Metrics for a single TTS generation."""

    sentence_index: int
    text: str
    mode: str  # "sequential", "parallel2", "parallel4"
    ttfb_ms: float
    total_time_ms: float
    audio_duration_ms: float
    audio_bytes: int
    rtf: float
    filename: str


def save_wav(audio_bytes: bytes, filepath: Path):
    """Save raw PCM bytes as WAV file."""
    with wave.open(str(filepath), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)


async def generate_single(
    client: httpx.AsyncClient,
    text: str,
    sentence_idx: int,
    mode: str,
    run_idx: int = 0,
) -> GenerationMetrics:
    """Generate audio for a single sentence and measure metrics."""
    start_time = time.time()
    first_chunk_time: Optional[float] = None
    chunks = []

    async with client.stream(
        "POST",
        f"{SERVER_URL}/v1/audio/speech/stream",
        json={
            "input": text,
            "voice": "aria",
            "language": "en",
            "first_chunk_frames": 8,
            "chunk_size_frames": 16,
            "overlap_frames": 12,
        },
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first_chunk_time is None:
                first_chunk_time = time.time()
            chunks.append(chunk)

    end_time = time.time()
    audio_bytes = b"".join(chunks)

    ttfb_ms = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
    total_time_ms = (end_time - start_time) * 1000
    audio_duration_ms = len(audio_bytes) / (SAMPLE_RATE * 2) * 1000
    rtf = (total_time_ms / 1000) / (audio_duration_ms / 1000) if audio_duration_ms > 0 else 0

    # Generate filename
    if mode == "sequential":
        filename = f"sequential_{sentence_idx + 1:02d}.wav"
    elif mode == "parallel2":
        filename = f"parallel2_run{run_idx + 1}_{sentence_idx + 1:02d}.wav"
    else:
        filename = f"parallel4_{sentence_idx + 1:02d}.wav"

    # Save audio
    filepath = OUTPUT_DIR / filename
    save_wav(audio_bytes, filepath)

    return GenerationMetrics(
        sentence_index=sentence_idx,
        text=text,
        mode=mode,
        ttfb_ms=ttfb_ms,
        total_time_ms=total_time_ms,
        audio_duration_ms=audio_duration_ms,
        audio_bytes=len(audio_bytes),
        rtf=rtf,
        filename=filename,
    )


async def run_sequential_test(client: httpx.AsyncClient) -> list[GenerationMetrics]:
    """Run 4 sequential generations as baseline."""
    print("\n" + "=" * 60)
    print("BASELINE: Sequential Generation (4 sentences)")
    print("=" * 60)

    metrics = []
    for i, text in enumerate(TEST_SENTENCES):
        print(f"  Generating {i + 1}/4: '{text[:40]}...'")
        m = await generate_single(client, text, i, "sequential")
        metrics.append(m)
        print(f"    TTFB: {m.ttfb_ms:.0f}ms, Total: {m.total_time_ms:.0f}ms, RTF: {m.rtf:.2f}x")

    avg_ttfb = sum(m.ttfb_ms for m in metrics) / len(metrics)
    avg_rtf = sum(m.rtf for m in metrics) / len(metrics)
    print(f"\n  Average TTFB: {avg_ttfb:.0f}ms, Average RTF: {avg_rtf:.2f}x")

    return metrics


async def run_parallel2_test(client: httpx.AsyncClient) -> list[GenerationMetrics]:
    """Run 2 concurrent generations, twice."""
    print("\n" + "=" * 60)
    print("TEST: Parallel-2 Generation (2 concurrent, 2 runs)")
    print("=" * 60)

    all_metrics = []

    for run in range(2):
        print(f"\n  Run {run + 1}/2:")
        sentences = TEST_SENTENCES[run * 2 : run * 2 + 2]

        tasks = [
            generate_single(client, text, i, "parallel2", run)
            for i, text in enumerate(sentences)
        ]

        start = time.time()
        metrics = await asyncio.gather(*tasks)
        wall_time = (time.time() - start) * 1000

        for m in metrics:
            print(f"    [{m.sentence_index + 1}] TTFB: {m.ttfb_ms:.0f}ms, Total: {m.total_time_ms:.0f}ms, RTF: {m.rtf:.2f}x")
            all_metrics.append(m)

        print(f"    Wall-clock time for run: {wall_time:.0f}ms")

    avg_ttfb = sum(m.ttfb_ms for m in all_metrics) / len(all_metrics)
    avg_rtf = sum(m.rtf for m in all_metrics) / len(all_metrics)
    print(f"\n  Average TTFB: {avg_ttfb:.0f}ms, Average RTF: {avg_rtf:.2f}x")

    return all_metrics


async def run_parallel4_test(client: httpx.AsyncClient) -> list[GenerationMetrics]:
    """Run 4 concurrent generations."""
    print("\n" + "=" * 60)
    print("TEST: Parallel-4 Generation (4 concurrent)")
    print("=" * 60)

    tasks = [
        generate_single(client, text, i, "parallel4")
        for i, text in enumerate(TEST_SENTENCES)
    ]

    start = time.time()
    metrics = await asyncio.gather(*tasks)
    wall_time = (time.time() - start) * 1000

    for m in metrics:
        print(f"  [{m.sentence_index + 1}] TTFB: {m.ttfb_ms:.0f}ms, Total: {m.total_time_ms:.0f}ms, RTF: {m.rtf:.2f}x")

    print(f"\n  Wall-clock time: {wall_time:.0f}ms")

    avg_ttfb = sum(m.ttfb_ms for m in metrics) / len(metrics)
    avg_rtf = sum(m.rtf for m in metrics) / len(metrics)
    print(f"  Average TTFB: {avg_ttfb:.0f}ms, Average RTF: {avg_rtf:.2f}x")

    return metrics


def print_summary(
    sequential: list[GenerationMetrics],
    parallel2: list[GenerationMetrics],
    parallel4: list[GenerationMetrics],
):
    """Print comparison summary."""
    print("\n" + "=" * 60)
    print("SUMMARY: Parallel Generation Test Results")
    print("=" * 60)

    def stats(metrics: list[GenerationMetrics]) -> tuple[float, float, float, float]:
        ttfbs = [m.ttfb_ms for m in metrics]
        rtfs = [m.rtf for m in metrics]
        return (
            sum(ttfbs) / len(ttfbs),
            max(ttfbs),
            sum(rtfs) / len(rtfs),
            max(rtfs),
        )

    seq_avg_ttfb, seq_max_ttfb, seq_avg_rtf, seq_max_rtf = stats(sequential)
    p2_avg_ttfb, p2_max_ttfb, p2_avg_rtf, p2_max_rtf = stats(parallel2)
    p4_avg_ttfb, p4_max_ttfb, p4_avg_rtf, p4_max_rtf = stats(parallel4)

    print("\n  | Mode       | Avg TTFB | Max TTFB | Avg RTF | Max RTF |")
    print("  |------------|----------|----------|---------|---------|")
    print(f"  | Sequential | {seq_avg_ttfb:7.0f}ms | {seq_max_ttfb:7.0f}ms | {seq_avg_rtf:6.2f}x | {seq_max_rtf:6.2f}x |")
    print(f"  | Parallel-2 | {p2_avg_ttfb:7.0f}ms | {p2_max_ttfb:7.0f}ms | {p2_avg_rtf:6.2f}x | {p2_max_rtf:6.2f}x |")
    print(f"  | Parallel-4 | {p4_avg_ttfb:7.0f}ms | {p4_max_ttfb:7.0f}ms | {p4_avg_rtf:6.2f}x | {p4_max_rtf:6.2f}x |")

    # Calculate regression
    ttfb_regression_p2 = ((p2_avg_ttfb - seq_avg_ttfb) / seq_avg_ttfb) * 100
    ttfb_regression_p4 = ((p4_avg_ttfb - seq_avg_ttfb) / seq_avg_ttfb) * 100
    rtf_regression_p2 = ((p2_avg_rtf - seq_avg_rtf) / seq_avg_rtf) * 100
    rtf_regression_p4 = ((p4_avg_rtf - seq_avg_rtf) / seq_avg_rtf) * 100

    print("\n  Regression vs Sequential Baseline:")
    print(f"    Parallel-2: TTFB {ttfb_regression_p2:+.1f}%, RTF {rtf_regression_p2:+.1f}%")
    print(f"    Parallel-4: TTFB {ttfb_regression_p4:+.1f}%, RTF {rtf_regression_p4:+.1f}%")

    # Success criteria check
    print("\n  Success Criteria Check:")
    if abs(ttfb_regression_p2) < 50:
        print("    ✓ Parallel-2 TTFB regression < 50%")
    else:
        print(f"    ✗ Parallel-2 TTFB regression >= 50% ({ttfb_regression_p2:+.1f}%)")

    if abs(ttfb_regression_p4) < 100:
        print("    ✓ Parallel-4 TTFB regression < 100%")
    else:
        print(f"    ✗ Parallel-4 TTFB regression >= 100% ({ttfb_regression_p4:+.1f}%)")

    print("\n  Audio files saved to: samples_parallel/")
    print("  Listen to samples to evaluate quality degradation.")


async def main():
    """Run parallel generation tests."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check server health
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/health", timeout=5.0)
            health = response.json()
            if health.get("status") != "healthy":
                print(f"Server not ready: {health}")
                return
        except Exception as e:
            print(f"Cannot connect to server at {SERVER_URL}: {e}")
            print("Make sure the TTS server is running: python -m nemotron_speech.tts_server")
            return

    print("\n" + "=" * 60)
    print("Parallel Generation Test - Experiment 1")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Output: {OUTPUT_DIR}")

    async with httpx.AsyncClient() as client:
        # Run tests
        sequential_metrics = await run_sequential_test(client)
        parallel2_metrics = await run_parallel2_test(client)
        parallel4_metrics = await run_parallel4_test(client)

        # Print summary
        print_summary(sequential_metrics, parallel2_metrics, parallel4_metrics)

        # Save metrics to JSON
        all_metrics = {
            "sequential": [asdict(m) for m in sequential_metrics],
            "parallel2": [asdict(m) for m in parallel2_metrics],
            "parallel4": [asdict(m) for m in parallel4_metrics],
        }

        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())

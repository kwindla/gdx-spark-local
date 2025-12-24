#!/usr/bin/env python3
"""Experiment 2: Adaptive Buffer Strategy Testing.

Tests the adaptive streaming TTS endpoints with various scenarios:
- Bulk text (all at once)
- Sentence-by-sentence streaming
- Token-by-token streaming (simulating LLM output)
- Mixed timing patterns

Usage:
    python -m tests.adaptive_streaming.test_adaptive
"""

import asyncio
import json
import time
import wave
from dataclasses import asdict
from pathlib import Path

import httpx

from tests.adaptive_streaming.client import AdaptiveStreamClient, StreamMetrics

# Configuration
SERVER_URL = "http://localhost:8001"
SAMPLE_RATE = 22000
OUTPUT_DIR = Path("samples_adaptive")


def save_wav(audio_bytes: bytes, filepath: Path):
    """Save raw PCM bytes as WAV file."""
    with wave.open(str(filepath), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)


def print_metrics(name: str, metrics: StreamMetrics):
    """Print metrics for a test."""
    print(f"\n  {name}:")
    print(f"    TTFB: {metrics.ttfb_ms:.0f}ms")
    print(f"    Total time: {metrics.total_time_ms:.0f}ms")
    print(f"    Audio duration: {metrics.audio_duration_ms:.0f}ms")
    print(f"    RTF: {metrics.rtf:.2f}x")
    print(f"    Chunks: {len(metrics.chunk_sizes)}")


async def test_bulk_text(client: AdaptiveStreamClient) -> StreamMetrics:
    """Test: All text provided at once."""
    print("\n" + "=" * 60)
    print("TEST A: Bulk Text (all at once)")
    print("=" * 60)

    text = (
        "Hello! Welcome to the adaptive streaming TTS demonstration. "
        "This text is provided all at once, so the system should use "
        "streaming mode for the first segment to achieve fast time to first byte, "
        "then switch to batch mode for subsequent segments when the buffer is healthy."
    )

    audio, metrics = await client.synthesize_streaming(text)
    save_wav(audio, OUTPUT_DIR / "bulk_text.wav")
    print_metrics("Bulk text", metrics)

    return metrics


async def test_sentence_stream(client: AdaptiveStreamClient) -> StreamMetrics:
    """Test: Text arrives sentence by sentence."""
    print("\n" + "=" * 60)
    print("TEST B: Sentence-by-Sentence Streaming")
    print("=" * 60)

    sentences = [
        "This is the first sentence. ",
        "Here comes the second one! ",
        "And now the third sentence arrives. ",
        "Finally, the fourth and last sentence. ",
    ]

    # 200ms delay between sentences (simulating typical LLM sentence generation)
    audio, metrics = await client.synthesize_incremental(
        sentences, delay_between_ms=200
    )
    save_wav(audio, OUTPUT_DIR / "sentence_stream.wav")
    print_metrics("Sentence stream", metrics)

    return metrics


async def test_token_stream(client: AdaptiveStreamClient) -> StreamMetrics:
    """Test: Text arrives token by token (simulating LLM)."""
    print("\n" + "=" * 60)
    print("TEST C: Token-by-Token Streaming (LLM simulation)")
    print("=" * 60)

    # Simulate LLM tokens - words with varying delays
    full_text = "Hello there! I am simulating an LLM generating text token by token. This should trigger the text buffering logic to collect tokens before generating."
    tokens = full_text.split(" ")
    tokens = [t + " " for t in tokens[:-1]] + [tokens[-1]]  # Add spaces back

    # 30ms per token (typical LLM speed)
    audio, metrics = await client.synthesize_incremental(
        tokens, delay_between_ms=30
    )
    save_wav(audio, OUTPUT_DIR / "token_stream.wav")
    print_metrics("Token stream", metrics)

    return metrics


async def test_mixed_timing(client: AdaptiveStreamClient) -> StreamMetrics:
    """Test: Mixed timing - some fast, some slow arrivals."""
    print("\n" + "=" * 60)
    print("TEST D: Mixed Timing Pattern")
    print("=" * 60)

    stream_id, _ = await client.create_stream()
    metrics = StreamMetrics(stream_id=stream_id)
    metrics.create_time = time.time()

    audio_chunks = []

    async def receive():
        async for chunk in client.receive_audio(stream_id, metrics):
            audio_chunks.append(chunk)

    receive_task = asyncio.create_task(receive())
    await asyncio.sleep(0.01)

    # Fast burst of text
    await client.append_text(stream_id, "Quick burst of text! ")
    await asyncio.sleep(0.05)
    await client.append_text(stream_id, "More text immediately! ")

    # Pause (simulating user thinking)
    await asyncio.sleep(0.5)

    # Another burst
    await client.append_text(stream_id, "After a pause, here comes more text. ")
    await client.append_text(stream_id, "And even more following quickly. ")

    # Slow token-by-token
    await asyncio.sleep(0.3)
    for word in ["Finally", "some", "slow", "tokens."]:
        await client.append_text(stream_id, word + " ")
        await asyncio.sleep(0.1)

    await client.close_stream(stream_id)
    await receive_task
    metrics.completion_time = time.time()

    audio = b"".join(audio_chunks)
    save_wav(audio, OUTPUT_DIR / "mixed_timing.wav")
    print_metrics("Mixed timing", metrics)

    return metrics


async def test_cancel_stream(client: AdaptiveStreamClient):
    """Test: Stream cancellation."""
    print("\n" + "=" * 60)
    print("TEST E: Stream Cancellation")
    print("=" * 60)

    stream_id, _ = await client.create_stream()
    print(f"  Created stream: {stream_id[:8]}...")

    await client.append_text(stream_id, "This text will never be fully generated.")

    # Start receiving
    chunks_received = 0

    async def receive():
        nonlocal chunks_received
        try:
            async for chunk in client.receive_audio(stream_id):
                chunks_received += 1
                if chunks_received >= 2:
                    # Cancel after receiving some chunks
                    await client.cancel_stream(stream_id)
                    break
        except Exception as e:
            print(f"  Receive ended: {e}")

    await receive()
    print(f"  Chunks received before cancel: {chunks_received}")
    print("  ✓ Cancellation test passed")


async def main():
    """Run all adaptive streaming tests."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check server health
    async with httpx.AsyncClient() as http:
        try:
            response = await http.get(f"{SERVER_URL}/health", timeout=5.0)
            health = response.json()
            if health.get("status") != "healthy":
                print(f"Server not ready: {health}")
                return
        except Exception as e:
            print(f"Cannot connect to server at {SERVER_URL}: {e}")
            print("Make sure the TTS server is running")
            return

    print("\n" + "=" * 60)
    print("Adaptive Streaming TTS Test - Experiment 2")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Output: {OUTPUT_DIR}")

    all_metrics = {}

    async with AdaptiveStreamClient(SERVER_URL) as client:
        # Run tests
        all_metrics["bulk_text"] = await test_bulk_text(client)
        all_metrics["sentence_stream"] = await test_sentence_stream(client)
        all_metrics["token_stream"] = await test_token_stream(client)
        all_metrics["mixed_timing"] = await test_mixed_timing(client)
        await test_cancel_stream(client)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Adaptive Streaming Test Results")
    print("=" * 60)

    print("\n  | Scenario        | TTFB    | Total   | Audio   | RTF   |")
    print("  |-----------------|---------|---------|---------|-------|")
    for name, m in all_metrics.items():
        print(
            f"  | {name:15} | {m.ttfb_ms:5.0f}ms | {m.total_time_ms:5.0f}ms | "
            f"{m.audio_duration_ms:5.0f}ms | {m.rtf:.2f}x |"
        )

    # Success criteria
    print("\n  Success Criteria Check:")
    bulk_ttfb = all_metrics["bulk_text"].ttfb_ms
    if bulk_ttfb < 200:
        print(f"    ✓ TTFB < 200ms for first audio ({bulk_ttfb:.0f}ms)")
    else:
        print(f"    ✗ TTFB >= 200ms ({bulk_ttfb:.0f}ms)")

    # Check all RTFs are real-time
    all_realtime = all(m.rtf < 1.0 for m in all_metrics.values())
    if all_realtime:
        print("    ✓ All scenarios maintain real-time generation (RTF < 1.0)")
    else:
        print("    ✗ Some scenarios exceed real-time")

    print(f"\n  Audio files saved to: {OUTPUT_DIR}/")

    # Save metrics
    metrics_data = {k: asdict(v) for k, v in all_metrics.items()}
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    print(f"  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())

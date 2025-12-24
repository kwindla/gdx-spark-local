#!/usr/bin/env python3
"""
Integration test for Chunked LLM Server + WebSocket TTS.

Tests the full pipeline: LLM generates text in chunks, TTS converts each chunk to audio.
Measures end-to-end latency from inference start to first audio byte.

Architecture:
    ┌──────────────────────┐      WS (8002/ws)       ┌─────────────────────┐
    │   Integration Test   │◄──────────────────────►│  Chunked LLM Server │
    │                      │                         │   (llama.cpp)       │
    │   - Orchestrates     │      WS (8001/ws/tts)   └─────────────────────┘
    │   - Measures timing  │◄──────────────────────►│  TTS Server         │
    │   - Reports metrics  │                         │   (Magpie)          │
    └──────────────────────┘                         └─────────────────────┘

Usage:
    uv run python tests/test_llm_tts_integration.py
    uv run python tests/test_llm_tts_integration.py --save-audio output.wav
    uv run python tests/test_llm_tts_integration.py --verbose
"""

import argparse
import asyncio
import json
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
    from websockets.asyncio.client import connect as ws_connect
except ImportError:
    from websockets import connect as ws_connect


# Server endpoints
LLM_URI = "ws://localhost:8002/ws"
TTS_URI = "ws://localhost:8001/ws/tts/stream"

# Audio constants
SAMPLE_RATE = 22000
BYTES_PER_SAMPLE = 2


@dataclass
class ChunkMetrics:
    """Timing metrics for a single LLM chunk + TTS segment."""

    chunk_num: int
    text: str
    token_count: int
    audio_bytes: int

    # Timestamps (absolute, from time.time())
    inference_start_ts: float
    llm_first_token_ts: Optional[float]
    llm_done_ts: float
    tts_text_sent_ts: float
    tts_first_audio_ts: Optional[float]
    tts_segment_done_ts: float

    @property
    def llm_ttft_ms(self) -> float:
        """LLM time to first token."""
        if self.llm_first_token_ts is None:
            return 0.0
        return (self.llm_first_token_ts - self.inference_start_ts) * 1000

    @property
    def llm_gen_time_ms(self) -> float:
        """Total LLM chunk generation time."""
        return (self.llm_done_ts - self.inference_start_ts) * 1000

    @property
    def tts_ttfb_ms(self) -> float:
        """TTS time to first byte."""
        if self.tts_first_audio_ts is None:
            return 0.0
        return (self.tts_first_audio_ts - self.tts_text_sent_ts) * 1000

    @property
    def tts_gen_time_ms(self) -> float:
        """Total TTS segment generation time."""
        return (self.tts_segment_done_ts - self.tts_text_sent_ts) * 1000

    @property
    def audio_duration_ms(self) -> float:
        """Audio playout duration."""
        return self.audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000

    @property
    def combined_ttfb_ms(self) -> float:
        """Combined TTFB: inference start to first audio byte."""
        if self.tts_first_audio_ts is None:
            return 0.0
        return (self.tts_first_audio_ts - self.inference_start_ts) * 1000


def format_timestamp(t: float) -> str:
    """Format timestamp as HH:MM:SS.mmm"""
    dt = datetime.fromtimestamp(t)
    return dt.strftime("%H:%M:%S.") + f"{int((t % 1) * 1000):03d}"


async def receive_llm_chunk(
    ws, chunk_num: int, inference_start: float
) -> tuple[str, int, Optional[float], float, bool]:
    """Receive LLM tokens until paused/done message.

    Returns: (text, token_count, first_token_ts, done_ts, is_done)
    """
    tokens = []
    first_token_ts = None

    while True:
        msg = json.loads(await ws.recv())

        if msg.get("type") == "token":
            if first_token_ts is None:
                first_token_ts = time.time()
            tokens.append(msg["content"])

        elif msg.get("type") in ("paused", "done"):
            return (
                msg["text"],
                len(tokens),
                first_token_ts,
                time.time(),
                msg["type"] == "done",
            )

        elif "error" in msg:
            raise RuntimeError(f"LLM error: {msg['error']}")


async def wait_for_tts_segment(
    tts_ws,
    first_timeout_s: float = 10.0,
    gap_timeout_ms: float = 150,
) -> tuple[bytes, Optional[float], float, bool]:
    """Wait for TTS segment to complete using gap detection.

    With TARGET_AUDIO_MS=500ms, TTS flushes almost immediately when text arrives.
    We wait for first audio, then collect until a gap indicates segment is done.

    Returns: (audio_bytes, first_audio_ts, done_ts, is_stream_complete)
    """
    audio_chunks = []
    first_audio_ts = None

    # Phase 1: Wait for first audio (TTS generation takes ~300-600ms)
    while first_audio_ts is None:
        try:
            msg = await asyncio.wait_for(tts_ws.recv(), timeout=first_timeout_s)
        except asyncio.TimeoutError:
            # No audio generated - might be empty segment or threshold not reached
            return b"", None, time.time(), False

        if isinstance(msg, bytes):
            first_audio_ts = time.time()
            audio_chunks.append(msg)
        else:
            data = json.loads(msg)
            if data.get("type") == "done":
                return b"".join(audio_chunks), first_audio_ts, time.time(), True
            elif data.get("type") == "error":
                raise RuntimeError(f"TTS error: {data.get('message')}")

    # Phase 2: Collect remaining audio until gap (segment boundary)
    while True:
        try:
            msg = await asyncio.wait_for(tts_ws.recv(), timeout=gap_timeout_ms / 1000)
            if isinstance(msg, bytes):
                audio_chunks.append(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "done":
                    return b"".join(audio_chunks), first_audio_ts, time.time(), True
        except asyncio.TimeoutError:
            # Gap detected - segment complete, more may be coming
            return b"".join(audio_chunks), first_audio_ts, time.time(), False


def print_chunk_progress(metrics: ChunkMetrics, verbose: bool = False):
    """Print progress for a completed chunk."""
    text_preview = metrics.text[:40] + "..." if len(metrics.text) > 40 else metrics.text
    text_preview = text_preview.replace("\n", "\\n")

    print(f"\n[{format_timestamp(metrics.inference_start_ts)}] LLM #{metrics.chunk_num}")
    if metrics.llm_first_token_ts:
        print(f"  └─ First token (LLM TTFT: {metrics.llm_ttft_ms:.0f}ms)")
    print(f"  └─ Complete: {metrics.token_count} tokens, {metrics.llm_gen_time_ms:.0f}ms")

    print(f"[{format_timestamp(metrics.tts_text_sent_ts)}] TTS #{metrics.chunk_num}: \"{text_preview}\"")
    if metrics.tts_first_audio_ts:
        print(f"  └─ First audio (TTS TTFB: {metrics.tts_ttfb_ms:.0f}ms)")
    print(f"  └─ Complete: {metrics.tts_gen_time_ms:.0f}ms, {metrics.audio_duration_ms:.0f}ms audio")
    print(f"  ★ Combined TTFB: {metrics.combined_ttfb_ms:.0f}ms")


def print_summary(metrics: list[ChunkMetrics]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tokens = sum(m.token_count for m in metrics)
    total_audio_bytes = sum(m.audio_bytes for m in metrics)
    total_audio_ms = total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000

    # Wall time: from first inference start to last TTS segment done
    wall_time_ms = (metrics[-1].tts_segment_done_ts - metrics[0].inference_start_ts) * 1000

    print(f"  Total chunks:          {len(metrics)}")
    print(f"  Total LLM tokens:      {total_tokens}")
    print(f"  Total audio:           {total_audio_ms/1000:.1f} seconds")
    print(f"  Total wall time:       {wall_time_ms/1000:.1f} seconds")

    # First chunk metrics
    first = metrics[0]
    print(f"\n  First chunk:")
    print(f"    LLM TTFT:            {first.llm_ttft_ms:.0f}ms")
    print(f"    LLM gen time:        {first.llm_gen_time_ms:.0f}ms")
    print(f"    TTS TTFB:            {first.tts_ttfb_ms:.0f}ms")
    print(f"    TTS gen time:        {first.tts_gen_time_ms:.0f}ms")
    print(f"    Combined TTFB:       {first.combined_ttfb_ms:.0f}ms")
    print(f"    Audio duration:      {first.audio_duration_ms:.0f}ms")

    # Averages
    valid_metrics = [m for m in metrics if m.tts_first_audio_ts is not None]
    if valid_metrics:
        avg_llm_ttft = sum(m.llm_ttft_ms for m in valid_metrics) / len(valid_metrics)
        avg_llm_gen = sum(m.llm_gen_time_ms for m in valid_metrics) / len(valid_metrics)
        avg_tts_ttfb = sum(m.tts_ttfb_ms for m in valid_metrics) / len(valid_metrics)
        avg_tts_gen = sum(m.tts_gen_time_ms for m in valid_metrics) / len(valid_metrics)
        avg_combined = sum(m.combined_ttfb_ms for m in valid_metrics) / len(valid_metrics)

        print(f"\n  Averages (all chunks):")
        print(f"    Avg LLM TTFT:        {avg_llm_ttft:.0f}ms")
        print(f"    Avg LLM gen:         {avg_llm_gen:.0f}ms")
        print(f"    Avg TTS TTFB:        {avg_tts_ttfb:.0f}ms")
        print(f"    Avg TTS gen:         {avg_tts_gen:.0f}ms")
        print(f"    Avg Combined TTFB:   {avg_combined:.0f}ms")

    # Chunk breakdown table
    print("\n" + "-" * 110)
    print(f"{'#':<3} {'Tokens':<7} {'LLM TTFT':<10} {'LLM Gen':<10} {'TTS TTFB':<10} {'TTS Gen':<10} {'Audio':<10} {'Combined':<10} {'Text'}")
    print("-" * 110)

    for m in metrics:
        text_preview = m.text[:25] + "..." if len(m.text) > 25 else m.text
        text_preview = text_preview.replace("\n", "\\n")
        print(
            f"{m.chunk_num:<3} {m.token_count:<7} {m.llm_ttft_ms:<10.0f} {m.llm_gen_time_ms:<10.0f} "
            f"{m.tts_ttfb_ms:<10.0f} {m.tts_gen_time_ms:<10.0f} {m.audio_duration_ms:<10.0f} "
            f"{m.combined_ttfb_ms:<10.0f} {text_preview}"
        )


def save_audio(audio_bytes: bytes, filename: str):
    """Save raw PCM audio to WAV file."""
    with wave.open(filename, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)
    print(f"\nSaved audio to {filename}")


async def run_integration_test(verbose: bool = False, save_audio_path: Optional[str] = None):
    """Run the LLM + TTS integration test."""

    # Same messages as timeline_test.py for comparability
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant running on an NVIDIA DGX Spark. "
                "You are built with Nemotron 3 Nano, a large language model developed by NVIDIA. "
                "Your goal is to have a natural conversation with the user. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Avoid special characters. Use only simple, plain text sentences."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hello I am a helpful assistant. \nI run on NVIDIA DGX Spark. \nMy name is Nemotron 3 Nano."
            ),
        },
        {"role": "user", "content": "Tell me three jokes about unicorns."},
    ]

    stream_id = f"integration-{uuid.uuid4().hex[:8]}"

    print("=" * 80)
    print("LLM + TTS INTEGRATION TEST")
    print("=" * 80)
    print(f"\nSystem: {messages[0]['content'][:60]}...")
    print(f"User: {messages[-1]['content']}")
    print(f"Strategy: max_tokens=10 (first), sentence_boundary (subsequent)")
    print(f"\nLLM: {LLM_URI}")
    print(f"TTS: {TTS_URI}")
    print("\n" + "-" * 80)
    print("TIMELINE")
    print("-" * 80)

    all_metrics: list[ChunkMetrics] = []
    all_audio: list[bytes] = []

    async with await ws_connect(LLM_URI) as llm_ws:
        async with await ws_connect(TTS_URI) as tts_ws:
            # Initialize TTS stream (once per LLM response)
            await tts_ws.send(json.dumps({
                "type": "init",
                "voice": "aria",
                "language": "en"
            }))
            init_response = await tts_ws.recv()
            tts_stream_id = json.loads(init_response).get("stream_id", "")[:8]
            print(f"\nTTS stream created: {tts_stream_id}")

            # Start LLM stream with first chunk strategy
            chunk_num = 1
            inference_start = time.time()

            await llm_ws.send(json.dumps({
                "action": "start_stream",
                "stream_id": stream_id,
                "messages": messages,
                "pause": {"max_tokens": 10},
                "stream_tokens": True
            }))

            while True:
                # Receive LLM chunk
                text, token_count, llm_first_token_ts, llm_done_ts, is_llm_done = \
                    await receive_llm_chunk(llm_ws, chunk_num, inference_start)

                # Send text to TTS (don't close yet - more chunks may come)
                tts_text_sent_ts = time.time()
                await tts_ws.send(json.dumps({"type": "text", "text": text}))

                # If LLM is done, close TTS stream to flush any remaining buffer
                if is_llm_done:
                    await tts_ws.send(json.dumps({"type": "close"}))

                # Wait for TTS audio segment
                audio_bytes, tts_first_audio_ts, tts_done_ts, is_tts_stream_done = \
                    await wait_for_tts_segment(tts_ws)

                # Record metrics
                metrics = ChunkMetrics(
                    chunk_num=chunk_num,
                    text=text,
                    token_count=token_count,
                    audio_bytes=len(audio_bytes),
                    inference_start_ts=inference_start,
                    llm_first_token_ts=llm_first_token_ts,
                    llm_done_ts=llm_done_ts,
                    tts_text_sent_ts=tts_text_sent_ts,
                    tts_first_audio_ts=tts_first_audio_ts,
                    tts_segment_done_ts=tts_done_ts,
                )
                all_metrics.append(metrics)
                all_audio.append(audio_bytes)

                # Log progress
                print_chunk_progress(metrics, verbose)

                if is_llm_done:
                    break

                # Continue LLM to next chunk with sentence boundary strategy
                chunk_num += 1
                inference_start = time.time()

                await llm_ws.send(json.dumps({
                    "action": "continue_stream",
                    "stream_id": stream_id,
                    "pause": {"sentence_boundary": True}
                }))

            # End LLM stream
            await llm_ws.send(json.dumps({
                "action": "end_stream",
                "stream_id": stream_id
            }))
            await llm_ws.recv()

    # Print summary
    print_summary(all_metrics)

    # Save audio if requested
    if save_audio_path:
        save_audio(b"".join(all_audio), save_audio_path)

    # Print full generated text
    print("\n" + "-" * 80)
    print("FULL OUTPUT:")
    print("-" * 80)
    full_text = "".join(m.text for m in all_metrics)
    print(full_text)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for Chunked LLM + WebSocket TTS"
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        metavar="FILE",
        help="Save combined audio to WAV file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with all timestamps",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_integration_test(
            verbose=args.verbose,
            save_audio_path=args.save_audio,
        ))
    except ConnectionRefusedError as e:
        print(f"\nERROR: Could not connect to server: {e}")
        print("\nMake sure both servers are running:")
        print("  - LLM server: uv run python -m nemotron_speech.chunked_llm_server")
        print("  - TTS server: docker exec nemotron-asr python -m nemotron_speech.tts_server")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test WebSocket client for streaming ASR server."""

import asyncio
import json
import sys
import time
import wave

import websockets


async def test_asr_streaming(
    audio_path: str,
    server_url: str = "ws://localhost:8080",
    chunk_ms: int = 500,
):
    """Send audio file to streaming ASR server and show interim results."""

    print(f"Reading audio file: {audio_path}")

    # Read WAV file
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    duration = n_frames / sample_rate
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Size: {len(audio_data)} bytes")

    # Calculate chunk size in bytes (16kHz, 16-bit = 2 bytes/sample)
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = chunk_samples * sample_width  # sample_width = 2 for 16-bit

    print(f"\nChunk size: {chunk_ms}ms = {chunk_samples} samples = {chunk_bytes} bytes")
    print(f"Connecting to {server_url}...")

    start_time = time.time()

    async with websockets.connect(server_url) as ws:
        connect_time = time.time()
        print(f"  Connected in {(connect_time - start_time)*1000:.0f}ms")

        # Wait for ready message
        ready_msg = await ws.recv()
        ready_data = json.loads(ready_msg)
        if ready_data.get("type") != "ready":
            print(f"  WARNING: Expected 'ready', got: {ready_data}")
        else:
            print(f"  Server ready")

        ready_time = time.time()

        # Track interim results
        interim_count = 0
        last_interim = ""

        # Create a task to receive messages
        async def receive_messages():
            nonlocal interim_count, last_interim
            try:
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "transcript":
                        text = data.get("text", "")
                        is_final = data.get("is_final", False)

                        if is_final:
                            return text
                        else:
                            interim_count += 1
                            last_interim = text
                            # Show interim result (truncated)
                            display = text[:60] + "..." if len(text) > 60 else text
                            print(f"  [interim {interim_count}] {display}")
                    elif data.get("type") == "error":
                        print(f"  ERROR: {data.get('message')}")
                        return None
            except websockets.exceptions.ConnectionClosed:
                return last_interim

        # Start receiving in background
        receive_task = asyncio.create_task(receive_messages())

        # Send audio data in chunks
        total_sent = 0
        chunks_sent = 0

        print(f"\nSending audio in {chunk_ms}ms chunks...")
        send_start = time.time()

        for i in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[i:i+chunk_bytes]
            await ws.send(chunk)
            total_sent += len(chunk)
            chunks_sent += 1

            # Simulate real-time streaming
            await asyncio.sleep(chunk_ms / 1000)

        send_time = time.time()
        print(f"  Sent {chunks_sent} chunks ({total_sent} bytes) in {(send_time - send_start)*1000:.0f}ms")

        # Record time of last audio chunk sent
        last_audio_time = send_time

        # Signal end of audio
        end_signal_time = time.time()
        await ws.send(json.dumps({"type": "reset"}))

        # Wait for final transcript
        print("\nWaiting for final transcript...")
        transcript = await receive_task
        final_recv_time = time.time()

        # Calculate time-to-final-transcription
        time_to_final = (final_recv_time - last_audio_time) * 1000
        end_signal_to_final = (final_recv_time - end_signal_time) * 1000

        print(f"\n{'='*60}")
        print("FINAL TRANSCRIPT:")
        print(f"{'='*60}")
        print(transcript if transcript else "(empty)")
        print(f"{'='*60}")

        total_time = final_recv_time - start_time
        print(f"\nStatistics:")
        print(f"  Interim results: {interim_count}")
        print(f"  Total time: {total_time*1000:.0f}ms")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  Real-time factor: {total_time/duration:.2f}x")
        print(f"\nFinalization latency:")
        print(f"  Last audio chunk -> final transcript: {time_to_final:.0f}ms")
        print(f"  End signal -> final transcript: {end_signal_to_final:.0f}ms")

        return transcript


async def test_multiple_chunk_sizes(audio_path: str, server_url: str):
    """Test with different chunk sizes."""
    print("=" * 60)
    print("Testing Multiple Chunk Sizes")
    print("=" * 60)

    for chunk_ms in [500, 160, 80]:
        print(f"\n{'='*60}")
        print(f"CHUNK SIZE: {chunk_ms}ms")
        print(f"{'='*60}")

        try:
            await test_asr_streaming(audio_path, server_url, chunk_ms)
        except Exception as e:
            print(f"ERROR with {chunk_ms}ms chunks: {e}")

        # Small delay between tests
        await asyncio.sleep(1)


if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/harvard_16k.wav"
    server_url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8080"

    # Check for --all flag to test all chunk sizes
    if "--all" in sys.argv:
        asyncio.run(test_multiple_chunk_sizes(audio_path, server_url))
    else:
        chunk_ms = 500
        if "--chunk" in sys.argv:
            idx = sys.argv.index("--chunk")
            if idx + 1 < len(sys.argv):
                chunk_ms = int(sys.argv[idx + 1])

        asyncio.run(test_asr_streaming(audio_path, server_url, chunk_ms))

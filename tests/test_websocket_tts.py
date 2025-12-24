#!/usr/bin/env python3
"""Test client for WebSocket TTS endpoint with realistic LLM token streaming."""

import asyncio
import json
import re
import time
import wave

try:
    from websockets.asyncio.client import connect as ws_connect
except ImportError:
    from websockets import connect as ws_connect


def tokenize_text(text: str) -> list[str]:
    """Split text on punctuation and spaces, keeping delimiters attached."""
    # Split on punctuation followed by space, or just spaces
    # Keep punctuation attached to preceding word
    tokens = []
    current = ""

    for char in text:
        if char in " ":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)  # Keep the space as a separate token
        elif char in ".,!?;:—-":
            current += char
            tokens.append(current)
            current = ""
        else:
            current += char

    if current:
        tokens.append(current)

    # Merge spaces with following word for more realistic tokens
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i] == " " and i + 1 < len(tokens):
            merged.append(" " + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return merged


async def test_websocket_tts():
    """Test the WebSocket TTS endpoint with realistic LLM streaming."""
    ws_url = "ws://localhost:8001/ws/tts/stream"

    # Realistic LLM response text
    full_text = (
        "Hello! I'm Nemotron Nano, a large language model developed by NVIDIA. "
        "I'm designed to assist with a wide range of tasks — whether you have questions, "
        "need help writing or understanding something, or just want to explore ideas. "
        "I'm here to make our conversation smooth, helpful, and engaging. "
        "How can I assist you today?"
    )

    tokens = tokenize_text(full_text)
    tokens_per_second = 20
    token_delay = 1.0 / tokens_per_second

    print(f"Text: {full_text[:60]}...")
    print(f"Tokens: {len(tokens)} @ {tokens_per_second}/sec = {len(tokens)/tokens_per_second:.1f}s")
    print(f"Token delay: {token_delay*1000:.0f}ms")
    print(f"\nConnecting to {ws_url}...")

    audio_chunks = []
    chunk_times = []
    start_time = None
    first_audio_time = None
    tokens_sent = 0

    async with await ws_connect(ws_url) as ws:
        print("Connected!")

        # Send init message
        await ws.send(json.dumps({
            "type": "init",
            "voice": "aria",
            "language": "en"
        }))

        # Wait for stream_created
        response = await ws.recv()
        data = json.loads(response)
        print(f"Stream created: {data.get('stream_id', '')[:8]}")

        # Start timing
        start_time = time.time()

        # Background task to receive audio
        async def receive_audio():
            nonlocal first_audio_time
            try:
                async for message in ws:
                    now = time.time()
                    if isinstance(message, bytes):
                        if first_audio_time is None:
                            first_audio_time = now
                            ttfb = (first_audio_time - start_time) * 1000
                            print(f"\n>>> First audio! TTFB: {ttfb:.0f}ms, {len(message)} bytes")
                        audio_chunks.append(message)
                        chunk_times.append(now - start_time)
                    else:
                        data = json.loads(message)
                        if data.get("type") == "done":
                            print(f"\n>>> Stream done: {data.get('total_audio_ms', 0):.0f}ms audio")
                            break
                        elif data.get("type") == "metadata":
                            mode = data.get("mode", "?")
                            buffer = data.get("buffer_ms", 0)
                            print(f"\n>>> Metadata: mode={mode}, buffer={buffer:.0f}ms")
            except Exception as e:
                print(f"\nReceive error: {e}")

        # Start receiver task
        receiver = asyncio.create_task(receive_audio())

        # Send tokens at LLM rate
        print("\nSending tokens:", end=" ", flush=True)
        for i, token in enumerate(tokens):
            await ws.send(json.dumps({"type": "text", "text": token}))
            tokens_sent += 1
            # Print progress every 10 tokens
            if (i + 1) % 10 == 0:
                elapsed = (time.time() - start_time) * 1000
                print(f"[{i+1}/{len(tokens)} @ {elapsed:.0f}ms]", end=" ", flush=True)
            await asyncio.sleep(token_delay)

        # Send close to signal end of text
        close_time = time.time() - start_time
        await ws.send(json.dumps({"type": "close"}))
        print(f"\n\nClose sent at {close_time*1000:.0f}ms")

        # Wait for receiver to complete
        await receiver

    # Calculate stats
    total_audio = b"".join(audio_chunks)
    duration_ms = len(total_audio) / (22000 * 2) * 1000
    total_time = (time.time() - start_time) * 1000
    ttfb = (first_audio_time - start_time) * 1000 if first_audio_time else 0

    print(f"\n{'='*60}")
    print(f"TIMING ANALYSIS")
    print(f"{'='*60}")
    print(f"Tokens sent:     {tokens_sent} @ {tokens_per_second}/sec")
    print(f"Token stream:    {len(tokens)/tokens_per_second*1000:.0f}ms")
    print(f"TTFB:            {ttfb:.0f}ms")
    print(f"Total time:      {total_time:.0f}ms")
    print(f"Audio duration:  {duration_ms:.0f}ms")
    print(f"Audio chunks:    {len(audio_chunks)}")

    if chunk_times:
        print(f"\nChunk arrival times (from start):")
        for i, t in enumerate(chunk_times[:10]):
            chunk_size = len(audio_chunks[i])
            chunk_dur = chunk_size / (22000 * 2) * 1000
            print(f"  Chunk {i+1}: {t*1000:.0f}ms ({chunk_size} bytes = {chunk_dur:.0f}ms audio)")
        if len(chunk_times) > 10:
            print(f"  ... and {len(chunk_times) - 10} more chunks")

    # Save to WAV file
    output_file = "test_websocket_output.wav"
    with wave.open(output_file, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(22000)
        wav.writeframes(total_audio)
    print(f"\nSaved audio to {output_file}")


if __name__ == "__main__":
    asyncio.run(test_websocket_tts())

#!/usr/bin/env python3
"""
Example client for streaming ASR inference via WebSocket.

Usage:
    python examples/asr_client.py audio.wav
    python examples/asr_client.py audio.wav --url ws://localhost:8080
    python examples/asr_client.py audio.wav --realtime  # Simulate real-time streaming

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import json
import sys
import wave


async def transcribe_audio(
    audio_path: str,
    server_url: str = "ws://localhost:8080",
    realtime: bool = False,
    chunk_ms: int = 100,
):
    """Send audio file to ASR server and print transcription."""
    import websockets

    # Read WAV file
    print(f"Reading: {audio_path}")
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    duration = n_frames / sample_rate
    print(f"  Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")

    if sample_rate != 16000:
        print(f"  WARNING: Expected 16kHz, got {sample_rate}Hz")

    # Calculate chunk size
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = chunk_samples * sample_width

    print(f"Connecting to {server_url}...")

    async with websockets.connect(server_url) as ws:
        # Wait for ready
        ready_msg = await ws.recv()
        ready_data = json.loads(ready_msg)
        if ready_data.get("type") != "ready":
            print(f"Unexpected message: {ready_data}")
            return None

        print("Server ready, streaming audio...")

        final_transcript = None

        async def receive_messages():
            nonlocal final_transcript
            async for message in ws:
                data = json.loads(message)
                if data.get("type") == "transcript":
                    text = data.get("text", "")
                    is_final = data.get("is_final", False)

                    if is_final:
                        final_transcript = text
                        return
                    else:
                        # Show interim result (overwrite line)
                        display = text[:70] + "..." if len(text) > 70 else text
                        print(f"\r  [{display}]", end="", flush=True)
                elif data.get("type") == "error":
                    print(f"\nError: {data.get('message')}")
                    return

        # Start receiver task
        receive_task = asyncio.create_task(receive_messages())

        # Send audio in chunks
        for i in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[i : i + chunk_bytes]
            await ws.send(chunk)

            if realtime:
                await asyncio.sleep(chunk_ms / 1000)

        # Signal end of audio
        await ws.send(json.dumps({"type": "end"}))

        # Wait for final transcript
        await receive_task
        print()  # newline after interim results

    return final_transcript


def main():
    parser = argparse.ArgumentParser(description="Streaming ASR Client")
    parser.add_argument("audio", help="Path to WAV audio file (16kHz, 16-bit, mono)")
    parser.add_argument(
        "--url",
        default="ws://localhost:8080",
        help="WebSocket server URL (default: ws://localhost:8080)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Simulate real-time streaming (add delays between chunks)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Chunk size in milliseconds (default: 100)",
    )
    args = parser.parse_args()

    try:
        transcript = asyncio.run(
            transcribe_audio(
                audio_path=args.audio,
                server_url=args.url,
                realtime=args.realtime,
                chunk_ms=args.chunk_ms,
            )
        )

        if transcript:
            print("\nTranscription:")
            print("-" * 60)
            print(transcript)
            print("-" * 60)
        else:
            print("No transcription received")
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

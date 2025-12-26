#!/usr/bin/env python3
"""
Test TTS WebSocket cancel/interruption handling.

Tests that:
1. Cancel message immediately stops audio generation
2. Stale audio from cancelled stream is not received after new stream starts
3. Generation counter synchronization works correctly

Usage:
    uv run python tests/test_tts_cancel.py
"""

import asyncio
import json
import time

try:
    from websockets.asyncio.client import connect as ws_connect
except ImportError:
    from websockets import connect as ws_connect


TTS_URI = "ws://localhost:8001/ws/tts/stream"
SAMPLE_RATE = 22000
BYTES_PER_SAMPLE = 2


async def test_basic_tts():
    """Test basic TTS flow works."""
    print("\n=== Test 1: Basic TTS Flow ===")

    async with await ws_connect(TTS_URI) as ws:
        # Initialize stream
        await ws.send(json.dumps({
            "type": "init",
            "voice": "aria",
            "language": "en"
        }))
        response = json.loads(await ws.recv())
        assert response.get("type") == "stream_created", f"Expected stream_created, got {response}"
        stream_id = response.get("stream_id", "")[:8]
        print(f"  Stream created: {stream_id}")

        # Send text
        text = "Hello, this is a test."
        await ws.send(json.dumps({
            "type": "text",
            "text": text,
            "mode": "stream",
            "preset": "conservative"
        }))

        # Collect audio
        audio_bytes = 0
        first_audio_ts = None
        text_sent_ts = time.time()

        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            if isinstance(msg, bytes):
                if first_audio_ts is None:
                    first_audio_ts = time.time()
                audio_bytes += len(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "segment_complete":
                    break

        ttfb_ms = (first_audio_ts - text_sent_ts) * 1000 if first_audio_ts else 0
        audio_duration_ms = audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000

        print(f"  TTFB: {ttfb_ms:.0f}ms")
        print(f"  Audio: {audio_duration_ms:.0f}ms ({audio_bytes} bytes)")

        # Close stream
        await ws.send(json.dumps({"type": "close"}))

        # Wait for done
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "done":
                    break

        print("  PASSED: Basic TTS flow works")
        return True


async def test_cancel_stops_generation():
    """Test that cancel message stops audio generation immediately."""
    print("\n=== Test 2: Cancel Stops Generation ===")

    async with await ws_connect(TTS_URI) as ws:
        # Initialize stream
        await ws.send(json.dumps({
            "type": "init",
            "voice": "aria",
            "language": "en"
        }))
        response = json.loads(await ws.recv())
        stream_id = response.get("stream_id", "")[:8]
        print(f"  Stream created: {stream_id}")

        # Send long text to generate
        long_text = "This is a very long sentence that should take a while to generate because it has many words and will produce a significant amount of audio output."
        await ws.send(json.dumps({
            "type": "text",
            "text": long_text,
            "mode": "stream",
            "preset": "conservative"
        }))

        # Wait for some audio to arrive
        audio_before_cancel = 0
        start_ts = time.time()

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                if isinstance(msg, bytes):
                    audio_before_cancel += len(msg)
                    # Once we have some audio, send cancel
                    if audio_before_cancel > 5000:  # ~100ms of audio
                        break
            except asyncio.TimeoutError:
                if time.time() - start_ts > 2.0:
                    print("  WARNING: Timeout waiting for initial audio")
                    break

        print(f"  Audio before cancel: {audio_before_cancel} bytes")

        # Send cancel
        cancel_ts = time.time()
        await ws.send(json.dumps({"type": "cancel"}))
        print(f"  Cancel sent at {cancel_ts - start_ts:.3f}s")

        # Count any audio that arrives after cancel
        audio_after_cancel = 0

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                if isinstance(msg, bytes):
                    audio_after_cancel += len(msg)
            except asyncio.TimeoutError:
                # No more messages
                break

        print(f"  Audio after cancel: {audio_after_cancel} bytes")

        # We expect minimal audio after cancel (maybe a few bytes in flight)
        # but not the full remaining audio
        if audio_after_cancel < 10000:  # Less than ~200ms of leaked audio
            print("  PASSED: Cancel stopped generation")
            return True
        else:
            print(f"  FAILED: Too much audio leaked after cancel ({audio_after_cancel} bytes)")
            return False


async def test_no_stale_audio_after_cancel():
    """Test that stale audio doesn't appear after cancel and new stream."""
    print("\n=== Test 3: No Stale Audio After Cancel ===")

    async with await ws_connect(TTS_URI) as ws:
        # Initialize first stream
        await ws.send(json.dumps({
            "type": "init",
            "voice": "aria",
            "language": "en"
        }))
        response = json.loads(await ws.recv())
        stream1_id = response.get("stream_id", "")[:8]
        print(f"  Stream 1 created: {stream1_id}")

        # Send long text
        await ws.send(json.dumps({
            "type": "text",
            "text": "First stream with a long sentence that should be cancelled before completion to test interruption handling.",
            "mode": "stream",
            "preset": "conservative"
        }))

        # Wait for some audio
        audio_stream1 = 0
        start_ts = time.time()

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                if isinstance(msg, bytes):
                    audio_stream1 += len(msg)
                    if audio_stream1 > 5000:
                        break
            except asyncio.TimeoutError:
                if time.time() - start_ts > 2.0:
                    break

        print(f"  Stream 1 audio before cancel: {audio_stream1} bytes")

        # Cancel and immediately start new stream
        await ws.send(json.dumps({"type": "cancel"}))
        await asyncio.sleep(0.05)  # Brief delay

        # Initialize new stream
        await ws.send(json.dumps({
            "type": "init",
            "voice": "aria",
            "language": "en"
        }))

        # Wait for stream_created
        stream2_created = False
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if isinstance(msg, bytes):
                    # Stale audio from stream 1 - this is what we're testing against
                    print(f"  WARNING: Received {len(msg)} bytes audio before stream_created")
                else:
                    data = json.loads(msg)
                    if data.get("type") == "stream_created":
                        stream2_id = data.get("stream_id", "")[:8]
                        print(f"  Stream 2 created: {stream2_id}")
                        stream2_created = True
                        break
            except asyncio.TimeoutError:
                print("  FAILED: Timeout waiting for stream_created")
                return False

        if not stream2_created:
            print("  FAILED: Stream 2 not created")
            return False

        # Send new text
        new_text = "New stream test."
        await ws.send(json.dumps({
            "type": "text",
            "text": new_text,
            "mode": "stream",
            "preset": "conservative"
        }))

        # Collect audio and check for stale content
        audio_stream2 = 0
        first_audio_ts = None
        text_sent_ts = time.time()

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, bytes):
                    if first_audio_ts is None:
                        first_audio_ts = time.time()
                    audio_stream2 += len(msg)
                else:
                    data = json.loads(msg)
                    if data.get("type") == "segment_complete":
                        break
            except asyncio.TimeoutError:
                print("  FAILED: Timeout waiting for segment_complete")
                return False

        ttfb_ms = (first_audio_ts - text_sent_ts) * 1000 if first_audio_ts else 0
        audio_duration_ms = audio_stream2 / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000

        print(f"  Stream 2 TTFB: {ttfb_ms:.0f}ms")
        print(f"  Stream 2 audio: {audio_duration_ms:.0f}ms ({audio_stream2} bytes)")

        # The new stream audio should be appropriate for "New stream test."
        # Not the long cancelled sentence. Expected: ~500-800ms of audio.
        expected_min = 300  # ms
        expected_max = 1500  # ms (with some margin)

        if expected_min <= audio_duration_ms <= expected_max:
            print("  PASSED: No stale audio bleeding through")
            return True
        else:
            print(f"  FAILED: Unexpected audio duration {audio_duration_ms:.0f}ms (expected {expected_min}-{expected_max}ms)")
            return False


async def main():
    print("=" * 60)
    print("TTS Cancel/Interruption Test Suite")
    print("=" * 60)

    results = []

    try:
        results.append(("Basic TTS", await test_basic_tts()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        results.append(("Basic TTS", False))

    try:
        results.append(("Cancel stops generation", await test_cancel_stops_generation()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        results.append(("Cancel stops generation", False))

    try:
        results.append(("No stale audio", await test_no_stale_audio_after_cancel()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        results.append(("No stale audio", False))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

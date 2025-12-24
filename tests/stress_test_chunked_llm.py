#!/usr/bin/env python3
"""
Stress test for Chunked LLM Inference Server.

Tests that the 100ms cooldown prevents the llama.cpp slot crash bug.

Usage:
    # Start llama-server on port 8000
    # Start chunked server on port 8002
    uv run python tests/stress_test_chunked_llm.py
"""

import asyncio
import json
import time
import uuid
import websockets


async def test_rapid_continuation(num_iterations: int = 20):
    """
    Test rapid continuation requests to stress the slot handling.

    This should NOT crash if the 100ms cooldown is working.
    """
    print("=" * 70)
    print(f"STRESS TEST: {num_iterations} rapid continuations")
    print("=" * 70)

    uri = "ws://localhost:8002/ws"
    stream_id = f"stress-{uuid.uuid4().hex[:8]}"

    async with websockets.connect(uri) as ws:
        # Start stream
        print(f"\n1. Starting stream {stream_id}...")
        start_time = time.time()

        await ws.send(json.dumps({
            "action": "start_stream",
            "stream_id": stream_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 100, one number per line."}
            ],
            "pause": {"max_tokens": 5}
        }))

        resp = json.loads(await ws.recv())
        if "error" in resp:
            print(f"   ERROR: {resp['error']}")
            return False

        ttft = (time.time() - start_time) * 1000
        print(f"   Started: {resp.get('tokens')} tokens in {ttft:.0f}ms")
        print(f"   Text: {resp.get('text')!r}")

        # Rapid continuations with NO client-side delay
        # Server should enforce 100ms cooldown
        print(f"\n2. Sending {num_iterations} rapid continuation requests...")
        print("   (Server enforces 100ms cooldown between requests)")

        success_count = 0
        total_tokens = resp.get('tokens', 0)
        times = []

        for i in range(num_iterations):
            iter_start = time.time()

            await ws.send(json.dumps({
                "action": "continue_stream",
                "stream_id": stream_id,
                "pause": {"max_tokens": 5}
            }))

            try:
                resp = json.loads(await ws.recv())
                iter_time = (time.time() - iter_start) * 1000
                times.append(iter_time)

                if "error" in resp:
                    print(f"   [{i+1}] ERROR: {resp['error']}")
                    break

                tokens = resp.get('tokens', 0)
                total_tokens += tokens
                success_count += 1

                if resp.get('done'):
                    print(f"   [{i+1}] DONE (EOS reached)")
                    break

                # Progress indicator
                if (i + 1) % 5 == 0:
                    avg_time = sum(times[-5:]) / len(times[-5:])
                    print(f"   [{i+1}/{num_iterations}] {tokens} tokens, {iter_time:.0f}ms (avg: {avg_time:.0f}ms)")

            except Exception as e:
                print(f"   [{i+1}] EXCEPTION: {e}")
                break

        # End stream
        await ws.send(json.dumps({
            "action": "end_stream",
            "stream_id": stream_id
        }))
        await ws.recv()

        # Summary
        total_time = time.time() - start_time
        avg_time = sum(times) / len(times) if times else 0

        print(f"\n3. Summary:")
        print(f"   Successful iterations: {success_count}/{num_iterations}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Avg time per continuation: {avg_time:.0f}ms")
        print(f"   Min/Max time: {min(times):.0f}ms / {max(times):.0f}ms")

        return success_count == num_iterations or resp.get('done')


async def test_sentence_boundary_stress(num_sentences: int = 10):
    """
    Test sentence boundary detection with rapid requests.
    """
    print("\n" + "=" * 70)
    print(f"STRESS TEST: {num_sentences} sentence boundary requests")
    print("=" * 70)

    uri = "ws://localhost:8002/ws"
    stream_id = f"sentence-{uuid.uuid4().hex[:8]}"

    async with websockets.connect(uri) as ws:
        # Start stream
        print(f"\n1. Starting stream...")

        await ws.send(json.dumps({
            "action": "start_stream",
            "stream_id": stream_id,
            "messages": [
                {"role": "user", "content": "Tell me 5 interesting facts about space. Number each fact."}
            ],
            "pause": {"sentence_boundary": True}
        }))

        resp = json.loads(await ws.recv())
        if "error" in resp:
            print(f"   ERROR: {resp['error']}")
            return False

        print(f"   First chunk: {resp.get('text')!r}")

        # Continue until done or max sentences
        sentences = [resp.get('text', '')]

        for i in range(num_sentences):
            if resp.get('done'):
                break

            await ws.send(json.dumps({
                "action": "continue_stream",
                "stream_id": stream_id,
                "pause": {"sentence_boundary": True}
            }))

            resp = json.loads(await ws.recv())

            if "error" in resp:
                print(f"   ERROR at sentence {i+2}: {resp['error']}")
                break

            text = resp.get('text', '')
            if text:
                sentences.append(text)
                print(f"   Sentence {i+2}: {text[:60]}...")

        # End stream
        await ws.send(json.dumps({
            "action": "end_stream",
            "stream_id": stream_id
        }))
        await ws.recv()

        print(f"\n2. Got {len(sentences)} chunks")
        return len(sentences) > 1


async def test_concurrent_streams(num_streams: int = 3):
    """
    Test multiple concurrent streams (if slots available).
    """
    print("\n" + "=" * 70)
    print(f"STRESS TEST: {num_streams} concurrent streams")
    print("=" * 70)

    uri = "ws://localhost:8002/ws"

    async def run_stream(stream_num: int):
        stream_id = f"concurrent-{stream_num}-{uuid.uuid4().hex[:4]}"

        async with websockets.connect(uri) as ws:
            # Start
            await ws.send(json.dumps({
                "action": "start_stream",
                "stream_id": stream_id,
                "messages": [{"role": "user", "content": f"Count to 10. You are stream {stream_num}."}],
                "pause": {"max_tokens": 10}
            }))

            resp = json.loads(await ws.recv())
            if "error" in resp:
                return False, resp.get('error')

            # Few continuations
            for _ in range(3):
                if resp.get('done'):
                    break
                await ws.send(json.dumps({
                    "action": "continue_stream",
                    "stream_id": stream_id,
                    "pause": {"max_tokens": 10}
                }))
                resp = json.loads(await ws.recv())
                if "error" in resp:
                    return False, resp.get('error')

            # End
            await ws.send(json.dumps({
                "action": "end_stream",
                "stream_id": stream_id
            }))
            await ws.recv()

            return True, None

    print("\n1. Starting concurrent streams...")
    tasks = [run_stream(i) for i in range(num_streams)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n2. Results:")
    success = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"   Stream {i}: EXCEPTION - {result}")
        elif result[0]:
            print(f"   Stream {i}: OK")
            success += 1
        else:
            print(f"   Stream {i}: FAILED - {result[1]}")

    print(f"\n   {success}/{num_streams} streams succeeded")
    return success > 0


async def main():
    """Run all stress tests."""
    print("\n" + "=" * 70)
    print("CHUNKED LLM SERVER STRESS TEST")
    print("=" * 70)
    print("\nThis tests that the 100ms cooldown prevents llama.cpp slot crashes.")
    print("The server should enforce delays even if client sends rapid requests.\n")

    try:
        # Test 1: Rapid continuation
        result1 = await test_rapid_continuation(20)

        # Test 2: Sentence boundary stress
        result2 = await test_sentence_boundary_stress(10)

        # Test 3: Concurrent streams (will fail if only 1 slot, that's OK)
        result3 = await test_concurrent_streams(2)

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  Rapid continuation: {'PASS' if result1 else 'FAIL'}")
        print(f"  Sentence boundary:  {'PASS' if result2 else 'FAIL'}")
        print(f"  Concurrent streams: {'PASS' if result3 else 'FAIL (expected if 1 slot)'}")

        if result1 and result2:
            print("\n  100ms cooldown appears to prevent crashes!")
        else:
            print("\n  Some tests failed - check logs")

    except websockets.exceptions.ConnectionRefused:
        print("\nERROR: Could not connect to ws://localhost:8002")
        print("Make sure the chunked LLM server is running:")
        print("  uv run python -m nemotron_speech.chunked_llm_server")
    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())

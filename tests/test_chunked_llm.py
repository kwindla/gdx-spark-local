#!/usr/bin/env python3
"""
Test client for Chunked LLM Inference Server.

This tests the pause/resume functionality for voice agent GPU sharing.

Usage:
    # Make sure llama-server is running on port 8000
    # Start chunked server: uv run python -m nemotron_speech.chunked_llm_server
    # Run test: uv run python tests/test_chunked_llm.py
"""

import asyncio
import json
import time
import websockets


async def test_chunked_generation():
    """Test basic chunked generation with sentence boundaries."""

    print("=" * 70)
    print("TEST: Chunked LLM Generation with Sentence Boundaries")
    print("=" * 70)

    uri = "ws://localhost:8002/ws/generate"

    async with websockets.connect(uri) as ws:
        # Start session
        print("\n1. Starting session...")
        start_time = time.time()

        await ws.send(json.dumps({
            "action": "start",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me three interesting facts about the moon."}
            ],
            "chat_template_kwargs": {"enable_thinking": False}
        }))

        resp = json.loads(await ws.recv())
        ttft = (time.time() - start_time) * 1000

        print(f"   Status: {resp.get('status')}")
        print(f"   Session ID: {resp.get('session_id')}")
        print(f"   Slot ID: {resp.get('slot_id')}")
        print(f"   Prompt tokens: {resp.get('prompt_tokens')}")
        print(f"   TTFT: {ttft:.0f}ms")

        # Generate in chunks (sentence by sentence)
        sentences = []
        chunk_num = 0
        total_done = False

        while not total_done and chunk_num < 10:  # Safety limit
            chunk_num += 1
            print(f"\n2.{chunk_num}. Generating chunk (stop at sentence boundary)...")

            # Simulate TTS delay between chunks
            if chunk_num > 1:
                tts_delay = 0.5  # 500ms simulated TTS time
                print(f"   [Simulating {tts_delay*1000:.0f}ms TTS processing...]")
                await asyncio.sleep(tts_delay)

            chunk_start = time.time()

            await ws.send(json.dumps({
                "action": "generate",
                "stop_at": [".", "!", "?"],
                "max_tokens": 100
            }))

            resp = json.loads(await ws.recv())
            chunk_time = (time.time() - chunk_start) * 1000

            text = resp.get("text", "")
            tokens = resp.get("tokens", 0)
            done = resp.get("done", False)
            stop_reason = resp.get("stop_reason")

            if text.strip():
                sentences.append(text)
                tps = tokens / (chunk_time / 1000) if chunk_time > 0 else 0
                print(f"   Text: \"{text.strip()}\"")
                print(f"   Tokens: {tokens}, Time: {chunk_time:.0f}ms ({tps:.1f} tok/s)")
                print(f"   Stop reason: {stop_reason}")
                print(f"   --> TTS could synthesize this while LLM is paused!")

            total_done = done

        # End session
        print(f"\n3. Ending session...")
        await ws.send(json.dumps({"action": "end"}))
        resp = json.loads(await ws.recv())
        print(f"   Status: {resp.get('status')}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total sentences: {len(sentences)}")
        print(f"Full response:")
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s.strip()}")


async def test_fixed_token_chunks():
    """Test generation with fixed token counts."""

    print("\n" + "=" * 70)
    print("TEST: Fixed Token Chunk Generation")
    print("=" * 70)

    uri = "ws://localhost:8002/ws/generate"

    async with websockets.connect(uri) as ws:
        # Start session
        await ws.send(json.dumps({
            "action": "start",
            "messages": [
                {"role": "user", "content": "Count from 1 to 20."}
            ],
            "chat_template_kwargs": {"enable_thinking": False}
        }))

        resp = json.loads(await ws.recv())
        print(f"\nSession started: {resp.get('session_id')}")

        # Generate in fixed 10-token chunks
        all_text = ""
        chunk_times = []

        for i in range(5):
            chunk_start = time.time()

            await ws.send(json.dumps({
                "action": "generate",
                "n_tokens": 10
            }))

            resp = json.loads(await ws.recv())
            chunk_time = (time.time() - chunk_start) * 1000
            chunk_times.append(chunk_time)

            text = resp.get("text", "")
            all_text += text

            print(f"Chunk {i+1}: {repr(text[:40])}... ({chunk_time:.0f}ms)")

            if resp.get("done"):
                break

        # End session
        await ws.send(json.dumps({"action": "end"}))
        await ws.recv()

        print(f"\nAverage chunk time: {sum(chunk_times)/len(chunk_times):.0f}ms")
        print(f"Full text: {all_text[:200]}...")


async def test_cache_efficiency():
    """Test that KV cache is being reused between chunks."""

    print("\n" + "=" * 70)
    print("TEST: KV Cache Efficiency")
    print("=" * 70)

    uri = "ws://localhost:8002/ws/generate"

    async with websockets.connect(uri) as ws:
        # Start with a longer prompt to see cache benefit
        system_prompt = """You are Nemotron Nano, a large language model developed by NVIDIA.
You are helpful, harmless, and honest. When asked a question, provide a thoughtful response.
Keep your answers concise but informative."""

        user_prompt = "What are the benefits of renewable energy?"

        print(f"\n1. Starting session with {len(system_prompt) + len(user_prompt)} char prompt...")

        start_time = time.time()
        await ws.send(json.dumps({
            "action": "start",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "chat_template_kwargs": {"enable_thinking": False}
        }))

        resp = json.loads(await ws.recv())
        initial_time = (time.time() - start_time) * 1000
        prompt_tokens = resp.get("prompt_tokens", 0)

        print(f"   Initial prompt: {prompt_tokens} tokens in {initial_time:.0f}ms")

        # Generate multiple chunks and measure timing
        chunk_times = []
        print("\n2. Generating chunks (each should reuse KV cache):")

        for i in range(4):
            chunk_start = time.time()

            await ws.send(json.dumps({
                "action": "generate",
                "stop_at": [".", "!", "?"],
                "max_tokens": 50
            }))

            resp = json.loads(await ws.recv())
            chunk_time = (time.time() - chunk_start) * 1000
            chunk_times.append(chunk_time)

            tokens = resp.get("tokens", 0)
            tps = tokens / (chunk_time / 1000) if chunk_time > 0 else 0

            print(f"   Chunk {i+1}: {tokens} tokens in {chunk_time:.0f}ms ({tps:.1f} tok/s)")

            if resp.get("done"):
                break

        await ws.send(json.dumps({"action": "end"}))
        await ws.recv()

        print(f"\n   Average chunk time: {sum(chunk_times)/len(chunk_times):.0f}ms")
        print(f"   (Initial prompt: {initial_time:.0f}ms - includes KV cache population)")


async def main():
    """Run all tests."""
    try:
        await test_chunked_generation()
        await test_fixed_token_chunks()
        await test_cache_efficiency()

        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nConnection closed: {e}")
    except ConnectionRefusedError:
        print("\nError: Could not connect to chunked LLM server at ws://localhost:8002")
        print("Make sure to start the server first:")
        print("  uv run python -m nemotron_speech.chunked_llm_server")


if __name__ == "__main__":
    asyncio.run(main())

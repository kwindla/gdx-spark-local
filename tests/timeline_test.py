#!/usr/bin/env python3
"""
Timeline test for Chunked LLM Server with true token streaming.

Shows detailed timeline with:
- Server TTFT (reported by server)
- Client TTFT (when client receives first token)
- Chunk Time (total time for all tokens in chunk)
"""

import asyncio
import json
import time
import uuid
import websockets
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChunkResult:
    chunk_num: int
    text: str
    tokens: int
    start_time: float
    end_time: float
    done: bool
    stop_reason: str
    server_ttft_ms: float  # TTFT reported by server
    client_ttft_ms: float  # When client received first token
    chunk_time_ms: float   # Total time for chunk


def format_timestamp(t: float) -> str:
    """Format timestamp as HH:MM:SS.mmm"""
    dt = datetime.fromtimestamp(t)
    return dt.strftime("%H:%M:%S.") + f"{int((t % 1) * 1000):03d}"


async def receive_streaming_chunk(ws, chunk_num: int, chunk_start: float) -> ChunkResult:
    """
    Receive a streaming chunk (multiple token messages + paused/done message).

    Returns ChunkResult with timing info.
    """
    tokens = []
    client_ttft_ms = None
    server_ttft_ms = 0
    text = ""
    done = False
    stop_reason = ""

    while True:
        msg = json.loads(await ws.recv())

        if msg.get("type") == "token":
            # Token message
            tokens.append(msg["content"])
            if client_ttft_ms is None:
                client_ttft_ms = (time.time() - chunk_start) * 1000

        elif msg.get("type") in ("paused", "done"):
            # Final message for this chunk
            text = msg.get("text", "")
            server_ttft_ms = msg.get("ttft_ms", 0)
            done = msg.get("type") == "done"
            stop_reason = msg.get("reason", "")
            break

        elif "error" in msg:
            raise RuntimeError(f"Server error: {msg['error']}")

    chunk_end = time.time()
    chunk_time_ms = (chunk_end - chunk_start) * 1000

    return ChunkResult(
        chunk_num=chunk_num,
        text=text,
        tokens=len(tokens),
        start_time=chunk_start,
        end_time=chunk_end,
        done=done,
        stop_reason=stop_reason,
        server_ttft_ms=server_ttft_ms,
        client_ttft_ms=client_ttft_ms or 0,
        chunk_time_ms=chunk_time_ms
    )


async def run_timeline_test():
    """Run timeline test with true token streaming."""

    print("=" * 80)
    print("TIMELINE TEST: True Token Streaming with Sentence Chunking")
    print("=" * 80)

    uri = "ws://localhost:8002/ws"
    stream_id = f"timeline-{uuid.uuid4().hex[:8]}"
    client_pause_ms = 500

    # Multi-turn conversation
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
            "role": "user",
            "content": (
                "Introduce yourself. Tell the user that you are an assistant and what platform and model you are running on."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hello I am a helpful assistant. \nI run on NVIDIA DGX Spark. \nMy name is Nemotron 3 Nano."
            ),
        },
        {"role": "user", "content": "Tell me three jokes about unicorns."}
    ]

    print(f"\nSystem: {messages[0]['content'][:60]}...")
    print(f"User: {messages[-1]['content']}")
    print(f"Strategy: max_tokens=10 for first chunk, then sentence_boundary")
    print(f"Token streaming: ENABLED")
    print(f"Client pause: {client_pause_ms}ms between chunks")
    print()

    chunks: list[ChunkResult] = []
    test_start = time.time()

    async with websockets.connect(uri) as ws:
        # ========== START STREAM (with token streaming) ==========
        print("-" * 80)
        print("TIMELINE")
        print("-" * 80)

        chunk_start = time.time()
        print(f"[{format_timestamp(chunk_start)}] CLIENT: start_stream (stream_tokens=True, max_tokens=10)")

        await ws.send(json.dumps({
            "action": "start_stream",
            "stream_id": stream_id,
            "messages": messages,
            "pause": {"max_tokens": 10},  # First chunk: fixed token count
            "stream_tokens": True  # Enable true token streaming
        }))

        try:
            chunk = await receive_streaming_chunk(ws, 1, chunk_start)
        except RuntimeError as e:
            print(f"[{format_timestamp(time.time())}] ERROR: {e}")
            return

        chunks.append(chunk)
        tps = chunk.tokens / (chunk.chunk_time_ms / 1000) if chunk.chunk_time_ms > 0 else 0

        print(f"[{format_timestamp(chunk.end_time)}] SERVER: chunk 1 received")
        print(f"         └─ Srv TTFT: {chunk.server_ttft_ms:.0f}ms | Cli TTFT: {chunk.client_ttft_ms:.0f}ms | Chunk: {chunk.chunk_time_ms:.0f}ms")
        print(f"         └─ Tokens: {chunk.tokens} | TPS: {tps:.1f}")
        print(f"         └─ Text: {chunk.text!r}")

        # ========== CONTINUE WITH PAUSES ==========
        chunk_num = 1
        while not chunk.done:
            chunk_num += 1

            # Client pause
            pause_start = time.time()
            print(f"[{format_timestamp(pause_start)}] CLIENT: sleeping {client_pause_ms}ms...")
            await asyncio.sleep(client_pause_ms / 1000)

            # Continue
            chunk_start = time.time()
            print(f"[{format_timestamp(chunk_start)}] CLIENT: continue_stream (sentence_boundary)")

            await ws.send(json.dumps({
                "action": "continue_stream",
                "stream_id": stream_id,
                "pause": {"sentence_boundary": True}
            }))

            try:
                chunk = await receive_streaming_chunk(ws, chunk_num, chunk_start)
            except RuntimeError as e:
                print(f"[{format_timestamp(time.time())}] ERROR: {e}")
                break

            chunks.append(chunk)
            tps = chunk.tokens / (chunk.chunk_time_ms / 1000) if chunk.chunk_time_ms > 0 else 0

            status = "DONE (EOS)" if chunk.done else f"paused ({chunk.stop_reason})"
            print(f"[{format_timestamp(chunk.end_time)}] SERVER: chunk {chunk_num} received - {status}")
            print(f"         └─ Srv TTFT: {chunk.server_ttft_ms:.0f}ms | Cli TTFT: {chunk.client_ttft_ms:.0f}ms | Chunk: {chunk.chunk_time_ms:.0f}ms")
            print(f"         └─ Tokens: {chunk.tokens} | TPS: {tps:.1f}")
            print(f"         └─ Text: {chunk.text!r}")

            if chunk_num > 20:  # Safety limit
                print("         └─ (Safety limit reached)")
                break

        # ========== END STREAM ==========
        end_time = time.time()
        print(f"[{format_timestamp(end_time)}] CLIENT: end_stream")

        await ws.send(json.dumps({
            "action": "end_stream",
            "stream_id": stream_id
        }))
        await ws.recv()

        final_time = time.time()
        print(f"[{format_timestamp(final_time)}] SERVER: stream ended")

    # ========== SUMMARY ==========
    total_time = final_time - test_start
    total_tokens = sum(c.tokens for c in chunks)
    total_gen_time = sum(c.chunk_time_ms for c in chunks) / 1000
    total_pause_time = total_time - total_gen_time

    # Compute averages (excluding empty chunks)
    valid_chunks = [c for c in chunks if c.tokens > 0]
    avg_server_ttft = sum(c.server_ttft_ms for c in valid_chunks) / len(valid_chunks) if valid_chunks else 0
    avg_client_ttft = sum(c.client_ttft_ms for c in valid_chunks) / len(valid_chunks) if valid_chunks else 0
    avg_chunk_time = sum(c.chunk_time_ms for c in valid_chunks) / len(valid_chunks) if valid_chunks else 0

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total chunks:          {len(chunks)}")
    print(f"  Total tokens:          {total_tokens}")
    print(f"  Total time:            {total_time*1000:.0f}ms ({total_time:.2f}s)")
    print(f"  Generation time:       {total_gen_time*1000:.0f}ms")
    print(f"  Client pause time:     {total_pause_time*1000:.0f}ms")
    print()
    if valid_chunks:
        print(f"  First chunk:")
        print(f"    Server TTFT:         {valid_chunks[0].server_ttft_ms:.0f}ms")
        print(f"    Client TTFT:         {valid_chunks[0].client_ttft_ms:.0f}ms")
        print(f"    Chunk time:          {valid_chunks[0].chunk_time_ms:.0f}ms")
        print()
        print(f"  Avg per chunk:")
        print(f"    Avg Server TTFT:     {avg_server_ttft:.0f}ms")
        print(f"    Avg Client TTFT:     {avg_client_ttft:.0f}ms")
        print(f"    Avg Chunk time:      {avg_chunk_time:.0f}ms")
        print(f"    Avg TPS:             {total_tokens / total_gen_time:.1f} tok/s")
    print()
    print("  TTFT improvement: Client now receives first token when server does!")
    print(f"    Old Client TTFT:     = Chunk time (~{avg_chunk_time:.0f}ms)")
    print(f"    New Client TTFT:     ~{avg_client_ttft:.0f}ms ({avg_client_ttft/avg_chunk_time*100:.0f}% of chunk time)" if avg_chunk_time > 0 else "    New Client TTFT:     N/A")
    print()

    print("CHUNK BREAKDOWN:")
    print("-" * 110)
    print(f"{'#':<3} {'Tokens':<7} {'Srv TTFT':<10} {'Cli TTFT':<10} {'Chunk':<10} {'TPS':<8} {'Text (truncated)'}")
    print("-" * 110)
    for c in chunks:
        tps = c.tokens / (c.chunk_time_ms / 1000) if c.chunk_time_ms > 0 else 0
        text_preview = c.text[:35] + "..." if len(c.text) > 35 else c.text
        text_preview = text_preview.replace("\n", "\\n")
        print(f"{c.chunk_num:<3} {c.tokens:<7} {c.server_ttft_ms:<10.0f} {c.client_ttft_ms:<10.0f} {c.chunk_time_ms:<10.0f} {tps:<8.1f} {text_preview}")

    print()
    print("FULL OUTPUT:")
    print("-" * 80)
    full_text = "".join(c.text for c in chunks)
    print(full_text)


if __name__ == "__main__":
    asyncio.run(run_timeline_test())

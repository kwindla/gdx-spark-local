# Chunked LLM Server

WebSocket-based streaming LLM server designed for voice agents where LLM and TTS share GPU resources. Provides client-controlled pause/resume with true token streaming and intelligent sentence boundary detection.

## Overview

The Chunked LLM Server sits between your voice agent and llama.cpp, providing:

- **True token streaming**: Tokens sent to client as they're generated
- **Pause/resume control**: Client decides when to pause for TTS processing
- **Sentence boundary detection**: Natural pause points for spoken output
- **Mixed chunking strategy**: Fast first chunk + natural sentence breaks
- **Stop signal detection**: Prevents crashes from empty continuation requests

```
┌─────────────┐      WebSocket       ┌──────────────────┐       HTTP        ┌─────────────┐
│ Voice Agent │ ◄───────────────────►│  Chunked LLM     │◄─────────────────►│  llama.cpp  │
│  (Pipecat)  │   token streaming    │  Server (:8002)  │    streaming      │   (:8000)   │
└─────────────┘                      └──────────────────┘                   └─────────────┘
```

## Why Chunked Streaming?

In voice agents, we want to start TTS as soon as possible while the LLM continues generating. The chunked approach:

1. **First chunk (max_tokens)**: Get predictable audio started quickly
2. **Subsequent chunks (sentence_boundary)**: Natural pauses for speech prosody
3. **True streaming**: Client receives tokens immediately, not buffered

### Performance Characteristics

Measured on Nemotron-3-Nano-30B (Q4_K_M) running on DGX Spark with llama.cpp. Values are medians from 10 test runs.

| Metric | Median Value |
|--------|--------------|
| First chunk tokens | 10 |
| First chunk TTFT (server) | 111ms |
| First chunk TTFT (client) | 205ms |
| First chunk gen time | 337ms |
| Continuation TTFT | 30-100ms |
| Generation speed | ~50 TPS |

## Installation

```bash
# From the nemotron-speech directory
uv run python -m nemotron_speech.chunked_llm_server
```

### Command Line Options

```
--llama-url     URL of llama.cpp server (default: http://localhost:8000)
--num-slots     Number of parallel slots (default: 1)
--host          Bind address (default: 0.0.0.0)
--port, -p      Port number (default: 8002)
```

## WebSocket Protocol

Connect to `ws://localhost:8002/ws`

### Actions

#### start_stream

Start a new generation stream.

```json
{
  "action": "start_stream",
  "stream_id": "unique-id",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Hello!"}
  ],
  "pause": {"max_tokens": 10},
  "stream_tokens": true,
  "temperature": 0.7
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `action` | Yes | - | Must be `"start_stream"` |
| `stream_id` | Yes | - | Unique identifier for this stream |
| `messages` | Yes | - | Array of chat messages (see Message Format) |
| `pause` | No | `{}` | Pause configuration (see Pause Strategies) |
| `stream_tokens` | No | `false` | Enable true token streaming |
| `temperature` | No | `0.7` | Sampling temperature |

#### continue_stream

Continue an existing stream after pause.

```json
{
  "action": "continue_stream",
  "stream_id": "unique-id",
  "pause": {"sentence_boundary": true}
}
```

#### end_stream

End and cleanup a stream.

```json
{
  "action": "end_stream",
  "stream_id": "unique-id"
}
```

#### ping

Health check / keepalive.

```json
{"action": "ping"}
```

Response: `{"status": "pong"}`

### Message Format

Messages use ChatML format internally. Supported roles:

| Role | Description |
|------|-------------|
| `system` | System prompt defining assistant behavior |
| `user` | User input |
| `assistant` | Previous assistant responses (for multi-turn) |

The server automatically formats messages as ChatML and injects `<think></think>` to disable reasoning:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
<think></think>
```

### Pause Strategies

| Strategy | Config | Behavior | Max Tokens |
|----------|--------|----------|------------|
| Max tokens | `{"max_tokens": N}` | Pause after exactly N tokens | N |
| Sentence boundary | `{"sentence_boundary": true}` | Pause at `.` `!` `?` | 200 |
| None | `{}` or omit | Generate until EOS | 500 |

### Recommended Mixed Strategy

For voice agents, use max_tokens for the first chunk (predictable latency) and sentence_boundary for subsequent chunks (natural speech):

```python
# First chunk: fixed token count for fast audio start
await ws.send(json.dumps({
    "action": "start_stream",
    "stream_id": stream_id,
    "messages": messages,
    "pause": {"max_tokens": 10},
    "stream_tokens": True
}))

# Subsequent chunks: sentence boundaries for natural pauses
await ws.send(json.dumps({
    "action": "continue_stream",
    "stream_id": stream_id,
    "pause": {"sentence_boundary": True}
}))
```

### Server Response Messages

#### Token Streaming Mode (stream_tokens=True)

**Token message** - Sent for each token as it's generated:
```json
{"type": "token", "content": "Hello"}
```

**Paused message** - Sent when pause condition is met (more content available):
```json
{
  "type": "paused",
  "reason": "sentence_boundary",
  "text": "Hello! How can I help you?",
  "tokens": 8,
  "ttft_ms": 111,
  "elapsed_ms": 287
}
```

**Done message** - Sent when generation is complete (EOS or stop token):
```json
{
  "type": "done",
  "reason": "stop_word",
  "text": "Hello! How can I help you today?",
  "tokens": 9,
  "ttft_ms": 111,
  "elapsed_ms": 320
}
```

#### Buffered Mode (stream_tokens=False, default)

Backward-compatible mode that buffers tokens and sends complete chunks:

**Start response:**
```json
{
  "stream_id": "unique-id",
  "status": "started",
  "text": "Hello! How can I help you?",
  "tokens": 8,
  "paused": true,
  "reason": "sentence_boundary",
  "done": false,
  "ttft_ms": 111,
  "full_text": "Hello! How can I help you?"
}
```

**Continue response:**
```json
{
  "stream_id": "unique-id",
  "text": " I'm here to assist.",
  "tokens": 5,
  "paused": false,
  "reason": "eos",
  "done": true,
  "ttft_ms": 32,
  "full_text": "Hello! How can I help you? I'm here to assist."
}
```

**End response:**
```json
{
  "stream_id": "unique-id",
  "status": "ended"
}
```

### Error Responses

```json
{"error": "stream_id required"}
{"error": "Stream not found"}
{"error": "Unknown action: foo"}
{"stream_id": "unique-id", "error": "Stream already started"}
```

### Stop Reasons

| Reason | Meaning |
|--------|---------|
| `max_tokens` | Reached token limit |
| `sentence_boundary` | Hit sentence-ending punctuation, more content follows |
| `sentence_boundary_eos` | Sentence boundary was the natural end (no more content) |
| `eos` | End of sequence token |
| `stop_word` | Hit stop word (`<\|im_end\|>`) |
| `empty_response` | No tokens generated |
| `already_done` | Stream already complete |
| `connection_error` | Connection to llama.cpp failed |

## Client Example

```python
import asyncio
import json
import websockets

async def voice_agent():
    uri = "ws://localhost:8002/ws"
    stream_id = "voice-001"

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
        {"role": "user", "content": "Tell me a joke."}
    ]

    async with websockets.connect(uri) as ws:
        # Start with max_tokens for fast first audio
        await ws.send(json.dumps({
            "action": "start_stream",
            "stream_id": stream_id,
            "messages": messages,
            "pause": {"max_tokens": 10},
            "stream_tokens": True
        }))

        full_text = ""
        while True:
            msg = json.loads(await ws.recv())

            if msg["type"] == "token":
                # Stream token to TTS
                print(msg["content"], end="", flush=True)
                full_text += msg["content"]

            elif msg["type"] == "paused":
                print(f"\n[Paused: {msg['reason']}, {msg['tokens']} tokens]")

                # Let TTS catch up, then continue
                await asyncio.sleep(0.5)

                await ws.send(json.dumps({
                    "action": "continue_stream",
                    "stream_id": stream_id,
                    "pause": {"sentence_boundary": True}
                }))

            elif msg["type"] == "done":
                print(f"\n[Done: {msg['reason']}, {msg['tokens']} tokens]")
                break

        # Cleanup
        await ws.send(json.dumps({
            "action": "end_stream",
            "stream_id": stream_id
        }))
        await ws.recv()  # Wait for end confirmation

        print(f"\nFull response: {full_text}")

asyncio.run(voice_agent())
```

## Architecture

### Key Components

#### LLMStream
Maintains state for a single generation stream:
- Prompt and generated text
- Token count
- Slot assignment
- Stop signal tracking (`_received_eos` guard)

#### StreamManager
Manages multiple streams with slot allocation:
- Creates/removes streams
- Allocates llama.cpp slots
- Enforces slot limits

#### PauseConfig
Configuration for pause behavior:
- `max_tokens`: Optional token limit
- `sentence_boundary`: Boolean for sentence detection

### Slot Cooldown

To prevent llama.cpp slot assertion crashes, a 250ms cooldown is enforced between requests to the same slot:

```python
SLOT_COOLDOWN_MS = 250
```

This prevents race conditions when rapidly pausing/resuming generation.

### Stop Signal Detection

A key fix prevents crashes when continuing after the model is done:

1. When sentence boundary is detected, **peek at the next token**
2. If next token is EOS/stop word, mark stream as done
3. If next token has more content, stop with `sentence_boundary` reason
4. Client receives `type: "done"` and knows not to continue
5. Prevents empty continuation requests that crash llama.cpp

The sentence boundary check uses end-of-string matching:

```python
def ends_at_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary (not just contains one)."""
    if not text:
        return False
    text = text.rstrip()
    if not text:
        return False
    # Ends with .!? (possibly followed by closing quotes/parens)
    return bool(re.search(r'[.!?]["\'\)]*$', text))
```

This correctly handles cases like `"Hello!"` (boundary) vs `"Hello! I"` (not a boundary).

### Reasoning Disabled

For voice agents, reasoning tokens are always disabled by injecting empty think tags:

```python
prompt_parts.append("<|im_start|>assistant\n<think></think>")
```

This prevents `<think>...</think>` blocks from appearing in spoken output.

## Configuration

### Server Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SLOT_COOLDOWN_MS` | 250 | Minimum ms between slot requests |
| `llama_url` | http://localhost:8000 | llama.cpp server URL |
| `num_slots` | 1 | Parallel generation slots |

### Generation Parameters

These are hardcoded in the server:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `temperature` | 0.7 (configurable) | Sampling temperature |
| `top_p` | 0.95 | Nucleus sampling |
| `top_k` | 40 | Top-k sampling |
| `repeat_penalty` | 1.1 | Repetition penalty |
| `stop` | `["<\|im_end\|>"]` | Stop sequences (ChatML) |
| `cache_prompt` | true | Enable KV cache |

## Health Check

```bash
curl http://localhost:8002/health
```

Response:
```json
{
  "status": "ok",
  "llama_server": "healthy",
  "llama_url": "http://localhost:8000",
  "active_streams": 0,
  "slot_cooldown_ms": 250
}
```

Status is `"degraded"` if llama.cpp is unreachable.

## Integration with Voice Pipelines

### Pipecat Integration

The chunked server is designed to work with Pipecat voice agents:

```python
# In your Pipecat bot
class ChunkedLLMService(LLMService):
    async def process_frame(self, frame):
        # Connect to chunked server
        async with websockets.connect("ws://localhost:8002/ws") as ws:
            # Start stream with first chunk strategy
            await ws.send(json.dumps({
                "action": "start_stream",
                "stream_id": self.stream_id,
                "messages": self.messages,
                "pause": {"max_tokens": 10},
                "stream_tokens": True
            }))

            # Stream tokens to TTS
            async for msg in self._receive_tokens(ws):
                if msg["type"] == "token":
                    await self.push_frame(TextFrame(msg["content"]))
                elif msg["type"] == "paused":
                    # Wait for TTS buffer, then continue
                    await self._wait_for_tts()
                    await self._continue_stream(ws)
                elif msg["type"] == "done":
                    break
```

### TTS Coordination

The pause/resume pattern allows coordinating with TTS:

1. **Start generation** with `max_tokens=10`
2. **Receive first chunk** (10 tokens, ~337ms)
3. **Start TTS** on first chunk
4. **Continue generation** with `sentence_boundary`
5. **Queue TTS** for each sentence chunk
6. **Repeat** until done

This overlap maximizes GPU utilization while minimizing perceived latency.

## Benchmarking

Run the timeline test to measure performance:

```bash
uv run python tests/timeline_test.py
```

Output includes:
- Per-chunk timing (TTFT, generation time, TPS)
- Token counts
- Full generated text

### Sample Output

```
CHUNK BREAKDOWN:
--------------------------------------------------------------------------------------------------------------
#   Tokens  Srv TTFT   Cli TTFT   Chunk      TPS      Text (truncated)
--------------------------------------------------------------------------------------------------------------
1   10      112        127        258        38.7     Why did the unicorn get a promotion...
2   11      32         34         175        62.8       \nIt kept making up its own colors...
3   14      43         44         233        60.0      it kept using its horn to light up...
```

## Troubleshooting

### llama.cpp Crashes

If you see `GGML_ASSERT(!slot.is_processing())`:

1. This usually means a continuation request was made when the model was already done
2. The fix (stop signal detection) should prevent this
3. If it persists:
   - Increase `SLOT_COOLDOWN_MS` (edit in source)
   - Restart the llama.cpp container: `docker restart llama-q4`
   - Check server logs for `reason=empty_response`

### High Client TTFT

If client TTFT is much higher than server TTFT (>100ms gap):

1. Check WebSocket connection overhead
2. Verify `stream_tokens=True` is set (buffered mode has higher latency)
3. Ensure no buffering between server and client
4. Check network latency if not localhost

### Off-topic Responses

If the model ignores the prompt:

1. Use a clear system prompt with explicit instructions
2. Add constraints like "Keep responses concise"
3. For testing, use "Repeat exactly:" prefix

### Connection Refused

If the server can't connect to llama.cpp:

1. Check llama.cpp is running: `curl http://localhost:8000/health`
2. Verify the `--llama-url` matches
3. Check Docker networking if using containers

## Files

| File | Description |
|------|-------------|
| `src/nemotron_speech/chunked_llm_server.py` | Main server implementation (887 lines) |
| `tests/timeline_test.py` | Performance testing script |
| `docs/chunked_llm_server.md` | This documentation |

## Version History

- **2025-12-24**: Initial implementation with true token streaming
- **2025-12-24**: Added stop signal detection after sentence boundary (peek at next token)
- **2025-12-24**: Fixed `ends_at_sentence_boundary()` to check text end, not middle
- **2025-12-24**: Added mixed chunking strategy (max_tokens + sentence_boundary)
- **2025-12-24**: Increased slot cooldown from 100ms to 250ms

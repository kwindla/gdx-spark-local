# Integration Test: Chunked LLM + Adaptive TTS Streaming

**Status:** ✅ Implemented and tested (2025-12-24)

## Overview

A standalone test that integrates the chunked LLM streaming server with the WebSocket TTS server, measuring end-to-end latency from inference start to audio output.

## Architecture

```
┌──────────────────────┐      WS (8002/ws)       ┌─────────────────────┐
│   Integration Test   │◄──────────────────────►│  Chunked LLM Server │
│                      │                         │   (llama.cpp)       │
│   - Orchestrates     │      WS (8001/ws/tts)   └─────────────────────┘
│   - Measures timing  │◄──────────────────────►│  TTS Server         │
│   - Reports metrics  │                         │   (Magpie)          │
└──────────────────────┘                         └─────────────────────┘
```

## WebSocket Connection Lifecycle

**Critical Design Decision**: We maintain persistent WebSocket connections:

1. **TTS WebSocket** (`ws://localhost:8001/ws/tts/stream`):
   - Connect ONCE at test start
   - Create a new TTS stream (`init`) for each LLM response
   - Append text chunks as they arrive from LLM (`text`)
   - Close the stream (`close`) only when LLM response is complete
   - For multi-turn: create new stream for each turn, reuse connection

2. **LLM WebSocket** (`ws://localhost:8002/ws`):
   - Connect ONCE at test start
   - Start/continue streams as needed

## Processing Flow (Sequential with Implicit Sync)

```
1. Connect to both WebSockets
2. Create TTS stream (init)
3. Start LLM stream (start_stream, max_tokens=10)

For each LLM chunk:
  4. Receive LLM tokens until paused/done
  5. Send accumulated text to TTS (text message)
  6. Wait for TTS audio to arrive and stop (implicit segment boundary)
  7. If LLM not done, request next chunk (continue_stream, sentence_boundary)
  8. Repeat from step 4

9. When LLM done, close TTS stream (close)
10. Wait for final audio and done message
11. Report all timing metrics
```

## Synchronization Mechanism

### Design Principle: Client Controls Chunking

The LLM pause strategies (`max_tokens`, `sentence_boundary`) define text segments. The TTS server
should NOT impose its own buffering logic on top of this. Each LLM chunk IS a TTS segment.

### Required Server Change

**File:** `src/nemotron_speech/adaptive_stream.py`

Change `TARGET_AUDIO_MS` from 4000ms to 500ms:

```python
# Before
TARGET_AUDIO_MS: float = 4000.0  # Flush when buffer has ~10 words of audio

# After
TARGET_AUDIO_MS: float = 500.0  # Flush almost immediately - client controls chunking
```

With 500ms threshold (~1-2 words), any realistic LLM chunk triggers immediate flush.
The client controls segmentation via LLM pause strategies, not TTS buffering.

### Gap-Based Segment Detection

Since TTS flushes immediately, we detect segment completion by waiting for audio to stop:

```python
async def wait_for_tts_segment(tts_ws, first_timeout_s=10.0, gap_timeout_ms=150):
    """Wait for a TTS segment to complete using gap detection.

    With TARGET_AUDIO_MS=500ms, TTS flushes almost immediately when text arrives.
    We wait for first audio (up to 10s for batch generation), then collect
    until a 150ms gap indicates segment is done.
    """
    audio_chunks = []
    first_audio_ts = None

    # Phase 1: Wait for first audio (TTS generation takes ~300-600ms)
    while first_audio_ts is None:
        msg = await asyncio.wait_for(tts_ws.recv(), timeout=first_timeout_s)
        if isinstance(msg, bytes):
            first_audio_ts = time.time()
            audio_chunks.append(msg)
        elif json.loads(msg).get("type") == "done":
            return audio_chunks, first_audio_ts, True  # stream complete

    # Phase 2: Collect remaining audio until gap (segment boundary)
    while True:
        try:
            msg = await asyncio.wait_for(tts_ws.recv(), timeout=gap_timeout_ms/1000)
            if isinstance(msg, bytes):
                audio_chunks.append(msg)
            elif json.loads(msg).get("type") == "done":
                return audio_chunks, first_audio_ts, True
        except asyncio.TimeoutError:
            return audio_chunks, first_audio_ts, False  # segment complete, more coming
```

### Why This Works

1. Client sends LLM chunk text to TTS (e.g., "Why did the unicorn cross the road?")
2. TTS receives text, buffer now has ~8 words = 3200ms estimated
3. `flush_text_buffer()` checks: 3200ms >= 500ms → flush immediately
4. TTS generates audio (~300-600ms) and sends chunks
5. Client receives audio chunks
6. After 150ms with no new audio → segment complete
7. Client requests next LLM chunk

## Protocol Details

### LLM WebSocket (port 8002, `/ws`)

**Start stream (first chunk):**
```json
{
  "action": "start_stream",
  "stream_id": "test-xxx",
  "messages": [...],
  "pause": {"max_tokens": 10},
  "stream_tokens": true
}
```

**Continue stream (subsequent chunks):**
```json
{
  "action": "continue_stream",
  "stream_id": "test-xxx",
  "pause": {"sentence_boundary": true}
}
```

**Response messages:**
- `{"type": "token", "content": "Hello"}` - individual tokens
- `{"type": "paused", "text": "...", "ttft_ms": 111, ...}` - chunk complete, more available
- `{"type": "done", "text": "...", "ttft_ms": 111, ...}` - generation finished

### TTS WebSocket (port 8001, `/ws/tts/stream`)

**Initialize stream:**
```json
{"type": "init", "voice": "aria", "language": "en"}
```
Response: `{"type": "stream_created", "stream_id": "..."}`

**Append text (multiple times, no close between chunks):**
```json
{"type": "text", "text": "Hello! I'm Nemotron."}
```

**Close stream (only when LLM response complete):**
```json
{"type": "close"}
```

**Response messages:**
- Binary frames: Raw PCM audio chunks (4096 bytes each)
- `{"type": "done", "total_audio_ms": 5432}` - stream complete

## Timing Metrics Per Chunk

| Metric | Definition |
|--------|------------|
| `inference_start_ts` | Timestamp when start/continue sent to LLM |
| `llm_ttft_ms` | First LLM token received - inference_start |
| `llm_gen_time_ms` | LLM paused/done received - inference_start |
| `tts_text_sent_ts` | Timestamp when text sent to TTS |
| `tts_first_audio_ts` | Timestamp when first audio byte received |
| `tts_segment_done_ts` | Timestamp when segment audio stops (gap detected) |
| `tts_ttfb_ms` | First TTS audio - tts_text_sent |
| `tts_gen_time_ms` | Segment done - tts_text_sent |
| `audio_duration_ms` | Audio playout time (bytes / 22000 / 2 * 1000) |
| `combined_ttfb_ms` | First TTS audio - inference_start |

## Implementation Plan

### File: `tests/test_llm_tts_integration.py`

#### 1. Data Structures

```python
@dataclass
class ChunkMetrics:
    chunk_num: int
    text: str
    token_count: int
    audio_bytes: int

    # Timestamps (absolute)
    inference_start_ts: float
    llm_first_token_ts: Optional[float]
    llm_done_ts: float
    tts_text_sent_ts: float
    tts_first_audio_ts: Optional[float]
    tts_segment_done_ts: float

    # Derived metrics
    @property
    def llm_ttft_ms(self) -> float:
        if self.llm_first_token_ts is None:
            return 0
        return (self.llm_first_token_ts - self.inference_start_ts) * 1000

    @property
    def llm_gen_time_ms(self) -> float:
        return (self.llm_done_ts - self.inference_start_ts) * 1000

    @property
    def tts_ttfb_ms(self) -> float:
        if self.tts_first_audio_ts is None:
            return 0
        return (self.tts_first_audio_ts - self.tts_text_sent_ts) * 1000

    @property
    def tts_gen_time_ms(self) -> float:
        return (self.tts_segment_done_ts - self.tts_text_sent_ts) * 1000

    @property
    def audio_duration_ms(self) -> float:
        return self.audio_bytes / (22000 * 2) * 1000

    @property
    def combined_ttfb_ms(self) -> float:
        if self.tts_first_audio_ts is None:
            return 0
        return (self.tts_first_audio_ts - self.inference_start_ts) * 1000
```

#### 2. LLM Client Functions

```python
async def receive_llm_chunk(ws, chunk_num: int, inference_start: float) -> tuple[str, int, float, float, bool]:
    """Receive tokens until paused/done.

    Returns: (text, token_count, first_token_ts, done_ts, is_done)
    """
    tokens = []
    first_token_ts = None

    while True:
        msg = json.loads(await ws.recv())

        if msg["type"] == "token":
            if first_token_ts is None:
                first_token_ts = time.time()
            tokens.append(msg["content"])

        elif msg["type"] in ("paused", "done"):
            return (
                msg["text"],
                len(tokens),
                first_token_ts,
                time.time(),
                msg["type"] == "done"
            )
```

#### 3. TTS Client Functions

```python
async def wait_for_tts_segment(
    tts_ws,
    first_timeout_s: float = 10.0,
    gap_timeout_ms: float = 150
) -> tuple[bytes, Optional[float], float, bool]:
    """Wait for TTS segment using gap detection.

    Returns: (audio_bytes, first_audio_ts, done_ts, is_stream_complete)
    """
    audio_chunks = []
    first_audio_ts = None

    # Wait for first audio
    while first_audio_ts is None:
        try:
            msg = await asyncio.wait_for(tts_ws.recv(), timeout=first_timeout_s)
        except asyncio.TimeoutError:
            # No audio generated (empty segment?)
            return b"", None, time.time(), False

        if isinstance(msg, bytes):
            first_audio_ts = time.time()
            audio_chunks.append(msg)
        else:
            data = json.loads(msg)
            if data.get("type") == "done":
                return b"".join(audio_chunks), first_audio_ts, time.time(), True

    # Collect remaining audio until gap
    while True:
        try:
            msg = await asyncio.wait_for(tts_ws.recv(), timeout=gap_timeout_ms/1000)
            if isinstance(msg, bytes):
                audio_chunks.append(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "done":
                    return b"".join(audio_chunks), first_audio_ts, time.time(), True
        except asyncio.TimeoutError:
            # Gap detected - segment complete
            return b"".join(audio_chunks), first_audio_ts, time.time(), False
```

#### 4. Main Integration Loop

Key design points:
- **Single TTS connection** - connect once, create one stream per LLM response
- **Multiple text messages** - send `text` for each LLM chunk (no close between chunks)
- **Close only at the end** - send `close` when LLM is done to flush any remaining buffer
- **Gap-based sync** - wait for audio to stop (150ms gap) to detect segment completion

```python
async def run_integration_test():
    LLM_URI = "ws://localhost:8002/ws"
    TTS_URI = "ws://localhost:8001/ws/tts/stream"

    async with websockets.connect(LLM_URI) as llm_ws:
        async with websockets.connect(TTS_URI) as tts_ws:
            # Initialize TTS stream ONCE per LLM response
            await tts_ws.send(json.dumps({
                "type": "init", "voice": "aria", "language": "en"
            }))
            await tts_ws.recv()  # stream_created

            # Start LLM stream
            await llm_ws.send(json.dumps({
                "action": "start_stream",
                "stream_id": stream_id,
                "messages": messages,
                "pause": {"max_tokens": 10},
                "stream_tokens": True
            }))

            while True:
                # Receive LLM chunk
                text, token_count, ..., is_llm_done = await receive_llm_chunk(...)

                # Send text to TTS (no close - more chunks may come)
                await tts_ws.send(json.dumps({"type": "text", "text": text}))

                # If LLM is done, NOW close TTS to flush remaining
                if is_llm_done:
                    await tts_ws.send(json.dumps({"type": "close"}))

                # Wait for TTS segment (gap-based detection)
                audio_bytes, first_audio_ts, done_ts, is_stream_done = \
                    await wait_for_tts_segment(tts_ws)

                if is_llm_done:
                    break

                # Continue LLM with sentence_boundary
                await llm_ws.send(json.dumps({
                    "action": "continue_stream",
                    "stream_id": stream_id,
                    "pause": {"sentence_boundary": True}
                }))
```

#### 5. Output Format

```
================================================================================
LLM + TTS INTEGRATION TEST
================================================================================

System: You are a helpful AI assistant running on an NVIDIA DGX Spark...
User: Tell me three jokes about unicorns.
Strategy: max_tokens=10 (first), sentence_boundary (subsequent)

--------------------------------------------------------------------------------
TIMELINE
--------------------------------------------------------------------------------
[12:34:56.789] LLM #1: start_stream
[12:34:56.901]   └─ First token (LLM TTFT: 112ms)
[12:34:57.126]   └─ Paused, 10 tokens, 337ms
[12:34:57.130] TTS #1: Sending "Why did the unicorn..." (10 words)
[12:34:57.712]   └─ First audio (TTS TTFB: 582ms)
[12:34:57.945]   └─ Segment complete (815ms, 2340ms audio)
              ★ Combined TTFB: 923ms

[12:34:57.950] LLM #2: continue_stream (sentence_boundary)
[12:34:57.982]   └─ First token (LLM TTFT: 32ms)
...

================================================================================
SUMMARY
================================================================================
  Total chunks:          8
  Total LLM tokens:      127
  Total audio:           19.8 seconds
  Total wall time:       12.4 seconds

  First chunk:
    LLM TTFT:            112ms
    LLM gen time:        337ms
    TTS TTFB:            582ms
    TTS gen time:        815ms
    Combined TTFB:       923ms
    Audio duration:      2340ms

  Averages (all chunks):
    Avg LLM TTFT:        45ms
    Avg LLM gen:         285ms
    Avg TTS TTFB:        295ms
    Avg TTS gen:         520ms
    Avg Combined TTFB:   580ms

CHUNK BREAKDOWN:
-------------------------------------------------------------------------------------------------
#   Tokens  LLM TTFT  LLM Gen   TTS TTFB  TTS Gen   Audio     Combined  Text
-------------------------------------------------------------------------------------------------
1   10      112       337       582       815       2340      923       Why did the unic...
2   11      32        175       289       612       1890      464       Because it always...
3   8       28        142       275       485       1450      417       What do you call...
...
```

## Edge Cases to Handle

1. **Empty LLM chunk**: LLM might produce empty text on some chunks (unlikely with sentence_boundary)
2. **TTS threshold not reached**: If text is too short, TTS won't flush until close
3. **LLM done on first chunk**: Handle single-chunk responses
4. **Network timeouts**: Graceful handling of slow LLM/TTS

## Testing Commands

```bash
# Ensure servers are running
curl http://localhost:8001/health  # TTS
curl http://localhost:8002/health  # LLM

# Run the integration test
uv run python tests/test_llm_tts_integration.py

# Save audio output
uv run python tests/test_llm_tts_integration.py --save-audio output.wav

# Verbose mode with all timestamps
uv run python tests/test_llm_tts_integration.py --verbose
```

## Files Modified/Created

| File | Action | Description | Status |
|------|--------|-------------|--------|
| `src/nemotron_speech/adaptive_stream.py` | Modified | Changed `TARGET_AUDIO_MS` from 4000.0 to 500.0 | ✅ Done |
| `tests/test_llm_tts_integration.py` | Created | Main integration test (318 lines) | ✅ Done |

## Implementation Order (Completed)

1. ✅ Modified `adaptive_stream.py` to set `TARGET_AUDIO_MS = 500.0`
2. ✅ Created the integration test
3. ✅ Tested with both servers active - results documented above

## Notes

- TTS WebSocket uses batch mode (not frame-by-frame streaming)
- With TARGET_AUDIO_MS = 500ms, TTS flushes immediately for any realistic text
- Gap detection timeout (150ms) tuned for local execution; may need adjustment for network latency
- Client controls chunking via LLM pause strategies, TTS just processes what it receives

## Test Results (2025-12-24)

**Test prompt:** "Tell me three jokes about unicorns."

### Per-Chunk Metrics

| Chunk | Tokens | LLM TTFT | LLM Gen | TTS TTFB | TTS Gen | Audio | Combined TTFB |
|-------|--------|----------|---------|----------|---------|-------|---------------|
| 1 | 10 | 241ms | 375ms | 509ms | 660ms | 1722ms | **884ms** |
| 2 | 10 | 36ms | 165ms | 873ms | 1026ms | 2886ms | **1038ms** |
| 3 | 14 | 55ms | 242ms | 775ms | 926ms | 2607ms | **1017ms** |
| 4 | 14 | 75ms | 263ms | 988ms | 1140ms | 3305ms | **1251ms** |
| 5 | 15 | 88ms | 289ms | 1061ms | 1213ms | 3537ms | **1350ms** |
| 6 | 13 | 103ms | 276ms | 735ms | 735ms | 2560ms | **1011ms** |

### Summary

| Metric | First Chunk | Average |
|--------|-------------|---------|
| LLM TTFT | 241ms | 100ms |
| LLM Gen Time | 375ms | 269ms |
| TTS TTFB | 509ms | 823ms |
| TTS Gen Time | 660ms | 950ms |
| **Combined TTFB** | **884ms** | **1092ms** |

**Totals:**
- 6 chunks, 76 tokens
- 16.6 seconds of audio
- 7.3 seconds wall time (2.3x faster than real-time)
- Audio saved to `integration_test_output.wav`

### Key Observations

1. **First chunk is critical**: Combined TTFB of 884ms means user hears first audio under 1 second
2. **LLM continuation is fast**: After first chunk, LLM TTFT drops to 36-103ms (cached context)
3. **TTS is the bottleneck**: TTS TTFB (509-1061ms) dominates combined latency
4. **Sequential processing works**: Gap-based detection (150ms) reliably detects segment boundaries

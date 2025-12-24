# WebSocket TTS Implementation Plan

**Date:** 2025-12-24
**Status:** Implementing simplified flush logic

## Problem Identified

The original flush logic waited 3 seconds before flushing "Hello!" because:
- `MIN_AUDIO_MS = 1500ms` (~4 words needed)
- "Hello!" = 1 word = 400ms estimated
- Fallback `long_timeout = 3000ms`

## Solution: Simplified Flush Logic

**Flush when:**
1. `estimated_audio >= TARGET_AUDIO_MS` (1500ms = ~4 words)
2. Stream is closed → flush whatever remains

**NOT flush on:**
- Sentence boundaries
- Arbitrary timeouts

## Timeline Example

```
  0ms: "Hello!"     → buffer="Hello!" (400ms) - not enough
 50ms: " I'm"       → buffer="Hello! I'm" (800ms) - not enough
100ms: " Nemotron"  → buffer="Hello! I'm Nemotron" (1200ms) - not enough
150ms: " Nano,"     → buffer="Hello! I'm Nemotron Nano," (1600ms) - FLUSH!
       → Start generating audio for "Hello! I'm Nemotron Nano,"
200ms: " a"         → buffer=" a" (new segment starts accumulating)
...
```

Expected first flush at ~150ms, not 3000ms.

## Architecture

### Receive Loop (pseudo-code)

```
while True:
    msg = await websocket.receive()

    if msg.type == "init":
        stream = create_stream()
        audio_task = start_background(send_audio)

    elif msg.type == "text":
        stream.text_buffer += msg.text  # just accumulate

    elif msg.type == "close":
        stream.state = CLOSED
        break

await audio_task  # wait for audio to finish
```

### Send Audio Task (pseudo-code)

```
TARGET_AUDIO_MS = 1500  # ~4 words

while True:
    # Check if we should flush
    estimated_audio = word_count(stream.text_buffer) * 400ms

    if estimated_audio >= TARGET_AUDIO_MS:
        # Flush buffer to pending queue
        segment = stream.text_buffer
        stream.text_buffer = ""
        stream.pending_queue.append(segment)

    # Process pending segments
    if stream.pending_queue:
        text = stream.pending_queue.pop()
        audio = await generate_tts(text)
        await websocket.send(audio)

    # Exit condition: closed AND buffer empty AND queue empty
    if stream.state == CLOSED:
        # Flush any remaining text
        if stream.text_buffer:
            audio = await generate_tts(stream.text_buffer)
            await websocket.send(audio)
        break

    # Brief yield to let receive loop run
    await sleep(10ms)

await websocket.send(done)
```

## Implementation Changes

### 1. TTSStream.flush_text_buffer() - SIMPLIFIED

**Before (complex):**
```python
def flush_text_buffer(self):
    has_sentence_end = bool(re.search(r"[.!?]\s*$", self.text_buffer))
    estimated_audio = self.estimate_audio_ms(self.text_buffer)
    has_enough_audio = estimated_audio >= self.MIN_AUDIO_MS
    buffer_is_healthy = self.buffer_ms >= self.BUFFER_LOW_THRESHOLD_MS

    if buffer_is_healthy:
        should_flush = has_sentence_end or buffer_age_ms >= timeout
    else:
        long_timeout = buffer_age_ms >= 3000
        should_flush = has_enough_audio or long_timeout
```

**After (simple):**
```python
def flush_text_buffer(self) -> Optional[str]:
    if not self.text_buffer:
        return None

    estimated_audio = self.estimate_audio_ms(self.text_buffer)
    should_flush = (
        estimated_audio >= self.TARGET_AUDIO_MS
        or self.state == StreamState.CLOSED
    )

    if should_flush:
        text = self.text_buffer.strip()
        self.text_buffer = ""
        if text:
            self.pending_text.append(text)
            return text
    return None
```

### 2. Remove Background Flush Task

Don't call `stream.start_flush_task()` - not needed since `send_audio` polls every 10ms.

### 3. Simplify send_audio()

```python
async def send_audio():
    while True:
        # Try to flush buffer → pending queue
        flushed = stream.flush_text_buffer()
        if flushed:
            logger.info(f"Flushed: '{flushed[:40]}...' ({len(flushed.split())} words)")

        # Generate audio for pending segments
        if stream.pending_text:
            text = stream.pending_text.pop(0)
            audio = await _generate_batch(model, text, language, speaker_idx)
            await websocket.send_bytes(audio)
            continue

        # Exit when closed AND nothing left
        if stream.state == StreamState.CLOSED and not stream.text_buffer:
            break

        # Yield to let receive loop run
        await asyncio.sleep(0.01)

    await websocket.send_json({"type": "done", ...})
```

### 4. Batch Mode Only

Remove adaptive streaming logic for now. Always use `_generate_batch()`.

## Complexity Removed

- Sentence boundary detection
- Buffer health tracking (`buffer_ms`, `BUFFER_LOW_THRESHOLD_MS`)
- Adaptive mode selection (streaming vs batch)
- Background `_flush_loop()` task
- Multiple timeout thresholds (100ms, 500ms, 3000ms)
- Buffer age tracking
- `text_available` event signaling

## Testing

```bash
# Restart server
docker restart nemotron-asr

# Wait for healthy
sleep 15 && curl -s http://localhost:8001/health

# Run test
timeout 60 uv run python tests/test_websocket_tts.py 2>&1

# Check timing in logs
docker logs nemotron-asr --tail 100 2>&1 | grep -E "(WS #|Flushed|generating)"
```

## Success Criteria

- First flush at ~150ms (after 4 words), not 3000ms
- Messages continue arriving at ~50ms intervals (no blocking)
- Audio generated and sent back
- TTFB < 500ms (batch mode TTFB ~300ms + 150ms accumulation)

## Results (2025-12-24)

**TTFB improved from 3153ms to 746ms (76% improvement!)**

Timeline:
```
WS #2 @1ms:    "Hello!"
WS #3 @52ms:   " I'm"
WS #4 @103ms:  " Nemotron"
WS #5 @154ms:  " Nano,"  → FLUSH! (4 words = 1600ms est)
               ↓
           Generating batch TTS (~590ms)
               ↓
First audio @746ms TTFB
```

**Metrics:**
- Accumulation time: 154ms (4 tokens × 50ms)
- Batch TTS generation: ~590ms
- Total TTFB: 746ms
- Messages arriving at 50ms intervals (no blocking!)
- Total audio generated: 19.8 seconds
- Total processing time: 6.8 seconds

**Next steps:**
- Consider reducing TARGET_AUDIO_MS to flush faster (fewer words)
- Consider using streaming TTS instead of batch for lower TTFB
- Monitor cold start behavior (batch TTS takes ~600ms on first call)

## Pipecat Service Integration

The existing `magpie_websocket_tts.py` was designed for the gRPC Magpie NIM, not our custom TTS server.

We need to create a new Pipecat TTS service that:
1. Connects to our WebSocket endpoint at `/ws/tts/stream`
2. Sends `init` message with voice/language on connection
3. Sends `text` messages as LLM tokens arrive (token-by-token streaming)
4. Receives audio chunks and yields them as `TTSAudioRawFrame`
5. Sends `close` when `LLMFullResponseEndFrame` is received
6. Handles interruption by sending `cancel` message

Reference implementation: Look at `magpie_streaming_tts.py` (HTTP streaming) as the template,
but adapt it to use WebSocket instead of HTTP streaming.

### Pipecat Service Implementation

Created `pipecat_bots/magpie_websocket_tts.py` following Deepgram's pattern:

```python
class MagpieWebSocketTTSService(WebsocketTTSService):
    # Extends WebsocketTTSService for auto-reconnection and task management

    # Key settings:
    aggregate_sentences=False  # Token-by-token, not sentences

    async def start(self, frame):
        # Connect WebSocket on pipeline start

    async def run_tts(self, text: str):
        # Called for each LLM token
        # Send {"type": "text", "text": text} message
        # Audio arrives via _receive_messages background task

    async def flush_audio(self):
        # Called on LLMFullResponseEndFrame
        # Sends {"type": "close"} to trigger server flush

    async def _receive_messages(self):
        # Receive loop using _receive_task_handler for auto-reconnection
        # Push TTSAudioRawFrame for binary audio
        # Push TTSStoppedFrame on "done" message

    async def _handle_interruption(self, frame, direction):
        # Disconnect and reconnect to clear server state
```

Key patterns from Deepgram:
- Uses `create_task()` / `cancel_task()` for receive task management
- Uses `_receive_task_handler()` for auto-reconnection on errors
- Implements `_connect_websocket()` / `_disconnect_websocket()` abstract methods
- Calls `flush_audio()` on `LLMFullResponseEndFrame`

### Usage in bot.py

```python
if USE_WEBSOCKET_TTS:
    tts = MagpieWebSocketTTSService(
        server_url=NVIDIA_TTS_URL,
        voice="aria",
        language="en",
    )
```

### Testing

```bash
# Set environment variable
export USE_WEBSOCKET_TTS=true

# Run the bot
uv run pipecat_bots/bot.py -t webrtc
```

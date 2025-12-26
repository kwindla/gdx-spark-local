# TTS Interruption Handling Fix Plan

## Problem Statement

Two issues occur during voice agent interruptions:

1. **Audio continues after interruption**: When a significant amount of audio is queued, interrupting the bot doesn't immediately stop playout.

2. **Old stream audio bleeds into new stream**: Occasionally after an interruption, a short segment of the previous response plays before the new response starts.

## Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌───────────────┐
│   LLM       │───►│   TTS       │───►│   Transport     │───►│   Audio Out   │
│  Service    │    │  Service    │    │   (Daily/etc)   │    │   (Speaker)   │
└─────────────┘    └─────────────┘    └─────────────────┘    └───────────────┘
                         │
                         │ WebSocket
                         ▼
                   ┌─────────────┐
                   │  TTS Server │
                   │  (Magpie)   │
                   └─────────────┘
```

### InterruptionFrame Flow

When the user speaks during bot output:
1. VAD/STT detects user speech
2. InterruptionFrame is pushed through the pipeline
3. Each component handles the interruption

## Root Cause Analysis

### Key Discovery: Transport Properly Clears Buffers

Looking at Pipecat's `base_output.py` (lines 490-508):

```python
async def handle_interruptions(self, _: InterruptionFrame):
    if not self._transport.interruptions_allowed:
        return

    # Cancel tasks - this abandons the old audio queue
    await self._cancel_audio_task()
    await self._cancel_clock_task()
    await self._cancel_video_task()

    # Create tasks - this creates a fresh empty audio queue
    self._create_video_task()
    self._create_clock_task()
    self._create_audio_task()

    await self._bot_stopped_speaking()
```

**The transport correctly clears its audio buffers on InterruptionFrame.** The old `_audio_queue` is abandoned (via task cancellation) and a fresh empty queue is created.

### The Real Problem: TTS Client Keeps Pushing

The issue is in `magpie_websocket_tts.py`:

1. **`_receive_messages()` runs continuously** and pushes ALL received audio as `TTSAudioRawFrame`, regardless of interruption state:
   ```python
   async def _receive_messages(self):
       async for message in self._get_websocket():
           if isinstance(message, bytes):
               # No check for stream validity!
               await self.push_frame(TTSAudioRawFrame(message, self.sample_rate, 1))
   ```

2. **`_handle_interruption()` only resets local state**:
   ```python
   async def _handle_interruption(self, frame, direction):
       await self.stop_all_metrics()
       if self._stream_active:
           await self.flush_audio()  # Sends "close" to server
       self._stream_active = False
       # ... reset other state
   ```

### Timeline of the Bug

```
T=0:    Bot is speaking, TTS receiving audio from server
T=1:    User speaks → InterruptionFrame generated
T=2:    Transport receives InterruptionFrame → clears buffer ✓
T=3:    TTS receives InterruptionFrame → sends "close", resets state
T=4:    Server receives "close" but is mid-generation
T=5:    Server continues sending audio for current segment
T=6:    TTS _receive_messages() receives audio, pushes TTSAudioRawFrame
T=7:    Transport receives frames → plays OLD audio after buffer was cleared!
T=8:    New response starts, new audio arrives
T=9:    User hears: [old audio snippet] → [new audio]
```

### Why "close" Isn't Enough

The server's "close" handler lets audio generation finish:
```python
elif msg_type == "close":
    if stream:
        stream.close()  # Just sets state, doesn't stop generation
    queue_event.set()
    # Audio task continues until queue is empty!
```

This is correct for normal response completion (we want the last sentence to finish), but wrong for interruption (we want immediate stop).

## Proposed Fixes

### Fix 1: Add "cancel" Message Type (Server-Side)

Add a new message type that immediately stops generation:

```python
# In tts_server.py websocket handler

elif msg_type == "cancel":
    logger.debug(f"[{stream.stream_id[:8] if stream else 'none'}] Cancel received")

    if stream:
        stream.cancel()  # Sets state to CANCELLED

    # Immediately cancel audio task
    if audio_task:
        audio_task.cancel()
        try:
            await asyncio.wait_for(audio_task, timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # Clear queue
    async with queue_lock:
        segment_queue.clear()

    # Clean up
    if stream:
        await stream_manager.remove_stream(stream.stream_id)

    # Reset for next stream
    stream = None
    audio_task = None
    queue_event.clear()
```

### Fix 2: Client Sends "cancel" on Interruption

Modify `_handle_interruption()` in `magpie_websocket_tts.py`:

```python
async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
    """Handle interruption by cancelling server generation immediately."""
    await self.stop_all_metrics()

    # Increment stream generation to invalidate old audio
    self._stream_gen += 1

    # Send CANCEL (not close) for immediate stop
    if self._websocket:
        try:
            await self._websocket.send(json.dumps({"type": "cancel"}))
            logger.debug("WS cancel sent (interruption)")
        except Exception as e:
            logger.debug(f"Failed to send cancel: {e}")

    # Reset local state
    self._stream_active = False
    self._stream_start_time = None
    self._first_audio_received = False
    self._is_first_segment = True
    self._segment_sentence_boundary_queue.clear()
```

### Fix 3: Client-Side Audio Gating (Defense in Depth)

**Key insight**: Audio bytes don't carry metadata about which stream they belong to. But WebSocket messages are **ordered** - after receiving `stream_created` for stream N, all subsequent audio is for stream N.

**Solution**: Discard all audio until we receive `stream_created` for the current generation. This handles:
- Multiple `run_tts()` calls for the same response (LLM chunks)
- In-flight audio that arrives after interruption but before new `stream_created`

```python
class MagpieWebSocketTTSService(WebsocketTTSService):
    def __init__(self, ...):
        ...
        # Generation tracking for interruption handling
        self._gen = 0            # Incremented on interruption
        self._confirmed_gen = 0  # Set to _gen when stream_created received
        # self._stream_active already exists

async def _handle_interruption(self, frame, direction):
    ...
    self._gen += 1  # Invalidate current generation
    self._stream_active = False  # Will need new stream
    ...

# run_tts() - existing logic works unchanged:
# if not self._stream_active:
#     self._stream_active = True
#     # First text for this stream - server will send stream_created

async def _receive_messages(self):
    async for message in self._get_websocket():
        # Handle stream_created - synchronization point
        if isinstance(message, str):
            data = json.loads(message)
            if data.get("type") == "stream_created":
                self._confirmed_gen = self._gen  # Now receiving for current gen
                ...

        elif isinstance(message, bytes):
            # Discard audio until stream_created confirms current generation
            if self._confirmed_gen != self._gen:
                logger.debug(f"Discarding stale audio (confirmed={self._confirmed_gen}, current={self._gen})")
                continue

            # ... rest of audio handling
            await self.push_frame(TTSAudioRawFrame(message, self.sample_rate, 1))
```

**Why this works with multiple run_tts() calls:**

| Event | `_gen` | `_confirmed_gen` | `_stream_active` | Audio accepted? |
|-------|--------|------------------|------------------|-----------------|
| Initial | 0 | 0 | False | - |
| run_tts("chunk1") | 0 | 0 | True | - |
| stream_created | 0 | 0 | True | - |
| audio | 0 | 0 | True | ✓ (0==0) |
| run_tts("chunk2") | 0 | 0 | True | - |
| audio | 0 | 0 | True | ✓ (0==0) |
| **Interruption** | **1** | 0 | False | - |
| old audio arrives | 1 | 0 | False | ✗ (0≠1) |
| run_tts("new") | 1 | 0 | True | - |
| more old audio | 1 | 0 | True | ✗ (0≠1) |
| stream_created | 1 | **1** | True | - |
| new audio | 1 | 1 | True | ✓ (1==1) |

## Server-Side Audio Task Cancellation Timing

### The Blocking Problem

Previous implementations had issues where waiting for audio task cancellation blocked new stream creation:

```python
# PROBLEMATIC: Blocks receive loop
if audio_task:
    await asyncio.wait_for(audio_task, timeout=30.0)  # Blocks for up to 30s!
```

### Current Approach (Non-Blocking Close)

The current implementation doesn't wait on close:

```python
elif msg_type == "close":
    if stream:
        stream.close()
    queue_event.set()
    # Don't wait - let audio_task finish in background
```

New streams clean up old tasks:

```python
elif msg_type == "text":
    if need_new_stream:
        if audio_task is not None and not audio_task.done():
            audio_task.cancel()
            await asyncio.wait_for(audio_task, timeout=0.5)  # Brief wait
        ...
        stream = await stream_manager.create_stream(...)
        audio_task = asyncio.create_task(send_audio())
```

### Critical Bug: send_audio() Exit Condition

The current `send_audio()` exit condition only checks for CLOSED:

```python
# Current code (line 901-905 in tts_server.py)
if stream.state == StreamState.CLOSED:
    async with queue_lock:
        if not segment_queue:
            break
```

**Problem**: If we call `stream.cancel()` (sets state to CANCELLED), this check fails and the loop continues spinning until CancelledError is raised. This means:
1. The task doesn't exit cleanly on cancel
2. It keeps looping with 0.01s timeouts waiting for CancelledError
3. This wastes CPU and delays proper cleanup

**Fix**: Update the exit condition to include CANCELLED:

```python
# Fixed code
if stream.state in (StreamState.CLOSED, StreamState.CANCELLED):
    async with queue_lock:
        if not segment_queue:
            break
```

### Proposed Approach for "cancel"

For the new "cancel" message, we use a **non-blocking** approach:

```python
elif msg_type == "cancel":
    logger.debug(f"[{stream.stream_id[:8] if stream else 'none'}] Cancel received")

    if stream:
        stream.cancel()  # Sets state to CANCELLED

    if audio_task:
        audio_task.cancel()  # Schedule CancelledError
        # DON'T WAIT - the task will exit via:
        # 1. The updated exit condition (state == CANCELLED)
        # 2. Or CancelledError at next await point

    # Clear queue so task has nothing to generate
    async with queue_lock:
        segment_queue.clear()

    # Clean up stream
    if stream:
        await stream_manager.remove_stream(stream.stream_id)

    # Reset state for next stream
    stream = None
    audio_task = None
    queue_event.clear()
```

**Why non-blocking is safe**:
1. **Queue is cleared**: Task has no segments to process
2. **State is CANCELLED**: Task will exit at next loop iteration (with the fix)
3. **CancelledError scheduled**: Task will exit at next await point regardless
4. **stream = None**: Even if old task accesses `stream`, `send_audio()` checks `if stream is None: return`
5. **Client-side gating**: Any stray audio that leaks through is discarded by client

### Task Cancellation Points in send_audio()

The `send_audio()` function has multiple await points where cancellation can occur:
- `async with queue_lock` - Can be cancelled while waiting for lock
- `await websocket.send_bytes()` - Can be cancelled during send
- `await websocket.send_json()` - Can be cancelled during send
- `await asyncio.wait_for(queue_event.wait(), ...)` - Can be cancelled while waiting

When `audio_task.cancel()` is called, CancelledError is raised at the next await point. The task's exception handler logs and re-raises:

```python
except asyncio.CancelledError:
    logger.debug(f"[{stream.stream_id[:8]}] WS audio task cancelled")
    raise
```

### In-Flight TTS Generation

When cancel occurs, send_audio() might be in the middle of generation. The two modes behave differently:

**Streaming mode** (segment 1): Can be cancelled with ~50ms latency by adding a cancellation flag:

```python
async def _generate_streaming_with_preset(
    model, text, language, speaker_idx, preset,
    cancel_event: Optional[threading.Event] = None  # NEW
):
    def run_generation():
        try:
            streamer = StreamingMagpieTTS(model, config)
            for chunk in streamer.synthesize_streaming(...):
                # Check cancellation between chunks (~46ms granularity)
                if cancel_event and cancel_event.is_set():
                    logger.debug("Streaming generation cancelled")
                    break
                chunk_queue.put(chunk)
        finally:
            chunk_queue.put(None)
            generation_done = True
```

In send_audio(), check stream state and signal cancellation:

```python
segment_cancel = threading.Event()

if mode == "stream":
    async for audio_chunk in _generate_streaming_with_preset(
        model, text, stream.language, speaker_idx, preset,
        cancel_event=segment_cancel
    ):
        # Check if cancelled while iterating
        if stream.state == StreamState.CANCELLED:
            segment_cancel.set()  # Signal thread to stop
            break
        await websocket.send_bytes(audio_chunk)
```

**Batch mode** (segments 2+): Cannot be cancelled - `model.do_tts()` is a single blocking call with no internal cancellation points. The generation runs to completion (1-2s max).

**Rationale for keeping current pattern**:
- Most interruptions happen early in the response (during segment 1)
- Segment 1 uses streaming mode → can be cancelled within ~50ms
- Segments 2+ use batch mode → 1-2s delay is acceptable for rare late interruptions
- GPU contention from orphaned batch generation is brief and infrequent

### Summary of Server-Side Changes

1. **Fix send_audio() exit condition**: Check for CANCELLED in addition to CLOSED
2. **Add "cancel" message handler**: Non-blocking cleanup
3. **Don't wait for audio_task on cancel**: Task exits cleanly via state check or CancelledError
4. **Add streaming cancellation flag**: Stop streaming generation within ~50ms on cancel

## Implementation Order

1. **Server: Fix send_audio() exit condition** - Check CANCELLED state
2. **Server: Add streaming cancellation flag** - Stop generation within ~50ms
3. **Server: Add "cancel" message handler** - Non-blocking cleanup
4. **Client: Add generation counter** - Discard stale audio in _receive_messages()
5. **Client: Send "cancel" on interrupt** - Tell server to stop immediately

## Testing

After implementation:
1. Start bot, let it speak a long response
2. Interrupt mid-sentence (during segment 1 - streaming)
3. Verify: Old audio stops within ~100ms
4. Verify: New response starts without old audio bleeding through
5. Interrupt during segment 2+ (batch mode)
6. Verify: Delay is acceptable (1-2s max)
7. Verify: No blocking delays when starting new response after interruption

## Files to Modify

### Server (`src/nemotron_speech/tts_server.py`)
- Fix `send_audio()` exit condition: add CANCELLED to state check
- Add `cancel_event` parameter to `_generate_streaming_with_preset()`
- Add cancellation check in streaming generation loop
- Pass cancel event from `send_audio()` to streaming generator
- Add "cancel" message handler in WebSocket loop

### Client (`pipecat_bots/magpie_websocket_tts.py`)
- Add `_gen` counter (incremented on interruption only)
- Add `_confirmed_gen` (set to `_gen` when `stream_created` received)
- Update `_handle_interruption()`: increment `_gen`, send cancel
- Update `_receive_messages()`:
  - Set `_confirmed_gen = _gen` on `stream_created`
  - Discard audio when `_confirmed_gen != _gen`
- No changes needed to `run_tts()` - existing `_stream_active` logic works

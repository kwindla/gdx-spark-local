# Chunked LLM Inference: Engineering Deep Dive

This document provides a comprehensive technical analysis of the chunked LLM inference implementation, explaining why mid-stream HTTP connection cancellation is necessary for voice agents, how the llama.cpp race condition manifests, and how the two-slot alternation pattern solves it.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Why Chunked Generation?](#why-chunked-generation)
3. [The llama.cpp Race Condition](#the-llamacpp-race-condition)
4. [Solution: Two-Slot Alternation](#solution-two-slot-alternation)
5. [Implementation Details](#implementation-details)
6. [TTS Synchronization Protocol](#tts-synchronization-protocol)
7. [Metrics and Observability](#metrics-and-observability)
8. [Configuration Reference](#configuration-reference)

---

## Problem Statement

Voice-to-voice conversational AI requires extremely low latency. The time from when a user finishes speaking to when the bot starts responding (voice-to-voice latency) directly impacts the naturalness of the conversation. Our target is 500-800ms.

Traditional LLM serving waits for the full response before sending to TTS:

```
User: "Hello"
       ↓
[LLM generates full response: "Hello! How can I help you today?" ~800ms]
       ↓
[TTS synthesizes full response: ~600ms]
       ↓
User hears: "Hello! How can I help you today?"

Total latency: ~1400ms (too slow for natural conversation)
```

Our solution: **interleaved streaming** where LLM and TTS run concurrently:

```
User: "Hello"
       ↓
[LLM chunk 1: "Hello!" ~150ms] → [TTS chunk 1: ~370ms] → User hears "Hello!"
       ↓ (parallel)
[LLM chunk 2: "How can I help you today?" ~200ms] → [TTS chunk 2: ~300ms]
       ↓
User hears: "How can I help you today?"

Total latency to first audio: ~520ms (natural conversation)
```

---

## Why Chunked Generation?

### The Sentence-Boundary Approach

TTS models produce more natural speech when given complete sentences rather than token fragments. Sending "Hel" then "lo" produces different prosody than sending "Hello!". Our chunked LLM service (`llama_cpp_chunked_llm.py`) generates text at sentence boundaries:

```python
def ends_at_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary.

    Matches text ending with .!? optionally followed by closing quotes/parens.
    """
    return bool(re.search(r'[.!?]["\'\)]*$', text.rstrip()))
```

### Why HTTP Connection Cancellation is Required

The critical design constraint: **we must stop LLM generation mid-stream** to synchronize with TTS.

Consider this flow:

1. LLM generates chunk 1: "Hello!"
2. We send chunk 1 to TTS (takes ~370ms to start speaking, ~800ms total)
3. LLM continues generating: "How can I help you today?"
4. LLM finishes the full response while TTS is still on chunk 1

If we let the LLM stream complete, it generates the entire response before TTS finishes chunk 1. This causes:
- Audio buffer overflow (LLM outpaces TTS)
- Lost synchronization between text and audio
- Inability to interrupt cleanly

**Solution**: Close the HTTP connection after each chunk to stop generation, wait for TTS to complete, then resume.

```python
# From llama_cpp_chunked_llm.py
async def _process_context(self, context: LLMContext):
    while self._generating and not self._cancelled:
        # Generate one chunk (closes HTTP connection when boundary found)
        chunk_text, is_done = await self._generate_chunk()

        if chunk_text:
            await self.push_frame(LLMTextFrame(text=chunk_text))

        if is_done:
            break

        # Wait for TTS to finish before generating next chunk
        await asyncio.wait_for(self._continue_event.wait(), timeout=30.0)
        self._continue_event.clear()
```

The HTTP connection closure happens implicitly when we exit the `async with` block mid-stream:

```python
async with self._client.stream("POST", f"{self._llama_url}/completion", json=payload) as response:
    async for line in response.aiter_lines():
        # ... collect tokens until sentence boundary ...
        if ends_at_sentence_boundary(collected_text):
            break  # <-- This exits the stream, closing the HTTP connection
```

---

## The llama.cpp Race Condition

### Symptoms

When HTTP connections are closed mid-stream, llama.cpp occasionally crashes with:

```
GGML_ASSERT(!slot.is_processing()) failed
src/server-context.cpp:1011
```

This crash occurs on a subsequent request to the same slot, not immediately when the connection closes.

### Root Cause Analysis

The crash occurs due to a race condition in llama.cpp's slot cleanup:

1. **Connection Close Detection**: When our HTTP client closes the connection, llama.cpp's `cpp-httplib` eventually detects the disconnection.

2. **Async Cancel Signal**: The cancellation is processed asynchronously. The slot receives a cancel signal, but cleanup happens in a different execution context.

3. **KV Cache State**: The slot's KV cache and `is_processing` flag are not atomically updated. The slot may report as idle (`/slots` API shows `"is_processing": false`) while internal cleanup is still in progress.

4. **TCP Delayed Close**: With HTTP connection pooling, TCP connections return to the pool and may not be closed immediately. When the pool later closes a stale connection, llama.cpp interprets the TCP FIN as a cancel signal for whatever is currently using that slot.

5. **GGML_ASSERT Crash**: If a new request starts on the slot before cleanup completes, the assertion `!slot.is_processing()` fails because the slot is simultaneously in two states.

This issue is documented in the llama.cpp GitHub issues:
- [Issue #11414: Cancel task message not stopping generation](https://github.com/ggml-org/llama.cpp/issues/11414)
- [Issue #16448: Server stops responding after many requests](https://github.com/ggml-org/llama.cpp/issues/16448)

### Why the Crash is Intermittent

The crash only occurs when:
1. A connection is closed mid-generation
2. A new request arrives at the same slot
3. Before async cleanup completes

This timing is rare in normal usage but common with chunked generation where we deliberately close connections every ~200ms.

---

## Solution: Two-Slot Alternation

### Design Overview

Instead of using one slot and hoping cleanup completes before the next request, we alternate between two slots with a reuse guard:

```
Chunk 1 → Slot 0 → [close] → [cleanup in progress]
          [TTS plays ~1-2s]
Chunk 2 → Slot 1 → [close] → [cleanup in progress]    (Slot 0 still cleaning)
          [TTS plays ~1-2s]
Chunk 3 → Slot 0 → (2-4+ seconds have passed, cleanup complete) ✓
```

### Two-Slot Implementation

```python
# Constants
DEFAULT_NUM_SLOTS = 2
DEFAULT_MIN_SLOT_REUSE_DELAY_S = 2.0

class LlamaCppChunkedLLMService(AIService):
    def __init__(self, ...):
        # Two-slot management
        self._num_slots = self._params.num_slots  # 2
        self._current_slot: int = 0
        self._slot_last_used: dict[int, float] = {}  # slot_id -> timestamp
        self._last_used_slot: Optional[int] = None

    async def _get_next_slot(self) -> tuple[int, bool]:
        """Get next slot, preferring reuse for KV cache, with reuse guard."""
        min_delay = self._params.min_slot_reuse_delay_s

        # Optimization: Try to reuse the last slot if it's been idle long enough
        # This improves KV cache hit ratio
        if self._last_used_slot is not None:
            elapsed = time.time() - self._slot_last_used.get(self._last_used_slot, 0)
            if elapsed >= min_delay:
                return self._last_used_slot, True  # slot_reused=True for metrics

        # Otherwise, rotate to the next slot
        slot = self._current_slot
        self._current_slot = (self._current_slot + 1) % self._num_slots

        # Enforce reuse guard: wait if this slot was used too recently
        last_used = self._slot_last_used.get(slot, 0)
        elapsed = time.time() - last_used
        if elapsed < min_delay:
            wait_time = min_delay - elapsed
            logger.warning(f"Slot {slot} reuse guard triggered, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        return slot, False

    def _mark_slot_used(self, slot: int):
        """Record when a slot's request completed or was cancelled."""
        self._slot_last_used[slot] = time.time()
        self._last_used_slot = slot
```

### Connection Pooling Workaround

An additional workaround: disable HTTP connection pooling to ensure connections close immediately:

```python
# From llama_cpp_chunked_llm.py:258-261
self._client = httpx.AsyncClient(
    timeout=300.0,
    limits=httpx.Limits(max_keepalive_connections=0),  # Disable pooling
)
```

Without this, pooled connections may close unpredictably, sending cancel signals to slots that have been reassigned.

### Server Configuration

The llama.cpp server must be configured with multiple parallel slots:

```bash
llama-server \
    -m model.gguf \
    --parallel 2 \              # REQUIRED: Enable 2 slots
    --ctx-size 16384 \
    --flash-attn on \
    --n-gpu-layers 99
```

---

## Implementation Details

### Chunk Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          _generate_chunk() Flow                             │
│                                                                             │
│  1. _get_next_slot()                                                        │
│     ├─ Check if last slot idle >= 2s → reuse for KV cache                   │
│     └─ Otherwise → rotate to next slot, wait if reuse guard triggers        │
│                                                                             │
│  2. Build payload with cache_prompt=True, id_slot=slot_id                   │
│     └─ Prompt = base prompt + all generated text so far                     │
│                                                                             │
│  3. HTTP stream to /completion                                              │
│     ├─ First chunk: min=10, max=24 tokens (fast TTFB)                       │
│     └─ Subsequent: pure sentence boundary detection                         │
│                                                                             │
│  4. Token accumulation                                                      │
│     ├─ Accumulate tokens until sentence boundary                            │
│     ├─ Peek at next token to handle edge cases (closing quotes)             │
│     └─ Save pending token for next chunk if starts new sentence             │
│                                                                             │
│  5. Exit stream (closes HTTP connection)                                    │
│     └─ _mark_slot_used(slot) in finally block                               │
│                                                                             │
│  6. Return (chunk_text, is_done)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### First Chunk Optimization

The first chunk uses aggressive token bounds for fast time-to-first-byte:

```python
if self._is_first_chunk:
    min_tokens = 10   # Don't emit too-short first chunk
    max_tokens = 24   # Cap length for fast TTFB (~100-150ms)
else:
    min_tokens = None  # Pure sentence boundary detection
    max_tokens = None
```

### Pending Token Handling

When we detect a sentence boundary, we peek at the next token to handle edge cases:

```python
# If sentence boundary found, peek at next token
if hit_sentence_boundary:
    peek_text = collected_text + token_text
    if ends_at_sentence_boundary(peek_text):
        # Token is part of sentence ending (e.g., closing quote)
        collected_text = peek_text
    else:
        # Token starts next sentence - save for next chunk
        self._pending_token = token_text
        break
```

This ensures proper handling of patterns like `He said "Hello!"` where the closing quote should stay with the sentence.

### Cache State Management

The service maintains cache state for KV cache reuse:

```python
# Build prompt including all generated text for cache hit
full_prompt = self._prompt + self._generated_text

# After chunk generation:
# Only add NEW tokens (exclude prepended pending token)
new_tokens = collected_text[len(prepended_from_pending):]
self._generated_text += new_tokens
if self._pending_token:
    self._generated_text += self._pending_token
```

---

## TTS Synchronization Protocol

### The Continue Signal

TTS and LLM synchronize via a custom Pipecat frame:

```python
class ChunkedLLMContinueGenerationFrame(SystemFrame):
    """Signal frame sent upstream by TTS when a segment completes.

    Tells the LLM service that TTS has finished processing
    the current chunk and generation can continue.
    """
    pass
```

### Flow Diagram

```
LLM Service                              TTS Service
     │                                        │
     │──LLMTextFrame("Hello!")───────────────►│
     │                                        │
     │                                   synthesize
     │                                   stream audio
     │                                        │
     │  (waiting on _continue_event)          │
     │                                        │
     │                                   segment_complete
     │◄──ChunkedLLMContinueGenerationFrame────│
     │                                        │
     │  _continue_event.set()                 │
     │                                        │
     │──LLMTextFrame("How are you?")─────────►│
     │                                        │
```

### TTS Side Implementation

```python
# From magpie_websocket_tts.py
async def _receive_messages(self):
    async for message in self._get_websocket():
        if isinstance(message, str):
            msg = json.loads(message)

            if msg.get("type") == "segment_complete":
                # Inject inter-sentence pause if needed
                if self._segment_sentence_boundary_queue:
                    ended_with_sentence = self._segment_sentence_boundary_queue.popleft()
                    if ended_with_sentence and self._params.sentence_pause_ms > 0:
                        silence = self._generate_silence_frames(self._params.sentence_pause_ms)
                        await self.push_frame(TTSAudioRawFrame(silence, self.sample_rate, 1))

                # Signal LLM to continue
                await self.push_frame(
                    ChunkedLLMContinueGenerationFrame(),
                    FrameDirection.UPSTREAM
                )
```

### LLM Side Implementation

```python
# From llama_cpp_chunked_llm.py
async def process_frame(self, frame: Frame, direction: FrameDirection):
    # Handle TTS continue signal (upstream)
    if isinstance(frame, ChunkedLLMContinueGenerationFrame):
        if self._continue_event:
            self._continue_event.set()
        return  # Don't propagate
```

---

## Metrics and Observability

### LLM Slot Metrics

The service emits detailed metrics after each response:

```python
class LLMSlotMetricsFrame(SystemFrame):
    """Frame containing LLM slot usage and cache metrics."""
    slot_id: int              # Which slot was used
    slot_reused: bool         # True if same slot as last time (better cache)
    total_chunks: int         # Number of chunks in response
    total_time_ms: float      # Total generation time
    tokens_cached: int        # Tokens from KV cache (cache HIT)
    tokens_evaluated: int     # Tokens that needed evaluation (cache MISS)

    @property
    def cache_hit_ratio(self) -> float:
        total = self.tokens_cached + self.tokens_evaluated
        return self.tokens_cached / total if total > 0 else 0.0
```

Example log output:

```
LlamaCppChunkedLLM: LLMSlotMetrics(slot=0, reused=True, chunks=3, time=850ms, cached=245, eval=120, hit=67%)
```

### Token Usage Metrics

Standard Pipecat token usage metrics for cost tracking:

```python
token_usage = LLMTokenUsage(
    prompt_tokens=self._generation_tokens_cached + self._generation_tokens_evaluated,
    completion_tokens=self._generation_tokens_predicted,
    total_tokens=prompt_tokens + completion_tokens,
    cache_read_input_tokens=self._generation_tokens_cached,
)
```

### Voice-to-Voice Metrics

Measured by `V2VMetricsProcessor`:

```python
# Time from VADUserStoppedSpeaking to BotStartedSpeaking
# Plus vad_stop_secs (200ms) to account for VAD delay
v2v_time = frame_to_frame_time + self._vad_stop_secs
```

---

## Configuration Reference

### LLM Service Parameters

```python
class InputParams(BaseModel):
    # First chunk bounds (for low TTFB)
    first_chunk_min_tokens: int = 10    # Don't emit too-short chunks
    first_chunk_max_tokens: int = 24    # Fast first response

    # LLM generation parameters
    temperature: float = 0.0            # Deterministic for punctuation consistency
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

    # Two-slot management
    num_slots: int = 2                          # Requires --parallel 2
    min_slot_reuse_delay_s: float = 2.0         # Cleanup buffer time
```

### Server Requirements

```bash
llama-server \
    -m /path/to/model.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --parallel 2 \                    # REQUIRED for two-slot alternation
    --ctx-size 16384 \                # Multi-turn conversation context
    --flash-attn on \                 # Fast attention (Blackwell GPUs)
    --n-gpu-layers 99                 # All layers on GPU
```

### TTS Service Parameters

```python
class InputParams(BaseModel):
    language: str = "en"
    streaming_preset: str = "conservative"    # ~370ms TTFB
    use_adaptive_mode: bool = True            # Stream first, batch after
    sentence_pause_ms: int = 250              # Inter-sentence pause
```

---

## Appendix: Edge Cases

### 1. Very Fast Responses

If all chunks complete in < 2 seconds, the reuse guard may trigger:

```
Chunk 1 (slot 0): 500ms
Chunk 2 (slot 1): 500ms
Chunk 3 (slot 0): Would reuse slot 0 after only 1s → guard waits 1s
```

This is rare because TTS typically takes 1-2s per segment.

### 2. Rapid Interruptions

When users interrupt quickly multiple times:

```
Response 1, Chunk 1 (slot 0): interrupted after 200ms
Response 2, Chunk 1 (slot 1): interrupted after 200ms
Response 3, Chunk 1 (slot 0): guard triggers, wait ~1.6s
```

The guard ensures cleanup completes even under stress.

### 3. Pending Token Across Interruption

If an interruption occurs when a pending token is saved:

```python
async def _handle_interruption(self, frame: InterruptionFrame):
    self._cancelled = True
    self._generating = False
    self._is_first_chunk = True
    # Note: _pending_token is cleared when _process_context starts
```

The pending token from the previous response is harmless because `_process_context` resets all state for the new response.

### 4. Single-Slot Server

If the server is started with `--parallel 1`:

- `_get_next_slot()` always returns slot 0
- No alternation benefit
- Reuse guard still provides some protection (2s cooldown)
- Higher crash risk than with two slots

Detection: Could query `/slots` on startup to verify slot count, but not currently implemented.

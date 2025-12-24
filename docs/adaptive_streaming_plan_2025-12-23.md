# Adaptive Streaming TTS - Experiment Plan

**Date:** 2025-12-23
**Status:** Phase 2 Complete, Phase 3 Ready to Implement

---

## Current State Summary

### Completed
- [x] **Phase 1: Parallel Generation Testing** - Confirmed GPU contention requires serialization
- [x] **Phase 2: Adaptive Server Implementation** - Five endpoints working, tests passing
- [x] **Research: Pipecat Integration** - DeepgramTTSService pattern identified

### Ready to Implement
- [ ] **Phase 3: Pipecat Integration** - See "Next Steps" below

### Key Files Created
- `src/nemotron_speech/adaptive_stream.py` - Stream state management
- `src/nemotron_speech/audio_boundary.py` - Crossfade handling
- `tests/adaptive_streaming/client.py` - Test client
- `tests/adaptive_streaming/test_parallel.py` - Experiment 1 tests
- `tests/adaptive_streaming/test_adaptive.py` - Experiment 2 tests

### Server Endpoints (Ready)
```
POST   /v1/tts/stream              - Create stream
POST   /v1/tts/stream/{id}/append  - Append text
POST   /v1/tts/stream/{id}/close   - Signal completion
DELETE /v1/tts/stream/{id}         - Cancel stream
GET    /v1/tts/stream/{id}/audio   - Receive audio (streaming)
```

---

## Next Steps (Phase 3: Pipecat Integration)

### Step 1: Update Server Threshold
**File:** `src/nemotron_speech/adaptive_stream.py`
- Change `BUFFER_THRESHOLD_MS` from 500ms to 1000ms
- Make `TEXT_BUFFER_TIMEOUT_MS` adaptive based on buffer status:
  - Buffer < 500ms: shorter timeout (50ms) for faster response
  - Buffer ≥ 500ms: normal timeout (100ms) for efficiency

### Step 2: Create Pipecat Adaptive TTS Service
**File:** `pipecat_bots/magpie_adaptive_tts.py`

Key design points:
- Extend `TTSService` with `aggregate_sentences=False`
- Override `process_frame` to manage stream lifecycle
- Background task receives audio via `push_frame(TTSAudioRawFrame(...))`
- On `LLMFullResponseEndFrame`: close stream
- On `InterruptionFrame`: cancel stream
- Follow DeepgramTTSService pattern (see analysis in this doc)

### Step 3: Update bot.py
- Add `USE_ADAPTIVE_TTS` environment variable (default: false)
- Import and instantiate `MagpieAdaptiveTTSService` when enabled
- Keep `MagpieStreamingTTSService` as fallback

### Step 4: Test and Validate
- Start TTS server in Docker container
- Run bot with `USE_ADAPTIVE_TTS=true`
- Test: normal conversation, interruptions, long responses
- Compare TTFB and quality vs existing streaming service

---

## Overview

This document outlines two experiments to improve TTS quality and robustness:

1. **Parallel Generation Testing**: Determine if concurrent TTS requests cause quality degradation
2. **Adaptive Buffer Strategy**: Implement server-side adaptive generation that switches between streaming and batch modes based on buffer status

## Experiment 1: Parallel Generation Testing

### Objective

Determine whether running multiple TTS generations in parallel causes:
- Quality degradation (artifacts, distortion)
- TTFB regression
- RTF slowdown

### Hypothesis

The current implementation may have issues with concurrent generation due to:
- KV cache sharing (`model.decoder.reset_cache()` may affect global state)
- GPU memory pressure
- CUDA stream contention

### Test Parameters

| Parameter | Value |
|-----------|-------|
| Concurrency levels | 2, 4 |
| Text variation | Different text per request |
| Quality evaluation | Subjective listening |
| Metrics | TTFB, total time, RTF |

### Test Sentences (Different per Request)

```python
PARALLEL_TEST_SENTENCES = [
    "Hello! How can I help you today?",
    "The weather is beautiful this morning.",
    "I'd be happy to explain that concept.",
    "Let me think about the best approach.",
]
```

### Test Procedure

1. Run baseline: 4 sequential generations, measure TTFB/RTF, save audio
2. Run parallel-2: 2 concurrent generations, measure TTFB/RTF, save audio
3. Run parallel-4: 4 concurrent generations, measure TTFB/RTF, save audio
4. Compare metrics and listen to audio quality

### Expected Outputs

```
samples_parallel/
├── sequential_01.wav ... sequential_04.wav
├── parallel2_01.wav ... parallel2_02.wav  (x2 runs)
├── parallel4_01.wav ... parallel4_04.wav
└── metrics.json  # TTFB, RTF for each run
```

### Results (2025-12-23)

**Test completed successfully.** All audio samples generated and saved.

#### Performance Metrics

| Mode       | Avg TTFB | Max TTFB | Avg RTF | Max RTF | Real-time? |
|------------|----------|----------|---------|---------|------------|
| Sequential | 148ms    | 166ms    | 0.39x   | 0.40x   | ✓          |
| Parallel-2 | 267ms    | 306ms    | 0.70x   | 0.75x   | ✓          |
| Parallel-4 | 1114ms   | 1452ms   | 1.86x   | 2.19x   | ✗          |

#### Regression Analysis

- **Parallel-2**: TTFB +81%, RTF +82% (exceeds 50% threshold)
- **Parallel-4**: TTFB +654%, RTF +381% (not real-time capable)

#### Quality Assessment

- **Subjective audio quality**: No perceived degradation across all concurrency levels
- Parallel execution affects timing/throughput but not audio fidelity

#### Conclusions

1. **GPU contention is significant**: Even 2 concurrent requests nearly double latency
2. **Parallel-4 is not viable**: RTF > 1.0x means audio can't keep up with playback
3. **Serialization required**: Adaptive streaming must process one segment at a time
4. **Buffer threshold validated**: 500ms buffer provides necessary headroom for serialized generation

#### Recommendation

**Proceed with serialized adaptive streaming.** The server should:
- Process one text segment at a time (no parallel generation)
- Use streaming mode (~148ms TTFB) for first segment to build initial buffer
- Switch to batch mode when buffer exceeds 500ms for quality benefits
- Maintain real-time RTF (~0.39x) by avoiding concurrent generation

---

## Experiment 2: Adaptive Buffer Strategy

### Objective

Implement a server-side adaptive TTS streaming system that:
- Achieves fast TTFB (~157ms) using streaming mode initially
- Switches to batch mode (higher quality) once buffer is healthy
- Handles text appending at any granularity
- Produces seamless audio across segment boundaries

### Key Metrics

| Mode | TTFB | RTF | Quality |
|------|------|-----|---------|
| Streaming | ~157ms | 0.38x | Good |
| Batch | ~600-3000ms | 0.31x | Better |

### Buffer Threshold Calculation

```
Switch to batch when: buffer_ms >= 500ms

Rationale:
- Batch TTFB for typical sentence: ~600ms
- During batch generation, playback consumes buffer
- 500ms threshold provides safety margin
- First segment always uses streaming for fast TTFB
```

### Text Buffering Strategy

**Hybrid approach:**
- Buffer incoming text for a short window (~100ms)
- Start generation when:
  - Timeout expires, OR
  - Sentence boundary detected (. ! ?)
- This allows efficient batching while bounding latency

The buffer window (~100ms) is chosen to be less than our streaming TTFB (~157ms), ensuring we don't add significant latency.

---

## Protocol Design

### Five Client Operations

```
1. CREATE STREAM
   POST /v1/tts/stream
   Body: {"voice": "aria", "language": "en"}
   Response: {"stream_id": "uuid", "audio_url": "/v1/tts/stream/{id}/audio"}

2. APPEND TEXT
   POST /v1/tts/stream/{id}/append
   Body: {"text": "Hello! How are you?"}
   Response: 202 Accepted

3. CLOSE STREAM (no more text)
   POST /v1/tts/stream/{id}/close
   Response: 200 OK

4. CANCEL STREAM
   DELETE /v1/tts/stream/{id}
   Response: 200 OK

5. RECEIVE AUDIO (long-lived connection)
   GET /v1/tts/stream/{id}/audio
   Response: Chunked PCM audio bytes
   Headers: X-Sample-Rate: 22000, X-Channels: 1

   On error: Returns error JSON instead of audio
   {"error": "generation_failed", "message": "..."}
```

### Connection Management

- Client uses `httpx.AsyncClient` with connection pooling
- Control operations (1-4) use short-lived requests with keep-alive
- Audio stream (5) is long-lived chunked response
- Idle timeout: 30 seconds of no activity → stream cleanup

---

## Server-Side Architecture

### Stream State

```python
@dataclass
class TTSStream:
    stream_id: str
    voice: str
    language: str
    created_at: float
    last_activity: float

    # Text management
    pending_text: list[str]           # Queue of text segments
    text_buffer: str                  # Accumulating buffer for incoming text
    text_buffer_start: float          # When buffering started

    # Audio tracking
    generated_audio_ms: float         # Total audio duration generated

    # State flags
    is_closed: bool                   # Client signaled no more text
    is_cancelled: bool                # Client cancelled
    has_error: bool                   # Generation failed
    error_message: str                # Error details

    # Constants
    BUFFER_THRESHOLD_MS: float = 500  # Switch to batch above this
    TEXT_BUFFER_TIMEOUT_MS: float = 100  # Max time to buffer text
    IDLE_TIMEOUT_S: float = 30        # Cleanup after inactivity

    @property
    def buffer_ms(self) -> float:
        """Virtual buffer: audio generated minus elapsed time."""
        elapsed = (time.time() - self.created_at) * 1000
        return self.generated_audio_ms - elapsed

    @property
    def should_use_batch(self) -> bool:
        """Determine generation mode based on buffer status."""
        return self.buffer_ms >= self.BUFFER_THRESHOLD_MS
```

### Generation Loop (Pseudocode)

```python
async def generate_audio(stream: TTSStream):
    """Main generation loop for a stream."""

    while not stream.is_cancelled:
        # Check for pending text
        text = await get_next_text_segment(stream)

        if text is None:
            if stream.is_closed:
                break  # Done
            await asyncio.sleep(0.01)
            continue

        try:
            # Choose generation mode based on buffer
            if stream.should_use_batch:
                logger.info(f"Using BATCH mode (buffer={stream.buffer_ms:.0f}ms)")
                audio_bytes = generate_batch(text, stream.voice, stream.language)
            else:
                logger.info(f"Using STREAMING mode (buffer={stream.buffer_ms:.0f}ms)")
                async for chunk in generate_streaming(text, stream.voice, stream.language):
                    yield chunk
                    stream.generated_audio_ms += chunk_duration_ms(chunk)
                continue  # Streaming already yielded

            # For batch mode, yield the complete audio
            yield audio_bytes
            stream.generated_audio_ms += audio_duration_ms(audio_bytes)

        except Exception as e:
            stream.has_error = True
            stream.error_message = str(e)
            raise

async def get_next_text_segment(stream: TTSStream) -> Optional[str]:
    """Get next text segment, handling buffering logic."""

    # If we have queued segments, return the first one
    if stream.pending_text:
        return stream.pending_text.pop(0)

    # Check if text buffer should be flushed
    if stream.text_buffer:
        buffer_age = (time.time() - stream.text_buffer_start) * 1000
        has_sentence_end = re.search(r'[.!?]\s*$', stream.text_buffer)

        if buffer_age >= stream.TEXT_BUFFER_TIMEOUT_MS or has_sentence_end:
            text = stream.text_buffer
            stream.text_buffer = ""
            return text

    return None
```

### Seamless Audio Boundaries

When concatenating audio from separate generations:

```python
class AudioBoundaryHandler:
    """Handle seamless transitions between audio segments."""

    CROSSFADE_MS = 30        # Crossfade duration
    TAIL_TRIM_MS = 50        # Trim from end of each segment

    def __init__(self, sample_rate: int = 22000):
        self.sample_rate = sample_rate
        self.prev_tail: Optional[np.ndarray] = None

    def process_segment(self, audio_bytes: bytes) -> bytes:
        """Process audio segment with crossfade at boundaries."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Trim tail (removes artifacts)
        trim_samples = int(self.TAIL_TRIM_MS * self.sample_rate / 1000)
        if len(audio) > trim_samples * 2:
            audio = audio[:-trim_samples]

        # Apply crossfade with previous segment
        if self.prev_tail is not None:
            crossfade_samples = int(self.CROSSFADE_MS * self.sample_rate / 1000)
            crossfade_samples = min(crossfade_samples, len(self.prev_tail), len(audio))

            if crossfade_samples > 0:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)

                # Blend the overlap region
                audio[:crossfade_samples] = (
                    self.prev_tail[-crossfade_samples:] * fade_out +
                    audio[:crossfade_samples] * fade_in
                )

        # Save tail for next segment
        tail_samples = int(self.CROSSFADE_MS * self.sample_rate / 1000)
        self.prev_tail = audio[-tail_samples:].copy()

        return audio.astype(np.int16).tobytes()
```

---

## Test Harness Design

### Components

```
tests/
├── adaptive_streaming/
│   ├── __init__.py
│   ├── client.py           # AdaptiveStreamClient class
│   ├── test_parallel.py    # Experiment 1: parallel generation
│   ├── test_adaptive.py    # Experiment 2: adaptive streaming
│   └── metrics.py          # Metrics collection and reporting
```

### AdaptiveStreamClient

```python
class AdaptiveStreamClient:
    """Test client for adaptive streaming TTS."""

    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        # Connection pooling for efficiency
        self._client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=10)
        )

    async def create_stream(self, voice: str = "aria", language: str = "en") -> str:
        """Create a new TTS stream. Returns stream_id."""
        response = await self._client.post(
            f"{self.server_url}/v1/tts/stream",
            json={"voice": voice, "language": language}
        )
        response.raise_for_status()
        return response.json()["stream_id"]

    async def append_text(self, stream_id: str, text: str) -> None:
        """Append text to a stream."""
        response = await self._client.post(
            f"{self.server_url}/v1/tts/stream/{stream_id}/append",
            json={"text": text}
        )
        response.raise_for_status()

    async def close_stream(self, stream_id: str) -> None:
        """Signal no more text will be appended."""
        response = await self._client.post(
            f"{self.server_url}/v1/tts/stream/{stream_id}/close"
        )
        response.raise_for_status()

    async def cancel_stream(self, stream_id: str) -> None:
        """Cancel a stream."""
        response = await self._client.delete(
            f"{self.server_url}/v1/tts/stream/{stream_id}"
        )
        response.raise_for_status()

    async def receive_audio(
        self,
        stream_id: str,
        on_chunk: Optional[Callable[[bytes, float], None]] = None
    ) -> AsyncGenerator[bytes, None]:
        """Receive audio bytes from a stream."""
        async with self._client.stream(
            "GET",
            f"{self.server_url}/v1/tts/stream/{stream_id}/audio"
        ) as response:
            if response.status_code != 200:
                error = await response.aread()
                raise RuntimeError(f"Stream error: {error.decode()}")

            async for chunk in response.aiter_bytes():
                if on_chunk:
                    on_chunk(chunk, time.time())
                yield chunk

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
```

### Metrics Collection

```python
@dataclass
class StreamMetrics:
    """Metrics for a single stream."""
    stream_id: str
    text_segments: list[str]

    # Timing
    create_time: float
    first_audio_time: float          # TTFB
    completion_time: float

    # Per-chunk metrics
    chunk_times: list[float]         # Arrival time of each chunk
    chunk_sizes: list[int]           # Bytes per chunk

    # Derived metrics
    @property
    def ttfb_ms(self) -> float:
        return (self.first_audio_time - self.create_time) * 1000

    @property
    def total_time_ms(self) -> float:
        return (self.completion_time - self.create_time) * 1000

    @property
    def audio_duration_ms(self) -> float:
        total_bytes = sum(self.chunk_sizes)
        return total_bytes / (22000 * 2) * 1000  # 16-bit, 22kHz

    @property
    def rtf(self) -> float:
        return (self.total_time_ms / 1000) / (self.audio_duration_ms / 1000)

    @property
    def buffer_over_time(self) -> list[tuple[float, float]]:
        """Calculate buffer status at each chunk arrival."""
        results = []
        audio_ms = 0
        for t, size in zip(self.chunk_times, self.chunk_sizes):
            elapsed_ms = (t - self.create_time) * 1000
            chunk_audio_ms = size / (22000 * 2) * 1000
            audio_ms += chunk_audio_ms
            buffer_ms = audio_ms - elapsed_ms
            results.append((elapsed_ms, buffer_ms))
        return results
```

### Test Scenarios

```python
# Experiment 1: Parallel Generation
async def test_parallel_generation():
    """Test concurrent generation quality and performance."""

    sentences = [
        "Hello! How can I help you today?",
        "The weather is beautiful this morning.",
        "I'd be happy to explain that concept.",
        "Let me think about the best approach.",
    ]

    # Baseline: Sequential
    sequential_metrics = []
    for i, text in enumerate(sentences):
        metrics = await generate_single(text, f"sequential_{i+1:02d}.wav")
        sequential_metrics.append(metrics)

    # Parallel-2: Two concurrent
    parallel2_metrics = await generate_parallel(sentences[:2], "parallel2")

    # Parallel-4: Four concurrent
    parallel4_metrics = await generate_parallel(sentences, "parallel4")

    # Report
    print_comparison(sequential_metrics, parallel2_metrics, parallel4_metrics)

# Experiment 2: Adaptive Streaming
async def test_adaptive_streaming():
    """Test adaptive buffer strategy."""

    # Scenario A: All text at once
    await test_scenario_bulk_text()

    # Scenario B: Sentence-by-sentence
    await test_scenario_sentence_stream()

    # Scenario C: Word-by-word (simulating LLM tokens)
    await test_scenario_token_stream()

    # Scenario D: Mixed (some fast, some slow arrivals)
    await test_scenario_mixed()
```

---

## Implementation Order

### Phase 1: Parallel Generation Test (Experiment 1)
1. Create `tests/adaptive_streaming/test_parallel.py`
2. Implement parallel request logic using existing streaming endpoint
3. Collect metrics and save audio files
4. Run tests and analyze results
5. Document findings

### Phase 2: Adaptive Server Implementation
1. Create stream state management in `tts_server.py`
2. Implement five HTTP endpoints
3. Implement text buffering logic
4. Implement adaptive mode selection
5. Implement audio boundary handling (crossfade)
6. Add logging for debugging

### Phase 3: Test Harness
1. Create `AdaptiveStreamClient` class
2. Create metrics collection utilities
3. Implement test scenarios
4. Run tests and collect audio samples
5. Document results in checkpoint

### Phase 4: Documentation
1. Document test harness in checkpoint document
2. Include all metrics and findings
3. Include code snippets for key components
4. List audio samples for quality evaluation

---

## Success Criteria

### Experiment 1: Parallel Generation
- [x] No significant quality degradation at 2x concurrency (confirmed: audio quality fine)
- [ ] TTFB regression < 50% at 2x concurrency (failed: 81% regression, but quality OK)
- [x] Document any issues found (serialization required due to GPU contention)

### Experiment 2: Adaptive Streaming
- [x] TTFB < 200ms for first audio (achieved: 150-173ms)
- [x] No audio gaps (buffer never goes negative)
- [x] Seamless transitions between segments (no audible clicks/pops)
- [x] Batch mode engages when buffer > 500ms
- [x] All five client operations work correctly
- [x] Idle timeout cleans up stale streams

### Phase 2 Results (2025-12-23)

**All adaptive streaming tests passed.**

#### Performance Metrics

| Scenario        | TTFB    | Total   | Audio   | RTF   |
|-----------------|---------|---------|---------|-------|
| Bulk text       | 173ms   | 7434ms  | 14289ms | 0.52x |
| Sentence stream | 150ms   | 2202ms  | 7025ms  | 0.31x |
| Token stream    | 254ms   | 3551ms  | 9728ms  | 0.37x |
| Mixed timing    | 153ms   | 2957ms  | 9116ms  | 0.32x |

#### Key Observations

1. **Fast TTFB achieved**: 150-173ms for most scenarios, meeting < 200ms target
2. **Token streaming slightly slower**: 254ms TTFB due to text buffering (100ms timeout)
3. **Real-time generation maintained**: All RTFs well under 1.0x (0.31-0.52x)
4. **Cancellation works correctly**: Stream can be cancelled mid-generation

#### Clarification: Bulk Text vs Batch Generation

| Mode | Text Input | Audio Output | TTFB |
|------|-----------|--------------|------|
| **Batch** (`/v1/audio/speech`) | All at once | All at once (wait for complete) | ~600-3000ms |
| **Bulk text** (adaptive stream) | All at once | Streamed in chunks | ~173ms |

"Bulk text" scenario uses adaptive streaming - audio is streamed out in chunks as generated.
The server uses streaming mode for the first segment (fast TTFB), then may switch to batch
mode for later segments if buffer > 500ms. This is NOT the same as batch generation which
waits for complete audio before returning anything.

#### Implementation Notes

- Fixed empty audio array handling in `streaming_tts.py`
- Stream cleanup moved from `finally` block to explicit completion/error handlers
- Text buffering with 100ms timeout effectively batches LLM tokens

---

## Phase 3: Pipecat Integration

### Current Architecture Analysis

**Pipecat TTS Flow:**
```
LLM (tokens) → [Pipecat aggregation] → TTSService.run_tts(sentence) → Audio frames
```

- Pipecat's `TTSService` base class aggregates LLM tokens into sentences
- Each sentence triggers one `run_tts(text)` call
- `run_tts` is designed for independent, one-shot synthesis
- Current `MagpieStreamingTTSService` achieves ~150ms TTFB per sentence

**Key Insight:** Since Pipecat already aggregates text into sentences before calling `run_tts`,
each call receives complete text (not tokens). This is similar to our "bulk text" test scenario.

### Integration Approaches

#### Approach A: Simple Drop-in (Conservative)

Each `run_tts` call uses adaptive streaming API independently:
```python
async def run_tts(self, text: str):
    stream_id = await create_stream()
    await append_text(stream_id, text)
    await close_stream(stream_id)
    async for chunk in receive_audio(stream_id):
        yield TTSAudioRawFrame(audio=chunk, ...)
```

**Pros:**
- Minimal changes to existing code
- No changes to Pipecat frame handling
- Low regression risk
- Easy to test and validate

**Cons:**
- Multiple HTTP calls per sentence (overhead)
- Doesn't leverage cross-segment mode switching
- Similar TTFB to current streaming (~150-173ms)

**Expected Result:** Functionally equivalent to current implementation

#### Approach B: Persistent Stream (Advanced)

Keep a single stream open across entire LLM response:
```
LLM starts → create_stream()
Sentence 1 → append_text() → audio streams...
Sentence 2 → append_text() → audio continues (batch mode if buffer healthy)
Sentence 3 → append_text() → audio continues...
LLM ends → close_stream()
```

**Pros:**
- First sentence: streaming mode (~150ms TTFB)
- Later sentences: batch mode (potentially higher quality)
- Single HTTP connection for audio
- Leverages full adaptive streaming benefits

**Cons:**
- Requires understanding Pipecat's frame lifecycle
- Need to detect LLM start/stop events
- More complex interruption handling
- Higher risk of edge case bugs

**Implementation requirements:**
1. Track active stream across `run_tts` calls
2. Detect LLM response start (create stream)
3. Detect LLM response end (close stream)
4. Handle interruptions (cancel stream)
5. Coordinate audio receiving with text appending

### Regression Risk Analysis

| Risk | Mitigation |
|------|------------|
| Stream not closed → resource leak | Idle timeout (30s), explicit cleanup |
| Audio gaps between segments | Buffer threshold ensures continuous playback |
| Interruption handling broken | Cancel stream on interrupt frame |
| TTFB regression | First segment always uses streaming mode |
| Connection issues | Error handling, reconnection logic |

### Pipecat TTSService Analysis

**Key Constructor Parameter (line 90):**
```python
aggregate_sentences: bool = True
```

When `aggregate_sentences=False` (lines 538-541):
```python
if not self._aggregate_sentences:
    text = frame.text
    includes_inter_frame_spaces = frame.includes_inter_frame_spaces
    aggregated_by = "token"
```

Each token from LLM is passed directly without sentence aggregation!

**Key Frames Identified:**
- `LLMFullResponseStartFrame` - LLM response begins
- `LLMFullResponseEndFrame` - LLM response ends (flushes pending text, resets aggregator)
- `InterruptionFrame` - User interruption (handled via `_handle_interruption`)
- `TextFrame` - Individual text chunks from LLM

**Frame Processing Flow (process_frame, lines 394-459):**
```
TextFrame → _process_text_frame → _push_tts_frames → run_tts(text)
LLMFullResponseEndFrame → flush pending → reset aggregator
InterruptionFrame → _handle_interruption → cancel processing
```

### Implementation Design

**Problem:** With `aggregate_sentences=False`, `run_tts` would be called per-token.
This doesn't fit our stream model (one stream per LLM response, incremental appends).

**Solution:** Override `process_frame` to manage stream lifecycle directly.

```python
class MagpieAdaptiveTTSService(TTSService):
    def __init__(self, ...):
        super().__init__(
            aggregate_sentences=False,  # Disable Pipecat aggregation
            push_text_frames=True,
            ...
        )
        self._active_stream_id: Optional[str] = None
        self._audio_receiver_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, TextFrame) and not frame.skip_tts:
            # Append to stream (creates stream on first text)
            await self._append_to_stream(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Close stream, wait for audio to complete
            await self._close_stream()
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            # Cancel stream immediately
            await self._cancel_stream()
            await self.push_frame(frame, direction)

        else:
            await super().process_frame(frame, direction)

    async def _append_to_stream(self, text: str):
        if not self._active_stream_id:
            await self._create_stream()
        await self._http_client.post(
            f"{self._server_url}/v1/tts/stream/{self._active_stream_id}/append",
            json={"text": text}
        )

    async def _create_stream(self):
        response = await self._http_client.post(...)
        self._active_stream_id = response.json()["stream_id"]
        # Start audio receiver in background
        self._audio_receiver_task = asyncio.create_task(self._receive_audio())

    async def _receive_audio(self):
        first_chunk = True
        async with self._http_client.stream("GET", f".../audio") as response:
            async for chunk in response.aiter_bytes():
                if first_chunk:
                    await self.push_frame(TTSStartedFrame())
                    await self.stop_ttfb_metrics()
                    first_chunk = False
                await self.push_frame(TTSAudioRawFrame(audio=chunk, ...))
        await self.push_frame(TTSStoppedFrame())

    async def _close_stream(self):
        if self._active_stream_id:
            await self._http_client.post(f".../close")
            await self._audio_receiver_task  # Wait for audio completion
            self._active_stream_id = None

    async def _cancel_stream(self):
        if self._active_stream_id:
            await self._http_client.delete(f".../{self._active_stream_id}")
            self._audio_receiver_task.cancel()
            self._active_stream_id = None

    async def run_tts(self, text: str):
        # Not used in this implementation
        raise NotImplementedError("Use process_frame directly")
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Create stream on | First TextFrame | Lazy creation, handles empty responses |
| TTSStartedFrame | First audio chunk | Accurate TTFB measurement |
| TTSStoppedFrame | After all audio | Correct end signaling |
| Close stream | LLMFullResponseEndFrame | Natural end boundary |
| Cancel stream | InterruptionFrame | Immediate cancellation |

### Edge Cases to Handle

1. **Empty LLM response**: LLMFullResponseEndFrame with no prior TextFrames
   - Solution: Only close stream if one was created

2. **Rapid interruption**: InterruptionFrame before any audio received
   - Solution: Cancel receiver task, cleanup stream state

3. **Server connection error**: HTTP request fails mid-stream
   - Solution: ErrorFrame, cleanup state, allow retry

4. **Overlapping responses**: New LLM response before previous completes
   - Solution: Close/cancel previous stream before creating new

### Open Questions (Resolved)

~~1. Which approach to implement first?~~
   - **Answer: Approach B with `aggregate_sentences=False`**

~~2. How does Pipecat signal LLM lifecycle?~~
   - **Answer: `LLMFullResponseStartFrame`, `LLMFullResponseEndFrame`, `InterruptionFrame`**

~~3. How does Pipecat handle interruptions?~~
   - **Answer: `InterruptionFrame` + `_handle_interruption` method**

### Remaining Questions

1. **TTSTextFrame handling**: Should we push TTSTextFrame for context tracking?
   - Current design skips this, but Pipecat uses it for assistant context

2. **Metrics integration**: How to properly integrate TTFB/processing metrics?
   - Need to call `start_ttfb_metrics()`, `stop_ttfb_metrics()` at right times

3. **Replace or add?**: Should this replace `MagpieStreamingTTSService` or be a new option?

---

## DeepgramTTSService Analysis (Reference Implementation)

**Key Pattern Discovery:**

DeepgramTTSService (WebSocket version) uses exactly the pattern we need:

```python
# run_tts just sends text, doesn't yield audio directly
async def run_tts(self, text: str):
    yield TTSStartedFrame()
    await self._websocket.send({"type": "Speak", "text": text})
    yield None  # Placeholder - audio comes from background task

# Background task receives audio and pushes frames
async def _receive_messages(self):
    async for message in self._websocket:
        if isinstance(message, bytes):
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSAudioRawFrame(message, ...))

# Flush on LLM response end
async def process_frame(self, frame, direction):
    if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
        await self.flush_audio()  # Sends "Flush" command

# Clear on interruption
async def _handle_interruption(self, frame, direction):
    await self._websocket.send({"type": "Clear"})  # Stop audio
```

**Key insights:**
1. `run_tts` yields TTSStartedFrame but audio comes from background task
2. `push_text_frames=True` (default) - base class handles TTSTextFrame
3. Flush on LLMFullResponseEndFrame signals "generate remaining audio"
4. Clear on InterruptionFrame signals "stop immediately"

---

## Adaptive Strategy Analysis

### Current Server Implementation

```python
BUFFER_THRESHOLD_MS = 500  # Switch to batch above this

def should_use_batch(self) -> bool:
    if self.is_first_segment:
        return False  # Always streaming for TTFB
    return self.buffer_ms >= BUFFER_THRESHOLD_MS
```

### Problem: 500ms Threshold Too Aggressive

From measurements:
- Batch for short sentence: ~600ms total generation
- Buffer = 500ms, batch takes 600ms
- During generation: 600ms playback occurs
- Buffer: 500ms → -100ms = **UNDERRUN**

### Proposed Fix: Increase Threshold

```python
BUFFER_THRESHOLD_MS = 1000  # Need 1s buffer to safely use batch
```

With 1000ms:
- Buffer = 1000ms, batch takes 600ms
- Buffer: 1000ms → 400ms = **SAFE**

### Quality Consideration: Batch >> Streaming

User confirmed: "batch seems to be much higher quality"

Quality differences:
- **Batch**: Full utterance context for prosody planning
- **Batch**: No chunk boundary artifacts
- **Streaming**: Limited lookahead, potential chunk artifacts

**Goal**: Maximize batch usage while maintaining real-time playback

### Advanced: Text-Length-Aware Threshold

```python
def should_use_batch(self) -> bool:
    if self.is_first_segment:
        return False

    # Estimate generation time: ~50 chars ≈ 3s audio ≈ 1s gen (RTF 0.33)
    estimated_gen_ms = len(pending_text) * 20
    safety_margin = 300

    return self.buffer_ms >= estimated_gen_ms + safety_margin
```

### Three-Tier Approach (User Suggestion)

| Buffer Status | Mode | Rationale |
|---------------|------|-----------|
| First segment | Streaming (small chunks) | Fast TTFB |
| Buffer < 500ms | Streaming | Rebuild buffer quickly |
| Buffer ≥ 1000ms | Batch | Quality when we can afford latency |

### Why Larger Streaming Chunks Don't Help Quality

**How streaming works (from `streaming_tts.py`):**
- Model generates **frame-by-frame autoregressively** with KV cache
- `chunk_size_frames` just controls how many frames accumulate before yielding
- The model doesn't "see ahead" - it generates each frame based on prior frames

**Why batch is higher quality:**
- Batch mode: model sees **entire text at once**, plans prosody/pacing holistically
- Streaming mode: model generates incrementally with limited lookahead

**Conclusion:** Larger streaming chunks won't improve quality because:
1. Model still generates frame-by-frame (no additional context)
2. KV cache is maintained across frames regardless of chunk size
3. Quality difference is fundamentally about full-utterance context, not chunk size

### Final Adaptive Strategy

| Buffer Status | Mode | Text Buffer | Rationale |
|---------------|------|-------------|-----------|
| First segment | Streaming | Adaptive (shorter) | Fast TTFB to start audio |
| < 500ms | Streaming | Adaptive (shorter) | Rebuild buffer quickly |
| 500-1000ms | Streaming | Normal (100ms) | Can't afford batch, maintain playback |
| ≥ 1000ms | Batch | Normal (100ms) | Quality when we can afford latency |

The 500-1000ms zone is "waiting for batch opportunity" - streaming maintains playback while building toward batch.

### Configuration Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BUFFER_THRESHOLD_MS` | 1000ms | Safe margin for batch generation |
| `first_chunk_frames` | 8 | ~140ms TTFB, good balance |
| `chunk_size_frames` | 16 | ~740ms chunks |
| `TEXT_BUFFER_TIMEOUT_MS` | Adaptive | Shorter when buffer low, normal otherwise |

### Implementation Plan

**Step 1: Update server threshold**
- Change `BUFFER_THRESHOLD_MS` from 500ms to 1000ms in `adaptive_stream.py`
- Make text buffer timeout adaptive based on buffer status

**Step 2: Create Pipecat adaptive TTS service**
- New file: `pipecat_bots/magpie_adaptive_tts.py`
- Follow DeepgramTTSService pattern (background audio receiver)
- Use `aggregate_sentences=False` to receive tokens directly
- Manage stream lifecycle based on LLM response frames

**Step 3: Update bot.py**
- Add `USE_ADAPTIVE_TTS` environment variable
- Keep existing `MagpieStreamingTTSService` as fallback
- Select service based on environment variable

**Step 4: Test and validate**
- Run with real LLM to test token-by-token flow
- Verify TTFB, quality, interruption handling
- Compare with existing streaming service

---

## Known Issues / Future Work

### Connection Error Handling (TODO)

During Phase 3 testing, we discovered that when a service (ASR, LLM, or TTS) fails to connect
at startup, the pipeline continues running but emits "StartFrame not received yet" errors.
This causes confusing error messages that appear to come from unrelated processors.

**Recommended fix:** The pipeline should exit cleanly if connection to any critical service
(ASR, LLM, TTS) fails during startup. This would provide clear error messages and prevent
the cascading "StartFrame not received" errors.

**Files to investigate:**
- `pipecat_bots/bot.py` - Add connection validation before pipeline start
- `nvidia_stt.py` - Improve error handling in `_connect_websocket`

---

## File Structure After Implementation

```
src/nemotron_speech/
├── tts_server.py              # Updated with adaptive streaming endpoints
├── streaming_tts.py           # Existing streaming implementation
├── adaptive_stream.py         # DONE: Stream state and adaptive logic
└── audio_boundary.py          # DONE: Crossfade and boundary handling

pipecat_bots/
├── bot.py                     # UPDATE: Add USE_ADAPTIVE_TTS selection
├── magpie_streaming_tts.py    # Existing streaming service (keep as fallback)
├── magpie_http_tts.py         # Existing batch service
└── magpie_adaptive_tts.py     # NEW: Adaptive streaming service (Phase 3)

tests/adaptive_streaming/
├── __init__.py                # DONE
├── client.py                  # DONE: AdaptiveStreamClient
├── test_parallel.py           # DONE: Experiment 1
└── test_adaptive.py           # DONE: Experiment 2

samples_parallel/              # DONE: Experiment 1 outputs (12 wav files)
samples_adaptive/              # DONE: Experiment 2 outputs (4 wav files)
```

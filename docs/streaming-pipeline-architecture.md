# Streaming Voice Pipeline Architecture

This document explains in detail how the STT, LLM, and TTS components work together to deliver fast voice-to-voice response times in the interleaved streaming pipeline.

## Code Files Overview

| File | Description |
|------|-------------|
| **Client-side (Pipecat)** | |
| `pipecat_bots/nvidia_stt.py` | WebSocket STT client that streams audio to Parakeet ASR and receives transcriptions |
| `pipecat_bots/llama_cpp_buffered_llm.py` | Buffered HTTP client to llama.cpp with single-slot operation for 100% KV cache reuse |
| `pipecat_bots/sentence_buffer.py` | Sentence boundary detection and text accumulation for buffered LLM output |
| `pipecat_bots/magpie_websocket_tts.py` | WebSocket TTS client with adaptive streaming/batch mode and sentence splitting |
| `pipecat_bots/frames.py` | Shared frame types (ChunkedLLMContinueGenerationFrame) to avoid circular imports |
| `pipecat_bots/v2v_metrics.py` | Measures voice-to-voice response time (VADUserStoppedSpeaking → BotStartedSpeaking) |
| `pipecat_bots/bot_interleaved_streaming.py` | Main bot that assembles the pipeline with all services |
| **Server-side (Inference)** | |
| `src/nemotron_speech/server.py` | WebSocket ASR server running NVIDIA Parakeet with true incremental streaming |
| `src/nemotron_speech/tts_server.py` | FastAPI server with WebSocket endpoint for adaptive TTS streaming |
| `src/nemotron_speech/streaming_tts.py` | Frame-by-frame streaming TTS inference with configurable presets |
| `src/nemotron_speech/adaptive_stream.py` | Stream state management for TTS sessions |

---

## Introduction: Optimizing Voice-to-Voice Latency

The key to low voice-to-voice latency is **pipelining** - starting each stage as early as possible rather than waiting for the previous stage to complete. This pipeline achieves ~500-700ms voice-to-voice response time through three optimizations:

### 1. Streaming STT with Frame Ordering Fix

The STT service streams audio continuously to the ASR server. When the user stops speaking (detected by VAD after ~200ms of silence), a reset signal triggers final transcription. A critical frame ordering fix ensures the `TranscriptionFrame` arrives at the aggregator *before* `UserStoppedSpeakingFrame`, preventing a 500ms aggregation timeout.

### 2. Buffered LLM with Sentence Boundaries

Instead of waiting for the complete LLM response, the buffered LLM service emits text at **sentence boundaries**. The first segment uses a 24-token limit for fast time-to-first-chunk, while subsequent segments accumulate up to 96 tokens waiting for natural sentence endings. Single-slot operation achieves **100% KV cache reuse** across turns, eliminating context re-evaluation. This enables TTS to start speaking the first sentence while the LLM is still generating.

### 3. Adaptive TTS with Streaming First Segment

The TTS service uses **adaptive mode**: the first segment uses streaming mode (~370ms TTFB) while subsequent segments use batch mode for higher quality. This optimizes for fast first-audio while maintaining quality for the rest of the response.

### End-to-End Timing

```
User speaks     VAD detects    STT sends     LLM receives    LLM first      TTS first
  "Hello"       silence        final text    transcript     segment ready   audio out
    │              │               │              │              │              │
    ├──────────────┤───────────────┤──────────────┤──────────────┤──────────────┤
    │   ~speech    │   ~200ms      │   ~30-50ms   │   ~0ms       │  ~100-150ms  │  ~370ms
    │   duration   │   VAD delay   │   STT proc   │  (cached)    │  LLM TTFB    │  TTS TTFB
    │              │               │              │              │              │
    └──────────────┴───────────────┴──────────────┴──────────────┴──────────────┘
                                                                        │
                                          Total V2V: ~500-700ms ────────┘
```

---

## STT Service: NVIDIA Parakeet Streaming ASR

### Server-Side Implementation (`src/nemotron_speech/server.py`)

The ASR server uses NVIDIA's Parakeet model with true incremental streaming. It processes audio in 160ms chunks (16 mel frames) and maintains encoder/decoder cache state across chunks for continuous transcription.

#### Encoder Context Configuration

The Conformer encoder uses attention with configurable left and right context:

```
att_context_size = [70, 1]  # [left_context, right_context]
```

| Parameter | Value | Duration | Purpose |
|-----------|-------|----------|---------|
| **Left context** | 70 frames | 700ms | Past audio for accuracy (fixed) |
| **Right context** | 1 frame | 160ms | Future lookahead (configurable) |

The left context of 70 frames is fixed and provides the encoder with substantial history for accurate recognition. The right context controls the latency/accuracy tradeoff and can be changed at runtime without retraining:

| Right Context | Latency | Description |
|---------------|---------|-------------|
| 0 | ~80ms | Ultra-low latency, slightly lower accuracy |
| **1** | **~160ms** | **Low latency (default, recommended)** |
| 6 | ~560ms | Higher accuracy for noisy environments |
| 13 | ~1.12s | Maximum accuracy, conversational use only |

Configure via CLI:

```bash
python -m nemotron_speech.server --right-context 1   # 160ms (default)
python -m nemotron_speech.server --right-context 0   # 80ms ultra-low
python -m nemotron_speech.server --right-context 6   # 560ms balanced
```

#### Streaming Chunk Parameters

These values come from the model's streaming configuration and were tuned for optimal latency:

| Parameter | Value | Duration | Purpose |
|-----------|-------|----------|---------|
| `shift_frames` | 16 | 160ms | Inference runs every 16 mel frames |
| `pre_encode_cache` | 9 | 90ms | Overlap frames for chunk boundary context |
| `hop_samples` | 160 | 10ms | Samples per mel frame at 16kHz |

The 160ms chunk size balances inference efficiency (larger chunks = better GPU utilization) against latency (smaller chunks = faster interim results).

#### VAD Alignment with ASR Trailing Context

The VAD `stop_secs` parameter is carefully aligned with ASR requirements:

```python
VAD_STOP_SECS = 0.2  # Fire after 200ms of silence
```

**Why this matters:** The encoder needs trailing silence to finalize the last word(s). The required padding is:

```
final_padding = (right_context + 1) * shift_frames * hop_samples
             = (1 + 1) * 16 * 160
             = 5120 samples = 320ms
```

By setting VAD to fire after ~200ms of silence, most of the required trailing context has already been streamed to the server. The server adds the remaining padding on reset, minimizing additional delay for final transcription.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ASR Server                                         │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ WebSocket    │───►│ Audio        │───►│ Preprocessor │───►│ Streaming  │  │
│  │ Handler      │    │ Accumulator  │    │ (Mel Spec)   │    │ Encoder    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘  │
│         │                                                           │        │
│         │ reset signal                                              ▼        │
│         │                                                    ┌────────────┐  │
│         └───────────────────────────────────────────────────►│ Decoder    │  │
│                                                              │ (Greedy)   │  │
│                                                              └────────────┘  │
│                                                                     │        │
│                                                                     ▼        │
│                                                              ┌────────────┐  │
│                                                              │ Transcript │  │
│                                                              │ Output     │  │
│                                                              └────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Key Implementation Details

**Session State (`ASRSession`):**
- `accumulated_audio`: All audio received since last reset
- `emitted_frames`: Number of mel frames already processed by encoder
- `cache_last_channel/time/len`: Encoder cache tensors for incremental processing
- `previous_hypotheses`, `pred_out_stream`: Decoder state for continuous decoding

**Streaming Inference Loop:**

1. Audio bytes arrive via WebSocket → convert to float32 → accumulate
2. When enough audio for a new 160ms chunk:
   - Preprocess ALL accumulated audio to mel spectrogram
   - Extract only NEW frames (skip already-emitted frames)
   - Run `conformer_stream_step()` with cached encoder state
   - Emit interim transcript if changed
3. On reset signal:
   - Pad with ~320ms silence for trailing context
   - Run final chunk with `keep_all_outputs=True`
   - Emit final transcript

**Reset Handling for Final Transcription:**

The VAD fires after ~200ms of silence, but the encoder needs trailing context to finalize the last word. The server pads with `(right_context + 1) * shift_frames` of silence (~320ms with default settings) before the final transcription pass.

### Client-Side Implementation (`pipecat_bots/nvidia_stt.py`)

The Pipecat client extends `WebsocketSTTService` to handle the streaming protocol and implement a critical frame ordering fix.

#### Frame Ordering Problem and Solution

The LLM context aggregator expects frames in this order:
1. `TranscriptionFrame` (final text)
2. `UserStoppedSpeakingFrame`

Without intervention, the transport may send `UserStoppedSpeakingFrame` before the server returns the final transcript, causing a 500ms aggregation timeout.

**Solution:** Hold `UserStoppedSpeakingFrame` until the final transcript arrives:

```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    if isinstance(frame, VADUserStoppedSpeakingFrame):
        self._waiting_for_final = True
        await self._send_reset()  # Trigger final transcription

    if isinstance(frame, UserStoppedSpeakingFrame):
        if self._waiting_for_final:
            # Hold until final transcript arrives
            self._pending_user_stopped_frame = frame
            return
```

When the final transcript arrives via WebSocket:
```python
async def _handle_transcript(self, data: dict):
    if is_final:
        await self.push_frame(TranscriptionFrame(...))  # Push transcript first
        await self._release_pending_frame()  # Then release held UserStoppedSpeakingFrame
```

#### Timing Diagram

```
Audio Stream                    VAD            STT Service              Aggregator
     │                           │                  │                       │
     │────audio bytes───────────►│                  │                       │
     │                           │                  │                       │
     │                      detects silence         │                       │
     │                      (~200ms)                │                       │
     │                           │                  │                       │
     │◄──VADUserStoppedSpeaking──│                  │                       │
     │                           │                  │                       │
     │──────────────────reset signal──────────────►│                       │
     │                           │                  │                       │
     │                           │            +320ms padding               │
     │                           │            final inference               │
     │                           │                  │                       │
     │◄─────────────────TranscriptionFrame─────────│                       │
     │                           │                  │──TranscriptionFrame──►│
     │                           │                  │                       │
     │◄──UserStoppedSpeaking─────│                  │──UserStoppedSpeaking─►│
     │   (held until transcript) │                  │  (released after)     │
     │                           │                  │                       │
```

---

## LLM Service: Buffered Sentence-Boundary Streaming

### Server-Side: llama.cpp

The LLM uses llama.cpp's HTTP API directly (no custom server). Key features used:

- **SSE Streaming**: `/completion` endpoint with `stream: true`
- **KV Cache**: `cache_prompt: true` + `id_slot` pinning for 100% cache reuse
- **Single Slot**: `--parallel 1` for maximum cache efficiency (no slot contention)
- **Stop Tokens**: `<|im_end|>` for ChatML format

### Client-Side Implementation (`pipecat_bots/llama_cpp_buffered_llm.py`)

The buffered LLM service uses a **run-to-completion** approach that achieves 100% KV cache reuse across conversation turns. Unlike the previous chunked approach that cancelled mid-stream, this service lets each generation complete fully, accumulates output in a `SentenceBuffer`, and emits text at natural sentence boundaries.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LlamaCppBufferedLLMService                             │
│                                                                             │
│  ┌────────────────┐      ┌─────────────────┐      ┌───────────────────┐    │
│  │  LLM Generator │      │  SentenceBuffer │      │   TTS Emitter     │    │
│  │                │      │                 │      │                   │    │
│  │  - Single slot │─────►│  - Accumulates  │─────►│  - Emits complete │    │
│  │  - max_tokens  │      │  - Extracts at  │      │    sentences      │    │
│  │  - Runs to     │      │    boundaries   │      │  - Waits for      │    │
│  │    completion  │      │  - Keeps tail   │      │    continue       │    │
│  └────────────────┘      └─────────────────┘      └───────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Benefits vs Previous Chunked Approach

| Aspect | Old (Chunked) | New (Buffered) |
|--------|---------------|----------------|
| Slots | 2 (alternating) | 1 (single) |
| GPU Memory | Higher (`--parallel 2`) | Lower (`--parallel 1`) |
| KV Cache Reuse | ~80-90% | **100%** |
| Mid-stream Cancel | Yes (race condition risk) | No (runs to completion) |
| GGML_ASSERT Crashes | Possible | Eliminated |

#### First Segment Optimization

The first segment uses tight token bounds for fast TTFB:

```python
# First segment: quick TTFB, single generation then emit
first_segment_max_tokens: int = 24
first_segment_hard_max_tokens: int = 24

# Subsequent segments: allow accumulation for complete sentences
segment_max_tokens: int = 32
segment_hard_max_tokens: int = 96
```

#### Sentence Buffer (`pipecat_bots/sentence_buffer.py`)

The `SentenceBuffer` class accumulates LLM output and extracts text at natural boundaries:

```python
class SentenceBuffer:
    def extract_complete_sentences(self) -> Optional[str]:
        """Extract all complete sentences, keep incomplete tail.

        Finds the LAST sentence boundary (.!? followed by space) and returns
        all text up to and including that boundary.
        """
        pattern = r'[.!?]["\'\)]*\s'
        matches = list(re.finditer(pattern, self.text))
        if not matches:
            return None

        last_match = matches[-1]
        boundary = last_match.end()
        sentences = self.text[:boundary].lstrip()
        self.text = self.text[boundary:]  # Keep incomplete tail
        return sentences if sentences else None

    def extract_at_boundary(self) -> str:
        """Force extraction at best boundary when hitting token limit.

        Priority: sentence > clause (, ;) > word > everything
        """
```

#### Generation Loop

The service runs a simple loop: generate → buffer → check → emit:

```python
while not self._cancelled:
    # Step 1: Generate tokens (runs to completion)
    new_text, new_tokens, hit_eos = await self._generate(max_tokens)
    self._buffer.add(new_text, new_tokens)

    # Step 2: Check buffer and decide action
    sentences = self._buffer.extract_complete_sentences()
    if sentences:
        await self._emit_and_wait(sentences)  # Emit and wait for TTS
        continue

    if self._buffer.token_count >= hard_max_tokens:
        text = self._buffer.extract_at_boundary()  # Force at best boundary
        await self._emit_and_wait(text)
        continue

    if hit_eos:
        # Emit remainder and finish
        break
```

#### TTS Synchronization

The LLM waits for TTS to finish each segment before generating the next, preventing audio buffer overflow:

```python
async def _emit_and_wait(self, text: str):
    await self.push_frame(LLMTextFrame(text=text))
    self._buffer.reset_token_count()

    await asyncio.wait_for(self._continue_event.wait(), timeout=30.0)
    self._continue_event.clear()
```

TTS sends `ChunkedLLMContinueGenerationFrame` (from `frames.py`) upstream when a segment completes.

#### 100% KV Cache Reuse

With single-slot operation, the KV cache is never invalidated between turns:

```
Turn 1: [system prompt][user: Hello][assistant: Hi there!]
        └──────────────── cached ────────────────────────┘

Turn 2: [system prompt][user: Hello][assistant: Hi there!][user: How are you?][assistant: ...]
        └──────────────── 100% cache hit ─────────────────┘└── new tokens ──────────────────┘
```

The `LLMSlotMetricsFrame` tracks cache performance:
- `first_segment_cache_hit_ratio`: High (>90%) = Fast TTFB (context cached)
- `cache_hit_ratio`: Overall cache efficiency across all generations

#### Timing Diagram

```
User Input                   LLM Service                      TTS Service
     │                            │                                │
     │──LLMMessagesFrame─────────►│                                │
     │                            │                                │
     │                       format ChatML                         │
     │                       generate 24 tokens (runs to end)      │
     │                            │                                │
     │                       buffer.add(text)                      │
     │                       buffer.extract_complete_sentences()   │
     │                            │                                │
     │◄──────LLMTextFrame─────────│ "Hello! I'm happy to help."    │
     │   (first segment, ~100ms)  │────────────────────────────────┤
     │                            │                                │
     │                       wait for continue signal...           │
     │                            │                                │
     │                            │◄─ChunkedLLMContinueGeneration──│
     │                            │   (segment complete)           │
     │                            │                                │
     │                       generate 32 more tokens               │
     │                       buffer.extract_complete_sentences()   │
     │                            │                                │
     │◄──────LLMTextFrame─────────│ "What would you like to know?" │
     │   (second segment)         │────────────────────────────────┤
     │                            │                                │
```

---

## TTS Service: Adaptive WebSocket Streaming

### Server-Side Implementation (`src/nemotron_speech/tts_server.py`)

The TTS server provides a WebSocket endpoint for full-duplex text-to-speech with adaptive mode selection.

#### WebSocket Protocol

**Client → Server:**
```json
{"type": "init", "voice": "aria", "language": "en"}
{"type": "text", "text": "Hello!", "mode": "stream", "preset": "conservative"}
{"type": "text", "text": "How are you?", "mode": "batch"}
{"type": "close"}
```

**Server → Client:**
```json
{"type": "stream_created", "stream_id": "..."}
{"type": "segment_complete", "segment": 1, "audio_ms": 1234}
{"type": "done", "total_audio_ms": 5432}
```
Plus binary frames containing raw PCM audio (16-bit, 22kHz, mono).

#### Streaming vs Batch Mode

| Mode | TTFB | Quality | Use Case |
|------|------|---------|----------|
| **Streaming** | ~185-370ms | Good | First segment (low latency priority) |
| **Batch** | ~200-400ms | Best | Subsequent segments (quality priority) |

Streaming mode uses frame-by-frame generation with configurable presets:
- `aggressive`: ~185ms TTFB, 8-frame chunks
- `balanced`: ~280ms TTFB, 10-frame chunks
- `conservative`: ~370ms TTFB, 12-frame chunks (default)

#### Streaming TTS Implementation (`src/nemotron_speech/streaming_tts.py`)

The `StreamingMagpieTTS` class generates audio frame-by-frame:

```python
for idx in range(max_decoder_steps):
    # Generate next token(s) with optional CFG
    audio_codes_next = model.sample_codes_from_logits(logits, ...)

    # Check if we should yield a chunk
    total_pending_frames = len(pending_tokens) * frame_stacking_factor

    if not first_chunk_yielded:
        if total_pending_frames >= min_first_chunk_frames:
            should_yield = True  # Fast first chunk
    else:
        if total_pending_frames >= chunk_size_frames:
            should_yield = True  # Regular chunk size

    if should_yield:
        # Decode accumulated tokens with overlap for smooth boundaries
        audio = model.codes_to_audio(decode_codes, decode_lens)
        yield audio_bytes
```

#### Overlap-Add for Chunk Boundaries

The HiFi-GAN vocoder is non-causal, so different chunks produce different waveforms for the same time period. The server uses COLA-compliant overlap-add to blend boundaries:

```python
def _overlap_add(chunk1_tail: bytes, chunk2_head: bytes) -> bytes:
    # Measure correlation to choose blending strategy
    corr = np.corrcoef(a1, a2)[0, 1]

    if corr > 0.5:
        # High correlation: Hann overlap-add (COLA)
        w1 = 0.5 * (1.0 + np.cos(np.pi * t))  # 1.0 → 0.0
        w2 = 0.5 * (1.0 - np.cos(np.pi * t))  # 0.0 → 1.0
    else:
        # Low correlation: Equal-power crossfade
        w1 = np.cos(np.pi * t / 2)
        w2 = np.sin(np.pi * t / 2)

    return a1 * w1 + a2 * w2
```

### Client-Side Implementation (`pipecat_bots/magpie_websocket_tts.py`)

The Pipecat client extends `WebsocketTTSService` with adaptive mode selection and sentence splitting.

#### Sentence Splitting for GPU Memory Safety

The LLM may emit multiple sentences in a single `LLMTextFrame`. To prevent GPU OOM on long text, the TTS client splits incoming text into individual sentences before sending to the server:

```python
# Sentence boundary pattern - matches .!? followed by optional quotes/parens and space
SENTENCE_BOUNDARY_PATTERN = re.compile(r'([.!?]["\'\)]*\s)')

def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences for TTS."""
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)
    # Recombine: each sentence = content + delimiter
    ...
```

Each sentence is sent separately to the TTS server, keeping GPU memory usage bounded.

#### Adaptive Mode Logic

```python
async def run_tts(self, text: str):
    # Split into individual sentences to keep TTS chunks small (avoid GPU OOM)
    sentences = split_into_sentences(text)

    for sentence in sentences:
        msg = {"type": "text", "text": sentence}

        if self._params.use_adaptive_mode:
            if self._is_first_segment:
                msg["mode"] = "stream"
                msg["preset"] = self._params.streaming_preset
            else:
                msg["mode"] = "batch"

        await self._get_websocket().send(json.dumps(msg))
```

After receiving `segment_complete`, the client:
1. Sets `_is_first_segment = False` for subsequent segments
2. Optionally injects silence for sentence pauses
3. Sends `ChunkedLLMContinueGenerationFrame` (from `frames.py`) upstream to resume LLM generation

#### Interruption Handling

On interruption, the client:
1. Increments a generation counter to invalidate stale audio
2. Sends `cancel` message to immediately stop server generation
3. Resets local state for the next response

```python
async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
    self._gen += 1  # Invalidate in-flight audio

    if self._websocket:
        await self._websocket.send(json.dumps({"type": "cancel"}))

    self._stream_active = False
    self._is_first_segment = True
```

Audio received after interruption is discarded until `stream_created` confirms the new generation.

#### Timing Diagram

```
LLM Service               TTS Client                TTS Server              Audio Out
     │                        │                          │                      │
     │──LLMTextFrame─────────►│                          │                      │
     │  "Hello!"              │                          │                      │
     │                        │──{"type":"text",         │                      │
     │                        │   "mode":"stream"}──────►│                      │
     │                        │                          │                      │
     │                        │                     generate frames             │
     │                        │                     (~185-370ms)                │
     │                        │                          │                      │
     │                        │◄────binary audio─────────│                      │
     │                        │──TTSAudioRawFrame───────►│──────────────────────┤
     │                        │                          │                      │
     │                        │◄───segment_complete──────│                      │
     │                        │                          │                      │
     │◄──ChunkedLLMContinue───│                          │                      │
     │                        │                          │                      │
     │──LLMTextFrame─────────►│                          │                      │
     │  "How are you?"        │──{"type":"text",         │                      │
     │                        │   "mode":"batch"}───────►│                      │
     │                        │                          │                      │
```

---

## V2V Metrics: Measuring End-to-End Latency

The `V2VMetricsProcessor` measures the critical voice-to-voice response time:

```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    if isinstance(frame, VADUserStoppedSpeakingFrame):
        self._vad_stopped_time = time.time()

    elif isinstance(frame, BotStartedSpeakingFrame):
        if self._vad_stopped_time is not None:
            frame_to_frame_time = time.time() - self._vad_stopped_time
            v2v_time = frame_to_frame_time + self._vad_stop_secs

            # Emit metric
            await self.push_frame(MetricsFrame(
                data=[TTFBMetricsData(processor="ServerVoiceToVoice", value=v2v_time)]
            ))
```

The `vad_stop_secs` is added because the user actually stopped speaking ~200ms before `VADUserStoppedSpeakingFrame` was emitted.

---

## Pipeline Assembly

The bot assembles all components in `bot_interleaved_streaming.py`:

```python
pipeline_processors = [
    transport.input(),          # Audio from user
    rtvi,                       # Client communication
    stt,                        # NVidiaWebSocketSTTService
    context_aggregator.user(),  # Accumulate transcription
    context_timing,             # Log LLMMessagesFrame timing (debug)
    llm,                        # LlamaCppBufferedLLMService (single-slot, 100% cache)
    tts,                        # MagpieWebSocketTTSService
    v2v_metrics,                # V2VMetricsProcessor
    transport.output(),         # Audio to user
    audiobuffer,                # Optional recording
    context_aggregator.assistant(),  # Accumulate response
]
```

The pipeline uses:
- **Silero VAD** with 200ms stop threshold for responsive turn detection
- **SmartTurn analyzer** for intelligent turn-taking decisions
- **RTVI protocol** for client communication

---

## Summary: Latency Budget

| Stage | Typical Latency | Notes |
|-------|-----------------|-------|
| VAD silence detection | 200ms | Configurable via `stop_secs` |
| STT final transcription | 30-50ms | Including 320ms padding for context |
| LLM context processing | ~0ms | 100% KV cache hit on subsequent turns |
| LLM first segment | 100-150ms | 24-token limit with sentence boundary |
| TTS first audio | 370ms | Conservative streaming preset |
| **Total V2V** | **~500-700ms** | Measured at BotStartedSpeakingFrame |

The interleaved pipeline achieves sub-second response times by:
1. Streaming audio continuously (no batch-then-transcribe)
2. Emitting LLM text at sentence boundaries (not waiting for full response)
3. Using adaptive TTS (streaming for first segment, batch for quality thereafter)
4. Synchronizing LLM/TTS to prevent buffer overflow
5. Single-slot LLM operation for 100% KV cache reuse across turns

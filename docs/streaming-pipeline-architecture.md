# Streaming Voice Pipeline Architecture

This document explains in detail how the STT, LLM, and TTS components work together to deliver fast voice-to-voice response times in the interleaved streaming pipeline.

## Code Files Overview

| File | Description |
|------|-------------|
| **Client-side (Pipecat)** | |
| `pipecat_bots/nvidia_stt.py` | WebSocket STT client that streams audio to Parakeet ASR and receives transcriptions |
| `pipecat_bots/llama_cpp_chunked_llm.py` | Direct HTTP client to llama.cpp with sentence-boundary chunking and TTS synchronization |
| `pipecat_bots/magpie_websocket_tts.py` | WebSocket TTS client with adaptive streaming/batch mode selection |
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

### 2. Chunked LLM with Sentence Boundaries

Instead of waiting for the complete LLM response, the chunked LLM service emits text at **sentence boundaries**. The first chunk has aggressive token bounds (10-24 tokens) for fast time-to-first-chunk, while subsequent chunks wait for natural sentence endings. This enables TTS to start speaking the first sentence while the LLM is still generating.

### 3. Adaptive TTS with Streaming First Segment

The TTS service uses **adaptive mode**: the first segment uses streaming mode (~370ms TTFB) while subsequent segments use batch mode for higher quality. This optimizes for fast first-audio while maintaining quality for the rest of the response.

### End-to-End Timing

```
User speaks     VAD detects    STT sends     LLM receives    LLM first      TTS first
  "Hello"       silence        final text    transcript      chunk ready    audio out
    │              │               │              │              │              │
    ├──────────────┤───────────────┤──────────────┤──────────────┤──────────────┤
    │   ~speech    │   ~200ms      │   ~30-50ms   │   ~0ms       │  ~100-150ms  │  ~370ms
    │   duration   │   VAD delay   │   STT proc   │              │  LLM TTFB    │  TTS TTFB
    │              │               │              │              │              │
    └──────────────┴───────────────┴──────────────┴──────────────┴──────────────┘
                                                                        │
                                          Total V2V: ~500-700ms ────────┘
```

---

## STT Service: NVIDIA Parakeet Streaming ASR

### Server-Side Implementation (`src/nemotron_speech/server.py`)

The ASR server uses NVIDIA's Parakeet model with true incremental streaming. It processes audio in 160ms chunks (16 mel frames) and maintains encoder/decoder cache state across chunks for continuous transcription.

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

## LLM Service: Chunked Sentence-Boundary Streaming

### Server-Side: llama.cpp

The LLM uses llama.cpp's HTTP API directly (no custom server). Key features used:

- **SSE Streaming**: `/completion` endpoint with `stream: true`
- **KV Cache**: `cache_prompt: true` + `id_slot` pinning for cache reuse across turns
- **Stop Tokens**: `<|im_end|>` for ChatML format

### Client-Side Implementation (`pipecat_bots/llama_cpp_chunked_llm.py`)

The chunked LLM service generates text in **sentence-boundary chunks** rather than token-by-token, enabling TTS to process natural units of speech.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LlamaCppChunkedLLMService                               │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ LLMContext   │───►│ ChatML       │───►│ HTTP Stream  │───►│ Sentence   │ │
│  │ Frame        │    │ Formatter    │    │ to llama.cpp │    │ Boundary   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    │ Detection  │ │
│                                                              └────────────┘ │
│                                                                     │       │
│                                                                     ▼       │
│  ┌──────────────┐    ┌──────────────┐                        ┌────────────┐ │
│  │ Continue     │◄───│ TTS Sync     │◄───────────────────────│ LLMText    │ │
│  │ Generation   │    │ Event        │                        │ Frame      │ │
│  └──────────────┘    └──────────────┘                        └────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### First Chunk Optimization

The first chunk uses aggressive token bounds for fast TTFB:

```python
if self._is_first_chunk:
    min_tokens = 10   # Don't emit too-short first chunk
    max_tokens = 24   # Cap length for fast TTFB
else:
    min_tokens = None  # Pure sentence boundary detection
    max_tokens = None
```

#### Sentence Boundary Detection with Token Peeking

The service doesn't just check for `.!?` - it peeks at the next token to handle edge cases like closing quotes:

```python
def ends_at_sentence_boundary(text: str) -> bool:
    """Matches: .!? optionally followed by closing quotes/parens"""
    return bool(re.search(r'[.!?]["\'\)]*$', text.rstrip()))
```

When a sentence boundary is detected, the service peeks at the next token:
- If it's part of the sentence ending (e.g., closing quote), include it
- If it starts the next sentence, save it as `_pending_token` for the next chunk

#### TTS Synchronization

The LLM waits for TTS to finish each segment before generating the next chunk, preventing audio buffer overflow:

```python
# After emitting a chunk, wait for TTS signal
await asyncio.wait_for(self._continue_event.wait(), timeout=30.0)
self._continue_event.clear()
```

TTS sends `ChunkedLLMContinueGenerationFrame` upstream when a segment completes.

#### Two-Slot Management

To avoid llama.cpp race conditions when cancelling requests, the service alternates between two slots:

```python
async def _get_next_slot(self) -> tuple[int, bool]:
    # Prefer reusing last slot for KV cache hits
    if self._last_used_slot is not None:
        elapsed = time.time() - self._slot_last_used[self._last_used_slot]
        if elapsed >= min_delay:
            return self._last_used_slot, True  # Reuse for cache

    # Otherwise rotate to avoid race conditions
    slot = self._current_slot
    self._current_slot = (self._current_slot + 1) % self._num_slots
    return slot, False
```

#### Timing Diagram

```
User Input                   LLM Service                      TTS Service
     │                            │                                │
     │──LLMContextFrame──────────►│                                │
     │                            │                                │
     │                       format ChatML                         │
     │                       start HTTP stream                     │
     │                            │                                │
     │                       accumulate tokens                     │
     │                       check sentence boundary               │
     │                            │                                │
     │◄──────LLMTextFrame─────────│ "Hello! I'm happy to help."    │
     │   (first chunk, ~100ms)    │────────────────────────────────┤
     │                            │                                │
     │                       wait for continue signal...           │
     │                            │                                │
     │                            │◄─ChunkedLLMContinueGeneration──│
     │                            │   (segment complete)           │
     │                            │                                │
     │                       resume generation                     │
     │                            │                                │
     │◄──────LLMTextFrame─────────│ "What would you like to know?" │
     │   (second chunk)           │────────────────────────────────┤
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

The Pipecat client extends `WebsocketTTSService` with adaptive mode selection.

#### Adaptive Mode Logic

```python
async def run_tts(self, text: str):
    msg = {"type": "text", "text": text}

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
3. Sends `ChunkedLLMContinueGenerationFrame` upstream to resume LLM generation

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
    llm,                        # LlamaCppChunkedLLMService
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
| LLM context processing | ~0ms | Immediate forwarding |
| LLM first chunk | 100-150ms | 10-24 tokens with sentence boundary |
| TTS first audio | 370ms | Conservative streaming preset |
| **Total V2V** | **~500-700ms** | Measured at BotStartedSpeakingFrame |

The interleaved pipeline achieves sub-second response times by:
1. Streaming audio continuously (no batch-then-transcribe)
2. Emitting LLM text at sentence boundaries (not waiting for full response)
3. Using adaptive TTS (streaming for first segment, batch for quality thereafter)
4. Synchronizing LLM/TTS to prevent buffer overflow

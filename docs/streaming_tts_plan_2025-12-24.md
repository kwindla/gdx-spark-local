# Streaming TTS Checkpoint & Integration Plan

**Date:** 2025-12-24
**Status:** Draft - Awaiting Approval

---

## Step 1: Analysis of Work Since Last Commit (74785a0)

### Summary

Since "magpie oss prerelease support", two major systems have been built:

1. **Streaming TTS Infrastructure** - Multiple TTS modes (batch, streaming, adaptive, WebSocket)
2. **Chunked LLM Server** - Pausable LLM streaming for voice agent coordination

### New Files Created

#### Streaming TTS (Server-side)

| File | Lines | Purpose |
|------|-------|---------|
| `src/nemotron_speech/streaming_tts.py` | 571 | Frame-by-frame streaming TTS inference |
| `src/nemotron_speech/adaptive_stream.py` | 279 | Stream state management, text buffering |
| `src/nemotron_speech/audio_boundary.py` | 152 | Crossfade handling at segment boundaries |

#### Chunked LLM Server

| File | Lines | Purpose |
|------|-------|---------|
| `src/nemotron_speech/chunked_llm_server.py` | 886 | WebSocket LLM server with pause/resume control |

#### Pipecat TTS Services

| File | Lines | Purpose |
|------|-------|---------|
| `pipecat_bots/magpie_streaming_tts.py` | 246 | HTTP streaming TTS |
| `pipecat_bots/magpie_adaptive_tts.py` | 369 | HTTP adaptive streaming |
| `pipecat_bots/magpie_websocket_tts.py` | 341 | WebSocket TTS (primary) |

#### Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `Dockerfile.vllm-llamacpp` | ~50 | llama.cpp + vLLM container for chunked LLM |

#### Tests

| File | Lines | Purpose |
|------|-------|---------|
| `tests/adaptive_streaming/` | ~750 | Adaptive streaming test suite |
| `tests/test_websocket_tts.py` | ~150 | WebSocket TTS test |
| `tests/test_llm_tts_integration.py` | 446 | LLM + TTS end-to-end integration |
| `tests/timeline_test.py` | 278 | Chunked LLM performance timing |
| `tests/test_chunked_llm.py` | ~250 | Chunked LLM basic tests |
| `tests/stress_test_chunked_llm.py` | ~280 | Chunked LLM stress test |
| `tests/test_streaming_tts.py` | ~600 | Streaming TTS development tests |
| `tests/measure_streaming_ttfb.py` | ~400 | TTFB measurement utility |

#### Documentation

| File | Lines | Purpose |
|------|-------|---------|
| `docs/adaptive_streaming_plan_2025-12-23.md` | ~1100 | Adaptive streaming implementation plan |
| `docs/websocket_debugging_plan.md` | ~300 | WebSocket debugging notes |
| `docs/llm_tts_integration_test_plan.md` | 516 | LLM+TTS integration with results |
| `docs/chunked_llm_server.md` | 569 | Chunked LLM server documentation |
| `docs/magpie-tts-local-plan.md` | ~300 | Initial Magpie TTS planning |
| `docs/magpie-tts-streaming-plan.md` | ~350 | Streaming architecture plan |
| `docs/streaming_tts_plan_2025-12-24.md` | ~350 | This checkpoint plan |

### Modified Files

| File | Change | Description |
|------|--------|-------------|
| `src/nemotron_speech/tts_server.py` | +835 lines | HTTP + WebSocket TTS endpoints |
| `pipecat_bots/bot.py` | +56 lines | 4 TTS modes + updated system prompt |

### Total New Code

| Category | Lines |
|----------|-------|
| Server-side (TTS + LLM) | ~2,800 |
| Pipecat services | ~950 |
| Tests | ~3,000 |
| Documentation | ~3,500 |
| **Total** | **~10,250 lines** |

### Key Endpoints Added

#### TTS Server (port 8001)

```
POST   /v1/audio/speech              - Batch TTS (keep for testing)
POST   /v1/audio/speech/stream       - HTTP streaming TTS
POST   /v1/tts/stream                - Create adaptive stream
POST   /v1/tts/stream/{id}/append    - Append text
POST   /v1/tts/stream/{id}/close     - Signal completion
DELETE /v1/tts/stream/{id}           - Cancel stream
GET    /v1/tts/stream/{id}/audio     - Receive audio (HTTP streaming)
WS     /ws/tts/stream                - WebSocket full-duplex TTS
```

#### Chunked LLM Server (port 8002)

```
WS     /ws                           - WebSocket with pause/resume control
GET    /health                       - Health check
```

### Experimental Results

#### Parallel TTS (Phase 1)
- GPU contention requires serialization
- 2 concurrent requests = 81% TTFB regression
- 4 concurrent = not real-time capable

#### WebSocket TTS (Phase 2)
- TTFB improved from 3153ms → 746ms (76% improvement)
- Best streaming config: 8/12/16 frames

#### LLM + TTS Integration (Phase 3)
- Combined TTFB: **884ms** (first audio from inference start)
- LLM TTFT: 241ms (first chunk), 36-103ms (continuations)
- TTS TTFB: 509-1061ms (batch mode)
- Total: 16.6s audio in 7.3s wall time (2.3x faster than real-time)

---

## Step 2: Checkpoint Commit

### Files to Stage

```bash
# Core TTS infrastructure
src/nemotron_speech/tts_server.py
src/nemotron_speech/streaming_tts.py
src/nemotron_speech/adaptive_stream.py
src/nemotron_speech/audio_boundary.py

# Chunked LLM Server
src/nemotron_speech/chunked_llm_server.py

# Pipecat services
pipecat_bots/bot.py
pipecat_bots/magpie_streaming_tts.py
pipecat_bots/magpie_adaptive_tts.py
pipecat_bots/magpie_websocket_tts.py

# Infrastructure
Dockerfile.vllm-llamacpp

# Tests - Core
tests/adaptive_streaming/
tests/test_websocket_tts.py
tests/test_llm_tts_integration.py
tests/timeline_test.py
tests/test_chunked_llm.py
tests/stress_test_chunked_llm.py

# Tests - Development (streaming TTS)
tests/test_streaming_tts.py
tests/measure_streaming_ttfb.py

# Documentation
docs/adaptive_streaming_plan_2025-12-23.md
docs/websocket_debugging_plan.md
docs/llm_tts_integration_test_plan.md
docs/chunked_llm_server.md
docs/magpie-tts-local-plan.md
docs/magpie-tts-streaming-plan.md
docs/streaming_tts_plan_2025-12-24.md
```

### Files NOT to Stage

```
# Secrets
.env

# Test artifacts (regeneratable)
samples_parallel/
samples_adaptive/
samples/
samples_http/
integration_test_output.wav
quality_test_*.wav
test_*.wav
tts_truncation_logs.txt

# Unrelated/older work
Dockerfile.combined
scripts/start_all.sh
nano_v3_pr_description.md
nano_v3_streaming_fix.patch
test_riva_concurrent.py

# Intermediate test files (can add later if needed)
tests/generate_samples.py
tests/generate_samples_http.py
tests/test_actual_streaming_ttfb.py
tests/test_cold_start.py
tests/test_generation_timing.py
tests/test_http_timing.py
tests/test_streaming_endpoint.py
tests/test_streaming_server.py
tests/test_magpie_tts.py
tests/test_magpie_pipecat_service.py
tests/test_csm_tts.py
tests/run_asr_latency_tests.py

# Older planning docs (superseded)
docs/csm-tts-implementation-plan.md
docs/streaming_tts_implementation_2025-12-23_1550.md
```

### Commit Message

```
WIP: Add streaming TTS and chunked LLM infrastructure

Streaming TTS:
- Frame-by-frame streaming with 8/12/16 frame config
- HTTP adaptive streaming with buffer-based mode selection
- WebSocket full-duplex TTS for token-by-token integration
- Immediate flush (TARGET_AUDIO_MS=500) for client-controlled chunking

Chunked LLM Server:
- WebSocket server wrapping llama.cpp with pause/resume
- True token streaming to client
- Sentence boundary detection for natural speech chunks
- Mixed strategy: max_tokens first, sentence_boundary after

Integration tested:
- Combined TTFB: 884ms (inference start → first audio)
- 2.3x faster than real-time generation

Includes Dockerfile.vllm-llamacpp for llama.cpp inference support.
```


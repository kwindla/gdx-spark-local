# Nemotron-Speech

Local voice agent infrastructure for NVIDIA DGX Spark or RTX 5090. Runs ASR, TTS, and LLM entirely on-device in a unified container.

## Architecture

```
Host: uv run pipecat_bots/bot_interleaved_streaming.py
  ├── ASR:  ws://localhost:8080   (Parakeet 600M)
  ├── TTS:  ws://localhost:8001   (Magpie 357M WebSocket)
  └── LLM:  http://localhost:8000 (llama.cpp, Nemotron-3-Nano Q8)

┌─────────────────────────────────────────────────┐
│  nemotron unified container                     │
│  ├─ ASR server (port 8080) - ~3GB VRAM          │
│  ├─ TTS server (port 8001) - ~2GB VRAM          │
│  └─ LLM server (port 8000) - ~32GB VRAM (Q8)    │
│     (llama.cpp, --parallel 2 for two-slot)      │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build the Unified Container

```bash
docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
```

Build time: 2-3 hours (builds PyTorch, NeMo, vLLM, llama.cpp from source for CUDA 13.1 / Blackwell).

### 2. Start the Container

```bash
# Start with default Q8 model (auto-detected from HuggingFace cache)
./scripts/nemotron.sh start

# Or specify a model explicitly
./scripts/nemotron.sh start --model ~/.cache/huggingface/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/.../Q8_0.gguf

# Start with vLLM instead of llama.cpp (requires ~72GB VRAM)
./scripts/nemotron.sh start --mode vllm

# Start ASR + TTS only (no LLM)
./scripts/nemotron.sh start --no-llm
```

**Note**: Set `HUGGINGFACE_ACCESS_TOKEN` environment variable for gated model access.

### 3. Run the Voice Bot

```bash
uv run pipecat_bots/bot_interleaved_streaming.py
```

Open `http://localhost:7860/client` in your browser.

## Bot Variants

Three bot implementations are available:

| Bot | Description | Use Case |
|-----|-------------|----------|
| `bot_interleaved_streaming.py` | Chunked LLM + adaptive TTS + SmartTurn | **Production** - lowest latency |
| `bot_simple_vad.py` | Same as above, but simple VAD (no SmartTurn) | Testing ASR accuracy |
| `bot_vllm.py` | vLLM + batch TTS + SmartTurn | Higher quality inference (~72GB VRAM) |

### bot_interleaved_streaming.py (Recommended)

Production bot with interleaved streaming for lowest latency:
- **LLM**: `LlamaCppChunkedLLMService` - sentence-boundary streaming
- **TTS**: `MagpieWebSocketTTSService` - adaptive mode (streaming first chunk, batch after)
- **VAD**: SmartTurn analyzer with 340ms silence threshold

```bash
uv run pipecat_bots/bot_interleaved_streaming.py            # SmallWebRTC transport
uv run pipecat_bots/bot_interleaved_streaming.py -t daily   # Daily WebRTC transport
```

### bot_simple_vad.py

Testing/debugging bot without Smart Turn:
- Same services as `bot_interleaved_streaming.py`
- Uses simple Silero VAD with 800ms silence threshold


```bash
uv run pipecat_bots/bot_simple_vad.py
```

### bot_vllm.py

LLM inference with vLLM, supports full BF16 Nemotron 3 Nanoweights:
- **LLM**: `OpenAILLMService` - vLLM with full BF16 weights
- **TTS**: `MagpieWebSocketTTSService` - batch mode
- **VAD**: SmartTurn analyzer with 340ms silence threshold
- Requires ~72GB VRAM (vs ~32GB for llama.cpp Q8)

```bash
# Start container in vLLM mode
./scripts/nemotron.sh start --mode vllm

# Run the bot
uv run pipecat_bots/bot_vllm.py
```

## Environment Variables

### bot_interleaved_streaming.py / bot_simple_vad.py

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_ASR_URL` | `ws://localhost:8080` | ASR WebSocket endpoint |
| `NVIDIA_LLAMA_CPP_URL` | `http://localhost:8000` | llama.cpp API endpoint |
| `NVIDIA_TTS_URL` | `http://localhost:8001` | Magpie TTS endpoint |

### bot_vllm.py

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_ASR_URL` | `ws://localhost:8080` | ASR WebSocket endpoint |
| `NVIDIA_LLM_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `NVIDIA_LLM_MODEL` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Model name/path |
| `NVIDIA_LLM_API_KEY` | `not-needed` | API key (if required) |
| `NVIDIA_TTS_URL` | `http://localhost:8001` | Magpie TTS endpoint |

## Pipecat Bot Components

Custom services in `pipecat_bots/`:

| Service | File | Description |
|---------|------|-------------|
| `LlamaCppChunkedLLMService` | `llama_cpp_chunked_llm.py` | Sentence-boundary chunking, two-slot alternation with 2s reuse guard |
| `MagpieWebSocketTTSService` | `magpie_websocket_tts.py` | Adaptive streaming (fast TTFB first chunk, batch quality after) |
| `NVidiaWebSocketSTTService` | `nvidia_stt.py` | Real-time streaming ASR |

### Key Features

**Chunked LLM** (`bot_interleaved_streaming.py`, `bot_simple_vad.py`):
- Generates responses in sentence-boundary chunks for natural TTS
- Two-slot alternation prevents llama.cpp race condition crashes
- 2-second reuse guard ensures slot cleanup completes

**WebSocket TTS** (all bots):
- Adaptive mode: streaming for first segment (~370ms TTFB), batch for subsequent (quality)
- Generation counter discards stale audio after interruptions
- Cancel message immediately stops server generation

## Server-Side Services

### TTS Server (`src/nemotron_speech/tts_server.py`)

WebSocket endpoint at `/ws/tts/stream`:

| Message | Direction | Description |
|---------|-----------|-------------|
| `{"type": "init", "voice": "aria", "language": "en"}` | Client | Initialize stream |
| `{"type": "text", "text": "...", "mode": "stream\|batch"}` | Client | Send text for synthesis |
| `{"type": "close"}` | Client | Flush remaining text, complete stream |
| `{"type": "cancel"}` | Client | Immediately stop generation |
| Binary audio | Server | PCM audio (22kHz, 16-bit, mono) |
| `{"type": "segment_complete"}` | Server | Segment done, more may follow |
| `{"type": "done"}` | Server | Stream complete |

### ASR Server (`src/nemotron_speech/server.py`)

WebSocket endpoint at `ws://localhost:8080`:

| Message | Direction | Description |
|---------|-----------|-------------|
| `{"type": "ready"}` | Server | Ready to receive audio |
| Binary audio | Client | PCM audio (16kHz, 16-bit, mono) |
| `{"type": "transcript", "text": "...", "is_final": bool}` | Server | Transcription result |

## Container Management

Use `./scripts/nemotron.sh` to manage the container:

```bash
# Start the container
./scripts/nemotron.sh start [OPTIONS]
  --mode MODE         LLM mode: llamacpp-q8 (default), llamacpp-q4, vllm
  --model PATH        Path to model file
  --no-asr            Disable ASR service
  --no-tts            Disable TTS service
  --no-llm            Disable LLM service
  -f, --foreground    Run in foreground (default: detached)

# Stop the container
./scripts/nemotron.sh stop

# Restart the container
./scripts/nemotron.sh restart [OPTIONS]

# Check status
./scripts/nemotron.sh status

# View logs
./scripts/nemotron.sh logs          # All logs interleaved
./scripts/nemotron.sh logs asr      # ASR logs only
./scripts/nemotron.sh logs tts      # TTS logs only
./scripts/nemotron.sh logs llm      # LLM logs only

# Open shell in container
./scripts/nemotron.sh shell

# Show help
./scripts/nemotron.sh help
```

## Building the Container

```bash
# Build the unified container (2-3 hours)
docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
```

The build compiles from source for CUDA 13.1 / Blackwell (sm_121):
- PyTorch (with NVRTC support)
- torchaudio
- NeMo ASR/TTS
- vLLM
- llama.cpp

## Model Requirements

| Model | Source | Size |
|-------|--------|------|
| Parakeet ASR | `models/Parakeet_Realtime_En_600M.nemo` | ~2.4GB |
| Nemotron-3-Nano Q8 | HuggingFace `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` | ~32GB |
| Nemotron-3-Nano BF16 | HuggingFace `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | ~72GB |
| Magpie TTS | Auto-downloaded from HuggingFace | ~1.4GB |

Download models:

```bash
# Q8 GGUF (for llama.cpp)
huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF

# BF16 (for vLLM)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Streaming Architecture & Design Decisions

This section describes the streaming architecture, frame flow, and key design decisions that enable low-latency voice interaction.

### Pipeline Frame Flow

```
User speaks → VAD → STT → LLM → TTS → Transport → Audio Out
                ↑                   │
                └───────────────────┘
                  Interruption Loop
```

**Downstream flow (user input → bot response):**
1. **VAD** detects speech start → `UserStartedSpeakingFrame`
2. **STT** sends audio via WebSocket, receives transcripts
3. **VAD** detects silence → `VADUserStoppedSpeakingFrame` triggers reset
4. **STT** sends `{"type": "reset"}` → server returns final transcript
5. **LLM** receives context, generates sentence chunks via `LLMTextFrame`
6. **TTS** sends chunks via WebSocket, receives binary audio
7. **Transport** plays `TTSAudioRawFrame` to output device

**Upstream flow (TTS → LLM synchronization):**
1. TTS receives `segment_complete` from server
2. TTS pushes `ChunkedLLMContinueGenerationFrame` upstream
3. LLM receives signal, continues generating next chunk

### Adaptive TTS Mode Selection

The TTS service uses adaptive mode for optimal latency/quality tradeoff:

| Segment | Mode | TTFB | Quality | Rationale |
|---------|------|------|---------|-----------|
| First | Streaming | ~370ms | Good | Fast response to user |
| Subsequent | Batch | ~800ms | Better | User already hearing audio |

**Streaming mode** generates audio frame-by-frame (~46ms chunks) with CFG enabled.
**Batch mode** generates the full segment before sending.

### Interruption Handling

When the user speaks during bot output:

```
T=0: User speaks → InterruptionFrame generated
T=1: Transport clears audio buffers (Pipecat built-in)
T=2: TTS increments generation counter (_gen++)
T=3: TTS sends {"type": "cancel"} to server
T=4: Server cancels audio task, clears queue
T=5: Stale audio discarded (confirmed_gen != gen)
T=6: New response starts, stream_created sets confirmed_gen = gen
T=7: New audio accepted, plays cleanly
```

**Key design: Generation counter gating**
- `_gen`: Incremented on interruption
- `_confirmed_gen`: Set to `_gen` when `stream_created` received
- Audio only accepted when `_confirmed_gen == _gen`

This handles in-flight audio that arrives after interruption but before the new stream starts.

### TTS Server Protocol

WebSocket at `/ws/tts/stream`:

```
Client → Server:
  {"type": "init", "voice": "aria", "language": "en"}
  {"type": "text", "text": "...", "mode": "stream|batch"}
  {"type": "close"}   ← Flush remaining, complete normally
  {"type": "cancel"}  ← Stop immediately (interruption)

Server → Client:
  {"type": "stream_created", "stream_id": "..."}
  Binary audio frames (PCM 22kHz 16-bit mono)
  {"type": "segment_complete", "audio_ms": 1234}
  {"type": "done", "total_audio_ms": 5432}
```

**Design decision: Single persistent connection**
- WebSocket connects on pipeline start, persists until end
- Interruptions reset server state via `cancel`, don't reconnect
- Avoids connection overhead (~50ms per reconnect)

### STT Server Protocol

WebSocket at `ws://localhost:8080`:

```
Client → Server:
  Binary audio (PCM 16kHz 16-bit mono)
  {"type": "reset"}  ← Finalize current utterance

Server → Client:
  {"type": "ready"}
  {"type": "transcript", "text": "...", "is_final": true|false}
```

**Design decision: VAD-triggered reset**
- Reset sent on `VADUserStoppedSpeakingFrame` (after ~200ms silence)
- Server adds 480ms silence padding for trailing context
- Lock ensures audio/reset message ordering

### LLM Chunked Generation

The LLM generates responses in sentence-boundary chunks for natural TTS:

```
LLM generates: "Hello! How can I help you today?"
         ↓
Chunk 1: "Hello!"          → TTS → Audio → segment_complete
         ↓ (wait for TTS)
Chunk 2: "How can I help you today?" → TTS → Audio → done
```

**First chunk optimization:**
- `min_tokens: 10`, `max_tokens: 24`
- Balances TTFB (~300ms) with natural phrasing

**Two-slot alternation (llama.cpp):**
- Prevents `GGML_ASSERT(!slot.is_processing())` crash
- Alternates slots with 2s reuse guard
- Requires `--parallel 2` on llama-server

### Inter-Sentence Pauses

For natural speech rhythm, 250ms silence is injected after sentence-ending segments:

```
Segment ends with .!? → segment_complete received
                      → Check sentence boundary queue
                      → Inject silence frame if true
                      → Push ChunkedLLMContinueGenerationFrame
```

The pause happens *after* the audio plays, not before, preserving low TTFB.

## Troubleshooting

**LLM crashes with `GGML_ASSERT(!slot.is_processing())`**:
- Ensure `--parallel 2` is set on llama-server (default in unified container)
- The two-slot implementation prevents this by alternating slots

**vLLM takes 10-15 minutes to start**:
- This is normal for first startup (model loading, kernel compilation)
- Set `SERVICE_TIMEOUT=900` if needed

**vLLM DNS resolution issues**:
- The container uses `--network=host` in vLLM mode to avoid DNS issues with HuggingFace

## License

MIT

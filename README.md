# Nemotron-Speech

Local voice agent infrastructure for NVIDIA DGX Spark (Blackwell GB10). Runs ASR, TTS, and LLM entirely on-device.

## Architecture

```
Host: uv run pipecat_bots/bot.py
  ├── ASR:  ws://localhost:8080   (Parakeet 600M)
  ├── TTS:  ws://localhost:8001   (Magpie 357M WebSocket)
  └── LLM:  http://localhost:8000 (llama.cpp, Nemotron-3-Nano Q8)

┌─────────────────────────────────────────────────┐
│  nemotron-asr container                         │
│  ├─ ASR server (port 8080) - ~3GB VRAM          │
│  └─ TTS server (port 8001) - ~2GB VRAM          │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  llama-q4 container                             │
│  └─ LLM server (port 8000) - ~32GB VRAM         │
│     (llama.cpp, --parallel 2 for two-slot)      │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the ASR + TTS Container

```bash
docker run -d --name nemotron-asr --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -p 8080:8080 -p 8001:8001 \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-asr:cuda13-full \
  bash /workspace/scripts/start_asr_tts.sh
```

### 2. Start the LLM Container (llama.cpp)

```bash
docker run -d --name llama-q4 \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/hf_cache:ro \
  vllm-llamacpp:cuda13 \
  llama-server \
    -m /hf_cache/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/b5f797f8fabe2c732d27ffbb91d91fb1d3ea0b56/Nemotron-3-Nano-30B-A3B-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --n-gpu-layers 99 \
    --ctx-size 65536 \
    --flash-attn on \
    --parallel 2 \
    --verbose-prompt
```

**Important**: `--parallel 2` is required for the two-slot alternation that prevents llama.cpp crashes during chunked generation.

### 3. Configure Environment

Add to `.env`:

```bash
USE_WEBSOCKET_TTS=true
USE_CHUNKED_LLM=true
```

### 4. Run the Voice Bot

```bash
uv run pipecat_bots/bot.py
```

Open `http://localhost:7860/client` in your browser.

## Pipecat Bot Components

The bot uses these custom services in `pipecat_bots/`:

| Service | File | Description |
|---------|------|-------------|
| `LlamaCppChunkedLLMService` | `llama_cpp_chunked_llm.py` | Sentence-boundary chunking, two-slot alternation with 2s reuse guard |
| `MagpieWebSocketTTSService` | `magpie_websocket_tts.py` | Adaptive streaming (fast TTFB first chunk, batch quality after) |
| `NVidiaWebSocketSTTService` | `nvidia_stt.py` | Real-time streaming ASR |

### Key Features

**Chunked LLM** (`USE_CHUNKED_LLM=true`):
- Generates responses in sentence-boundary chunks for natural TTS
- Two-slot alternation prevents llama.cpp race condition crashes
- 2-second reuse guard ensures slot cleanup completes

**WebSocket TTS** (`USE_WEBSOCKET_TTS=true`):
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

```bash
# View logs
docker logs -f nemotron-asr
docker logs -f llama-q4

# Restart containers
docker restart nemotron-asr llama-q4

# Stop and remove
docker stop nemotron-asr llama-q4
docker rm nemotron-asr llama-q4
```

## Model Requirements

| Model | Source | Size |
|-------|--------|------|
| Parakeet ASR | `models/Parakeet_Realtime_En_600M.nemo` | ~2.4GB |
| Nemotron-3-Nano Q8 | HuggingFace `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` | ~32GB |
| Magpie TTS | Auto-downloaded from HuggingFace | ~1.4GB |

## Building Containers

```bash
# ASR + TTS container (1-2 hours, builds PyTorch from source)
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .

# LLM container (llama.cpp)
# See vllm-llamacpp:cuda13 build instructions
```

## Troubleshooting

**LLM crashes with `GGML_ASSERT(!slot.is_processing())`**:
- Ensure `--parallel 2` is set on llama-server
- The two-slot implementation prevents this by alternating slots

**Stale audio after interruption**:
- Fixed by generation counter in `MagpieWebSocketTTSService`
- Audio is discarded until `stream_created` confirms current generation

**CUDA errors after long idle**:
- Restart the affected container: `docker restart nemotron-asr`

## Alternative: vLLM Inference (Full Weights)

For higher quality inference, use vLLM with the full BF16 model instead of the Q8 GGUF quantization.

### Download Model

```bash
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir='models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir_use_symlinks=False
)
"

# Copy reasoning parser to model directory
cp vllm/nano_v3_reasoning_parser.py models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
```

### Run vLLM Server

```bash
docker run -d --name vllm-nemotron --gpus all --ipc=host \
  -p 8000:8000 \
  -v $(pwd)/models:/workspace/models:ro \
  -e USE_LIBUV=0 \
  vllm:cuda13-full \
  bash -c "pip uninstall -y torchvision torchaudio --break-system-packages -q && \
           pip install --break-system-packages -q triton && \
           rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
           ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
           python -m vllm.entrypoints.openai.api_server \
               --model /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
               --host 0.0.0.0 --port 8000 \
               --dtype bfloat16 --trust-remote-code \
               --gpu-memory-utilization 0.60 \
               --max-num-seqs 1 --max-model-len 100000 \
               --enforce-eager --disable-log-requests \
               --enable-prefix-caching"
```

### Bot Configuration for vLLM

Set in `.env`:

```bash
USE_CHUNKED_LLM=false
NVIDIA_LLM_URL=http://localhost:8000/v1
NVIDIA_LLM_MODEL=/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

**Note**: vLLM uses ~72GB VRAM (vs ~32GB for llama.cpp Q8). The `--gpu-memory-utilization 0.60` setting leaves room for ASR/TTS.

## License

MIT

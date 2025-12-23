# Nemotron-Speech

Streaming Speech-to-Text, Text-to-Speech, and LLM inference services for NVIDIA DGX Spark (Blackwell GB10).

- **ASR**: Streaming Speech-to-Text using NVIDIA's Parakeet ASR model with NeMo
- **TTS**: Text-to-Speech using open-source NVIDIA Magpie TTS (357M) with NeMo
- **LLM**: Nemotron-3-Nano-30B inference using vLLM with OpenAI-compatible API

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/yourusername/nemotron-speech.git
cd nemotron-speech

# 2. Download model weights (see "Model Weights" section for details)
#    - ASR: models/Parakeet_Reatime_En_600M.nemo (~2.4GB)
#    - LLM: models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/ (~60GB)
#    - TTS: Downloaded automatically from HuggingFace on first run

# 3. Build containers (takes 1-3 hours each due to PyTorch compilation)
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .
docker build -f Dockerfile.vllm-cuda13-build -t vllm:cuda13-full .

# 4. Start combined ASR + TTS server (ports 8080 and 8001)
docker run -d --name nemotron-asr --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -p 8080:8080 -p 8001:8001 \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-asr:cuda13-full \
  bash /workspace/scripts/start_asr_tts.sh

# 5. Start LLM server (port 8000) - see LLM section for full command

# 6. Run the Pipecat voice bot
uv run pipecat_bots/bot.py
```

## Requirements

- NVIDIA DGX Spark (ARM64/Blackwell GB10) or compatible NVIDIA GPU
- Docker with NVIDIA Container Toolkit
- CUDA 13.0+ (for Blackwell/sm_121 support)
- HuggingFace account with access to `nvidia/magpie_tts_multilingual_357m`

## Architecture

The system runs three services that communicate over the network:

```
Host (uv run pipecat_bots/bot.py)
    ├─→ ASR:  ws://localhost:8080      (WebSocket, Parakeet 600M)
    ├─→ LLM:  http://localhost:8000/v1 (HTTP, Nemotron-3-Nano-30B)
    └─→ TTS:  http://localhost:8001/v1 (HTTP, Magpie TTS 357M)

┌─────────────────────────────────────────────────┐
│  nemotron-asr container (shared CUDA context)   │
│  ├─ ASR server (port 8080) - ~3GB VRAM          │
│  └─ TTS server (port 8001) - ~2GB VRAM          │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  vllm-nemotron container                        │
│  └─ LLM server (port 8000) - ~72GB VRAM         │
└─────────────────────────────────────────────────┘
```

ASR and TTS run in the same container to share a single CUDA context, avoiding memory fragmentation when running alongside the memory-hungry LLM.

## ASR + TTS Container

This container provides both streaming ASR and TTS services. It builds PyTorch and NeMo from source with full CUDA 13.1 and sm_121 support for Blackwell GPUs.

### Build the Container

```bash
# Build (takes 1-2 hours due to PyTorch compilation)
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .
```

### Run the Combined Server

```bash
docker run -d --name nemotron-asr --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -p 8080:8080 -p 8001:8001 \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-asr:cuda13-full \
  bash /workspace/scripts/start_asr_tts.sh
```

This starts both servers:
- **ASR**: WebSocket server on port 8080
- **TTS**: HTTP server on port 8001

### Test the Endpoints

```bash
# Check TTS health
curl http://localhost:8001/health

# Get TTS configuration (sample rate, voices, languages)
curl http://localhost:8001/v1/audio/config

# Synthesize speech (returns raw PCM audio)
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "aria", "language": "en"}' \
  --output hello.pcm

# Play the audio (22kHz, 16-bit, mono)
ffplay -f s16le -ar 22000 -ac 1 hello.pcm
```

### Container Management

```bash
# View logs
docker logs -f nemotron-asr

# Stop the container
docker stop nemotron-asr

# Start again
docker start nemotron-asr

# Remove container
docker rm -f nemotron-asr
```

## TTS Server (Open-Source Magpie TTS)

The TTS server uses NVIDIA's open-source Magpie TTS model for high-quality multilingual speech synthesis. Unlike the Magpie TTS NIM, this runs the model directly via NeMo, avoiding the audio truncation bugs that affect the NIM on DGX Spark.

### Model Specifications

| Attribute | Value |
|-----------|-------|
| Model ID | `nvidia/magpie_tts_multilingual_357m` |
| Parameters | 357M |
| Codec | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` |
| Sample Rate | 22kHz |
| License | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (commercial OK) |

### Available Voices

| Voice | Speaker Index | Description |
|-------|---------------|-------------|
| `john` | 0 | Male voice |
| `sofia` | 1 | Female voice |
| `aria` | 2 | Female voice (default) |
| `jason` | 3 | Male voice |
| `leo` | 4 | Male voice |

### Supported Languages

| Code | Language |
|------|----------|
| `en` | English |
| `es` | Spanish |
| `de` | German |
| `fr` | French |
| `vi` | Vietnamese |
| `it` | Italian |
| `zh` | Mandarin Chinese |

### HTTP API

The TTS server exposes an OpenAI-compatible API:

**POST /v1/audio/speech**

```json
{
  "input": "Text to synthesize",
  "voice": "aria",
  "language": "en",
  "response_format": "pcm"
}
```

Returns raw PCM audio (16-bit signed, mono, 22kHz) with headers:
- `X-Sample-Rate`: 22000
- `X-Channels`: 1
- `X-Duration-Ms`: audio duration in milliseconds

**GET /health**

Returns server health status and model loading state.

**GET /v1/audio/config**

Returns TTS configuration (sample rate, available voices, languages).

### Performance (DGX Spark)

| Metric | Value |
|--------|-------|
| Model Load Time | ~52s (includes codec download) |
| Warm-up Latency | ~230ms |
| Average Latency | ~800ms |
| Real-time Factor | 0.29x (3.4x faster than real-time) |

### NeMo Version Requirement

The Magpie TTS model requires **NeMo main branch** (not the r2.6.0 release). The `MagpieTTSModel` class only exists in the main branch. The Dockerfile pins to a specific commit for reproducibility:

```dockerfile
ARG NEMO_COMMIT=661af02e105662bd5b61881054608ea44944572c
```

### Why Open-Source Instead of NIM?

| Aspect | Magpie NIM | Open-Source Magpie |
|--------|------------|-------------------|
| DGX Spark Support | Buggy (audio truncation) | Works correctly |
| Dependencies | Separate container, gRPC | Same container as ASR |
| Latency | Network + inference | Inference only |
| Control | Black box | Full control, debuggable |
| Memory | Separate CUDA context | Shared with ASR |

## Pipecat Voice Bot

The `pipecat_bots/` directory contains a complete voice bot that uses all three services.

### Run the Bot

```bash
# Start the bot (requires ASR, TTS, and LLM containers running)
uv run pipecat_bots/bot.py
```

Open `http://localhost:7860/client` in your browser to start a conversation.

### Configuration

The bot reads configuration from environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_ASR_URL` | `ws://localhost:8080` | ASR WebSocket server |
| `NVIDIA_LLM_URL` | `http://localhost:8000/v1` | LLM API server |
| `NVIDIA_TTS_URL` | `http://localhost:8001` | TTS HTTP server |
| `USE_LOCAL_TTS` | `true` | Use Magpie TTS (false = Cartesia cloud) |

### Pipecat Services

**MagpieHTTPTTSService** (`pipecat_bots/magpie_http_tts.py`)

HTTP client that connects to the Magpie TTS server. Runs on the host without NeMo/PyTorch dependencies.

```python
from magpie_http_tts import MagpieHTTPTTSService

tts = MagpieHTTPTTSService(
    server_url="http://localhost:8001",
    voice="aria",
    language="en",
)
```

**NVidiaWebSocketSTTService** (`pipecat_bots/nvidia_stt.py`)

WebSocket client for the streaming ASR server.

```python
from nvidia_stt import NVidiaWebSocketSTTService

stt = NVidiaWebSocketSTTService(
    url="ws://localhost:8080",
    sample_rate=16000,
)
```

## LLM Container (Nemotron-3-Nano-30B)

The vLLM container serves the Nemotron-3-Nano-30B model with an OpenAI-compatible API.

### Build the Container

```bash
# Build (takes 2-3 hours due to PyTorch and vLLM compilation)
docker build -f Dockerfile.vllm-cuda13-build -t vllm:cuda13-full .
```

### Download Model Weights

```bash
# Download from HuggingFace
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir='models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir_use_symlinks=False
)
"
```

### Run the LLM Server

```bash
docker run -d \
    --name vllm-nemotron \
    --gpus all \
    --ipc=host \
    --restart unless-stopped \
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
                 --host 0.0.0.0 \
                 --port 8000 \
                 --dtype bfloat16 \
                 --trust-remote-code \
                 --gpu-memory-utilization 0.60 \
                 --swap-space 0 \
                 --max-num-seqs 1 \
                 --max-num-batched-tokens 2048 \
                 --scheduling-policy fcfs \
                 --max-model-len 100000 \
                 --enforce-eager \
                 --disable-log-requests \
                 --enable-auto-tool-choice \
                 --tool-call-parser qwen3_coder \
                 --reasoning-parser-plugin /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/nano_v3_reasoning_parser.py \
                 --reasoning-parser nano_v3 \
                 --enable-prefix-caching"
```

The reasoning parser (`nano_v3_reasoning_parser.py`) enables structured reasoning output. We use a patched version that fixes streaming behavior when `enable_thinking: false` (the upstream version incorrectly routes content to `reasoning_content` in streaming mode).

#### Prefix Caching

Prefix caching reduces Time-To-First-Token (TTFT) by caching the KV cache for common prefixes like system prompts. When subsequent requests share the same prefix, the cached values are reused instead of recomputing.

**Note:** Nemotron-3-Nano uses a hybrid Mamba-Transformer architecture. vLLM's prefix caching support for Mamba layers is experimental but functional.

**Observed behavior (DGX Spark):**

| Prompt Size | First Request TTFT | Cached Request TTFT | Improvement |
|-------------|-------------------|---------------------|-------------|
| Short (~200 tokens) | ~210ms | ~210ms | None |
| Medium (~1000 tokens) | ~610ms | ~420ms | ~30% |

For multi-turn conversations with the same system prompt, you should see reduced TTFT after the first request.

#### Key Arguments Explained

| Argument | Value | Purpose |
|----------|-------|---------|
| `--gpu-memory-utilization 0.60` | 0.60 | Use 60% of GPU memory (~72GB), leaving room for ASR/TTS |
| `--max-num-seqs 1` | 1 | Single request at a time (optimized for latency) |
| `--max-model-len 100000` | 100K | Maximum context length for multi-turn conversations |
| `--enforce-eager` | - | Required for sm_121a compatibility |
| `--swap-space 0` | 0 | Disable CPU swap for consistent latency |
| `--reasoning-parser nano_v3` | nano_v3 | Use NVIDIA's reasoning parser for structured output |
| `--enable-auto-tool-choice` | - | Enable tool/function calling support |
| `--enable-prefix-caching` | - | Cache common prefixes (system prompts) to reduce TTFT |

### Test the Endpoint

```bash
# Check server health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Chat completion (reasoning disabled, streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "messages": [{"role": "user", "content": "Write a haiku about GPUs"}],
        "max_tokens": 256,
        "stream": true,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

### Reasoning Mode

Nemotron-3-Nano is a "thinking" model with structured reasoning output. The reasoning parser separates chain-of-thought from the final answer.

**Response Structure (with reasoning parser):**
```json
{
  "content": "The final answer",
  "reasoning": "Chain-of-thought reasoning...",
  "reasoning_content": "Chain-of-thought reasoning..."
}
```

**Performance (DGX Spark):**

| Metric | Value |
|--------|-------|
| Time to First Token (TTFT) | ~150 ms |
| Generation Speed | ~27 tokens/sec |

**Example - Reasoning ON (default):**
```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 50
    }'
```

Response has both fields populated:
- `content`: "2 + 2 = 4"
- `reasoning_content`: "User asks a simple math question..."

**Example - Reasoning OFF (for chat):**
```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

Response has only content (no reasoning generated):
- `content`: "2 + 2 = 4"
- `reasoning_content`: null

Streaming works correctly with both modes - content appears in the appropriate field.

### Container Management

```bash
# View logs
docker logs -f vllm-nemotron

# Stop the container
docker stop vllm-nemotron

# Start again
docker start vllm-nemotron

# Remove container
docker rm -f vllm-nemotron
```

## Memory Allocation

The DGX Spark has 128GB unified memory. vLLM pre-allocates GPU memory for KV cache, so careful tuning is required to run all services simultaneously.

**Recommended settings:**

| Container | Memory Budget | Key Settings |
|-----------|--------------|--------------|
| LLM (vLLM) | ~72 GB | `--gpu-memory-utilization 0.60 --max-model-len 100000` |
| ASR (Parakeet) | ~3 GB | Load to CPU first, then move to GPU |
| TTS (Magpie) | ~2 GB | Shares CUDA context with ASR |
| Free | ~45 GB | Available for inference |

**Important notes:**
- vLLM's `--gpu-memory-utilization` controls total GPU allocation (model + KV cache)
- The LLM model weights alone consume ~60GB; set utilization high enough for weights plus KV cache
- ASR and TTS run in the same container to share a CUDA context
- Running TTS in a separate container causes OOM due to CUDA context overhead

Adjust based on your needs:
- `0.60` + `max-model-len 100000` = ~72GB for LLM, ~48GB free (tested working)
- `0.70` + `max-model-len 100000` = ~84GB for LLM, more KV cache headroom
- Lower utilization will fail: model weights alone need ~60GB

## Model Weights

Model weight directories are excluded from git due to their size. Download them before running the containers.

### ASR Model (Parakeet)

```bash
mkdir -p models
# Download Parakeet_Realtime_En_600M.nemo (~2.4GB) to models/
# Available from NVIDIA NGC or HuggingFace
```

### LLM Model (Nemotron-3-Nano)

```bash
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir='models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir_use_symlinks=False
)
"

# Copy the custom reasoning parser to the model directory
cp vllm/nano_v3_reasoning_parser.py models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
```

### TTS Model (Magpie)

The Magpie TTS model is downloaded automatically from HuggingFace on first run. Set your token:

```bash
export HUGGINGFACE_ACCESS_TOKEN=hf_your_token_here
```

The model requires access to:
- `nvidia/magpie_tts_multilingual_357m` (~1.4GB)
- `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` (~100MB, downloaded automatically)

## WebSocket Streaming ASR Server

The server provides real-time streaming speech recognition via WebSocket with progressive interim results.

### Latency Options

The `--right-context` parameter controls the accuracy/latency tradeoff:

| Right Context | Latency | Use Case |
|---------------|---------|----------|
| `0` | ~80ms | Ultra-low latency |
| `1` | ~160ms | Low latency (recommended) |
| `6` | ~560ms | Balanced |
| `13` | ~1.12s | Highest accuracy |

### WebSocket Protocol

**Connect:** `ws://localhost:8080`

**Messages from server:**
- `{"type": "ready"}` - Server ready to receive audio
- `{"type": "transcript", "text": "...", "is_final": false}` - Interim result
- `{"type": "transcript", "text": "...", "is_final": true}` - Final result

**Messages to server:**
- Binary audio data (16-bit PCM, 16kHz, mono)
- `{"type": "end"}` or `{"type": "reset"}` - Signal end of audio stream

### Performance (DGX Spark)

| Metric | Value |
|--------|-------|
| Transcript Accuracy | 100% (matches batch mode) |
| Real-time Factor | 0.14x (7x faster than real-time) |
| Finalization Latency | 24ms (real-time streaming) |
| Interim Updates | ~52 progressive results for 18s audio |

### Test the WebSocket Server

```bash
# Run test (sends audio in 500ms chunks)
uv run --with websockets tests/test_websocket_client.py tests/fixtures/harvard_16k.wav ws://localhost:8080

# Test with real-time streaming simulation
uv run --with websockets tests/test_websocket_client.py tests/fixtures/harvard_16k.wav ws://localhost:8080 --chunk 160
```

## Client Examples

The `examples/` directory contains standalone Python clients for inference from outside the containers.

### ASR Client (Audio to Text)

```bash
# Transcribe an audio file
uv run --with websockets examples/asr_client.py audio.wav

# Simulate real-time streaming
uv run --with websockets examples/asr_client.py audio.wav --realtime

# Custom server URL
uv run --with websockets examples/asr_client.py audio.wav --url ws://192.168.1.100:8080
```

### LLM Client (Text to Text)

```bash
# Simple query
uv run --with openai examples/llm_client.py "What is the capital of France?"

# Streaming output
uv run --with openai examples/llm_client.py --stream "Write a poem about AI"

# Enable chain-of-thought reasoning
uv run --with openai examples/llm_client.py --reasoning "What is 15 * 23?"

# Custom server URL
uv run --with openai examples/llm_client.py --url http://192.168.1.100:8000/v1 "Hello"
```

## Development

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Or run commands directly without explicit venv activation
uv run python -m nemotron_speech.server --help
```

## Project Structure

```
nemotron-speech/
├── Dockerfile.asr-cuda13-build   # ASR + TTS container (CUDA 13.1, Blackwell)
├── Dockerfile.vllm-cuda13-build  # LLM container (vLLM, CUDA 13.1)
├── pyproject.toml
├── README.md
├── .gitignore
├── scripts/
│   └── start_asr_tts.sh          # Combined ASR + TTS startup script
├── src/
│   └── nemotron_speech/
│       ├── __init__.py
│       ├── server.py             # WebSocket ASR server
│       ├── tts_server.py         # HTTP TTS server (Magpie)
│       ├── stt_service.py        # STT service implementation
│       └── cli.py                # Command-line interface
├── pipecat_bots/
│   ├── bot.py                    # Pipecat voice bot
│   ├── nvidia_stt.py             # Pipecat ASR service (WebSocket client)
│   ├── magpie_http_tts.py        # Pipecat TTS service (HTTP client)
│   └── magpie_local_tts.py       # Pipecat TTS service (in-process, for container)
├── examples/
│   ├── asr_client.py             # ASR WebSocket client example
│   └── llm_client.py             # LLM OpenAI-compatible client example
├── vllm/
│   └── nano_v3_reasoning_parser.py  # Custom reasoning parser for Nemotron
├── tests/
│   ├── test_streaming_blackwell.py   # Streaming inference test
│   ├── test_websocket_client.py      # WebSocket client test
│   ├── test_magpie_tts.py            # TTS inference test
│   ├── benchmark_inference.py        # Performance benchmark
│   └── fixtures/
│       └── harvard_16k.wav           # Test audio file (16kHz)
├── docs/
│   └── magpie-tts-local-plan.md      # TTS implementation notes
└── models/                           # Model weights (not in git)
    ├── Parakeet_Reatime_En_600M.nemo     # ASR model (~2.4GB)
    └── NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/  # LLM model (~60GB)
```

## Technical Details

### Blackwell (sm_121) Support

The DGX Spark uses the NVIDIA GB10 chip with compute capability 12.1 (sm_121). Standard PyTorch wheels don't include sm_121 kernels, causing CUDA Error 222 during NVRTC compilation.

The `Dockerfile.asr-cuda13-build` solves this by:

1. **CUDA 13.1 Base**: Provides full sm_121 support via NVRTC
2. **PyTorch from Source**: Built with `TORCH_CUDA_ARCH_LIST="12.1"`
3. **NeMo Main Branch**: Required for MagpieTTSModel (pinned to specific commit)
4. **NCCL Support**: Required by NeMo's distributed training infrastructure
5. **NeMo NVRTC Patch**: Dynamic GPU architecture detection for runtime compilation

### Key Build Configuration

```dockerfile
# CUDA architecture for Blackwell
ENV TORCH_CUDA_ARCH_LIST="12.1"

# Enable distributed (required by NeMo)
ENV USE_DISTRIBUTED=1
ENV USE_NCCL=1
ENV USE_SYSTEM_NCCL=1

# NeMo main branch for Magpie TTS support
ARG NEMO_COMMIT=661af02e105662bd5b61881054608ea44944572c
```

### Streaming Configuration

The ASR model uses cache-aware streaming with:
- `att_context_size=[70, 1]` - attention context window
- `chunk_size=[9, 16]` - audio chunk size in frames
- `shift_size=[9, 16]` - shift between chunks
- Greedy decoding (CUDA graphs disabled for Blackwell compatibility)

## Troubleshooting

### CUDA Error 222 on Blackwell

If you see `CUDA error 222` or `NVRTC compilation failed`, you need the CUDA 13.1 build:

```bash
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .
```

### Missing torch.distributed

If NeMo fails to import with `ModuleNotFoundError: torch._C._distributed_c10d`, the PyTorch build is missing NCCL support. Rebuild with the cache buster:

```bash
# Increment the cache buster in Dockerfile.asr-cuda13-build
ARG PYTORCH_CACHE_BUSTER=v3-nccl

# Rebuild
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .
```

### Verify PyTorch Configuration

```bash
docker run --rm nemotron-asr:cuda13-full python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('NCCL:', torch.distributed.is_nccl_available())
print('Gloo:', torch.distributed.is_gloo_available())
"
```

Expected output:
```
PyTorch: 2.11.0a0+git6493f23
CUDA: 13.1
NCCL: True
Gloo: True
```

### TTS Model Not Loading

If the TTS server fails to load the model:

1. **Check HuggingFace token**: Ensure `HUGGINGFACE_ACCESS_TOKEN` is set and has access to the model
2. **Check NeMo version**: The model requires NeMo main branch, not r2.6.0

```bash
# Verify NeMo has MagpieTTSModel
docker run --rm nemotron-asr:cuda13-full python3 -c "
from nemo.collections.tts.models import MagpieTTSModel
print('MagpieTTSModel available')
"
```

### TTS Out of Memory

If TTS fails with OOM when running in a separate container:

- **Solution**: Run TTS in the same container as ASR using `scripts/start_asr_tts.sh`
- **Cause**: Separate containers create separate CUDA contexts, fragmenting memory

### vLLM Triton/PTXAS Errors

If you see `PTXASError: ptxas fatal: Value 'sm_121a' is not defined`, the bundled Triton ptxas doesn't support Blackwell. The run command above handles this by symlinking to the system ptxas.

### vLLM Out of Memory

If the LLM container fails with OOM errors, reduce memory utilization or context length:

```bash
--gpu-memory-utilization 0.55  # Use less memory
--max-model-len 32000          # Reduce max context length
```

Note: The model weights require ~60GB minimum. Setting utilization below 0.55 will fail.

## License

MIT

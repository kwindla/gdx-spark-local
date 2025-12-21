# Nemotron-Speech

Streaming Speech-to-Text and LLM inference services for NVIDIA DGX Spark (Blackwell GB10).

- **ASR**: Streaming Speech-to-Text using NVIDIA's Parakeet ASR model with NeMo
- **LLM**: Nemotron-3-Nano-30B inference using vLLM with OpenAI-compatible API

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/yourusername/nemotron-speech.git
cd nemotron-speech

# 2. Download model weights (see "Model Weights" section for details)
#    - ASR: models/Parakeet_Reatime_En_600M.nemo (~2.4GB)
#    - LLM: models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/ (~60GB)

# 3. Build containers (takes 1-3 hours each due to PyTorch compilation)
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .
docker build -f Dockerfile.vllm-cuda13-build -t vllm:cuda13-full .

# 4. Start ASR server (port 8080)
docker run -d --name nemotron-asr --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models:ro -p 8080:8080 \
  nemotron-asr:cuda13-full python -m nemotron_speech.server --port 8080

# 5. Start LLM server (port 8000) - see README for full command

# 6. Test from host (using uv)
uv run --with websockets examples/asr_client.py tests/fixtures/harvard_16k.wav
uv run --with openai examples/llm_client.py "Hello, who are you?"
```

## Requirements

- NVIDIA DGX Spark (ARM64/Blackwell GB10) or compatible NVIDIA GPU
- Docker with NVIDIA Container Toolkit
- CUDA 13.0+ (for Blackwell/sm_121 support)

## Container Options

### Option 1: CUDA 13.1 Build (Recommended for Blackwell/DGX Spark)

This container builds PyTorch from source with full CUDA 13.1 and sm_121 support. Required for cache-aware streaming inference on Blackwell GPUs.

```bash
# Build (takes 1-2 hours due to PyTorch compilation)
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .

# Run the ASR server
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -p 8080:8080 \
  nemotron-asr:cuda13-full \
  python -m nemotron_speech.server --port 8080
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
                 --reasoning-parser nano_v3"
```

The reasoning parser (`nano_v3_reasoning_parser.py`) enables structured reasoning output. We use a patched version that fixes streaming behavior when `enable_thinking: false` (the upstream version incorrectly routes content to `reasoning_content` in streaming mode).

#### Key Arguments Explained

| Argument | Value | Purpose |
|----------|-------|---------|
| `--gpu-memory-utilization 0.60` | 0.60 | Use 60% of GPU memory (~72GB), leaving room for ASR |
| `--max-num-seqs 1` | 1 | Single request at a time (optimized for latency) |
| `--max-model-len 100000` | 100K | Maximum context length for multi-turn conversations |
| `--enforce-eager` | - | Required for sm_121a compatibility |
| `--swap-space 0` | 0 | Disable CPU swap for consistent latency |
| `--reasoning-parser nano_v3` | nano_v3 | Use NVIDIA's reasoning parser for structured output |
| `--enable-auto-tool-choice` | - | Enable tool/function calling support |

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

### Memory Allocation for Running Both Containers

The DGX Spark has 128GB unified memory. vLLM pre-allocates GPU memory for KV cache, so careful tuning is required to run both ASR and LLM simultaneously.

**Recommended settings for running both:**

| Container | Memory Budget | Key Settings |
|-----------|--------------|--------------|
| LLM (vLLM) | ~72 GB | `--gpu-memory-utilization 0.60 --max-model-len 100000` |
| ASR (NeMo) | ~3 GB | Load to CPU first, then move to GPU |
| Free | ~45 GB | Available for inference |

**Important notes:**
- vLLM's `--gpu-memory-utilization` controls total GPU allocation (model + KV cache)
- The model weights alone consume ~60GB; set utilization high enough to fit them plus KV cache
- ASR must load to CPU first (`map_location='cpu'`) then move to GPU to avoid OOM
- Reduce `--max-model-len` to minimize KV cache size if memory is tight

Adjust based on your needs:
- `0.60` + `max-model-len 100000` = ~72GB for LLM, ~48GB free (tested working)
- `0.70` + `max-model-len 100000` = ~84GB for LLM, more KV cache headroom
- Lower utilization will fail: model weights alone need ~60GB

The model natively supports up to 262K tokens. The KV cache at 0.60 utilization can handle ~408K tokens, so 100K context is well within capacity.

## Model Weights

Both model weight directories are excluded from git due to their size. Download them before running the containers.

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

## WebSocket Streaming ASR Server

The server provides real-time streaming speech recognition via WebSocket with progressive interim results.

### Start the Server

```bash
docker run -d --name nemotron-asr --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models:ro \
  -p 8080:8080 \
  nemotron-asr:cuda13-full \
  python -m nemotron_speech.server --port 8080 --right-context 1
```

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

**Basic test with test client:**

```bash
# Run test (sends audio in 500ms chunks)
uv run --with websockets tests/test_websocket_client.py tests/fixtures/harvard_16k.wav ws://localhost:8080
```

**Test with real-time streaming simulation:**

```bash
uv run --with websockets tests/test_websocket_client.py tests/fixtures/harvard_16k.wav ws://localhost:8080 --chunk 160
```

**Test multiple chunk sizes:**

```bash
uv run --with websockets tests/test_websocket_client.py tests/fixtures/harvard_16k.wav ws://localhost:8080 --all
```

### Example Output

```
Reading audio file: tests/fixtures/harvard_16k.wav
  Sample rate: 16000 Hz
  Duration: 18.36s

Sending audio in 500ms chunks...
  [interim 1] The
  [interim 2] The stale
  [interim 3] The stale smell
  ...
  [interim 52] The stale smell of old beer lingers It takes heat to bring o...

FINAL TRANSCRIPT:
The stale smell of old beer lingers It takes heat to bring out the odor.
A cold dip restores health and zest. A salt pickle tastes fine with ham.
Tacos al pastor are my favorite. A zestful food is the hot cross bun.

Statistics:
  Interim results: 52
  Real-time factor: 0.14x

Finalization latency:
  Last audio chunk -> final transcript: 24ms
  End signal -> final transcript: 24ms
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

## Testing Streaming Inference

### Run the Blackwell Streaming Test

This test validates cache-aware streaming ASR on Blackwell GPUs:

```bash
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/tests:/workspace/tests \
  nemotron-asr:cuda13-full \
  python3 /workspace/tests/test_streaming_blackwell.py
```

Expected output:
```
============================================================
Cache-Aware Streaming ASR Test on Blackwell GPU
============================================================
GPU: NVIDIA GB10
CUDA: 13.1

Loading model: /workspace/models/Parakeet_Reatime_En_600M.nemo
...
Streaming Inference
  Chunk 1: cache_len=2, pred_shapes=[torch.Size([0])]
  Chunk 2: cache_len=4, pred_shapes=[torch.Size([0])]
  ...
  Chunk 115: cache_len=70, pred_shapes=[torch.Size([85])]

Final transcript (214 chars):
"The stale smell of old beer lingers It takes heat to bring out the odor..."

SUCCESS - Cache-aware streaming inference completed!
```

### Run Offline Benchmark

```bash
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/tests:/workspace/tests \
  nemotron-asr:cuda13-full \
  python3 /workspace/tests/benchmark_inference.py
```

## Technical Details

### Blackwell (sm_121) Support

The DGX Spark uses the NVIDIA GB10 chip with compute capability 12.1 (sm_121). Standard PyTorch wheels don't include sm_121 kernels, causing CUDA Error 222 during NVRTC compilation.

The `Dockerfile.asr-cuda13-build` solves this by:

1. **CUDA 13.1 Base**: Provides full sm_121 support via NVRTC
2. **PyTorch from Source**: Built with `TORCH_CUDA_ARCH_LIST="12.1"`
3. **NCCL Support**: Required by NeMo's distributed training infrastructure
4. **NeMo NVRTC Patch**: Dynamic GPU architecture detection for runtime compilation

### Key Build Configuration

```dockerfile
# CUDA architecture for Blackwell
ENV TORCH_CUDA_ARCH_LIST="12.1"

# Enable distributed (required by NeMo)
ENV USE_DISTRIBUTED=1
ENV USE_NCCL=1
ENV USE_SYSTEM_NCCL=1

# NCCL paths
ENV NCCL_ROOT=/usr
ENV NCCL_LIB_DIR=/usr/lib/aarch64-linux-gnu
ENV NCCL_INCLUDE_DIR=/usr/include
```

### Streaming Configuration

The model uses cache-aware streaming with:
- `att_context_size=[70, 1]` - attention context window
- `chunk_size=[9, 16]` - audio chunk size in frames
- `shift_size=[9, 16]` - shift between chunks
- Greedy decoding (CUDA graphs disabled for Blackwell compatibility)

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
├── Dockerfile.asr-cuda13-build   # ASR container (CUDA 13.1, Blackwell)
├── Dockerfile.vllm-cuda13-build  # LLM container (vLLM, CUDA 13.1)
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── nemotron_speech/
│       ├── __init__.py
│       ├── server.py             # WebSocket ASR server
│       ├── stt_service.py        # STT service implementation
│       └── cli.py                # Command-line interface
├── examples/
│   ├── asr_client.py             # ASR WebSocket client example
│   └── llm_client.py             # LLM OpenAI-compatible client example
├── vllm/
│   └── nano_v3_reasoning_parser.py  # Custom reasoning parser for Nemotron
├── tests/
│   ├── test_streaming_blackwell.py   # Streaming inference test
│   ├── test_websocket_client.py      # WebSocket client test
│   ├── benchmark_inference.py        # Performance benchmark
│   └── fixtures/
│       └── harvard_16k.wav           # Test audio file (16kHz)
└── models/                           # Model weights (not in git, ~120GB total)
    ├── Parakeet_Reatime_En_600M.nemo     # ASR model (~2.4GB)
    └── NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/  # LLM model (~60GB)
```

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

### vLLM Triton/PTXAS Errors

If you see `PTXASError: ptxas fatal: Value 'sm_121a' is not defined`, the bundled Triton ptxas doesn't support Blackwell. The run command above handles this by symlinking to the system ptxas.

For a permanent fix, rebuild the container with triton configured:

```bash
# The Dockerfile already includes the fix, but if running manually:
rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas
ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas
```

### vLLM Out of Memory

If the LLM container fails with OOM errors, reduce memory utilization or context length:

```bash
--gpu-memory-utilization 0.55  # Use less memory
--max-model-len 32000          # Reduce max context length
```

Note: The model weights require ~60GB minimum. Setting utilization below 0.55 will fail.

## License

MIT

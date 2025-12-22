# Pipecat Bot with Local NVIDIA ASR/LLM

A Pipecat voice bot using local NVIDIA Nemotron-Speech infrastructure:
- **ASR**: WebSocket server on `ws://localhost:8080` (Parakeet model)
- **LLM**: OpenAI-compatible API on `http://localhost:8000/v1` (Nemotron-3-Nano-30B)
- **TTS**: Cartesia cloud API

## Files

```
pipecat_bots/
├── __init__.py           # Package marker
├── nvidia_stt.py         # NVidiaWebSocketSTTService implementation
├── bot.py                # Main Pipecat bot (uses runner pattern)
└── PLAN.md               # This file
```

## Usage

### 1. Ensure Docker containers are running

```bash
# ASR container on port 8080
docker ps | grep nemotron-asr

# LLM container on port 8000
docker ps | grep vllm-nemotron
```

### 2. Set environment variables in `.env`

```bash
CARTESIA_API_KEY=your-cartesia-key
DAILY_API_KEY=your-daily-key  # Only needed for Daily transport

# Optional overrides
NVIDIA_ASR_URL=ws://localhost:8080
NVIDIA_LLM_URL=http://localhost:8000/v1
NVIDIA_LLM_MODEL=/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### 3. Run the bot

```bash
# Default: WebRTC transport (opens browser UI at http://localhost:7860)
uv run pipecat_bots/bot.py

# With Daily transport
uv run pipecat_bots/bot.py -t daily

# Direct Daily connection (no web server)
uv run pipecat_bots/bot.py -d
```

## Architecture

```
User Audio → [Transport] → [VAD] → [NVidiaWebSocketSTT] → [LLM] → [CartesiaTTS] → [Transport] → User Audio
                              ↓
                    ws://localhost:8080 (Parakeet ASR)
                              ↓
                    http://localhost:8000/v1 (Nemotron-3-Nano)
```

## NVidiaWebSocketSTTService

Custom STT service connecting to the NVIDIA Parakeet ASR WebSocket server.

### Protocol

**Client → Server:**
- Binary audio: 16-bit PCM, 16kHz, mono
- `{"type": "reset"}` - End of utterance (triggers final transcription)

**Server → Client:**
- `{"type": "ready"}` - Ready to receive audio
- `{"type": "transcript", "text": "...", "is_final": false}` - Interim
- `{"type": "transcript", "text": "...", "is_final": true}` - Final

## LLM Configuration

Uses `OpenAILLMService` with vLLM's OpenAI-compatible API:
- Reasoning disabled via `chat_template_kwargs: {"enable_thinking": False}`
- Max 256 tokens for concise responses

## Troubleshooting

### ASR not connecting
```bash
# Check container
docker logs -f nemotron-asr

# Test WebSocket
uv run --with websockets tests/test_websocket_client.py tests/fixtures/harvard_16k.wav
```

### LLM not responding
```bash
# Check container
docker logs -f vllm-nemotron

# Test API
curl http://localhost:8000/health
```

# Voice-to-Voice Latency Testing

This document describes how to run the 20-turn voice-to-voice latency test and extract metrics from the logs.

## Prerequisites

1. The container must be running with all services (ASR, LLM, TTS)
2. Environment variables configured in `.env`:
   - `NVIDIA_ASR_URL` (default: ws://localhost:8080)
   - `NVIDIA_LLAMA_CPP_URL` (default: http://localhost:8000)
   - `NVIDIA_TTS_URL` (default: http://localhost:8001)

## Test Architecture

The test uses a **persistent WebRTC connection** across all 20 turns. This is critical for:

1. **LLM KV Cache Reuse**: Conversation history is cached, reducing LLM TTFB from ~800ms to ~150ms on later turns
2. **Realistic Conversation Flow**: Simulates actual multi-turn voice conversations
3. **Proper Turn Timing**: Measures time from user audio completion to bot audio start

The test client (`scripts/voice_agent_test_client.py`) synthesizes user audio via TTS, sends it over WebRTC at realtime pace, and measures when the bot starts/stops speaking.

## Running the Test

### Step 1: Start the Bot with Logging

Start the bot in the background with logs redirected to a file:

```bash
uv run pipecat_bots/bot_interleaved_streaming.py -t webrtc > /tmp/bot_v2v_test.log 2>&1 &
```

Wait for the bot to start (check for "Uvicorn running on http://localhost:7860"):

```bash
tail -f /tmp/bot_v2v_test.log
```

### Step 2: Run the 20-Turn Test

```bash
uv run scripts/run_20_turn_test.py
```

The test runs 20 utterances (10 short, 10 long) with a 1-second pause between turns.

### Step 3: Stop the Bot

```bash
pkill -f "bot_interleaved_streaming.py"
```

## Extracting Metrics

### Log Line Patterns

The following patterns can be used to extract metrics from the log file:

| Metric | Log Pattern | Unit |
|--------|-------------|------|
| ASR TTFB | `NVidiaWebSocketSTTService#N NemotronSTT TTFB: Xms` | milliseconds |
| LLM TTFB | `LlamaCppBufferedLLMService#N TTFB: X.XXX` | seconds |
| TTS TTFB | `MagpieWebSocketTTSService#N TTFB: X.XXX` | seconds |
| V2V TTFB | `V2VMetrics: ServerVoiceToVoice TTFB: Xms` | milliseconds |

### Extract V2V TTFB (Primary Metric)

```bash
grep "V2VMetrics: ServerVoiceToVoice TTFB" /tmp/bot_v2v_test.log | sed 's/.*TTFB: //' | sed 's/ms//'
```

### Extract Per-Turn Detailed Breakdown

```bash
awk '
/NemotronSTT TTFB:/ { gsub(/.*TTFB: /, ""); gsub(/ms/, ""); asr = $0 }
/LlamaCppBufferedLLMService.*TTFB:/ { gsub(/.*TTFB: /, ""); llm = $0 }
/MagpieWebSocketTTSService.*TTFB:/ { gsub(/.*TTFB: /, ""); tts = $0 }
/V2VMetrics: ServerVoiceToVoice TTFB:/ {
    gsub(/.*TTFB: /, ""); gsub(/ms/, ""); v2v = $0;
    printf "Turn %2d: V2V=%4dms | ASR=%3dms | LLM=%4dms | TTS=%3dms\n", ++turn, v2v, asr, int(llm*1000), int(tts*1000)
}
' /tmp/bot_v2v_test.log
```

### Calculate Statistics

```bash
grep "V2VMetrics: ServerVoiceToVoice TTFB" /tmp/bot_v2v_test.log | \
  sed 's/.*TTFB: //' | sed 's/ms//' | \
  awk '{ sum += $1; count++; if (min == "" || $1 < min) min = $1; if (max == "" || $1 > max) max = $1 }
       END { printf "V2V TTFB: Min=%dms, Max=%dms, Avg=%.0fms\n", min, max, sum/count }'
```

### Generate Percentile Summary Table

After running tests, generate a summary table with Min, P50, P90, and Max for each metric:

```bash
# Extract per-turn metrics to a temp file
awk '
/NemotronSTT TTFB:/ { gsub(/.*TTFB: /, ""); gsub(/ms/, ""); asr = $0 }
/LlamaCppBufferedLLMService.*TTFB:/ { gsub(/.*TTFB: /, ""); llm = $0 * 1000 }
/MagpieWebSocketTTSService.*TTFB:/ { gsub(/.*TTFB: /, ""); tts = $0 * 1000 }
/V2VMetrics: ServerVoiceToVoice TTFB:/ {
    gsub(/.*TTFB: /, ""); gsub(/ms/, ""); v2v = $0
    print asr, int(llm), int(tts), v2v
}
' /tmp/bot_v2v_test.log > /tmp/metrics_raw.txt

# Generate percentile table
echo "| Metric | Min | P50 | P90 | Max |"
echo "|--------|-----|-----|-----|-----|"
for i in 1 2 3 4; do
  name=$(echo "ASR LLM TTS V2V" | cut -d' ' -f$i)
  cut -d' ' -f$i /tmp/metrics_raw.txt | sort -n > /tmp/col.txt
  min=$(head -1 /tmp/col.txt)
  max=$(tail -1 /tmp/col.txt)
  p50=$(sed -n '10p' /tmp/col.txt)
  p90=$(sed -n '18p' /tmp/col.txt)
  printf "| %-6s | %4dms | %4dms | %4dms | %4dms |\n" $name $min $p50 $p90 $max
done
```

Example output:

| Metric | Min | P50 | P90 | Max |
|--------|-----|-----|-----|-----|
| ASR    |  25ms |  33ms | 183ms | 247ms |
| LLM    | 518ms | 808ms | 847ms | 1004ms |
| TTS    | 163ms | 193ms | 234ms | 301ms |
| V2V    | 1215ms | 1276ms | 1596ms | 1730ms |

## Understanding the Metrics

### V2V TTFB (Voice-to-Voice Time to First Byte)

The end-to-end latency from when the user finishes speaking to when the bot starts speaking. This includes:

1. **VAD stop delay** (200ms): Time waiting after voice activity ends to confirm turn completion
2. **ASR processing**: Speech-to-text transcription
3. **LLM processing**: Text generation (first segment)
4. **TTS processing**: Text-to-speech synthesis (first audio chunk)

Formula: `V2V = VAD_STOP_SECS + ASR + LLM_first_segment + TTS_first_chunk`

### Service TTFBs

- **ASR TTFB**: Time from audio input to first transcript (streaming, so typically fast for longer utterances)
- **LLM TTFB**: Time from prompt submission to first token (using buffered/sentence-boundary streaming)
- **TTS TTFB**: Time from text input to first audio chunk

## Test Results (2026-01-02, Persistent Connection)

### Summary Statistics

| Metric | Min | P50 | P90 | Max |
|--------|-----|-----|-----|-----|
| ASR    |  16ms |  18ms |  21ms |  24ms |
| LLM    |  88ms | 162ms | 184ms | 205ms |
| TTS    | 100ms | 107ms | 112ms | 113ms |
| V2V    | 432ms | 508ms | 544ms | 3463ms |

**Note**: Max V2V of 3463ms is due to SmartTurn INCOMPLETE detection (see Known Behaviors below).

### Per-Turn Results

| Turn | Utterance | TTR (ms) | Response Duration |
|------|-----------|----------|-------------------|
| 1 | Hello there. | 424 | 2.6s |
| 2 | What time is it? | 3465* | 7.3s |
| 3 | Tell me a joke. | 540 | 4.7s |
| 4 | How are you? | 447 | 3.3s |
| 5 | What is two plus two? | 374 | 2.3s |
| 6 | Goodbye. | 437 | 2.5s |
| 7 | Thanks very much! | 421 | 2.7s |
| 8 | Can you help me? | 3378* | 3.2s |
| 9 | What do you think? | 452 | 3.4s |
| 10 | That sounds good. | 445 | 2.7s |
| 11-20 | (Long utterances) | 459-544 | 3.6-32.5s |

*Turns marked with * had SmartTurn INCOMPLETE detection, adding ~3s delay.

### LLM Cache Performance

With persistent connection, LLM KV cache reuse improves significantly:

| Turn Range | Cached Tokens | Cache Hit Rate |
|------------|---------------|----------------|
| 1-5 | 0-268 | 0-70% |
| 6-10 | 156-316 | 52-77% |
| 11-20 | 3000+ | 85-99% |

### Observations

1. **LLM cache reuse is critical**: With persistent connection, LLM TTFB drops from ~800ms (cold) to ~150ms (warm) due to KV cache reuse.

2. **ASR is consistently fast**: 16-24ms across all turns with streaming ASR.

3. **TTS is consistent**: 100-113ms TTFB with WebSocket streaming.

4. **SmartTurn adds latency for ambiguous utterances**: Short questions like "What time is it?" and "Can you help me?" are classified as INCOMPLETE, causing a 3-second silence wait.

## Known Behaviors

### SmartTurn INCOMPLETE Detection

The SmartTurn analyzer may classify short, ambiguous utterances as INCOMPLETE and wait up to 3 seconds for more input. This affects:
- "What time is it?" - could be followed by more context
- "Can you help me?" - open-ended question
- Similar short questions that might have follow-up

This is expected behavior to avoid cutting off users mid-thought.

### Multi-Sentence Utterance Splitting

Long utterances with natural pauses may be split by ASR into multiple transcripts:
1. First sentence arrives, bot starts responding
2. User continues speaking, interrupting the bot
3. Full utterance arrives, bot gives complete response

The test client handles this by measuring from audio completion to final bot response.

### TTS GPU Memory

The TTS warmup text must be long enough to pre-allocate GPU memory. Current warmup is 183 characters (matching longest test utterance). If TTS OOMs during inference, increase the warmup text length in `src/nemotron_speech/tts_server.py`.

### Short Utterance Audio Detection

Very short single-word utterances (like "Thanks!") may not be reliably detected by VAD due to WebRTC buffering. Use slightly longer phrases (e.g., "Thanks very much!") for reliable detection.

## Configuration Options

### Test Client Timeout

The test client waits for bot responses with a configurable timeout (default: 90 seconds). For conversations with long bot responses, this may need to be increased:

```python
# In scripts/voice_agent_test_client.py
async def send_turn(self, text: str, timeout: float = 90.0) -> TurnMetrics:
```

### TTS Warmup Text

To prevent TTS OOM errors, ensure warmup text matches the longest expected input:

```python
# In src/nemotron_speech/tts_server.py
warmup_text = os.environ.get("TTS_WARMUP_TEXT", (
    "I just finished reading a fascinating book about the history of computing. "
    "It discussed how early computers filled entire rooms. "
    "Tell me more about the evolution of computer hardware."
))
```

### Reduce VAD Stop Time

In `bot_interleaved_streaming.py`, modify `VAD_STOP_SECS`:

```python
VAD_STOP_SECS = 0.15  # More aggressive (default: 0.2)
```

### LLM Buffering Parameters

Adjust first segment size in `LlamaCppBufferedLLMService.InputParams`:

```python
params=LlamaCppBufferedLLMService.InputParams(
    first_segment_max_tokens=24,      # Target tokens for first TTS segment
    first_segment_hard_max_tokens=24, # Hard limit
)
```

### TTS Streaming Preset

Available presets in `MagpieWebSocketTTSService.InputParams`:
- `"conservative"`: Higher quality, slightly higher latency
- `"balanced"`: Balance of quality and latency
- `"aggressive"`: Lower latency, may affect quality

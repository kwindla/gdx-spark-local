# Voice-to-Voice Latency Testing

This document describes how to run the 20-turn voice-to-voice latency test and extract metrics from the logs.

## Prerequisites

1. The container must be running with all services (ASR, LLM, TTS)
2. Environment variables configured in `.env`:
   - `NVIDIA_ASR_URL` (default: ws://localhost:8080)
   - `NVIDIA_LLAMA_CPP_URL` (default: http://localhost:8000)
   - `NVIDIA_TTS_URL` (default: http://localhost:8001)

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

The test runs 20 utterances (10 short, 10 long) with a 2-second pause between turns.

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

## Test Results (2026-01-02)

### Per-Turn Breakdown

| Turn | V2V (ms) | ASR (ms) | LLM (ms) | TTS (ms) | Utterance Type |
|------|----------|----------|----------|----------|----------------|
| 1 | 1399 | 165 | 774 | 256 | Short |
| 2 | 1485 | 247 | 824 | 211 | Short |
| 3 | 1596 | 183 | 1004 | 206 | Short |
| 4 | 1540 | 196 | 946 | 194 | Short |
| 5 | 1219 | 161 | 662 | 193 | Short |
| 6 | 1317 | 70 | 724 | 301 | Short |
| 7 | 1342 | 57 | 847 | 234 | Short |
| 8 | 1704 | 135 | 518 | 177 | Short |
| 9 | 1730 | 81 | 567 | 171 | Short |
| 10 | 1388 | 175 | 821 | 189 | Short |
| 11 | 1247 | 26 | 808 | 191 | Long |
| 12 | 1215 | 26 | 797 | 163 | Long |
| 13 | 1247 | 25 | 813 | 185 | Long |
| 14 | 1225 | 27 | 805 | 164 | Long |
| 15 | 1257 | 28 | 809 | 194 | Long |
| 16 | 1272 | 27 | 826 | 203 | Long |
| 17 | 1231 | 32 | 814 | 174 | Long |
| 18 | 1215 | 31 | 803 | 166 | Long |
| 19 | 1292 | 28 | 802 | 196 | Long |
| 20 | 1276 | 33 | 842 | 194 | Long |

### Statistics

| Metric | Min | Max | Average |
|--------|-----|-----|---------|
| V2V TTFB | 1215ms | 1730ms | 1360ms |

### Observations

1. **ASR is faster for longer utterances**: Short utterances (turns 1-10) have ASR TTFB of 57-247ms, while long utterances (turns 11-20) have ASR TTFB of 25-33ms. This is because streaming ASR can return results while audio is still being received.

2. **LLM is the dominant latency**: LLM TTFB accounts for ~60% of the total V2V time (600-1000ms out of 1200-1700ms total).

3. **TTS is consistent**: TTS TTFB stays in the 160-300ms range across all turns.

4. **VAD overhead**: The 200ms VAD stop delay is a fixed cost that could be reduced with more aggressive VAD settings (at the risk of cutting off users).

## Configuration Options

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

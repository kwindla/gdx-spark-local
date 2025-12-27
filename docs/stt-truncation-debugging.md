# STT Truncation Issue - Debugging Guide

## Problem Description

The NVIDIA Parakeet streaming ASR sometimes truncates the final word(s) of an utterance. For example, "What's four plus nine?" might be transcribed as "What's 4 plus..." with "nine" missing.

## Root Causes Identified

### 1. Server-Side Insufficient Trailing Context

**Issue:** The RNN-T streaming decoder buffers output until it has enough context to commit. When reset arrives, the final chunk may not have enough trailing context for the model to finalize the last word.

**Fix in `server.py`:** Add silence padding before processing the final chunk. The padding is calculated dynamically based on the streaming configuration:

```python
# Padding = (right_context + 1) * shift_frames
# With right_context=1, shift_frames=16: (1+1)*16 = 32 frames = 320ms
padding_samples = self.final_padding_frames * self.hop_samples
silence_padding = np.zeros(padding_samples, dtype=np.float32)
```

**Why this formula:**
- `right_context` chunks (160ms with right_context=1) for encoder lookahead
- 1 additional chunk (160ms) for RNN-T decoder to gain confidence
- Total: 320ms with the default right_context=1 setting

### 2. Client-Side Reset Timing

**Background:** VAD fires `VADUserStoppedSpeakingFrame` after detecting ~200ms of silence. By this point, all speech audio has already been sent to the server.

**Approach in `nvidia_stt.py`:** Send reset immediately when VADFrame arrives. The audio frame that triggered VAD is silence (not speech), so there's no risk of truncating speech.

### 3. VAD Alignment with ASR Model Requirements

**Solution:** Set `stop_secs=0.34` (340ms) so that the VAD-accumulated silence naturally provides the ~320ms trailing context the model needs, eliminating the need for server-side padding in the normal case.

## Debugging

### Server-Side Logs

The ASR server logs to `/var/log/nemotron/asr.log` inside the container. View with:

```bash
docker exec nemotron tail -100 /var/log/nemotron/asr.log
```

Or follow in real-time:

```bash
docker exec nemotron tail -f /var/log/nemotron/asr.log
```

### Log Messages to Look For

#### 1. Reset Received
```
Session XXXX reset received: accumulated=85760 samples (5360ms), emitted=528 frames
```
- `accumulated`: Total audio samples received (before padding)
- `emitted`: Mel frames already processed by streaming chunks

#### 2. Final Chunk Processing
```
Session XXXX final chunk: total_mel=537, emitted=528, remaining=9
```
- `total_mel`: Total mel frames from all audio (including padding)
- `remaining`: New frames to process in final chunk
- **If remaining is small (<16), the model may not have enough context**

#### 3. Interim Transcripts
```
Session XXXX interim: What's 4 plus
```
Shows what the model outputs during streaming. If the final word never appears in interim transcripts, the model isn't recognizing it at all.

#### 4. Final Chunk Output
```
Session XXXX final chunk output: 'What's 4 plus 4' (was: 'What's 4 plus')
```
Shows what the final chunk processing produced vs what was buffered. If these are the same, the final chunk didn't add anything.

#### 5. Processing Time
```
Session XXXX final chunk processed in 45.2ms
```

### Bot-Side Logs

Key log messages in the pipecat bot:

```
NVidiaWebSocketSTTService#0 sent reset (audio: 4740ms)
NVidiaWebSocketSTTService#0 final transcript: What's 4 plus 4...
```

The reset is sent immediately when VAD fires. All speech audio has already been sent (VAD fires after 340ms of silence with our configuration).

## Timeline of a Typical Utterance

```
t=0ms:      User starts speaking
t=0-800ms:  User says "What's four plus four"
t=800ms:    User stops speaking
t=800-1140ms: Silence (audio frames being sent to server)
t=1140ms:   VAD detects 340ms of silence, fires VADUserStoppedSpeakingFrame
t=1140ms:   STT receives VADFrame, sends reset immediately
t=1140ms:   Server has ~320ms of silence (340ms minus triggering chunk)
t=1140ms:   Server processes final chunk with keep_all_outputs=True
t=1180ms:   Server sends final transcript
```

Note: With `stop_secs=0.34`, the VAD-accumulated silence (~320ms at server) matches the model's trailing context requirements exactly. The server-side padding formula is kept as a safety net but is rarely needed.

## Tuning Parameters

### Silence Padding (server.py)

```python
# Calculated as (right_context + 1) * shift_frames
self.final_padding_frames = (self.right_context + 1) * self.shift_frames
```

- Current: 320ms (2 chunks) with right_context=1
- Formula: `(right_context + 1) * 160ms`
- Increase if final words are still being cut off (bump the +1 to +2)
- Trade-off: More padding = slightly higher latency for final transcript

### Right Context (server.py)

```python
self.model.encoder.set_default_att_context_size([70, self.right_context])
```

- Current: 1 (160ms lookahead)
- Options: 0=80ms, 1=160ms, 6=560ms, 13=1.12s
- Higher = better accuracy but more streaming latency

### VAD Silence Threshold (stop_secs)

Configured in the pipecat transport VAD analyzer. This setting directly affects how much silence the server receives naturally through normal audio flow.

**Alignment formula:**
```
optimal_stop_secs = ((right_context + 1) * 0.16) - 0.02
```

With `right_context=1`: `(2 * 0.16) - 0.02 = 0.30` seconds = **300ms**

**Why this works:**
- At 16kHz with 20ms chunks, when VAD fires after `S` seconds of silence
- Server has received approximately `S*1000 - 20`ms of silence (minus the triggering chunk)
- With `stop_secs=0.3`, server gets ~280ms naturally, needs only ~40ms padding
- With `stop_secs=0.34`, server gets ~320ms naturally, needs zero padding

**Recommended settings:**

| stop_secs | Silence at server | Padding needed | Use case |
|-----------|-------------------|----------------|----------|
| 0.2       | ~180ms            | ~140ms         | Fast response, more padding overhead |
| 0.3       | ~280ms            | ~40ms          | Good balance |
| 0.34      | ~320ms            | ~0ms           | **Configured** - perfect alignment |

**Note:** If users pause mid-sentence longer than `stop_secs`, VAD will fire prematurely. 300ms is a natural conversational pause that avoids most false triggers.

## Files Modified

- `pipecat_bots/nvidia_stt.py`: Immediate reset on VAD silence detection
- `pipecat_bots/bot.py`: VAD stop_secs=0.34 for ASR alignment
- `src/nemotron_speech/server.py`: Dynamic silence padding and debug logging

## Testing

1. Restart the container to pick up server changes:
   ```bash
   docker restart nemotron
   ```

2. Restart the bot to pick up client changes

3. Test with phrases that end in short words:
   - "What's two plus two?"
   - "What's four plus nine?"
   - "Count to five"

4. Check logs for the full pipeline:
   - Bot: Reset sent with audio duration
   - Server: Interim transcripts, final chunk output

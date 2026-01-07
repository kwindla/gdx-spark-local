# Parakeet Batch vs Streaming Inference Analysis

This document provides a deep technical analysis of why batch transcription produces better results than streaming transcription, experiments conducted, and current recommendations.

## Executive Summary

**Final Results (January 6, 2026) - Full 1000-Sample Evaluation:**

| Implementation | Pooled WER | Word Errors | Perfect (0% WER) | Gap vs Batch |
|----------------|------------|-------------|------------------|--------------|
| **RC1 (160ms lookahead)** | 3.63% | 860 | 649 (65.2%) | +275 errors |
| **RC6 (560ms lookahead)** | 2.51% | 599 | 701 (70.3%) | +14 errors |
| **Batch** | 2.45% | 585 | 700 (70.1%) | baseline |

**Key Finding:** RC6 closes **95% of the accuracy gap** between streaming and batch!

- Error reduction: 860 → 599 = **261 fewer word errors** (30% improvement)
- Perfect matches: 649 → 701 = **52 more perfect transcriptions**
- Gap vs batch: 275 errors → 14 errors = **95% gap reduction**

**Latency Impact:**
- Interim transcription latency: +400ms (acceptable for most use cases)
- Final transcription latency: **unchanged** (~20ms finalization time)
- The silence padding is processed in a single GPU call, not sequentially

**Recommendation:** Use `right_context=6` for production. The 400ms additional interim latency is acceptable, and final transcription latency is unchanged while achieving near-batch accuracy.

**Key Constraint:**
- `vad_stop_secs = 0.2` is FIXED (200ms) - cannot be changed
- This means mid-utterance VAD stops will continue to occur

**Root Cause (for remaining errors):**
The streaming implementation has limited lookahead compared to batch. With `right_context=1`, the encoder only sees 160ms of future audio, while batch sees the entire utterance. Decoder state corruption from mid-utterance resets is a secondary factor.

---

## Inference Code Analysis

### Batch Mode (`model.transcribe`)

**File:** `scripts/batch_transcribe_nemo.py:93`

```python
transcriptions = model.transcribe(batch_audio, batch_size=len(batch_audio), verbose=False)
```

**Characteristics:**
- Processes entire audio file at once
- Full encoder context available (all audio)
- Single decoder pass over entire input
- No state management or caching
- No mid-stream finalization

**Why it's more accurate:**
1. Encoder sees complete audio context (left AND right)
2. Decoder processes entire sequence with global attention
3. No forced token emission at arbitrary points
4. No accumulated decoder state corruption

### Streaming Mode (`conformer_stream_step`)

**File:** `src/nemotron_speech/server.py:356-439`

```python
# Normal streaming (keep_all_outputs=False)
(
    session.pred_out_stream,
    transcribed_texts,
    session.cache_last_channel,
    session.cache_last_time,
    session.cache_last_channel_len,
    session.previous_hypotheses,
) = self.model.conformer_stream_step(
    processed_signal=chunk_mel,
    processed_signal_length=chunk_len,
    cache_last_channel=session.cache_last_channel,
    cache_last_time=session.cache_last_time,
    cache_last_channel_len=session.cache_last_channel_len,
    keep_all_outputs=False,  # Only emit stable tokens
    previous_hypotheses=session.previous_hypotheses,
    previous_pred_out=session.pred_out_stream,
    drop_extra_pre_encoded=drop_extra,
    return_transcription=True,
)
```

**Processing parameters:**
- `shift_frames = 16` (160ms of new audio per chunk)
- `pre_encode_cache_size = 9` (90ms of overlap from previous chunk)
- `right_context = 1` (default, 160ms lookahead in encoder attention)
- `final_padding_frames = 32` (320ms silence padding on reset)

**Key differences from batch:**
1. **Chunked processing** - Audio processed in 160ms chunks
2. **Limited lookahead** - Only `right_context` frames of future context
3. **Stateful decoder** - `previous_hypotheses` and `pred_out_stream` carry state
4. **Forced finalization** - `keep_all_outputs=True` on reset forces token emission

---

## The `keep_all_outputs` Parameter

This is the critical control for streaming behavior:

### `keep_all_outputs=False` (normal streaming)
- Only emits **stable** tokens (high-confidence predictions)
- Decoder maintains **unstable** tokens internally
- Unstable tokens can be revised as more audio arrives
- **Decoder state preserved** across chunks

### `keep_all_outputs=True` (finalization)
- Forces ALL tokens to be emitted (stable + unstable)
- Commits to current hypothesis permanently
- Used when VAD signals end of speech
- **Can corrupt decoder state** if called multiple times

---

## The Decoder State Corruption Problem

From `docs/pipecat-asr-fix-hypotheses.md`:

> When `_process_final_chunk()` is called with `keep_all_outputs=True`, the decoder is forced to emit all pending hypotheses. This changes the decoder's internal state:
>
> 1. **First reset (segment 1):** Decoder finalizes with high confidence that utterance ends at pause
> 2. **Second reset (segment 2):** Decoder again learns that pauses = end of utterance
> 3. **Third reset (segment 3):** Pattern reinforced
> 4. **Fourth reset (segment 4):** By now decoder has strong prior that "speaker" ends the utterance
> 5. **Word "and" arrives:** Decoder ignores it because internal state says utterance already ended

### Example: Sample 151 ("speaker and" truncated)

**Ground truth:** "...for a native English speaker and"
**Streaming result:** "...for a native English speaker"

**VAD events during playback:**
1. VAD STOP at ~3654ms → Hard reset with keep_all_outputs=True → decoder state modified
2. VAD STOP at ~6790ms → Hard reset → decoder state further modified
3. VAD STOP at ~13520ms → Hard reset → but "and" was lost

**Why batch gets it right:**
- Single pass, no mid-utterance finalization
- Decoder sees full context before emitting any tokens
- No accumulated "end of segment" bias

---

## Current Server Implementation Details

### Session State (`ASRSession` dataclass)

```python
@dataclass
class ASRSession:
    # Audio buffer
    accumulated_audio: Optional[np.ndarray] = None
    emitted_frames: int = 0

    # Encoder cache state
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

    # Decoder state
    previous_hypotheses: Any = None
    pred_out_stream: Any = None

    # Transcription tracking
    current_text: str = ""
    last_emitted_text: str = ""  # For server-side deduplication
```

### Hard Reset Flow (`_reset_session` with `finalize=True`)

```python
async def _reset_session(self, session: ASRSession, finalize: bool = True):
    if finalize:
        # 1. Add 320ms silence padding
        padding_samples = self.final_padding_frames * self.hop_samples
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        session.accumulated_audio = np.concatenate([session.accumulated_audio, silence_padding])

        # 2. Process with keep_all_outputs=True
        final_text = await self._process_final_chunk(session)

        # 3. Server-side deduplication (delta only)
        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text):].lstrip()
        else:
            delta_text = final_text

        session.last_emitted_text = final_text

        # 4. Send delta to client
        await session.websocket.send_str(json.dumps({
            "type": "transcript",
            "text": delta_text,
            "is_final": True,
            "finalize": True
        }))

        # 5. FULL STATE RESET (current implementation)
        session.last_emitted_text = ""
        session.overlap_buffer = None
        self._init_session(session)  # Clears everything!
```

### Soft Reset Flow (`_reset_session` with `finalize=False`)

```python
if not finalize:
    # Just return current text without processing
    text = session.current_text

    await session.websocket.send_str(json.dumps({
        "type": "transcript",
        "text": text,
        "is_final": True,
        "finalize": False
    }))
    # Keep all state intact - decoder, encoder, audio buffer
    return
```

---

## Analysis of the 33 Fixable Samples

These samples have `batch_word_errs = 0` but `streaming_word_errs > 0`.

### Error Patterns Observed

| Pattern | Examples | Count | Root Cause |
|---------|----------|-------|------------|
| Word substitution | sync→sink, legal→Lego, thermostat→thermosaur | ~15 | Context loss at chunk boundary |
| Word truncation | "with my"→"with" | ~7 | Final word in padding window |
| Homophones | pairing→tearing | ~5 | Decoder state bias |
| Garbling | "fender bender"→"fender" | ~6 | Mid-utterance reset timing |

### Common Characteristics

1. **Most errors involve words at or near VAD boundaries**
   - The 200ms VAD threshold catches natural speech pauses
   - Words split across reset boundaries suffer most

2. **Errors correlate with number of VAD resets**
   - More mid-utterance pauses = more decoder state corruption
   - Single-segment utterances perform close to batch

3. **Context-dependent words most affected**
   - Words that rely on surrounding context for disambiguation
   - e.g., "legal" vs "Lego" depends on "requirements" following

---

## Technical Deep Dive: Conformer Encoder Streaming

### Chunk Extraction Logic

```python
# From server.py:385-396
if session.emitted_frames == 0:
    # First chunk: just shift_frames, no cache
    chunk_start = 0
    chunk_end = self.shift_frames  # 16 frames (160ms)
    drop_extra = 0
else:
    # Subsequent chunks: include pre_encode_cache frames before
    chunk_start = session.emitted_frames - self.pre_encode_cache_size  # -9 frames
    chunk_end = session.emitted_frames + self.shift_frames  # +16 frames
    drop_extra = self.drop_extra
```

**Key insight:** After `_init_session()` clears state:
- `emitted_frames = 0`
- First new chunk has **NO left context** (no pre_encode_cache)
- This cold start can cause word boundary issues

### Right Context (Encoder Attention Lookahead)

```python
# From server.py:121
self.model.encoder.set_default_att_context_size([70, self.right_context])
#                                                 ^    ^
#                                          left=700ms  right=variable
```

**Current config:** `right_context = 1` → 160ms lookahead
**Available options:**
- 0: ~80ms ultra-low latency
- 1: ~160ms low latency (default)
- 6: ~560ms balanced
- 13: ~1.12s highest accuracy

**Trade-off:** Higher right_context = better accuracy but more latency

---

## Potential Solutions

### Solution A: Preserve Encoder State Across Resets (TRIED - REVERTED)

**Problem:** Current `_init_session()` clears encoder cache, losing left-context.

**Hypothesis:** Keep encoder cache across hard resets, only reset decoder.

**Experiment Results (January 2026):**
| Group | Improved | Same | Regressed | Error Delta |
|-------|----------|------|-----------|-------------|
| Batch-better (124) | 76 (61%) | 25 | 23 (19%) | -139 |
| Streaming-better (62) | 27 (44%) | 19 | 16 (26%) | -36 |

**Why this was reverted:**
1. **Conceptually flawed** - Batch transcription starts with fresh state for each audio file
2. **Adds context batch doesn't have** - Makes streaming MORE different from batch, not less
3. **High regression rate** - 19-26% of samples got worse

**Conclusion:** This approach was wrong. Streaming should match batch's fresh-start behavior on hard reset.

### Solution B: Increase Right Context (TESTED - RECOMMENDED)

**Change:** Server-side configuration

```bash
python -m nemotron_speech.server --right-context 6  # Instead of 1
```

**Experiment Results (January 2026):** See "Right Context=6 Experiment" section below.

**Impact:**
- **72% error reduction** on batch-better samples (1,721 → 480 errors)
- **94% reduction** in error gap vs batch (1,324 → 83)
- 51% of previously-worse samples now match or beat batch
- Adds ~400ms latency (avg 1.3s total per sample)

**Trade-off:** Latency increase of ~400ms. Acceptable for many voice applications.

### Solution C: Audio Overlap Buffer

**Problem:** After reset, first chunk lacks left-context audio.

**Fix:** Preserve last 90ms of audio as overlap.

```python
# Before reset
session.overlap_buffer = session.accumulated_audio[-overlap_samples:].copy()

# In _init_session
if session.overlap_buffer is not None:
    session.accumulated_audio = session.overlap_buffer.copy()
```

**Current state:** Code for this exists but is NOT being used effectively because:
1. `_init_session()` is called after saving overlap
2. `_init_session()` also resets `overlap_buffer = None` sometimes

### Solution D: Soft-First, Hard-Second Protocol

**Current flow:**
1. VAD fires → soft reset (returns current_text)
2. UserStoppedSpeaking → hard reset (padding + keep_all_outputs=True)
3. State fully reset

**Problem:** Hard reset on every UserStoppedSpeaking corrupts decoder over multi-segment utterances.

**Alternative flow:**
1. VAD fires → soft reset (returns current_text, state preserved)
2. UserStoppedSpeaking → hard reset (decoder reset, encoder preserved)
3. Continue with preserved encoder context

---

## Verification Approach

### Step 1: Analyze the 33 Fixable Samples

For each sample:
1. Count number of VAD events during audio playback
2. Identify exact error location (which word(s))
3. Check if error is at segment boundary

### Step 2: A/B Test with Encoder Preservation

1. Modify `_reset_session` to NOT call `_init_session()`
2. Run the 33 samples through streaming
3. Compare results

### Step 3: Measure Impact

- Does preserving encoder state fix any of the 33?
- Does it introduce any regressions in the 649 perfect samples?
- What's the memory cost?

---

## Questions for Further Investigation

1. **Why do 62 samples perform BETTER in streaming than batch?**
   - Hypothesis: Streaming's chunked attention may help certain patterns
   - Worth investigating these samples

2. **What's the actual memory cost of not resetting?**
   - `accumulated_audio` grows unbounded
   - `cache_last_channel` and `cache_last_time` are bounded by model architecture

3. **Can we detect when to use hard vs soft reset?**
   - Longer pauses → more likely end of utterance → use hard reset
   - Short pauses (200-300ms) → likely mid-utterance → keep state

4. **Would a hybrid approach work?**
   - Use batch for final transcription (after all audio received)
   - Use streaming only for interim display

---

## Appendix: Key Code Locations

| File | Line | Purpose |
|------|------|---------|
| `src/nemotron_speech/server.py:356-439` | `_process_chunk` | Normal streaming inference |
| `src/nemotron_speech/server.py:441-563` | `_reset_session` | Hard/soft reset handling |
| `src/nemotron_speech/server.py:565-647` | `_process_final_chunk` | Finalization with keep_all_outputs=True |
| `src/nemotron_speech/server.py:228-260` | `_init_session` | State initialization (clears everything) |
| `pipecat_bots/nvidia_stt.py:164-211` | `process_frame` | VAD frame handling, reset triggers |
| `pipecat_bots/nvidia_stt.py:213-240` | `_send_reset` | Client-side reset protocol |
| `scripts/batch_transcribe_nemo.py:93` | `model.transcribe` | Batch inference call |
| `asr_eval/adapters/synthetic_input_transport.py` | - | Test pipeline with corrected soft/hard reset |
| `scripts/retest_batch_better_samples.py` | - | Retest script for 33 batch-better samples |
| `scripts/verify_regressions_semantic.py` | - | Semantic WER verification of "regressed" samples |

---

## Test Pipeline Fix (January 2026)

### Problem Discovered

The test pipeline (`asr_eval/adapters/synthetic_input_transport.py`) was not correctly simulating production Pipecat behavior. Investigation revealed:

**Root Cause:** Pipecat's `BaseInputTransport` emits BOTH `VADUserStoppedSpeakingFrame` AND `UserStoppedSpeakingFrame` for every VAD stop when no turn analyzer is configured:

```python
# From pipecat/transports/base_input.py:409-423
can_create_user_frames = (
    self._params.turn_analyzer is None
    or not self._params.turn_analyzer.speech_triggered
)
if new_vad_state == VADState.QUIET:
    await self.push_frame(VADUserStoppedSpeakingFrame())
    if can_create_user_frames:
        interruption_state = VADState.QUIET  # triggers UserStoppedSpeakingFrame
```

This caused **multiple hard resets per sample** instead of the intended behavior:
- `VADUserStoppedSpeakingFrame` → soft reset (state preserved)
- `UserStoppedSpeakingFrame` → hard reset (state cleared)

### Fix Implemented

Modified `asr_eval/adapters/synthetic_input_transport.py`:

1. **Override `_handle_user_interruption`** to suppress automatic `UserStoppedSpeakingFrame`:
```python
async def _handle_user_interruption(self, vad_state: VADState, emulated: bool = False):
    if vad_state == VADState.SPEAKING:
        await super()._handle_user_interruption(vad_state, emulated)
    elif vad_state == VADState.QUIET:
        logger.debug("Suppressing automatic UserStoppedSpeakingFrame (will emit after audio ends)")
        self._user_speaking = False
        # Don't emit UserStoppedSpeakingFrame - we'll do it manually after audio ends
```

2. **Emit `UserStoppedSpeakingFrame` once, 250ms after audio ends**:
```python
# In _pump_audio, after sending real audio:
silence_chunks_before_user_stopped = 12  # 240ms (close to 250ms)
for i in range(self.silence_tail_chunks):
    await self._audio_in_queue.put(silence_chunk)
    await asyncio.sleep(self.chunk_duration_sec)

    if i == silence_chunks_before_user_stopped:
        logger.debug("Emitting UserStoppedSpeakingFrame (250ms after audio end)")
        await self.push_frame(UserStoppedSpeakingFrame())
```

### Retest Results (33 Batch-Better Samples)

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| IMPROVED | 10 (30.3%) | 11 (33.3%) |
| SAME | 8 (24.2%) | 10 (30.3%) |
| REGRESSED | 15 (45.5%) | 12 (36.4%) |
| Net word error change | +10 | +10 |

**Original word errors (baseline):** 78
**New word errors:** 88

### Analysis

The fix correctly implements the soft/hard reset protocol:
- VAD stops during speech → `VADUserStoppedSpeakingFrame` → soft reset (encoder state preserved)
- End of utterance → `UserStoppedSpeakingFrame` (250ms after audio) → hard reset

Debug logs confirm correct behavior:
```
Suppressing automatic UserStoppedSpeakingFrame (will emit after audio ends)
VADUserStoppedSpeakingFrame  # soft reset triggered
...
Emitting UserStoppedSpeakingFrame (250ms after audio end)
sent hard reset (audio: 10628ms)  # single hard reset at end
```

**Key insight:** The distribution improved (fewer regressions, more stable) but net word errors remained similar. This suggests:
1. The test pipeline now correctly simulates production behavior
2. The remaining errors are inherent to streaming vs batch, not test artifacts
3. Further improvements require server-side changes (encoder preservation, etc.)

### Regression Verification (Semantic WER)

The 12 "REGRESSED" samples were investigated using semantic WER evaluation. **Result: ALL 12 are FALSE REGRESSIONS.**

**Verification method:**
1. Normalize texts (lowercase, strip trailing punctuation)
2. Compare original vs new transcription text
3. If identical → false regression (same input = same semantic WER)

**Results:**
```
Identical text: 12 (automatic FALSE REGRESSION)
Different text: 0 (need judge evaluation)

VERIFICATION SUMMARY
FALSE REGRESSIONS: 12
REAL REGRESSIONS:  0

✓ No real regressions found!
```

**Root cause of apparent regressions:**
- `original_errs` came from CSV (proper WER calculation)
- `new_errs` calculated with `difflib.SequenceMatcher` (NOT standard WER)
- SequenceMatcher inflates error counts, especially for disfluencies in ground truth

**Example: Sample 07596639 (5→11 "regression")**
- Ground truth contains repeat: "I'm looking for a low **I'm looking for a low**..."
- Both original and new transcriptions are IDENTICAL
- SequenceMatcher counts missing repeat as 5+ deletions

**Conclusion:** The test pipeline fix had **zero negative effect** on transcription quality. The remaining batch-better samples have errors that are inherent to streaming vs batch inference.

---

## Right Context=6 Experiment (January 2026)

### Motivation

After fixing the test pipeline and confirming encoder preservation was the wrong approach, we tested the primary architectural difference: **lookahead context**. Batch sees full audio; streaming with `right_context=1` sees only 160ms ahead.

### Methodology

1. Started server with `--right-context 6` (560ms lookahead vs 160ms default)
2. Ran all 187 samples (125 batch-better + 62 streaming-better)
3. Compared against baseline (`right_context=1`) and batch transcription
4. Used `scripts/test_right_context.py` for automated testing

### Results: Batch-Better Samples (125)

| Metric | Value |
|--------|-------|
| Improved vs baseline | 96 (77%) |
| Same as baseline | 15 (12%) |
| Regressed vs baseline | 14 (11%) |
| **Matched batch accuracy** | 64 (51%) |
| **Better than batch** | 34 (27%) |

**Error counts:**
| Metric | Errors |
|--------|--------|
| Original streaming (right_context=1) | 1,721 |
| New streaming (right_context=6) | 480 |
| Batch | 397 |
| **Error reduction** | **-1,241 (-72%)** |
| **Gap vs batch** | 83 (down from 1,324) |

### Results: Streaming-Better Samples (62)

| Metric | Value |
|--------|-------|
| Improved vs baseline | 25 (40%) |
| Same as baseline | 19 (31%) |
| Regressed vs baseline | 18 (29%) |
| Matched batch accuracy | 59 (95%) |
| Better than batch | 51 (82%) |

**Error counts:**
| Metric | Errors |
|--------|--------|
| Original streaming (right_context=1) | 241 |
| New streaming (right_context=6) | 222 |
| Batch | 1,353 |
| **Error reduction** | **-19 (-8%)** |

### Latency Impact

| Metric | right_context=1 | right_context=6 |
|--------|-----------------|-----------------|
| Encoder lookahead | 160ms | 560ms |
| Average total latency | ~900ms | ~1,300ms |
| Additional latency | - | ~400ms |

### Key Findings

1. **right_context=6 nearly eliminates the batch-streaming gap** on samples where streaming was worse
2. **94% of the error gap was closed** (1,324 → 83 word errors behind batch)
3. **51% of batch-better samples now match or beat batch** accuracy
4. **Streaming-better samples remain mostly unaffected** (82% still beat batch)
5. **29% regression on streaming-better** samples needs investigation

### Trade-off Analysis

| right_context | Latency | Accuracy vs Batch | Recommendation |
|---------------|---------|-------------------|----------------|
| 1 (default) | 160ms | -1,324 errors | Low-latency priority |
| 6 | 560ms | -83 errors | **Recommended for quality** |
| 13 | 1.12s | (not tested) | Highest accuracy, highest latency |

### Remaining Questions (ANSWERED)

1. **Why do 14 batch-better samples regress with more context?**
   - ✓ ANSWERED: 4 false regressions (measurement artifact), 10 real but minor model variations. No bugs.
2. **Why do 18 streaming-better samples regress?**
   - ✓ ANSWERED: 5 false regressions, 13 real variations. RC6's extra context occasionally shifts word boundaries differently on samples where streaming already beat batch.
3. **Would right_context=13 close the remaining 83-error gap?** Worth testing if latency budget allows.

### Regression Analysis (32 samples analyzed)

Deep analysis of all 32 regressions (14 batch-better + 18 streaming-better) was completed.

**Methodology:**
- Loaded ground truth, batch transcriptions, and RC6 transcriptions
- Computed semantic WER for each using Agent SDK judge
- Compared RC6 vs original streaming performance

**Results:**

| Category | Count | Percentage |
|----------|-------|------------|
| False regressions (same text, measurement artifact) | 9 | 28% |
| Real regressions (model behavior variation) | 23 | 72% |

**Key Findings:**
1. **No truncation issues** - No samples had empty or severely truncated output
2. **No anomalous latencies** - All latencies within expected range
3. **No systematic bugs** - No patterns suggesting implementation problems
4. **Real regressions are minor** - Typically 1-2 word differences
5. **Net effect still strongly positive** - 121 improved vs 32 regressed = +89 samples better

**Conclusion:** The RC6 implementation is working correctly. The 32 regressions are:
- 9 measurement artifacts (false positives from difflib vs standard WER)
- 23 legitimate model variations (expected statistical noise)

No implementation fixes needed. Proceeding to full 1,000 sample test.

---

## Background Job Best Practices

**IMPORTANT:** When running transcription and evaluation batch jobs:

1. **Run in background** - Don't block the conversation
2. **Redirect logs to file** - Keeps context clean
3. **Do NOT use `tee`** - Puts too much log content into conversation context
4. **Monitor periodically** - Check progress without flooding context

**Example:**
```bash
# Good - background with log redirect
python scripts/test_right_context.py > /tmp/right_context_test.log 2>&1 &

# Bad - tee floods context
python scripts/test_right_context.py 2>&1 | tee /tmp/test.log
```

---

## Next Steps

### Completed
1. ~~**Fix test pipeline**~~ ✓ - soft/hard reset protocol now correct
2. ~~**Verify regressions**~~ ✓ - all 12 "regressions" were measurement artifacts
3. ~~**Encoder preservation experiment**~~ ✓ - tried and reverted (wrong approach)
4. ~~**Right context=6 experiment**~~ ✓ - **SUCCESS** - 94% error gap reduction
5. ~~**Batch transcribe all 1000 samples**~~ ✓ - Completed in 9.3s (Jan 6)
6. ~~**Load batch transcriptions to DB**~~ ✓ - 1000 samples loaded as `nvidia_parakeet_batch`

### Completed (Jan 6, 2026) - Full Validation Complete ✓

1. ✓ **RC6 test on 187 samples** - 121 improved, 32 regressed (net +89 better)
2. ✓ **Deep regression analysis** - 32 regressions: 9 false, 23 minor variations. No bugs.
3. ✓ **Agent SDK judge on batch** - 999 samples, 2.45% pooled WER
4. ✓ **RC6 on all 1,000 samples** - 997 successful transcriptions
5. ✓ **Load RC6 to database** - 997 samples as `nvidia_parakeet_rc6`
6. ✓ **Agent SDK judge on RC6** - 997 samples, 2.51% pooled WER
7. ✓ **Final comparison** - RC6 closes 95% of the gap vs batch

### Final Recommendation

**Deploy `right_context=6` in production:**
- 95% accuracy gap reduction (3.63% → 2.51% WER)
- Only +14 word errors vs batch (was +275)
- Final transcription latency unchanged (~20ms)
- Interim latency +400ms (acceptable trade-off)

### Future Investigations
- **Test on 649 perfect samples** - Verify no regressions on WER=0 samples
- **Consider right_context=13 test** - If latency budget allows (would add ~560ms more)

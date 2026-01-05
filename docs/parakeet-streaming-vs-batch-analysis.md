# Parakeet Streaming vs Batch Transcription Analysis

**Date:** 2026-01-05
**Model:** Parakeet_Realtime_En_600M.nemo

## Summary

This analysis compares Parakeet ASR results between:
- **Streaming mode**: WebSocket-based chunked inference (used in production)
- **Batch mode**: NeMo's native `model.transcribe()` with full audio context

The goal was to understand why Parakeet has a relatively high WER (3.63%) compared to other ASR services, and determine whether errors are due to the streaming implementation or the model itself.

## Methodology

1. Identified 346 samples where streaming Parakeet had WER > 0 (semantic errors)
2. Re-transcribed all 346 samples using NeMo's batch API inside the container
3. Compared streaming vs batch results against ground truth

## Results

### Overall Comparison

| Comparison | Count | Percentage |
|------------|-------|------------|
| Exact match (streaming == batch) | 68 | 19.7% |
| **Batch better than streaming** | **125** | **36.1%** |
| Streaming better than batch | 62 | 17.9% |
| Same quality (both have errors) | 159 | 46.0% |

### Key Finding

**Batch mode produces better results 2:1 compared to streaming mode.**

- Batch better: 125 samples
- Streaming better: 62 samples
- Net difference: **~63 errors are attributable to streaming implementation**

This suggests that approximately **18% of Parakeet's errors** (63 out of 346) could potentially be eliminated by improving the streaming implementation.

## Error Categories

### 1. Truncation Issues
Some streaming results are severely truncated while batch captures full content.

**Example:**
```
GT:     "Okay, please describe the geological process of erosion..."
Stream: "" (25 word errors - severe truncation)
Batch:  "Can you please describe the geological process of erosion..." (7 word errors)
```

### 2. Word Substitution Differences
Different words recognized between streaming and batch modes.

**Example:**
```
GT:     "What are the legal requirements..."
Stream: "What are the Lego requirements..." (error)
Batch:  "What are the legal requirements..." (correct)
```

### 3. Word Drops
Streaming sometimes drops words that batch captures.

**Example:**
```
GT:     "...so that they don't flag my credit card..."
Stream: "...so that I don't flag my credit card..."
Batch:  "...so that don't flag my credit card..." (missing "I")
```

### 4. Punctuation-Only Differences
57 samples differed only in punctuation (commas, periods, contractions).

**Example:**
```
Stream: "I have an upcoming trip to Europe next month, and I need..."
Batch:  "I have an upcoming trip to Europe next month and I need..."
```

## Implications

### Streaming Implementation Issues
The 2:1 ratio of batch-better vs streaming-better suggests the streaming implementation has room for improvement:

1. **Right-context limitations**: Streaming uses limited lookahead (160ms default) while batch sees full audio
2. **Chunk boundary effects**: Word recognition at chunk boundaries may be suboptimal
3. **State management**: Encoder/decoder cache handling may have edge cases

### Model Limitations
159 samples (46%) had the same quality in both modes, indicating these errors are inherent to the model:

- Acoustic confusion (e.g., "form" vs "forum")
- Vocabulary gaps
- Audio quality issues in source recordings

## Files Generated

- **`docs/parakeet_error_analysis.csv`** - Full analysis with all 346 samples
- `asr_eval_data/parakeet_batch_transcriptions.jsonl` - Raw batch transcription results

### CSV Columns

| Column | Description |
|--------|-------------|
| sample_id | Unique sample identifier |
| ground_truth | Reference transcription |
| streaming_transcription | Result from streaming WebSocket API |
| batch_transcription | Result from NeMo batch API |
| streaming_wer | WER from Agent SDK judge (semantic) |
| substitutions/deletions/insertions | Error breakdown |
| streaming_matches_batch | Boolean: exact string match |
| comparison | batch_better / streaming_better / same |
| streaming_word_errs | Simple word error count for streaming |
| batch_word_errs | Simple word error count for batch |

## Recommendations

1. **Investigate truncation cases**: The severe truncation in some streaming samples suggests potential issues with VAD or end-of-utterance detection

2. **Review right-context settings**: Consider testing with higher right-context values (6 or 13) to see if accuracy improves

3. **Analyze chunk boundary errors**: Look for patterns in where streaming makes errors vs batch to identify chunking issues

4. **Focus optimization efforts**: The 63 "fixable" errors represent low-hanging fruit for improving streaming WER

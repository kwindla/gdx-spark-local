# Ground Truth Transcription Evaluation

This document explains how to generate and evaluate ground truth transcriptions using Gemini 3.0 Flash.

## Overview

The ground truth system uses Gemini 3.0 Flash with "minimal" thinking level to transcribe audio samples. This configuration provides the best phonetic accuracy with minimal semantic substitution (e.g., won't substitute "daughter's" for "doctor's" based on context).

## Prerequisites

- Google API key set in environment (`GOOGLE_API_KEY`)
- Audio samples already downloaded in `asr_eval_data/audio/`

## Running a Transcription Iteration

Generate ground truth transcriptions for 100 samples:

```bash
uv run python -m asr_eval.ground_truth.run_iteration --samples 100 --clear
```

Options:
- `--samples N` or `-n N`: Number of samples to transcribe (default: 100)
- `--clear` or `-c`: Clear existing ground truths before running

The script:
1. Selects the first N samples by `dataset_index` (deterministic/repeatable)
2. Transcribes each using Gemini 3.0 Flash with minimal thinking
3. Saves results to `asr_eval_data/ground_truth_runs/{timestamp}.jsonl`

## Interactive Evaluation

List available runs:

```bash
uv run python -m asr_eval.ground_truth.evaluate_run --list
```

Evaluate a specific run:

```bash
uv run python -m asr_eval.ground_truth.evaluate_run <run_id>
```

Example:
```bash
uv run python -m asr_eval.ground_truth.evaluate_run 2026-01-03_16-29-16
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `p` | Play audio |
| `r` | Replay audio |
| `a` | Approve transcription |
| `n` | Add a note (prompts for text) |
| `Enter` | Skip to next sample |
| `q` | Quit and save progress |

### Output Files

- `asr_eval_data/ground_truth_runs/{run_id}.jsonl` - Transcription results
- `asr_eval_data/ground_truth_runs/{run_id}_notes.jsonl` - Evaluation notes

## JSONL Format

Run files use JSONL format for easy inspection with bash tools:

```bash
# View all transcriptions
cat asr_eval_data/ground_truth_runs/*.jsonl | jq 'select(.type=="sample")'

# Count total words
cat run.jsonl | jq -r 'select(.type=="sample") | .transcription' | wc -w

# Find samples with errors
cat run_notes.jsonl | jq 'select(.status=="noted")'
```

## Configuration

The transcription prompt and model settings are in:
- `asr_eval/ground_truth/gemini_transcriber.py`

Key settings:
- Model: `gemini-3-flash-preview`
- Thinking level: `minimal` (best for phonetic accuracy)
- Temperature: `0.0` (deterministic)

## Baseline Results

As of 2026-01-03:
- 100 samples, 2,290 total words
- 4 word-level errors
- **WER: 0.17%**

## Full Dataset Run

Complete 5000-sample ground truth run:
- **File**: `asr_eval_data/ground_truth_runs/2026-01-03_17-00-06.jsonl`
- **Samples**: 5000/5000 successful, 0 errors
- **Model**: Gemini 3.0 Flash (minimal thinking)

# Agent SDK Semantic WER Judge

A Claude-powered judge for evaluating ASR transcription quality using **semantic Word Error Rate** — counting only errors that would impact how an LLM agent understands and responds to the user.

## Why Semantic WER?

Traditional WER counts every word difference equally. But for conversational AI, we only care about errors that change meaning:

| Difference | Traditional WER | Semantic WER |
|------------|-----------------|--------------|
| `"driver's license"` → `"driver licenses"` | 2 errors | 0 errors (same meaning) |
| `"card"` → `"car"` | 1 error | 1 error (different meaning) |
| `"Wi-Fi"` → `"wi fire"` | 1 error | 1 error (nonsense) |
| `"offices"` → `"office"` | 1 error | 0 errors (same intent) |

**Key principle:** If an LLM would interpret both versions the same way, it's not an error.

---

## Quick Start

### Prerequisites

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# Install dependencies
uv sync --extra asr-eval
```

### Evaluate a Single Text Pair

```python
import asyncio
from asr_eval.evaluation.agent_sdk_judge import AgentSDKJudge

async def evaluate():
    judge = AgentSDKJudge()

    reference = "I need to change my driver's license after getting married."
    hypothesis = "I need to change my driver licenses after getting married."

    result, trace = await judge.evaluate(reference, hypothesis)

    print(f"Semantic WER: {result['wer']:.2%}")
    print(f"Errors: S={result['substitutions']}, D={result['deletions']}, I={result['insertions']}")

asyncio.run(evaluate())
```

Output:
```
Semantic WER: 0.00%
Errors: S=0, D=0, I=0
```

---

## End-to-End Evaluation Pipeline

**CRITICAL: Understand this data flow before running evaluations.**

```
Audio Files (LibriSpeech via HuggingFace)
    ↓
Batch Transcription (run_batch_transcription.py)
    ↓
JSONL Files (asr_eval_data/transcription_runs/*.jsonl)
    ↓
Import Script (scripts/import_transcriptions.py)  ← REQUIRED STEP
    ↓
SQLite Database (asr_eval_data/results.db)
    ↓
Judge Script (scripts/run_judge.py)
    ↓
Results JSON (asr_eval_data/judge_results_*.json)
```

---

## Supported STT Services

| Service | Model | Notes |
|---------|-------|-------|
| Deepgram Nova-3 | `nova-3` | Best accuracy for general use |
| Deepgram Whisper | `whisper-large` | OpenAI Whisper via Deepgram API |
| NVIDIA Parakeet | Local | Requires local ASR server |
| ElevenLabs | `scribe_v2_realtime` | Streaming transcription |
| Cartesia | `ink-whisper` | Streaming-first design |

---

## Step 1: Batch Transcription

### Running Batch Transcriptions

**Start a batch transcription job:**
```bash
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
nohup uv run python -m asr_eval.run_batch_transcription \
  --service deepgram --model nova-3 --samples 1000 \
  > "asr_eval_data/logs/deepgram_nova3_${TIMESTAMP}.log" 2>&1 &
echo "PID=$! LOG=asr_eval_data/logs/deepgram_nova3_${TIMESTAMP}.log"
```

**Available services:**
```bash
--service deepgram --model nova-3
--service deepgram --model whisper-large
--service nvidia_parakeet
--service elevenlabs
--service cartesia
```

**Monitor progress:**
```bash
tail -f asr_eval_data/logs/deepgram_nova3_*.log
```

### VAD Configuration

The transcription adapter uses **Silero VAD** for speech detection:

```python
adapter = PipelineSTTAdapter(
    service_name=ServiceName.DEEPGRAM,
    model="nova-3",
    vad_stop_secs=0.2,           # 200ms silence triggers VAD stop
    post_audio_silence_ms=2000,  # 2s silence keeps pipeline alive
)
```

### Latency Measurement

Latency measures time from **actual end of speech** to **final transcription**:

```
Latency = (Last_TranscriptionFrame_time - Last_VAD_stop_time) + vad_stop_secs
```

**Timeline:**
1. **T0**: Actual speech ends in audio
2. **T0 + 0.2s**: Silero VAD detects silence, emits VADUserStoppedSpeakingFrame
3. **Tf**: Final TranscriptionFrame received
4. **Latency** = (Tf - T_vad) + 0.2s

### Results Location

Batch transcription results are stored as JSONL:
```
asr_eval_data/transcription_runs/{service}_{model}_{timestamp}.jsonl
```

**IMPORTANT:** Results are NOT automatically imported into the database. See next section.

---

## Step 2: Import Transcriptions

### Warning: Transcription Jobs Write to JSONL, Not Database

Transcription batch jobs output to JSONL files in `asr_eval_data/transcription_runs/`, for example:
```
deepgram_nova-3_2026-01-04_07-15-19.jsonl
elevenlabs_scribe_v2_realtime_2026-01-04_07-15-35.jsonl
```

**The judge reads from the database, NOT from JSONL files.** You must import transcriptions before running the judge.

### Import Transcriptions to Database

```bash
# Import a single JSONL file
uv run --extra asr-eval python scripts/import_transcriptions.py \
  asr_eval_data/transcription_runs/deepgram_nova-3_2026-01-04_07-15-19.jsonl

# Import multiple files
uv run --extra asr-eval python scripts/import_transcriptions.py \
  asr_eval_data/transcription_runs/deepgram_nova-3_2026-01-04_07-15-19.jsonl \
  asr_eval_data/transcription_runs/elevenlabs_scribe_v2_realtime_2026-01-04_07-15-35.jsonl
```

The import script maps JSONL (service, model) to database service_name:
| JSONL Service | JSONL Model | DB service_name |
|---------------|-------------|-----------------|
| deepgram | nova-3 | `deepgram` |
| deepgram | whisper-large | `faster_whisper` |
| cartesia | default | `cartesia` |
| elevenlabs | scribe_v2_realtime | `elevenlabs` |

### Verify Database Has Fresh Data

Before running the judge, verify the database has recent transcriptions:

```bash
sqlite3 asr_eval_data/results.db "SELECT service_name, MAX(timestamp) FROM transcriptions GROUP BY service_name"
```

If timestamps are old, you need to import the latest JSONL files.

---

## Model Selection: Sonnet vs Opus

The judge defaults to **Sonnet 4.5** (fast, cost-effective) but also supports Opus 4.5:

```python
# Default: Sonnet (fast, cheap, 90% agreement with Opus)
judge = AgentSDKJudge()
judge = AgentSDKJudge(model="claude-sonnet-4-5-20250929")

# Opus (highest accuracy, for final validation)
judge = AgentSDKJudge(model="claude-opus-4-5-20251101")
```

### Comparison (tested on 20 samples)

| Model | Agreement | Avg Time | Cost/Sample |
|-------|-----------|----------|-------------|
| **Sonnet 4.5 (default)** | 90% match | ~3s | ~$0.02 |
| Opus 4.5 | baseline | ~18s | ~$0.11 |

**Recommendation:** Use Sonnet (default) for bulk evaluation. Use Opus when you need the most detailed reasoning traces or for final validation.

---

## CLI Usage

### Evaluate Samples in Database

```bash
# Evaluate all samples for a service (skips already-evaluated)
uv run --extra asr-eval python -m asr_eval.cli.run_evaluation agent-sdk-wer -s nvidia_parakeet

# Force re-evaluation of all samples
uv run --extra asr-eval python -m asr_eval.cli.run_evaluation agent-sdk-wer -s nvidia_parakeet --force

# Limit to N samples
uv run --extra asr-eval python -m asr_eval.cli.run_evaluation agent-sdk-wer -s nvidia_parakeet --limit 10
```

### Available Services

- `nvidia_parakeet`
- `deepgram`
- `faster_whisper`
- `cartesia`
- `elevenlabs`

---

## Data Storage Format

All evaluation data is stored in a SQLite database.

### Database Location

```
asr_eval_data/results.db
```

### Database Schema

#### `samples` - Audio sample metadata
```sql
sample_id        TEXT PRIMARY KEY   -- UUID for the sample
audio_path       TEXT NOT NULL      -- Path to PCM audio file
duration_seconds REAL NOT NULL      -- Audio duration
language         TEXT NOT NULL      -- Language code (e.g., "en")
synthetic        INTEGER NOT NULL   -- 1 if TTS-generated, 0 if real
dataset_index    INTEGER NOT NULL   -- Index in source dataset
```

#### `ground_truth` - Reference transcriptions
```sql
sample_id    TEXT PRIMARY KEY   -- References samples.sample_id
text         TEXT NOT NULL      -- The correct transcription
model_used   TEXT NOT NULL      -- Model that generated it (e.g., "gemini-2.0-flash")
generated_at TEXT NOT NULL      -- ISO timestamp
```

#### `transcriptions` - ASR service outputs
```sql
id                      INTEGER PRIMARY KEY
sample_id               TEXT NOT NULL      -- References samples.sample_id
service_name            TEXT NOT NULL      -- e.g., "nvidia_parakeet", "deepgram"
transcribed_text        TEXT NOT NULL      -- ASR output (empty string if failed)
time_to_transcription_ms REAL NOT NULL     -- Latency
audio_duration_ms       REAL NOT NULL      -- Audio length
rtf                     REAL NOT NULL      -- Real-time factor
timestamp               TEXT NOT NULL      -- ISO timestamp
error                   TEXT               -- Error message if transcription failed
adapter_type            TEXT NOT NULL      -- "direct" or "pipecat"
-- UNIQUE(sample_id, service_name)
```

#### `metrics` - Evaluation results
```sql
id              INTEGER PRIMARY KEY
sample_id       TEXT NOT NULL      -- References samples.sample_id
service_name    TEXT NOT NULL      -- e.g., "nvidia_parakeet"
wer             REAL NOT NULL      -- Raw word error rate
cer             REAL NOT NULL      -- Character error rate
substitutions   INTEGER NOT NULL   -- Raw substitution count
deletions       INTEGER NOT NULL   -- Raw deletion count
insertions      INTEGER NOT NULL   -- Raw insertion count
reference_words INTEGER NOT NULL   -- Total words in reference
semantic_wer    REAL               -- Legacy semantic WER
semantic_errors TEXT               -- Legacy JSON errors list
gemini_wer      REAL               -- Gemini-normalized WER
agent_sdk_wer   REAL               -- Agent SDK semantic WER (main metric)
timestamp       TEXT NOT NULL      -- ISO timestamp
-- UNIQUE(sample_id, service_name)
```

#### `agent_sdk_traces` - Full reasoning traces
```sql
id                   INTEGER PRIMARY KEY
sample_id            TEXT NOT NULL      -- References samples.sample_id
service_name         TEXT NOT NULL      -- e.g., "nvidia_parakeet"
session_id           TEXT NOT NULL      -- Unique evaluation session ID
conversation_trace   TEXT NOT NULL      -- JSON: full Claude message history
tool_calls           TEXT NOT NULL      -- JSON: all calculate_wer invocations
normalized_reference TEXT               -- Text after normalization
normalized_hypothesis TEXT              -- Text after normalization
alignment            TEXT               -- JSON: word-level alignment
agent_sdk_wer        REAL NOT NULL      -- Calculated semantic WER
substitutions        INTEGER NOT NULL   -- Semantic substitutions
deletions            INTEGER NOT NULL   -- Semantic deletions
insertions           INTEGER NOT NULL   -- Semantic insertions
reference_words      INTEGER NOT NULL   -- Words in normalized reference
errors               TEXT               -- JSON: list of identified errors
total_cost_usd       REAL               -- API cost for this evaluation
duration_ms          INTEGER            -- Evaluation time in milliseconds
num_turns            INTEGER NOT NULL   -- Number of conversation turns
model_used           TEXT NOT NULL      -- e.g., "claude-opus-4-5-20251101"
timestamp            TEXT NOT NULL      -- ISO timestamp
-- UNIQUE(sample_id, service_name)
```

#### `human_reviews` - Manual review tracking
```sql
id                INTEGER PRIMARY KEY
sample_id         TEXT NOT NULL      -- References samples.sample_id
service_name      TEXT NOT NULL
review_status     TEXT NOT NULL      -- "pending", "approved", "rejected"
flagged_reason    TEXT               -- Why flagged for review
agent_sdk_wer     REAL               -- Agent SDK result
gemini_wer        REAL               -- Gemini result for comparison
wer_disagreement  REAL               -- Absolute difference
human_approved_wer REAL              -- Human-verified WER
human_notes       TEXT               -- Reviewer comments
reviewed_by       TEXT               -- Reviewer name
reviewed_at       TEXT               -- ISO timestamp
code_version      TEXT               -- For tracking prompt changes
created_at        TEXT NOT NULL      -- ISO timestamp
-- UNIQUE(sample_id, service_name)
```

### Querying Results

```python
import asyncio
import aiosqlite

async def query_results():
    async with aiosqlite.connect('asr_eval_data/results.db') as db:
        db.row_factory = aiosqlite.Row

        # Get all Agent SDK WER results for a service
        cursor = await db.execute('''
            SELECT sample_id, agent_sdk_wer, wer as raw_wer
            FROM metrics
            WHERE service_name = "nvidia_parakeet"
            AND agent_sdk_wer IS NOT NULL
            ORDER BY agent_sdk_wer DESC
        ''')
        results = await cursor.fetchall()

        for row in results:
            print(f"{row['sample_id']}: semantic={row['agent_sdk_wer']:.2%}, raw={row['raw_wer']:.2%}")

asyncio.run(query_results())
```

---

## Running at Scale

### Prerequisites for Evaluation

Before running Agent SDK evaluation, you need:

1. **Samples in database** - Audio files registered in `samples` table
2. **Ground truth** - Reference transcriptions in `ground_truth` table
3. **Transcriptions** - ASR outputs in `transcriptions` table

Check readiness:
```python
async with aiosqlite.connect('asr_eval_data/results.db') as db:
    # Samples ready for evaluation (have both GT and transcription)
    cursor = await db.execute('''
        SELECT COUNT(*) FROM metrics m
        JOIN ground_truth g ON m.sample_id = g.sample_id
        JOIN transcriptions t ON m.sample_id = t.sample_id
            AND t.service_name = m.service_name
        WHERE m.service_name = "nvidia_parakeet"
        AND t.transcribed_text != ""
    ''')
    count = (await cursor.fetchone())[0]
    print(f"Ready for evaluation: {count} samples")
```

### Efficient Batch Evaluation

The CLI `--limit` flag limits total samples processed, not samples with ground truth. For efficient evaluation of only samples that have ground truth:

```python
import asyncio
import aiosqlite
from asr_eval.evaluation.agent_sdk_judge import AgentSDKJudge
from asr_eval.models import ServiceName

async def evaluate_with_ground_truth():
    # Get only sample IDs that have ground truth and non-empty transcriptions
    async with aiosqlite.connect('asr_eval_data/results.db') as db:
        cursor = await db.execute('''
            SELECT m.sample_id
            FROM metrics m
            JOIN ground_truth g ON m.sample_id = g.sample_id
            JOIN transcriptions t ON m.sample_id = t.sample_id
                AND t.service_name = m.service_name
            WHERE m.service_name = "nvidia_parakeet"
            AND t.transcribed_text != ""
            AND m.agent_sdk_wer IS NULL  -- Not yet evaluated
        ''')
        sample_ids = [row[0] for row in await cursor.fetchall()]

    print(f"Evaluating {len(sample_ids)} samples")

    judge = AgentSDKJudge()

    for i, sample_id in enumerate(sample_ids):
        metrics = await judge.evaluate_sample(sample_id, ServiceName.NVIDIA_PARAKEET)
        if metrics:
            print(f"[{i+1}/{len(sample_ids)}] {sample_id}: {metrics.agent_sdk_wer:.2%}")

asyncio.run(evaluate_with_ground_truth())
```

### Scale Estimates

| Samples | Time (Sonnet default) | Time (Opus) | Cost (Sonnet) | Cost (Opus) |
|---------|----------------------|-------------|---------------|-------------|
| 100 | ~5 min | ~30 min | ~$2 | ~$11 |
| 1,000 | ~50 min | ~5 hours | ~$20 | ~$110 |
| 5,000 | ~4 hours | ~25 hours | ~$100 | ~$550 |

### Resume After Interruption

Evaluation is idempotent. Results are stored after each sample. To resume:

```bash
# Without --force, skips already-evaluated samples
uv run --extra asr-eval python -m asr_eval.cli.run_evaluation agent-sdk-wer -s nvidia_parakeet
```

### Monitoring Progress

```python
# Check evaluation progress
async with aiosqlite.connect('asr_eval_data/results.db') as db:
    cursor = await db.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN agent_sdk_wer IS NOT NULL THEN 1 ELSE 0 END) as evaluated
        FROM metrics
        WHERE service_name = "nvidia_parakeet"
    ''')
    row = await cursor.fetchone()
    print(f"Progress: {row[1]}/{row[0]} ({row[1]/row[0]*100:.1f}%)")
```

---

## What Counts as an Error?

### NOT Errors (LLM understands both the same)

- Punctuation: `"Hello, world!"` → `"Hello world"`
- Possessives: `"driver's"` → `"drivers"`
- Singular/plural: `"offices"` → `"office"`
- Hyphenation: `"cross-country"` → `"cross country"`
- Contractions: `"I'm"` → `"I am"`
- Minor grammar: `"it is"` → `"it"`
- Truncated words: `"reme"` → `"remember"` (at end of truncated text)

### YES Errors (changes meaning for LLM)

- Different words: `"card"` → `"car"`, `"trace"` → `"trade"`
- Nonsense: `"lentil"` → `"landon"`, `"Wi-Fi"` → `"wi fire"`
- Key deletions: `"fender bender"` → `"fender"` (loses "minor accident" meaning)
- Subject changes: `"they don't flag"` → `"I don't flag"` (changes who acts)

### Special Rules

**Compound words:** Hyphenated compounds count as ONE error, not multiple.
- `"cross-country"` → `"koscanti"` = 1 substitution (not S=1, D=1)

**Truncated text:** When reference and hypothesis are both truncated, partial words at the truncation point are ignored.
- `"traveling abor"` vs `"traveling aboard"` = 0 errors (both incomplete)

---

## Pooled WER

For aggregate statistics, use pooled WER (sum of all errors / sum of all reference words):

```python
judge = AgentSDKJudge()
pooled = await judge.compute_pooled_wer(ServiceName.NVIDIA_PARAKEET)

print(f"Pooled WER: {pooled['pooled_wer']:.2%}")
print(f"Total errors: {pooled['total_errors']}")
print(f"Total words: {pooled['total_reference_words']}")
```

This weights longer utterances more heavily, giving a more accurate overall error rate.

---

## Viewing Reasoning Traces

Every evaluation stores the full Claude conversation for debugging:

```python
from asr_eval.storage.database import Database
from asr_eval.models import ServiceName

db = Database()
await db.initialize()

trace = await db.get_agent_sdk_trace(sample_id, ServiceName.NVIDIA_PARAKEET)

# Print the reasoning
for msg in trace.conversation_trace:
    if msg['role'] == 'assistant':
        for block in msg.get('content', []):
            if block.get('type') == 'text':
                print(block['text'])
```

The trace shows:
1. **Normalization** — lowercase, remove punctuation, expand contractions
2. **Alignment** — word-by-word comparison
3. **Semantic Check** — for each difference, "Would an LLM respond differently?"
4. **Count** — only count semantically meaningful errors
5. **Calculate** — final WER computation

---

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or add to `.env` file in project root.

### "No ground truth for sample_id"

The sample doesn't have a reference transcription. Generate ground truth first:

```bash
uv run --extra asr-eval python -m asr_eval.cli.generate_ground_truth --batch-size 100
```

### Empty transcriptions (100% WER)

Some samples may have empty transcriptions due to ASR service errors. Check:

```python
cursor = await db.execute('''
    SELECT sample_id, error FROM transcriptions
    WHERE service_name = "nvidia_parakeet"
    AND transcribed_text = ""
''')
```

### Need higher accuracy

Use Opus instead of Sonnet for maximum accuracy:

```python
judge = AgentSDKJudge(model="claude-opus-4-5-20251101")
```

---

## Output Files Summary

| File/Location | Contents |
|--------------|----------|
| `asr_eval_data/results.db` | SQLite database with all evaluation data |
| `asr_eval_data/audio/*.pcm` | Raw audio files (16kHz, 16-bit, mono) |
| `asr_eval_data/ground_truth_runs/*.jsonl` | Ground truth transcription runs |
| `asr_eval_data/transcription_runs/*.jsonl` | **Transcription batch job outputs (must be imported to DB before judging)** |
| `asr_eval_data/judge_results_*.json` | Judge evaluation results with S/D/I/N for pooled WER |

### Ground Truth Data

The 5,000 sample ground truth dataset is stored as JSONL:

```
asr_eval_data/ground_truth_runs/2026-01-03_17-00-06.jsonl
```

Format:
```json
{"type": "header", "run_id": "...", "model": "gemini-3-flash-preview", "num_samples": 5000, ...}
{"type": "sample", "sample_id": "uuid", "transcription": "...", "generated_at": "..."}
{"type": "sample", "sample_id": "uuid", "transcription": "...", "generated_at": "..."}
...
```

Load ground truth programmatically:
```python
import json

gt_data = {}
with open('asr_eval_data/ground_truth_runs/2026-01-03_17-00-06.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'sample':
            gt_data[data['sample_id']] = data['transcription']

print(f"Loaded {len(gt_data)} ground truth samples")
```

### Running the Judge

Use `scripts/run_judge.py` to evaluate any ASR service:

```bash
# Available services
python scripts/run_judge.py --service nvidia_parakeet
python scripts/run_judge.py --service deepgram
python scripts/run_judge.py --service elevenlabs
python scripts/run_judge.py --service cartesia
python scripts/run_judge.py --service faster_whisper
```

#### Foreground (see output live)
```bash
PYTHONPATH=/home/khkramer/src/gdx-spark-local uv run --extra asr-eval python scripts/run_judge.py --service nvidia_parakeet
```

#### Background (recommended for full runs)
```bash
PYTHONPATH=/home/khkramer/src/gdx-spark-local nohup uv run --extra asr-eval python scripts/run_judge.py --service nvidia_parakeet > asr_eval_data/judge_run_nvidia_parakeet_2026-01-04.log 2>&1 &
echo "PID: $!"
```

#### Monitor progress
```bash
tail -f asr_eval_data/judge_run_nvidia_parakeet_2026-01-04.log
```

#### Output files

| File | Contents |
|------|----------|
| `asr_eval_data/judge_run_<service>_<date>.log` | Progress log with per-sample WER |
| `asr_eval_data/judge_results_<service>_<date>.json` | Full results with S/D/I/N for pooled WER |

The JSON results file contains `reference_words` for each sample, required for pooled WER calculation:
```json
{
  "service": "nvidia_parakeet",
  "results": [
    {
      "sample_id": "...",
      "wer": 0.0323,
      "substitutions": 1,
      "deletions": 0,
      "insertions": 0,
      "reference_words": 31
    }
  ]
}
```

#### Performance estimates

~3 seconds per sample with Sonnet. For 1000 samples: ~50 minutes.


### Pooled WER Calculation (Post-Run)

After all evaluation runs complete, calculate **Pooled WER** correctly from the JSON result files:

```python
import json

def calculate_pooled_wer(results_file: str) -> dict:
    """
    Pooled WER = sum(all errors) / sum(all reference words)

    This weights each WORD equally (not each sample).
    A 50-word utterance with 2 errors contributes more to the
    denominator than a 10-word utterance with 0 errors.
    """
    total_errors = 0
    total_ref_words = 0

    with open(results_file, 'r') as f:
        results = json.load(f)

    for result in results:
        S = result['substitutions']
        D = result['deletions']
        I = result['insertions']
        N = result['reference_words']

        total_errors += (S + D + I)
        total_ref_words += N

    pooled_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

    return {
        'pooled_wer': pooled_wer,
        'total_errors': total_errors,
        'total_reference_words': total_ref_words,
        'num_samples': len(results)
    }

# Calculate for each service
for service in ['', '_elevenlabs', '_cartesia', '_deepgram', '_faster_whisper']:
    path = f'asr_eval_data/judge_results{service}_2026-01-03.json'
    result = calculate_pooled_wer(path)
    print(f"{service or 'nvidia_parakeet'}: {result['pooled_wer']:.2%} ({result['total_errors']}/{result['total_reference_words']} words)")
```

**Important:** Do NOT use naive WER averaging (averaging per-sample WER percentages). This incorrectly weights short utterances the same as long ones. Pooled WER is the standard method for aggregate ASR evaluation.

### Key Tables for Analysis

| Table | Use Case |
|-------|----------|
| `metrics` | Final WER scores (`agent_sdk_wer` column) |
| `agent_sdk_traces` | Full reasoning for debugging |
| `ground_truth` | Reference transcriptions |
| `transcriptions` | ASR outputs to compare |

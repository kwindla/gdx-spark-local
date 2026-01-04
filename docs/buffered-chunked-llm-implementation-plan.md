# Buffered Chunked LLM Implementation Plan

## Overview

This document describes a redesigned chunked LLM service that eliminates mid-stream HTTP cancellation by decoupling LLM generation boundaries from TTS emission boundaries. The key insight: let LLM generations run to completion (preserving KV cache), while a client-side buffer manages sentence-boundary chunking for TTS.

## Goals

1. **100% KV cache reuse**: Single slot, no mid-stream cancellation, continuous cache accumulation
2. **Sentence-boundary TTS**: Preserve prosody by emitting complete sentences to TTS
3. **Fast TTFB**: Optimized first-segment generation for quick initial response
4. **Simplified code**: Remove two-slot alternation, reuse guards, and race condition mitigations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LlamaCppBufferedLLMService                             │
│                                                                             │
│  ┌────────────────┐      ┌─────────────────┐      ┌───────────────────┐    │
│  │  LLM Generator │      │  SentenceBuffer │      │   TTS Emitter     │    │
│  │                │      │                 │      │                   │    │
│  │  - Single slot │─────►│  - Accumulates  │─────►│  - Emits complete │    │
│  │  - max_tokens  │      │  - Extracts at  │      │    sentences      │    │
│  │  - Runs to     │      │    boundaries   │      │  - Waits for      │    │
│  │    completion  │      │  - Keeps tail   │      │    continue       │    │
│  └────────────────┘      └─────────────────┘      └───────────────────┘    │
│         │                                                   │              │
│         │              ┌──────────────────┐                 │              │
│         └──────────────│  Cache State     │◄────────────────┘              │
│                        │  - prompt        │                                │
│                        │  - generated_text│  (accumulates all generated   │
│                        └──────────────────┘   text for cache continuity)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Difference from Current Approach

| Aspect | Current (Sentence Cancel) | New (Buffered) |
|--------|---------------------------|----------------|
| LLM generation | Cancel mid-stream at sentence | Run to max_tokens completion |
| HTTP connection | Closed mid-stream (race condition) | Completes naturally |
| Slot usage | Two slots with 2s reuse guard | Single slot always |
| KV cache hits | ~67% (slot alternation loses context) | ~100% (continuous) |
| Buffer location | Implicit (pending_token only) | Explicit SentenceBuffer |
| Complexity | High (slot management, guards) | Moderate (buffer management) |

## Configuration Parameters

```python
class InputParams(BaseModel):
    # First segment: quick TTFB, single generation then emit
    first_segment_max_tokens: int = 24      # Tokens to generate
    first_segment_hard_max_tokens: int = 24 # Force emit threshold (same as max)

    # Subsequent segments: allow accumulation for complete sentences
    segment_max_tokens: int = 32            # Tokens per generation
    segment_hard_max_tokens: int = 96       # Force emit after ~3 generations

    # LLM generation parameters
    # Note: With temperature=0.0, top_p/top_k have no effect (greedy decoding)
    # repeat_penalty=1.0 matches NVIDIA's defaults for Nemotron 3 Nano
    temperature: float = 0.0
    repeat_penalty: float = 1.0

    # Single slot (no alternation needed with buffered approach)
    slot_id: int = 0

    # Context management (platform-dependent)
    max_context_tokens: int = 16384         # 5090: 16384, DGX Spark: larger
    context_reserve_tokens: int = 2048      # Reserved for response generation
```

## Server Configuration

The buffered approach requires single-slot mode, which uses less GPU memory:

```bash
# 5090 (32GB GPU) - smaller context
llama-server \
    -m /path/to/model.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --parallel 1 \                    # Single slot (was --parallel 2)
    --ctx-size 16384 \                # 16K context for 5090
    --flash-attn on \
    --n-gpu-layers 99

# DGX Spark (larger GPU) - can use larger context
llama-server \
    -m /path/to/model.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --parallel 1 \
    --ctx-size 32768 \                # 32K+ context for DGX Spark
    --flash-attn on \
    --n-gpu-layers 99
```

**GPU Memory Savings**: With `--parallel 1` instead of `--parallel 2`, we save approximately
50% of KV cache memory. For a 16K context with Q8 quantization, this is roughly 1-2GB.

## Token Tracking

The buffer tracks token count using `tokens_predicted` from llama.cpp responses. This field
is at the top level of the response (not inside `timings`) and contains the number of tokens
generated in the current request.

```python
class SentenceBuffer:
    def __init__(self):
        self.text: str = ""
        self.token_count: int = 0  # Tokens accumulated since last emit

    def add(self, text: str, tokens: int) -> None:
        """Append text from LLM generation with token count."""
        self.text += text
        self.token_count += tokens

    def reset_token_count(self) -> None:
        """Reset token count after emit (regardless of remainder size)."""
        self.token_count = 0
```

**Important**: Use `data.get("tokens_predicted", 0)` from the stop event. This gives the
number of tokens generated in the current HTTP request. The `timings` object contains
`cache_n` and `prompt_n` for cache/evaluation metrics.

**Design decision**: On emit, we reset `token_count = 0` even if the buffer has a remainder. Rationale:
- The remainder is always an incomplete sentence tail (small)
- We're about to generate more tokens anyway
- Precise tracking of remainder tokens would require tokenization
- The hard_max limit is generous enough that slight undercounting is safe

## SentenceBuffer Design

### Data Structure

```python
class SentenceBuffer:
    """Accumulates LLM output and extracts at sentence boundaries."""

    def __init__(self):
        self.text: str = ""
        self.token_count: int = 0  # Tokens since last emit

    def add(self, text: str, tokens: int) -> None:
        """Append text from LLM generation."""
        self.text += text
        self.token_count += tokens

    def clear(self) -> None:
        """Reset buffer (on interruption or completion)."""
        self.text = ""
        self.token_count = 0

    def has_content(self) -> bool:
        """Check if buffer has any text."""
        return bool(self.text.strip())
```

### Extraction Methods

```python
def extract_complete_sentences(self) -> str | None:
    """Extract all complete sentences, keep incomplete tail.

    Returns:
        All complete sentences joined, or None if no complete sentence found.

    Example:
        Buffer: "Hello! How are you? I hope"
        Returns: "Hello! How are you?"
        Remaining buffer: "I hope"
    """
    # Find LAST sentence boundary (.!? with optional closing quotes/parens)
    # Requires trailing whitespace to avoid false positives on abbreviations
    # like "Dr." or decimals like "3.14". End-of-response is handled by EOS logic.
    pattern = r'[.!?]["\'\)]*\s'

    matches = list(re.finditer(pattern, self.text))
    if not matches:
        return None

    # Last match gives us the end of the last complete sentence
    last_match = matches[-1]
    boundary = last_match.end()

    sentences = self.text[:boundary].strip()
    self.text = self.text[boundary:].lstrip()

    return sentences if sentences else None


def extract_at_boundary(self) -> str:
    """Force extraction when buffer exceeds token limit.

    Extracts all content, finding the best break point using priority:
    1. Last sentence boundary (.!?)
    2. Last clause boundary (", " or "; " or newline)
    3. Last word boundary (space)
    4. Everything (fallback)

    Returns:
        Extracted text
    """
    if not self.text:
        return ""

    search_region = self.text

    # Priority 1: Last sentence boundary
    sentence_pattern = r'[.!?]["\'\)]*\s'
    sentence_matches = list(re.finditer(sentence_pattern, search_region))
    if sentence_matches:
        boundary = sentence_matches[-1].end()
        result = self.text[:boundary].strip()
        self.text = self.text[boundary:].lstrip()
        return result

    # Priority 2: Last clause boundary (", " or "; " or "\n")
    clause_idx = max(
        search_region.rfind(", "),
        search_region.rfind("; "),
        search_region.rfind("\n")
    )
    if clause_idx > 0:
        # Include the punctuation, exclude the space
        boundary = clause_idx + 1
        result = self.text[:boundary].strip()
        self.text = self.text[boundary:].lstrip()
        return result

    # Priority 3: Last word boundary
    space_idx = search_region.rfind(" ")
    if space_idx > 0:
        result = self.text[:space_idx].strip()
        self.text = self.text[space_idx:].lstrip()
        return result

    # Fallback: emit everything
    result = self.text.strip()
    self.text = ""
    return result
```

## State Machine

The state machine uses the same logic for all segments - the only difference is the
`max_tokens` and `hard_max_tokens` limits passed in. For first segment, these are equal
(24, 24), which naturally forces an emit after one generation. For subsequent segments,
hard_max is larger (32, 96), allowing multiple generations to find sentence boundaries.

```
┌─────────────────────────────────────────┐
│               GENERATE                  │
│  Request max_tokens from LLM            │
│  Add response to buffer (text + tokens) │
│  Track if EOS received                  │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│             CHECK_BUFFER                │
│  After generation, decide what to emit  │
└───────────────────┬─────────────────────┘
                    │
    ┌───────────────┼───────────────┬──────────────────┐
    ▼               ▼               ▼                  ▼
Has complete   token_count      No sentence,      Hit EOS?
sentence(s)?   >= hard_max?     under limit
    │               │               │                  │
    ▼               ▼               ▼                  ▼
┌─────────┐  ┌─────────────┐  ┌──────────┐      ┌───────────┐
│ EXTRACT │  │ EXTRACT AT  │  │ GENERATE │      │ EMIT      │
│ ALL     │  │ BOUNDARY    │  │ MORE     │      │ REMAINING │
│ COMPLETE│  │ (sent/clause│  │ (loop)   │      │ + DONE    │
│ SENTS   │  │  /word)     │  └────┬─────┘      └───────────┘
└────┬────┘  └──────┬──────┘       │
     │              │              │
     └──────┬───────┘              │
            ▼                      │
┌─────────────────────────────────────────┐
│           EMIT + WAIT_TTS               │
│  Push LLMTextFrame                      │
│  Reset buffer.token_count = 0           │
│  Wait for ChunkedLLMContinueFrame       │
└───────────────────┬─────────────────────┘
                    │
                    ▼
          Buffer has remainder?
          (always incomplete tail)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Has remainder           Buffer empty
   (incomplete)
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│    GENERATE     │    │ Select limits   │
│    (continue    │    │ for next segment│
│    building     │    │ then GENERATE   │
│    sentence)    │    └────────┬────────┘
└────────┬────────┘             │
         │                      │
         └──────────────────────┘


Key invariant: After extracting all complete sentences, the buffer
ONLY contains an incomplete sentence tail. Therefore, after emit + wait,
we ALWAYS generate more tokens - there is never a case where we emit
twice without generating.


Special cases:

┌─────────────────────────────────────────┐
│              HIT_EOS                    │
│  LLM returned stop token or <|im_end|>  │
│  Emit whatever remains in buffer        │
│  Push LLMFullResponseEndFrame           │
│  This is a complete LLM response        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│            INTERRUPTED                  │
│  Clear buffer (text + token_count)      │
│  Increment generation ID                │
│  In-flight generation completes but     │
│  results are discarded (stale ID)       │
│  Reset to IDLE                          │
└─────────────────────────────────────────┘
```

## Generation Loop Pseudocode

The loop uses identical logic for all segments. The behavioral difference between first
segment (one generation then emit) and subsequent segments (multiple generations allowed)
emerges naturally from the limit values: first segment has `max_tokens == hard_max_tokens`,
so the hard_max check triggers immediately after the first generation.

```python
async def _process_context(self, context: LLMContext):
    # Initialize state
    self._buffer = SentenceBuffer()
    self._generated_text = ""
    self._generation_id += 1
    my_generation_id = self._generation_id
    hit_eos = False

    # Initialize metrics for this response
    self._is_first_generation = True
    self._first_segment_tokens_cached = 0
    self._first_segment_tokens_evaluated = 0
    self._total_tokens_cached = 0
    self._total_tokens_evaluated = 0
    self._total_tokens_predicted = 0

    # Trim messages to fit context window, then format
    messages = context.get_messages()
    messages = self._trim_messages_to_fit_context(messages)
    self._prompt = self._format_messages(messages)

    # Segment limits - first segment uses equal max/hard_max
    max_tokens = self._params.first_segment_max_tokens          # 24
    hard_max_tokens = self._params.first_segment_hard_max_tokens # 24

    await self.push_frame(LLMFullResponseStartFrame())

    while not self._cancelled:
        # Step 1: Generate tokens (runs to completion, no mid-stream cancel)
        new_text, new_tokens, hit_eos = await self._generate(max_tokens, my_generation_id)

        if self._cancelled or self._generation_id != my_generation_id:
            break  # Interrupted, discard results

        self._buffer.add(new_text, new_tokens)

        # Step 2: Check buffer and decide action
        # Priority: sentences > hard_max > generate more > EOS

        sentences = self._buffer.extract_complete_sentences()
        if sentences:
            # Found complete sentences - emit all of them
            await self._emit_and_wait(sentences)

            if hit_eos and not self._buffer.has_content():
                break

            # Switch to subsequent segment limits for next iteration
            max_tokens = self._params.segment_max_tokens          # 32
            hard_max_tokens = self._params.segment_hard_max_tokens # 96
            continue

        if self._buffer.token_count >= hard_max_tokens:
            # Hit hard limit without sentence - emit at best boundary
            text = self._buffer.extract_at_boundary()
            await self._emit_and_wait(text)

            if hit_eos and not self._buffer.has_content():
                break

            # Switch to subsequent segment limits
            max_tokens = self._params.segment_max_tokens
            hard_max_tokens = self._params.segment_hard_max_tokens
            continue

        if hit_eos:
            # EOS reached - emit whatever remains and finish
            if self._buffer.has_content():
                await self._emit_and_wait(self._buffer.text.strip())
                self._buffer.clear()
            break

        # No sentence, under hard_max, not EOS - generate more tokens
        # (Loop continues with same limits until we emit something)

    await self.push_frame(LLMFullResponseEndFrame())


async def _emit_and_wait(self, text: str):
    """Emit text to TTS and wait for completion signal."""
    await self.push_frame(LLMTextFrame(text=text))
    self._buffer.reset_token_count()  # Reset after emit, regardless of remainder

    try:
        await asyncio.wait_for(self._continue_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for TTS continue signal")
    self._continue_event.clear()


async def _generate(self, max_tokens: int, expected_gen_id: int) -> tuple[str, int, bool]:
    """Generate tokens via llama.cpp HTTP API.

    Returns:
        (generated_text, tokens_generated, hit_eos)

    Note: hit_eos is True only for natural completion (EOS token or stop word).
    Hitting the n_predict limit returns hit_eos=False (we just hit our requested limit).
    """
    full_prompt = self._prompt + self._generated_text

    payload = {
        "prompt": full_prompt,
        "n_predict": max_tokens,
        "id_slot": self._params.slot_id,
        "cache_prompt": True,
        "temperature": self._params.temperature,
        "repeat_penalty": self._params.repeat_penalty,
        "stream": True,
        "stop": ["<|im_end|>"],
    }

    collected_text = ""
    tokens_generated = 0
    hit_eos = False

    # Stream runs to completion - NO early break for sentence boundaries
    async with self._client.stream("POST", f"{self._llama_url}/completion", json=payload) as response:
        async for line in response.aiter_lines():
            # Check for stale generation
            if self._generation_id != expected_gen_id:
                return "", 0, False

            if not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if data.get("stop"):
                # Generation stopped - check why
                stop_type = data.get("stop_type", "")
                token_text = data.get("content", "")
                if token_text:
                    collected_text += token_text

                # Capture metrics from response
                # Note: tokens_predicted is at top level, cache/prompt metrics in timings
                tokens_generated = data.get("tokens_predicted", 0)
                timings = data.get("timings", {})

                # Update aggregated metrics
                self._total_tokens_cached += timings.get("cache_n", 0)
                self._total_tokens_evaluated += timings.get("prompt_n", 0)
                self._total_tokens_predicted += tokens_generated

                # Capture first segment metrics
                if self._is_first_generation:
                    self._first_segment_tokens_cached = timings.get("cache_n", 0)
                    self._first_segment_tokens_evaluated = timings.get("prompt_n", 0)
                    self._is_first_generation = False

                # Only set hit_eos for natural completion, NOT for hitting n_predict limit
                # stop_type: "eos" = EOS token, "word" = stop word like <|im_end|>
                # stop_type: "limit" = hit n_predict limit (NOT end of response)
                if stop_type in ("eos", "word"):
                    hit_eos = True
                # For "limit", hit_eos stays False - we just hit our requested token limit

                break

            collected_text += data.get("content", "")

    # Handle empty generation (model immediately hit EOS)
    if not collected_text and tokens_generated == 0:
        hit_eos = True

    # Update cache state for next generation
    self._generated_text += collected_text

    return collected_text, tokens_generated, hit_eos


```

## Key Invariants

1. **No double-emit without generating**: After extracting all complete sentences, the
   buffer ONLY contains an incomplete sentence tail. Therefore, after emit + wait, we
   ALWAYS generate more tokens before emitting again.

2. **Token count reset on emit**: After any emission, `buffer.token_count` resets to 0,
   even if the buffer has a remainder. This is safe because:
   - The remainder is always an incomplete tail (small)
   - We're about to generate more tokens, which will add to the count
   - The hard_max is generous enough that slight undercounting won't cause issues

3. **Limits switch after first emit**: After the first emission (regardless of how it
   happened), we switch from first-segment limits (24, 24) to subsequent limits (32, 96).

4. **EOS is final**: When the LLM returns EOS/stop token, we emit whatever remains in
   the buffer and complete. This constitutes a full LLM response (LLMFullResponseEndFrame).

## Interruption Handling

```python
async def _handle_interruption(self, frame: InterruptionFrame):
    """Handle user interruption."""
    # Increment generation ID to invalidate in-flight results
    self._generation_id += 1

    # Clear buffer (text and token count)
    self._buffer.clear()

    # Reset state
    self._cancelled = True
    self._generating = False

    # Signal any waiting coroutines
    if self._continue_event:
        self._continue_event.set()
```

**Key behavior**: If a generation is in-flight when interrupted, it runs to completion
(we don't cancel the HTTP request), but results are discarded because `_generation_id`
has changed. This is intentional - avoiding mid-stream cancellation is the whole point
of this design.

## Cache Behavior Analysis

### Single-Slot Continuous Cache

```
Turn 1:
  Gen 1: P (100 tokens) → cache[0] = P, generate 24 tokens
  Gen 2: P + G1 (124 tokens) → FULL CACHE HIT on P + G1, generate 32 more
  Gen 3: P + G1 + G2 (156 tokens) → FULL CACHE HIT, generate 32 more

Turn 2:
  Gen 1: P + G_all + user + P' → FULL CACHE HIT on conversation history
  ...continues with perfect cache hits...
```

### Comparison to Two-Slot Approach

| Metric | Two-Slot (Current) | Single-Slot (New) |
|--------|-------------------|-------------------|
| Cache hit ratio | ~67% | ~100% |
| Extra tokens evaluated | ~300 per response | ~0 |
| Time saved | - | ~400ms per response |
| Race condition risk | Mitigated by guard | None |
| Server config | `--parallel 2` required | `--parallel 1` (less GPU memory) |
| GPU memory | Higher (2 KV caches) | Lower (1 KV cache) |

## Segment Behavior

The same logic handles both first and subsequent segments. The difference is only in the limits:

| Segment | max_tokens | hard_max_tokens | Behavior |
|---------|------------|-----------------|----------|
| First | 24 | 24 | One generation, then must emit |
| Subsequent | 32 | 96 | Up to 3 generations before forced emit |

For the first segment, `max_tokens == hard_max_tokens`, so after one generation the
`token_count >= hard_max_tokens` check triggers immediately if no sentence was found.

**Example scenarios (first segment, max=24, hard=24):**

```
Scenario A: Sentence boundary found
  Generated: "Hello! How can I help you today?" (24 tokens)
  token_count: 24
  Sentence found: "Hello! How can I help you today?"
  Emit: "Hello! How can I help you today?"
  Remaining: ""

Scenario B: Sentence + incomplete tail
  Generated: "Hi there! I'm happy to help with" (24 tokens)
  token_count: 24
  Sentence found: "Hi there!"
  Emit: "Hi there!"
  Remaining: "I'm happy to help with"

Scenario C: No sentence, has clause
  Generated: "Well, let me think about that for" (24 tokens)
  token_count: 24 >= hard_max (24)
  No sentence found, extract at boundary
  Find clause: "Well,"
  Emit: "Well,"
  Remaining: "let me think about that for"

Scenario D: No sentence or clause
  Generated: "The answer to your question is that" (24 tokens)
  token_count: 24 >= hard_max (24)
  No sentence, no clause
  Find word boundary: "The answer to your question is"
  Emit: "The answer to your question is"
  Remaining: "that"
```

**Example scenarios (subsequent segment, max=32, hard=96):**

```
Scenario E: Multiple generations needed
  Buffer (from previous): "I hope"
  Gen 1: " you're doing well today and I was" (32 tokens)
  token_count: 32, no sentence, under hard_max (96)
  Gen 2: " wondering if you could help me with" (32 tokens)
  token_count: 64, no sentence, under hard_max
  Gen 3: " a programming question. Here's my" (32 tokens)
  token_count: 96 >= hard_max (96)
  Sentence found: "I hope you're doing well today and I was wondering if you could help me with a programming question."
  Emit that sentence
  Remaining: "Here's my"

Scenario F: Sentence found before hard_max
  Buffer (from previous): "The weather is"
  Gen 1: " beautiful today! I was thinking we" (32 tokens)
  token_count: 32, sentence found: "The weather is beautiful today!"
  Emit: "The weather is beautiful today!"
  Remaining: "I was thinking we"
  (Did not need to reach hard_max)
```

## Metrics

The service emits enhanced metrics with first-segment cache tracking:

```python
class LLMSlotMetricsFrame(SystemFrame):
    slot_id: int              # Always 0 (single slot)
    total_chunks: int         # Number of TTS emissions
    total_time_ms: float

    # Aggregated across ALL generations in the response
    tokens_cached: int        # From llama.cpp timings.cache_n
    tokens_evaluated: int     # From llama.cpp timings.prompt_n
    tokens_predicted: int     # Completion tokens generated

    # First segment metrics (critical for TTFB analysis)
    first_segment_tokens_cached: int
    first_segment_tokens_evaluated: int

    @property
    def cache_hit_ratio(self) -> float:
        """Overall cache hit ratio across all generations."""
        total = self.tokens_cached + self.tokens_evaluated
        return self.tokens_cached / total if total > 0 else 0.0

    @property
    def first_segment_cache_hit_ratio(self) -> float:
        """Cache hit ratio for first segment (affects TTFB).

        High ratio (>90%) = System prompt + history cached = Fast TTFB
        Low ratio (<50%) = Context shifted or cold start = Slower TTFB
        """
        total = self.first_segment_tokens_cached + self.first_segment_tokens_evaluated
        return self.first_segment_tokens_cached / total if total > 0 else 0.0
```

### Metric Collection

Metrics are collected in `_generate()` and aggregated across all generations. See the
`_process_context()` and `_generate()` pseudocode above for the complete implementation.

Key points:
- `self._is_first_generation` tracks whether to capture first segment metrics
- `tokens_predicted` is at the response top level (not inside `timings`)
- `cache_n` and `prompt_n` are inside the `timings` object
- Metrics are initialized in `_process_context()`, updated in `_generate()`

### Why First Segment Cache Hit Matters

The `first_segment_cache_hit_ratio` is the most important metric for understanding TTFB:

1. **High ratio (>90%)**: The system prompt and conversation history were already in the
   KV cache. Only the new user message needed evaluation. TTFB will be optimal.

2. **Low ratio (<50%)**: Either:
   - Cold start (first request after server restart)
   - Context was evicted (long conversation, slot was reused by another request)
   - Server config issue (cache disabled)

3. **Operational insight**: If first_segment_cache_hit drops suddenly in production,
   investigate whether context limits are being hit or if there's cache contention.

Expected improvement: `tokens_evaluated` should be near zero for all generations after the first, as the full conversation history is cached.

## TTS Integration Changes

The TTS service (`MagpieWebSocketTTSService`) must be updated to work correctly with the new
buffered LLM approach:

### Disable Built-in Sentence Chunking

The TTS service should send all received text directly to Magpie without internal sentence
aggregation. The LLM service now handles sentence boundary detection.

```python
class MagpieWebSocketTTSService(WebsocketTTSService):
    class InputParams(BaseModel):
        # ... existing params ...
        # NEW: Disable sentence chunking (LLM handles boundaries now)
        enable_sentence_chunking: bool = False

    def __init__(self, ...):
        super().__init__(
            # When enable_sentence_chunking=False, pass through all text as-is
            aggregate_sentences=self._params.enable_sentence_chunking,
            ...
        )
```

### Inter-Sentence Pause Injection

The TTS service should inject silence pauses ONLY when the received text chunk ends at a
sentence boundary. Since the LLM now sends complete sentences, we check if the `run_tts()`
argument ends with sentence-ending punctuation:

```python
async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
    # ... existing logic ...

    # Track whether this chunk ends at a sentence boundary
    # Silence is injected AFTER segment_complete in _receive_messages
    ends_with_sentence = self._ends_at_sentence_boundary(text)
    self._segment_sentence_boundary_queue.append(ends_with_sentence)

    # ...
```

**Key change**: Previously, every token was checked. Now, since the LLM sends complete
sentence chunks (e.g., "Hello! How are you?"), checking if the text ends with `.!?`
correctly identifies sentence-ending chunks.

### Adaptive Mode Still Works

The adaptive mode (streaming for first segment, batch for subsequent) continues to work
unchanged. Each `LLMTextFrame` from the LLM triggers a new TTS segment.

## Context Window Management

The client (LLM service) must manage context window overflow. llama.cpp's server-side
context shift behavior is unreliable and may error when context is exceeded.

### Querying Context Size from Server

On startup, query the `/props` endpoint to get the actual context size:

```python
async def _query_server_props(self) -> dict:
    """Query llama.cpp server for configuration properties.

    Returns:
        Server properties including n_ctx (context size per slot).
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{self._llama_url}/props")
        response.raise_for_status()
        return response.json()

async def start(self, frame: StartFrame):
    """Handle StartFrame - create HTTP client and query server config."""
    await super().start(frame)

    # Query actual context size from server
    try:
        props = await self._query_server_props()
        logger.debug(f"LlamaCppBufferedLLM: Server props: {props}")

        n_ctx = props.get("default_generation_settings", {}).get("n_ctx", 16384)
        total_slots = props.get("total_slots", 1)

        # With single-slot mode, we get the full context
        self._max_context_tokens = n_ctx
        logger.info(
            f"LlamaCppBufferedLLM: Server config - n_ctx={n_ctx}, "
            f"slots={total_slots}"
        )
    except Exception as e:
        logger.warning(f"Failed to query server props, using default: {e}")
        self._max_context_tokens = self._params.max_context_tokens

    # Create HTTP client with connection pooling disabled
    # This avoids issues with connection reuse after interruptions
    self._client = httpx.AsyncClient(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=0)
    )
```

### Configuration (Fallback)

If the server query fails, use configured defaults:

```python
class InputParams(BaseModel):
    # ... existing params ...
    # Context management fallback (queried from server on startup)
    max_context_tokens: int = 16384
    # Reserve tokens for response generation
    context_reserve_tokens: int = 2048
```

### Sliding Window Implementation

Before formatting messages, check if the conversation history exceeds the context limit.
If so, drop older messages (preserving system message):

```python
def _trim_messages_to_fit_context(self, messages: list) -> list:
    """Drop oldest messages if conversation exceeds context limit.

    Preserves:
    - System message (always first)
    - Most recent messages up to context limit

    Returns:
        Trimmed message list that fits within context window.
    """
    # Use actual context size from server (queried on startup), minus reserve for response
    max_tokens = self._max_context_tokens - self._params.context_reserve_tokens

    # Simple heuristic: ~4 chars per token (conservative estimate)
    # For production, consider using a tokenizer
    def estimate_tokens(msg):
        content = msg.get("content", "")
        return len(content) // 4 + 10  # +10 for role/formatting overhead

    # Always keep system message
    if messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        other_msgs = messages[1:]
        system_tokens = estimate_tokens(system_msg)
    else:
        system_msg = None
        other_msgs = messages
        system_tokens = 0

    # Calculate tokens for remaining messages (newest first)
    available = max_tokens - system_tokens
    kept_msgs = []
    for msg in reversed(other_msgs):
        msg_tokens = estimate_tokens(msg)
        if available >= msg_tokens:
            kept_msgs.insert(0, msg)
            available -= msg_tokens
        else:
            break  # No more room

    if system_msg:
        kept_msgs.insert(0, system_msg)

    if len(kept_msgs) < len(messages):
        dropped = len(messages) - len(kept_msgs)
        logger.warning(f"Context limit: dropped {dropped} oldest messages")

    return kept_msgs
```

## Migration Path

### Phase 1: New Service Implementation
1. Create `LlamaCppBufferedLLMService` with new architecture
2. Implement `SentenceBuffer` with extraction methods
3. Implement generation loop with proper state management
4. Add comprehensive logging for debugging

### Phase 2: Testing
1. Test with voice agent, verify sentence boundaries
2. Measure cache hit ratios (should be ~100%)
3. Measure TTFB (should be similar or better)
4. Test interruption handling
5. Stress test with rapid interruptions

### Phase 3: Rollout
1. Keep old `LlamaCppChunkedLLMService` as fallback
2. Add configuration to select implementation
3. Gradual rollout with monitoring

### Phase 4: Cleanup
1. Remove old implementation once validated
2. Remove two-slot configuration from server
3. Update documentation

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `pipecat_bots/llama_cpp_buffered_llm.py` | Create | New buffered implementation |
| `pipecat_bots/sentence_buffer.py` | Create | SentenceBuffer class |
| `pipecat_bots/magpie_websocket_tts.py` | Modify | Add enable_sentence_chunking param |
| `pipecat_bots/bot_interleaved_streaming.py` | Modify | Option to use new service |
| `scripts/start_unified.sh` | Modify | Change `--parallel 2` to `--parallel 1` |
| `docs/chunked-llm-engineering-deep-dive.md` | Update | Document new approach |

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TTS quality regression | Low | Medium | Same sentence boundaries, just different chunking |
| TTFB regression | Low | High | First segment optimized same as before |
| Long sentences causing TTS delay | Medium | Low | Hard max with clause/word fallback |
| Generation timeout during long wait | Low | Medium | 30s timeout on TTS wait |
| Stale generation confusion | Low | Medium | Generation ID tracking |

## Success Criteria

1. **Cache efficiency**: `tokens_evaluated` approaches 0 after first generation
2. **TTFB**: First audio within 500-700ms (same as current)
3. **No crashes**: Eliminate `GGML_ASSERT(!slot.is_processing())` crashes
4. **Simpler config**: Single-slot mode (`--parallel 1`), less GPU memory
5. **Clean code**: Remove two-slot logic, reuse guards
6. **First segment cache hit**: `first_segment_cache_hit_ratio` > 90% in steady state

## Implementation Progress

| Task | Status | Notes |
|------|--------|-------|
| Create `sentence_buffer.py` | ✅ Complete | SentenceBuffer class with extraction methods |
| Create `llama_cpp_buffered_llm.py` | ✅ Complete | Buffered LLM service with single-slot operation |
| Modify `magpie_websocket_tts.py` | ✅ Complete | Added `enable_sentence_chunking` param |
| Modify `bot_interleaved_streaming.py` | ✅ Complete | Option to use buffered LLM service |
| Update `start_unified.sh` | ✅ Complete | Changed default to `--parallel 1` |
| Update documentation | ✅ Complete | This progress section |

### Testing Checklist

- [ ] Run with buffered LLM service
- [ ] Verify sentence boundary detection
- [ ] Confirm cache hit ratio approaches 100%
- [ ] Test interruption handling
- [ ] Measure TTFB
- [ ] Stress test with rapid interruptions
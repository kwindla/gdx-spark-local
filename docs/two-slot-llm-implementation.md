# Two-Slot LLM Implementation Plan

## Problem Statement

When using llama.cpp server with chunked generation, closing the HTTP connection mid-stream triggers a "cancel" that can leave the slot's KV cache in an inconsistent state. Even after `/slots` reports the slot as idle, internal cleanup may not be complete, causing `GGML_ASSERT(!slot.is_processing())` crashes on the next request.

## Root Cause

1. **cpp-httplib limitation**: Client disconnect during streaming doesn't reliably trigger cleanup
2. **llama.cpp race condition**: KV cache state and `is_processing` flag are not atomically updated
3. **No reliable cancel mechanism**: No API endpoint to gracefully cancel and ensure cleanup

## Solution: Two-Slot Alternation with Reuse Guard

Alternate between two slots, ensuring each slot has at least 2 seconds before being reused. This gives ample time for any async cleanup to complete.

### Architecture

```
Chunk 1 → Slot 0 → close → [cleanup happening]
          [TTS ~1-2s]
Chunk 2 → Slot 1 → close → [cleanup happening]    (Slot 0 still cleaning, but unused)
          [TTS ~1-2s]
Chunk 3 → Slot 0 → Slot 0 has had 2-4+ seconds to clean up ✓
```

### Key Components

1. **Slot Alternation**: Round-robin between slot 0 and slot 1
2. **Reuse Guard**: If a slot would be reused within 2 seconds, wait for the remainder
3. **Usage Tracking**: Record timestamp when each slot's request completes/cancels

## Implementation Details

### New Constants

```python
# Minimum time before reusing the same slot (seconds)
# This ensures any async cleanup from cancelled requests is complete
MIN_SLOT_REUSE_DELAY_S = 2.0

# Number of slots to alternate between (requires --parallel N on server)
DEFAULT_NUM_SLOTS = 2
```

### New State Variables

```python
self._num_slots: int = 2
self._current_slot: int = 0
self._slot_last_used: dict[int, float] = {}  # slot_id -> timestamp
```

### Slot Selection Logic

```python
async def _get_next_slot(self) -> int:
    """Get next slot, waiting if reuse guard is triggered."""
    slot = self._current_slot
    self._current_slot = (self._current_slot + 1) % self._num_slots

    # Check reuse guard
    last_used = self._slot_last_used.get(slot, 0)
    elapsed = time.time() - last_used

    if elapsed < MIN_SLOT_REUSE_DELAY_S:
        wait_time = MIN_SLOT_REUSE_DELAY_S - elapsed
        logger.warning(
            f"LlamaCppChunkedLLM: Slot {slot} reuse guard triggered, "
            f"waiting {wait_time:.2f}s"
        )
        await asyncio.sleep(wait_time)

    return slot

def _mark_slot_used(self, slot: int):
    """Record when a slot's request completed/cancelled."""
    self._slot_last_used[slot] = time.time()
```

### Integration Points

1. **_generate_chunk()**:
   - Call `_get_next_slot()` to get slot with guard
   - Pass slot to payload as `id_slot`
   - Call `_mark_slot_used(slot)` in finally block

2. **Remove old cooldown logic**:
   - Remove `_wait_for_cooldown()`
   - Remove `_wait_for_slot_idle()`
   - Remove `DEFAULT_COOLDOWN_MS`, `POST_IDLE_DELAY_MS`, `MAX_SLOT_WAIT_MS`, `SLOT_POLL_INTERVAL_MS`
   - Remove `cooldown_ms` from InputParams

### Configuration

Add `num_slots` to InputParams:

```python
class InputParams(BaseModel):
    # ... existing params ...
    num_slots: int = DEFAULT_NUM_SLOTS  # Requires --parallel N on server
```

### Server Requirements

The llama.cpp server must be started with `--parallel 2` (or higher):

```bash
llama-server -m model.gguf --parallel 2 ...
```

## KV Cache Behavior with Two Slots

### Cache State Per Slot

| Chunk | Slot | Prompt Sent | Cache Hit | New Processing |
|-------|------|-------------|-----------|----------------|
| 1 | 0 | P | None | P + G1 |
| 2 | 1 | P + G1 | None (new slot) | P + G1 + G2 |
| 3 | 0 | P + G1 + G2 | P + G1 | G2 + G3 |
| 4 | 1 | P + G1 + G2 + G3 | P + G1 + G2 | G3 + G4 |
| 5 | 0 | P + G1 + ... + G4 | P + G1 | G2 + G3 + G4 + G5 |

### Cache Efficiency Analysis

- **Single slot (ideal)**: Each chunk only processes new tokens
- **Two slots**: Each slot misses every other chunk's generation, must reprocess

For a typical 5-chunk response:
- Single slot: ~100 extra tokens processed (just new content)
- Two slots: ~320 extra tokens processed (reprocessing missed chunks)

At ~1.3ms/token prompt processing: **~280ms extra total** across all chunks

This is **much better** than the 200-500ms delays we'd need per chunk with cooldowns.

## Edge Cases

### 1. Very Fast Responses

If all chunks complete in < 2 seconds total, the guard may trigger:

```
Chunk 1 (slot 0): 500ms
Chunk 2 (slot 1): 500ms
Chunk 3 (slot 0): Would reuse slot 0 after only 1s → guard waits 1s
```

This is rare in practice because TTS takes 1-2s per chunk.

### 2. Rapid Interruptions

User interrupts multiple times quickly:

```
Response 1, Chunk 1 (slot 0): interrupted after 200ms
Response 2, Chunk 1 (slot 1): interrupted after 200ms
Response 3, Chunk 1 (slot 0): guard triggers, wait ~1.6s
```

The guard ensures slot 0 has time to clean up.

### 3. Single-Slot Server

If server started with `--parallel 1`:
- `_get_next_slot()` always returns 0
- No alternation benefit
- Should log warning on init if possible

Detection: Could query `/slots` on startup to count available slots.

### 4. Interruption During Guard Wait

If interrupted while waiting on guard:
- `_cancelled` flag is set
- `_generate_chunk()` checks flag and returns early
- Guard wait should also check cancellation

```python
if elapsed < MIN_SLOT_REUSE_DELAY_S:
    wait_time = MIN_SLOT_REUSE_DELAY_S - elapsed
    # Check for cancellation during wait
    try:
        await asyncio.wait_for(
            asyncio.sleep(wait_time),
            timeout=wait_time
        )
    except asyncio.CancelledError:
        raise
```

Actually simpler: just check `self._cancelled` after the sleep.

## Testing

### Unit Tests

1. Slot alternation: Verify slots cycle 0 → 1 → 0 → 1
2. Reuse guard: Verify wait triggers when reusing slot within 2s
3. Guard bypass: Verify no wait when slot unused for > 2s

### Integration Tests

1. Normal conversation: Multiple chunks without guard triggering
2. Interruption: Cancel mid-response, new response uses different slot
3. Rapid interruptions: Guard triggers and prevents crash

### Manual Testing

1. Start server with `--parallel 2`
2. Run voice agent with chunked LLM
3. Interrupt frequently
4. Verify no crashes, minimal guard waits logged

## Rollback Plan

If issues arise:
1. Revert to single slot with increased `POST_IDLE_DELAY_MS` (500ms)
2. Or use `POST /slots/{id}?action=erase` after each cancel (loses caching)

## Files to Modify

### `pipecat_bots/llama_cpp_chunked_llm.py`

1. Add `MIN_SLOT_REUSE_DELAY_S`, `DEFAULT_NUM_SLOTS` constants
2. Remove old cooldown constants
3. Add `num_slots` to InputParams
4. Add `_num_slots`, `_current_slot`, `_slot_last_used` state
5. Add `_get_next_slot()` method
6. Add `_mark_slot_used()` method
7. Remove `_wait_for_cooldown()` method
8. Remove `_wait_for_slot_idle()` method
9. Update `_generate_chunk()` to use new slot logic
10. Update `__init__` logging to show num_slots

### Documentation

1. Update `docs/llama-cpp-chunked-llm-service.md` with two-slot info
2. Note `--parallel 2` requirement

## Implementation Checklist

- [ ] Add new constants
- [ ] Remove old cooldown constants
- [ ] Update InputParams
- [ ] Add slot tracking state to __init__
- [ ] Implement _get_next_slot()
- [ ] Implement _mark_slot_used()
- [ ] Remove _wait_for_cooldown()
- [ ] Remove _wait_for_slot_idle()
- [ ] Update _generate_chunk() to use slot logic
- [ ] Update logging in __init__
- [ ] Test with voice agent
- [ ] Update documentation

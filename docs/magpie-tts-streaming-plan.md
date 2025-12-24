# Magpie TTS Streaming Inference Plan

## Executive Summary

This document outlines a plan to implement streaming inference for NVIDIA's Magpie TTS model, optimizing Time-To-First-Byte (TTFB). Currently, the model generates all audio tokens before decoding to waveform, resulting in high latency for the first audio output.

## Current Architecture Analysis

### Pipeline Overview

```
Text Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  MagpieTTSModel.infer_batch()                       │
│  ├── prepare_context_tensors() - Text encoding     │
│  └── Autoregressive Loop:                          │
│      ├── for idx in range(max_decoder_steps):      │
│      │   ├── embed_audio_tokens()                  │
│      │   ├── decoder.forward() (with KV cache)     │
│      │   ├── sample_codes_from_logits()            │
│      │   └── append to all_predictions             │
│      └── codes_to_audio() [CALLED ONCE AT END]     │
└─────────────────────────────────────────────────────┘
    │
    ▼
Audio Output (only available after ALL tokens generated)
```

### Key Parameters (from NeMo source analysis)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sample_rate` | 22,000 Hz | Magpie/nano codec output rate |
| `samples_per_frame` | 1,024 | Audio samples per codec frame |
| `frame_stacking_factor` | 1 (default) | Frames per decoder step |
| `num_codebooks` | 8 | RVQ codebook count |
| `ms_per_frame` | ~46.4 ms | 1024/22000 * 1000 |
| `codec_decoder_type` | HiFiGANDecoder | **Non-causal** (standard) |

### Bottleneck Analysis

The current `infer_batch()` method at `magpietts.py:2730-3040`:

1. **Token Generation Loop** (lines 2800-2992): Generates tokens autoregressively with KV cache
2. **Final Decoding** (line 3004): `predicted_audio, predicted_audio_lens = self.codes_to_audio(predicted_codes, predicted_codes_lens)`

The entire audio decoding happens **after** all tokens are generated, creating the TTFB bottleneck.

### Codec Pipeline

```python
# codes_to_audio() implementation (magpietts.py:814-832)
def codes_to_audio(self, codes, codes_len):
    # 1. Dequantize: discrete tokens → continuous representation
    dequantized = self.dequantize(tokens=codes, tokens_len=codes_len)

    # 2. Decode: continuous → audio waveform
    audio, audio_len = self.decode_audio(inputs=dequantized, input_len=tokens_len)
    return audio, audio_len
```

The codec uses **HiFiGANDecoder** (non-causal), which processes the entire sequence at once. This is important for the streaming strategy.

## Streaming Implementation Strategy

### Approach: Chunked Codec Decoding

Since the HiFiGAN decoder is non-causal, we cannot decode frames independently. However, we can:

1. **Generate tokens incrementally** (already supported via KV cache)
2. **Decode in chunks** with overlap to handle boundary artifacts
3. **Stream audio** as soon as first chunk is decoded

### Proposed Architecture

```
Text Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  MagpieTTSModel.infer_batch_streaming()             │
│  ├── prepare_context_tensors()                      │
│  └── Streaming Loop:                                │
│      ├── Generate N frames of tokens                │
│      ├── If accumulated >= chunk_size:              │
│      │   ├── Decode chunk with overlap              │
│      │   ├── YIELD audio bytes immediately          │
│      │   └── Update overlap buffer                  │
│      └── Continue until EOS                         │
└─────────────────────────────────────────────────────┘
    │
    ▼
Audio chunks (available incrementally)
```

### Streaming Presets

| Preset | First Chunk | Chunk Size | Overlap | Target TTFB | Quality |
|--------|-------------|------------|---------|-------------|---------|
| `aggressive` | 4 frames | 8 frames | 2 frames | ~150-180ms | Good (may have subtle artifacts) |
| `balanced` | 6 frames | 10 frames | 2 frames | ~250-300ms | Excellent |
| `conservative` | 8 frames | 12 frames | 3 frames | ~350-400ms | Reference quality |

**Default: `aggressive`** - Optimized for lowest TTFB while maintaining acceptable quality.

### Implementation Plan

#### Phase 1: Chunked Decoding Prototype

Create a new method that yields audio chunks:

```python
def infer_batch_streaming(
    self,
    batch,
    chunk_size_frames: int = 15,  # ~700ms chunks
    overlap_frames: int = 3,       # ~140ms overlap
    min_first_chunk: int = 6,      # ~280ms TTFB
    **kwargs
) -> Generator[bytes, None, None]:
    """
    Streaming inference that yields audio bytes as soon as available.

    Yields:
        bytes: PCM audio data (int16, mono, 22kHz)
    """
    # Token generation with incremental decoding
    ...
```

#### Phase 2: HTTP Streaming Endpoint

Modify `tts_server.py` to support streaming:

```python
from fastapi.responses import StreamingResponse

@app.post("/v1/audio/speech/stream")
async def synthesize_speech_stream(request: SpeechRequest):
    """Streaming TTS endpoint with chunked transfer encoding."""

    async def audio_generator():
        for audio_chunk in model.infer_batch_streaming(...):
            yield audio_chunk

    return StreamingResponse(
        audio_generator(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "22000", ...}
    )
```

### Critical Implementation Details

#### 1. Overlap-Add for Chunk Boundaries

Non-causal HiFiGAN produces artifacts at chunk boundaries. Use overlap-add:

```python
def decode_with_overlap(self, tokens_chunk, overlap_buffer, overlap_frames):
    # Decode with extra context
    tokens_with_context = torch.cat([overlap_buffer, tokens_chunk], dim=-1)
    audio_full = self.codes_to_audio(tokens_with_context, ...)

    # Extract new audio, discarding overlap region
    new_audio_start = overlap_frames * self.samples_per_frame
    new_audio = audio_full[..., new_audio_start:]

    # Optional: apply crossfade at boundary
    # crossfade_samples = 512  # ~23ms
    # new_audio = apply_crossfade(prev_tail, new_audio, crossfade_samples)

    return new_audio
```

#### 2. KV Cache Preservation

The current implementation already uses KV caching. For streaming, ensure cache is preserved across chunk boundaries:

```python
# Already supported in infer_batch:
self.decoder.reset_cache(use_cache=self.use_kv_cache_for_inference)
```

#### 3. EOS Handling

Need to handle early termination properly:

```python
for idx in range(max_decoder_steps):
    # Generate tokens...

    if eos_detected:
        # Flush remaining buffered tokens
        if pending_tokens.size(-1) > 0:
            final_audio = self.codes_to_audio(pending_tokens, ...)
            yield final_audio.cpu().numpy().astype(np.int16).tobytes()
        break
```

## Expected Performance

### Latency Analysis (Aggressive Preset)

| Metric | Batch Mode | Streaming (Aggressive) |
|--------|------------|------------------------|
| TTFB | ~800ms (full synthesis) | **~150-180ms** |
| Total latency | ~800ms | ~850ms (+overlap overhead) |
| RTF | 0.29x | ~0.32x (slight overhead) |

### TTFB Breakdown (Aggressive Preset)

| Phase | Time |
|-------|------|
| Text encoding | ~10ms |
| First 4 frames generation | ~100ms |
| Codec decode (4 frames) | ~20ms |
| Network/buffering | ~30ms |
| **Total TTFB** | **~160ms** |

### Quality Metrics

The quality comparison test measures:
- **SNR (Signal-to-Noise Ratio)**: >20dB = good, >25dB = excellent
- **Correlation**: >0.9 = good, >0.95 = excellent

Run quality test: `python tests/test_streaming_tts.py --quality-test`

## Testing Strategy

### Local Development Environment

To test outside the Docker container, we need:

1. **PyTorch with CUDA 13.1/sm_121 support** (built from source)
2. **NeMo main branch** (for MagpieTTSModel)
3. **HuggingFace access token**

```bash
# Option 1: Use the existing container interactively
docker run -it --gpus all --ipc=host \
    -v $(pwd):/workspace \
    -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
    nemotron-asr:cuda13-full \
    bash

# Option 2: Build a dev environment with uv (requires sm_121 PyTorch)
# This is complex due to PyTorch compilation requirements
```

### Test Cases

1. **Unit Test: Chunked Decoding**
   - Compare chunk-decoded audio vs full decode
   - Measure SNR/similarity

2. **Integration Test: Streaming Latency**
   - Measure TTFB for various chunk sizes
   - Find optimal chunk_size/overlap trade-off

3. **End-to-End Test: Pipecat Integration**
   - Verify Pipecat can consume streamed audio
   - Test with VAD/interruption handling

## Open Questions

1. **Optimal chunk size**: Need empirical testing to find best TTFB/quality trade-off

2. **Crossfade necessity**: Do we need crossfade for HiFiGAN chunk boundaries, or is overlap sufficient?

3. **GPU memory**: Does chunked decoding increase peak memory (multiple forward passes)?

4. **Warmup latency**: First inference may be slower (JIT compilation) - need to measure

## Alternative Approaches Considered

### A. CausalHiFiGANDecoder Modification

Replace the codec with a causal version for true streaming.

**Pros**: Perfect frame-by-frame streaming, no overlap needed
**Cons**: Requires retraining codec, may degrade quality

### B. Speculative Decoding

Generate and decode tokens in parallel using speculative execution.

**Pros**: Lower latency if speculation accuracy is high
**Cons**: Complex implementation, memory overhead

### C. MaskGit Parallel Decoding

Use MaskGit-style parallel token generation instead of autoregressive.

**Pros**: Faster token generation
**Cons**: May reduce audio quality, requires model support

## Recommended Next Steps

1. **Implement chunked decoding prototype** in `infer_batch_streaming()`
2. **Benchmark TTFB and audio quality** with different chunk sizes
3. **Add streaming HTTP endpoint** to tts_server.py
4. **Update Pipecat integration** to use streaming endpoint
5. **Document optimal parameters** after tuning

## References

- NeMo Source: `/home/khkramer/src/nemo-source/`
- MagpieTTSModel: `nemo/collections/tts/models/magpietts.py`
- AudioCodecModel: `nemo/collections/tts/models/audio_codec.py`
- HiFiGANDecoder: `nemo/collections/tts/modules/audio_codec_modules.py:2188`

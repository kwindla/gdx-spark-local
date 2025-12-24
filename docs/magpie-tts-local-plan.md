# Magpie TTS Local Inference Plan

## Overview

This document outlines the plan to test and integrate NVIDIA's open-source Magpie TTS (357M) model for local inference on DGX Spark (Blackwell GB10), using the existing ASR container which has PyTorch and NeMo built from source with CUDA 13.1/sm_121 support.

## Why Magpie Open Source vs NIM

| Aspect | Magpie NIM (Current) | Magpie Open Source |
|--------|---------------------|-------------------|
| Platform | Buggy on DGX Spark | Should work (same container as working ASR) |
| Dependencies | gRPC to external container | In-process with NeMo |
| Latency | Network + inference | Inference only |
| Control | Black box | Full control, can debug |
| Audio truncation | Known bug, no fix | Can investigate/fix |

## Model Specifications

| Attribute | Value |
|-----------|-------|
| Model ID | `nvidia/magpie_tts_multilingual_357m` |
| Parameters | 357M |
| Codec Model | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` |
| Output Sample Rate | 22kHz |
| Languages | en, es, de, fr, vi, it, zh |
| Speakers | John (0), Sofia (1), Aria (2), Jason (3), Leo (4) |
| Max Duration | 30 seconds per inference |
| License | NVIDIA Open Model License (commercial OK) |

## Current ASR Container Analysis

The `Dockerfile.asr-cuda13-build` provides:

```
Base: nvidia/cuda:13.1.0-devel-ubuntu24.04
PyTorch: Built from source with TORCH_CUDA_ARCH_LIST="12.1"
NeMo: r2.6.0 from source, installed with [asr] extras
Python: 3.12
Key libs: torchaudio, pytorch-lightning, transformers, librosa, soundfile
```

**What's missing for TTS:**
1. NeMo TTS extras (currently only `[asr]`)
2. Codec model download
3. `kaldialign` package (required per README)

## Implementation Plan

### Phase 1: Verify Model Access

```bash
# Test that we can download the model with current HF token
python3 -c "
from huggingface_hub import snapshot_download
# Just check access, don't download yet
from huggingface_hub import model_info
info = model_info('nvidia/magpie_tts_multilingual_357m', token='$HUGGINGFACE_TOKEN')
print(f'Model: {info.modelId}')
print(f'Size: {info.safetensors.total if info.safetensors else \"unknown\"}')
"
```

### Phase 2: Test Script

Created `tests/test_magpie_tts.py` - run inside the container:

```bash
# Quick test (install deps at runtime)
docker run --rm -it --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-asr:cuda13-full \
  bash -c "uv pip install 'nemo_toolkit[tts]' kaldialign && \
           python /workspace/tests/test_magpie_tts.py --benchmark"

# Interactive testing
docker run --rm -it --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-asr:cuda13-full bash

# Inside container:
uv pip install 'nemo_toolkit[tts]' kaldialign
python /workspace/tests/test_magpie_tts.py --text "Hello world" --speaker Aria --output /workspace/test.wav
```

The script:
- Downloads model from HuggingFace (uses $HUGGINGFACE_TOKEN)
- Runs inference with GPU timing
- Saves output WAV at native 22kHz
- Optional `--benchmark` for latency testing

### Phase 3: Container Modification Options

**Option A: Modify existing Dockerfile (Recommended)**

Add TTS dependencies to `Dockerfile.asr-cuda13-build`:

```dockerfile
# Change NeMo installation to include TTS
RUN cd /opt/nemo && uv pip install --no-cache -e ".[asr,tts]"

# Add kaldialign (required by Magpie)
RUN uv pip install --no-cache kaldialign
```

**Option B: Runtime installation (for testing)**

```bash
docker run --rm -it --gpus all --ipc=host \
  -v $(pwd)/models:/workspace/models \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
  nemotron-asr:cuda13-full \
  bash -c "
    pip install kaldialign
    cd /opt/nemo && pip install -e '.[tts]'
    python3 /workspace/tests/test_magpie_tts.py
  "
```

### Phase 4: Model Weight Management

**Download location:** `models/magpie_tts_multilingual_357m/`

```bash
# Download model weights (~1.4GB estimated)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/magpie_tts_multilingual_357m',
    local_dir='models/magpie_tts_multilingual_357m',
    local_dir_use_symlinks=False,
    token='hf_...'
)
"

# Download codec model (~100MB estimated)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps',
    local_dir='models/nemo-nano-codec-22khz-1.89kbps-21.5fps',
    local_dir_use_symlinks=False,
    token='hf_...'
)
"
```

### Phase 5: Blackwell Compatibility Testing

**Potential issues:**
1. **NVRTC compilation** - Same issue as ASR, already patched in container
2. **CUDA graphs** - May fail with sm_121, use eager mode fallback
3. **Triton kernels** - May need ptxas symlink like vLLM

**Test matrix:**

| Test | Command | Expected |
|------|---------|----------|
| Basic load | `model.from_pretrained(...)` | Success |
| CPU inference | `model.cpu(); model.do_tts(...)` | Success (slow) |
| GPU inference | `model.cuda(); model.do_tts(...)` | May fail |
| Eager mode | `torch.backends.cudnn.enabled = False` | Fallback |

## Memory Budget

| Component | Memory | Notes |
|-----------|--------|-------|
| LLM (vLLM) | ~72GB | 0.60 utilization |
| ASR (Parakeet) | ~3GB | 600M params |
| TTS (Magpie) | ~2GB | 357M params + codec |
| **Total** | ~77GB | |
| **Available** | 128GB | DGX Spark unified memory |
| **Headroom** | ~51GB | Comfortable margin |

## Audio Format Considerations

| Property | Magpie Output | Pipecat Expected | Action |
|----------|--------------|------------------|--------|
| Sample Rate | 22kHz | 16kHz (configurable) | Resample if needed |
| Channels | Mono | Mono | None |
| Format | Float32 tensor | PCM bytes | Convert |
| Duration | Up to 30s | Streaming chunks | Split long text |

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CUDA kernel failures on sm_121 | Medium | Eager mode, NVRTC patch already in container |
| Codec model loading issues | Low | Download codec separately, verify NeMo path |
| Memory pressure with all 3 models | Low | 51GB headroom, can reduce LLM utilization |
| Audio quality differences from NIM | Low | Same underlying model |
| Long text handling | Medium | Implement chunking for >30s text |

## Pipecat Service (After Testing)

After validating inference works, create `pipecat_bots/magpie_local_tts.py` following the `riva_local_tts.py` pattern:

```python
# Key differences from Riva:
# - No gRPC, in-process NeMo inference
# - Use asyncio.to_thread() for GPU inference (non-blocking)
# - Output sample_rate=22000 in TTSAudioRawFrame (Pipecat resamples for transport)
# - No retry logic needed (no network/NIM bugs)

class MagpieLocalTTSService(TTSService):
    SAMPLE_RATE = 22000  # Native codec rate

    def __init__(self, *, speaker: str = "Aria", language: str = "en", **kwargs):
        super().__init__(sample_rate=self.SAMPLE_RATE, **kwargs)
        self._model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
        self._model.cuda().eval()
        self._speaker_idx = SPEAKERS[speaker]
        self._language = language

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame()

        # Run GPU inference in thread pool (non-blocking)
        audio, _ = await asyncio.to_thread(
            self._model.do_tts, text,
            language=self._language,
            speaker_index=self._speaker_idx,
            apply_TN=False,
        )

        # Convert to bytes (16-bit PCM)
        audio_bytes = (audio.cpu().numpy() * 32767).astype('int16').tobytes()

        yield TTSAudioRawFrame(
            audio=audio_bytes,
            sample_rate=self.SAMPLE_RATE,  # Pipecat resamples to transport rate
            num_channels=1,
        )
        yield TTSStoppedFrame()
```

## Open Questions

1. **Codec loading**: Does `MagpieTTSModel.from_pretrained()` auto-download the codec, or must we load it separately?

2. **Text normalization**: The model supports `apply_TN=True` for English. Does this require additional dependencies?

3. **Streaming**: Can we get streaming audio output, or is it batch-only (generate full audio then yield)?

4. **Voice consistency**: Unlike NIM, we have direct access to speaker embeddings. Can we customize or blend voices?

5. **Audio format**: Does `do_tts()` return float32 [-1,1] or int16? Need to verify for correct conversion.

## Test Results (2025-12-22)

**CONFIRMED WORKING on DGX Spark (Blackwell sm_121)**

```
GPU: NVIDIA GB10
CUDA: 13.1
PyTorch: 2.11.0a0+git6493f23 (built from source)
```

**Performance:**
| Metric | Value |
|--------|-------|
| Model Load | 52.4s (includes codec download) |
| Warm-up | 230ms |
| Average Latency | 798ms |
| Min Latency | 497ms |
| Max Latency | 1094ms |
| RTF | 0.29x (3.4x faster than real-time) |

**Key Finding:** Requires NeMo **main branch** (not r2.6.0):
```bash
uv pip install 'nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git@main' kaldialign
```

The model config references `MagpieTTSModel` which only exists in main branch.

## Next Steps

1. [x] Create `tests/test_magpie_tts.py` test script
2. [x] Run test in container - **SUCCESS**
3. [x] Update Dockerfile to use NeMo main branch for TTS
4. [x] Create `pipecat_bots/magpie_local_tts.py` service
5. [ ] Rebuild container and integrate with bot.py

## Dockerfile Changes (2025-12-22)

Updated `Dockerfile.asr-cuda13-build` to support both ASR and TTS:

```dockerfile
# Pin to specific commit for reproducibility
ARG NEMO_COMMIT=661af02e105662bd5b61881054608ea44944572c
RUN git clone https://github.com/NVIDIA/NeMo.git /opt/nemo && \
    cd /opt/nemo && git checkout ${NEMO_COMMIT}

# Added kaldialign for TTS
RUN uv pip install --no-cache ... kaldialign

# Install with both ASR and TTS extras
RUN cd /opt/nemo && uv pip install --no-cache -e ".[asr,tts]"
```

To rebuild:
```bash
docker build -f Dockerfile.asr-cuda13-build -t nemotron-asr:cuda13-full .
```

## References

- Model: https://huggingface.co/nvidia/magpie_tts_multilingual_357m
- Codec: https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps
- NeMo TTS docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html
- License: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

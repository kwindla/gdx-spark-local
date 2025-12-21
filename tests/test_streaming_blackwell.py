#!/usr/bin/env python3
"""
Test cache-aware streaming ASR inference on Blackwell GPU.

This script uses NeMo's CacheAwareStreamingAudioBuffer for proper streaming inference.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

MODEL_PATH = "/workspace/models/Parakeet_Reatime_En_600M.nemo"
AUDIO_PATH = "/workspace/tests/fixtures/harvard_16k.wav"


def extract_transcriptions(hyps):
    """Extract text from hypothesis objects."""
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    if isinstance(hyps[0], Hypothesis):
        return [hyp.text for hyp in hyps]
    return hyps


def main():
    print("=" * 60)
    print("Cache-Aware Streaming ASR Test on Blackwell GPU")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Load model - use CPU first to avoid OOM on unified memory systems
    print(f"\nLoading model: {MODEL_PATH}")
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location='cpu')
    model = model.cuda()
    print(f"Model type: {type(model).__name__}")

    # Set attention context size for streaming (use [70, 1] for low latency)
    # The model supports multiple lookaheads: [70,13], [70,6], [70,1], [70,0]
    print("\nConfiguring for streaming with att_context_size=[70, 1]...")
    model.encoder.set_default_att_context_size([70, 1])

    # Check streaming config after setting
    scfg = model.encoder.streaming_cfg
    print(f"  New chunk_size: {scfg.chunk_size}")
    print(f"  New shift_size: {scfg.shift_size}")
    print(f"  New drop_extra_pre_encoded: {scfg.drop_extra_pre_encoded}")

    # CRITICAL: Change to greedy (non-batch) strategy and disable CUDA graphs
    print("\nApplying Blackwell workaround: greedy decoding (no CUDA graphs)...")
    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create({
            'strategy': 'greedy',
            'greedy': {
                'max_symbols': 10,
                'loop_labels': False,  # Disable loop labels (uses CUDA graphs)
                'use_cuda_graph_decoder': False,  # Explicitly disable CUDA graphs
            }
        })
    )
    model.eval()

    # Check streaming capability
    if not hasattr(model, 'conformer_stream_step'):
        print("ERROR: Model does not support streaming")
        sys.exit(1)
    print("✓ Model supports cache-aware streaming")

    # Get streaming configuration
    streaming_cfg = model.encoder.streaming_cfg
    print(f"\nStreaming config:")
    print(f"  chunk_size: {streaming_cfg.chunk_size}")
    print(f"  shift_size: {streaming_cfg.shift_size}")

    # Load test audio
    print(f"\nLoading audio: {AUDIO_PATH}")
    audio_data, sr = sf.read(AUDIO_PATH, dtype='float32')
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio_data) / sr:.2f}s")

    # Create streaming buffer (AFTER setting att_context_size)
    print("\nCreating streaming buffer...")
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    streaming_buffer.append_audio_file(AUDIO_PATH)
    print(f"  Loaded {len(streaming_buffer.streams_length)} stream(s)")

    batch_size = len(streaming_buffer.streams_length)

    # First try offline mode through conformer_stream_step (no caching)
    print("\n" + "=" * 60)
    print("Offline Mode (full audio through conformer_stream_step)")
    print("=" * 60)

    with torch.inference_mode():
        processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
        print(f"  Full audio processed_signal shape: {processed_signal.shape}")
        print(f"  Full audio processed_signal dtype: {processed_signal.dtype}, device: {processed_signal.device}")
        print(f"  Full audio processed_signal_length: {processed_signal_length}")

        with torch.no_grad():
            (
                pred_out_offline,
                transcribed_texts_offline,
                _cache1, _cache2, _cache3,
                _hyps,
            ) = model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                return_transcription=True,
            )

        print(f"  pred_out_offline: {[p.shape for p in pred_out_offline] if pred_out_offline else 'None'}")
        if transcribed_texts_offline:
            text = extract_transcriptions(transcribed_texts_offline)[0]
            print(f"  Offline transcription: \"{text[:100]}...\"")
        else:
            print(f"  Offline transcription: EMPTY")

    # Reset buffer for streaming test (re-create to reset iterator)
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    streaming_buffer.append_audio_file(AUDIO_PATH)

    batch_size = len(streaming_buffer.streams_length)

    # Initialize cache for streaming test
    cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    # Process audio in streaming mode
    print("\n" + "=" * 60)
    print("Streaming Inference")
    print("=" * 60)
    print(f"  Initial cache_last_channel_len: {cache_last_channel_len}")

    previous_hypotheses = None
    pred_out_stream = None
    transcribed_texts = None
    step_num = 0

    streaming_buffer_iter = iter(streaming_buffer)

    with torch.inference_mode():
        for chunk_audio, chunk_lengths in streaming_buffer_iter:
            step_num += 1
            print(f"  [DEBUG] chunk_audio shape: {chunk_audio.shape}, dtype: {chunk_audio.dtype}, device: {chunk_audio.device}")

            # Check if buffer is empty (last chunk)
            keep_all_outputs = streaming_buffer.is_buffer_empty()

            # Calculate drop_extra_pre_encoded
            if step_num == 1:
                drop_extra_pre_encoded = 0
            else:
                drop_extra_pre_encoded = model.encoder.streaming_cfg.drop_extra_pre_encoded

            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=keep_all_outputs,
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=drop_extra_pre_encoded,
                    return_transcription=True,
                )

            # Show interim result
            print(f"  Chunk {step_num}: cache_len={cache_last_channel_len.item()}, pred_shapes={[p.shape for p in pred_out_stream] if pred_out_stream else 'None'}")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    if transcribed_texts:
        final_text = extract_transcriptions(transcribed_texts)[0]
        print(f"Final transcript ({len(final_text)} chars):")
        print(f'"{final_text}"')
    else:
        print("No transcription returned")

    print("\n✓ SUCCESS - Cache-aware streaming inference completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

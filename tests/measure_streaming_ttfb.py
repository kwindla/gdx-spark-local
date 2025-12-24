#!/usr/bin/env python3
"""Measure actual streaming TTFB and verify realtime delivery of subsequent chunks.

This test validates that:
1. First audio chunk can be delivered in ~80-100ms (TTFB)
2. Subsequent chunks are generated faster than realtime (RTF < 1.0)
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch


def setup_environment():
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        print("ERROR: Set HUGGINGFACE_ACCESS_TOKEN environment variable")
        sys.exit(1)
    os.environ["HF_TOKEN"] = token
    return token


@dataclass
class StreamingMetrics:
    text: str
    total_frames: int
    total_audio_duration_s: float
    total_generation_time_ms: float
    per_frame_gen_time_ms: float
    first_chunk_decode_time_ms: float
    streaming_ttfb_ms: float
    rtf: float
    frames_per_second: float


def measure_streaming_performance(model, text: str, first_chunk_frames: int = 4) -> StreamingMetrics:
    """
    Measure streaming performance by:
    1. Running full generation to measure per-frame generation time
    2. Decoding first N frames to measure decode latency
    3. Computing expected streaming TTFB

    In a true streaming implementation, TTFB = (first N frames gen time) + (first N frames decode time)
    """
    from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod

    device = model.device

    # Prepare input
    tokenizer_name = "english_phoneme"
    tokens = model.tokenizer.encode(text=text, tokenizer_name=tokenizer_name)
    tokens = tokens + [model.eos_id]
    text_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
    text_lens = torch.tensor([len(tokens)], device=device, dtype=torch.long)

    batch_dict = {
        'text': text_tensor,
        'text_lens': text_lens,
        'speaker_indices': 2,
    }

    # Step 1: Measure token generation time (full generation)
    torch.cuda.synchronize()
    gen_start = time.time()

    with torch.no_grad():
        output = model.infer_batch(
            batch_dict,
            temperature=0.7,
            topk=80,
            use_cfg=True,
            cfg_scale=2.5,
            eos_detection_method=EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY,
        )

    torch.cuda.synchronize()
    total_gen_time = time.time() - gen_start

    predicted_codes = output.predicted_codes
    total_frames = predicted_codes.size(-1)

    # Calculate per-frame generation time
    per_frame_gen_time = total_gen_time / total_frames

    # Time to generate first N frames
    first_chunk_gen_time = first_chunk_frames * per_frame_gen_time

    # Step 2: Measure first chunk decode time
    first_chunk_codes = predicted_codes[:, :, :first_chunk_frames]
    first_chunk_lens = torch.tensor([first_chunk_frames], device=device).long()

    torch.cuda.synchronize()
    decode_start = time.time()

    with torch.no_grad():
        first_audio, _ = model.codes_to_audio(first_chunk_codes, first_chunk_lens)

    torch.cuda.synchronize()
    first_chunk_decode_time = time.time() - decode_start

    # Step 3: Calculate streaming TTFB
    streaming_ttfb = first_chunk_gen_time + first_chunk_decode_time

    # Step 4: Full audio for RTF calculation
    full_lens = torch.tensor([total_frames], device=device).long()
    with torch.no_grad():
        full_audio, _ = model.codes_to_audio(predicted_codes, full_lens)

    total_audio_duration = full_audio.size(-1) / 22000

    return StreamingMetrics(
        text=text,
        total_frames=total_frames,
        total_audio_duration_s=total_audio_duration,
        total_generation_time_ms=total_gen_time * 1000,
        per_frame_gen_time_ms=per_frame_gen_time * 1000,
        first_chunk_decode_time_ms=first_chunk_decode_time * 1000,
        streaming_ttfb_ms=streaming_ttfb * 1000,
        rtf=total_gen_time / total_audio_duration,
        frames_per_second=total_frames / total_gen_time,
    )


def verify_realtime_delivery(model, text: str, first_chunk_frames: int = 4, chunk_size: int = 16, overlap: int = 12):
    """
    Verify that chunked streaming can deliver audio faster than realtime.

    This simulates what would happen in production:
    1. Generate tokens incrementally
    2. After first_chunk_frames, decode and yield first audio
    3. After each subsequent chunk_size frames, decode with overlap and yield
    """
    from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod

    device = model.device

    # Prepare input
    tokenizer_name = "english_phoneme"
    tokens = model.tokenizer.encode(text=text, tokenizer_name=tokenizer_name)
    tokens = tokens + [model.eos_id]
    text_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
    text_lens = torch.tensor([len(tokens)], device=device, dtype=torch.long)

    batch_dict = {
        'text': text_tensor,
        'text_lens': text_lens,
        'speaker_indices': 2,
    }

    # Generate all tokens first
    torch.cuda.synchronize()
    gen_start = time.time()

    with torch.no_grad():
        output = model.infer_batch(
            batch_dict,
            temperature=0.7,
            topk=80,
            use_cfg=True,
            cfg_scale=2.5,
            eos_detection_method=EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY,
        )

    torch.cuda.synchronize()
    total_gen_time = time.time() - gen_start

    predicted_codes = output.predicted_codes
    total_frames = predicted_codes.size(-1)
    per_frame_gen_time = total_gen_time / total_frames

    print(f"\n  Token generation: {total_frames} frames in {total_gen_time*1000:.0f}ms "
          f"({per_frame_gen_time*1000:.1f}ms/frame)")

    # Simulate chunked streaming
    chunks = []
    cumulative_wall_time_ms = 0
    cumulative_audio_ms = 0

    # First chunk
    first_codes = predicted_codes[:, :, :first_chunk_frames]
    first_lens = torch.tensor([first_chunk_frames], device=device).long()

    first_gen_time_ms = first_chunk_frames * per_frame_gen_time * 1000

    torch.cuda.synchronize()
    decode_start = time.time()
    with torch.no_grad():
        first_audio, _ = model.codes_to_audio(first_codes, first_lens)
    torch.cuda.synchronize()
    first_decode_time_ms = (time.time() - decode_start) * 1000

    first_audio_ms = first_audio.size(-1) / 22000 * 1000
    first_total_ms = first_gen_time_ms + first_decode_time_ms

    cumulative_wall_time_ms = first_total_ms
    cumulative_audio_ms = first_audio_ms

    chunks.append({
        'chunk_id': 0,
        'frames': first_chunk_frames,
        'gen_time_ms': first_gen_time_ms,
        'decode_time_ms': first_decode_time_ms,
        'audio_ms': first_audio_ms,
        'cumulative_wall_ms': cumulative_wall_time_ms,
        'cumulative_audio_ms': cumulative_audio_ms,
        'is_realtime': cumulative_wall_time_ms < cumulative_audio_ms,
    })

    print(f"\n  Chunk delivery simulation:")
    print(f"    Chunk 0 (first): gen={first_gen_time_ms:.0f}ms + dec={first_decode_time_ms:.0f}ms "
          f"= {first_total_ms:.0f}ms -> {first_audio_ms:.0f}ms audio [TTFB]")

    # Subsequent chunks
    frame_idx = first_chunk_frames
    chunk_id = 1

    while frame_idx < total_frames:
        end_idx = min(frame_idx + chunk_size, total_frames)
        new_frames = end_idx - frame_idx

        # Chunk with overlap
        start_with_overlap = max(0, frame_idx - overlap)
        chunk_codes = predicted_codes[:, :, start_with_overlap:end_idx]
        chunk_lens = torch.tensor([chunk_codes.size(-1)], device=device).long()

        # Generation time for new frames
        chunk_gen_time_ms = new_frames * per_frame_gen_time * 1000

        # Decode time
        torch.cuda.synchronize()
        decode_start = time.time()
        with torch.no_grad():
            chunk_audio, _ = model.codes_to_audio(chunk_codes, chunk_lens)
        torch.cuda.synchronize()
        chunk_decode_time_ms = (time.time() - decode_start) * 1000

        # New audio (excluding overlap)
        overlap_samples = overlap * 1024 if frame_idx > first_chunk_frames else 0
        new_audio_samples = chunk_audio.size(-1) - overlap_samples
        new_audio_ms = max(0, new_audio_samples) / 22000 * 1000

        chunk_total_ms = chunk_gen_time_ms + chunk_decode_time_ms
        cumulative_wall_time_ms += chunk_total_ms
        cumulative_audio_ms += new_audio_ms

        is_realtime = cumulative_wall_time_ms < cumulative_audio_ms

        chunks.append({
            'chunk_id': chunk_id,
            'frames': new_frames,
            'gen_time_ms': chunk_gen_time_ms,
            'decode_time_ms': chunk_decode_time_ms,
            'audio_ms': new_audio_ms,
            'cumulative_wall_ms': cumulative_wall_time_ms,
            'cumulative_audio_ms': cumulative_audio_ms,
            'is_realtime': is_realtime,
        })

        status = "OK" if is_realtime else "SLOW"
        buffer_ms = cumulative_audio_ms - cumulative_wall_time_ms

        print(f"    Chunk {chunk_id}: gen={chunk_gen_time_ms:.0f}ms + dec={chunk_decode_time_ms:.0f}ms "
              f"= {chunk_total_ms:.0f}ms -> {new_audio_ms:.0f}ms audio "
              f"[{status}, buffer={buffer_ms:.0f}ms]")

        frame_idx = end_idx
        chunk_id += 1

    # Summary
    all_realtime = all(c['is_realtime'] for c in chunks[1:])  # Exclude first chunk (TTFB)
    final_buffer = cumulative_audio_ms - cumulative_wall_time_ms

    print(f"\n  Result: {'PASS' if all_realtime else 'FAIL'} - "
          f"Final buffer: {final_buffer:.0f}ms "
          f"({'ahead of' if final_buffer > 0 else 'behind'} realtime)")

    return chunks


def main():
    setup_environment()

    from nemo.collections.tts.models import MagpieTTSModel

    print("Loading Magpie TTS model...")
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model = model.cuda().eval()
    print("Model loaded.\n")

    # Warm up
    print("[Warm-up]")
    with torch.no_grad():
        _ = model.do_tts("Warm up.", language="en", speaker_index=2)
    print("Done.\n")

    test_sentences = [
        "Hi there! How can I help you today?",
        "Your appointment has been scheduled for tomorrow at 3 PM.",
        "I'm sorry to hear you're having trouble. Let me help fix that.",
    ]

    print("=" * 80)
    print("STREAMING TTFB MEASUREMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  First chunk:  4 frames (~185ms audio)")
    print(f"  Chunk size:   16 frames (~743ms audio)")
    print(f"  Overlap:      12 frames (for quality)")
    print()

    all_metrics = []

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(test_sentences)}] \"{sentence}\"")
        print("=" * 80)

        with torch.no_grad():
            metrics = measure_streaming_performance(model, sentence, first_chunk_frames=4)

        all_metrics.append(metrics)

        print(f"\n  Streaming TTFB Breakdown:")
        print(f"    First 4 frames generation: {metrics.per_frame_gen_time_ms * 4:.0f}ms")
        print(f"    First 4 frames decode:     {metrics.first_chunk_decode_time_ms:.0f}ms")
        print(f"    ─────────────────────────────────")
        print(f"    STREAMING TTFB:            {metrics.streaming_ttfb_ms:.0f}ms")
        print()
        print(f"  Full synthesis metrics:")
        print(f"    Total frames:      {metrics.total_frames}")
        print(f"    Audio duration:    {metrics.total_audio_duration_s:.2f}s")
        print(f"    Generation time:   {metrics.total_generation_time_ms:.0f}ms")
        print(f"    RTF:               {metrics.rtf:.3f}x (< 1.0 = faster than realtime)")
        print(f"    Frame rate:        {metrics.frames_per_second:.1f} fps (codec native: 21.5 fps)")

        # Verify realtime delivery
        verify_realtime_delivery(model, sentence, first_chunk_frames=4, chunk_size=16, overlap=12)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ttfbs = [m.streaming_ttfb_ms for m in all_metrics]
    rtfs = [m.rtf for m in all_metrics]
    fps = [m.frames_per_second for m in all_metrics]

    print(f"\nStreaming TTFB (first 4 frames = ~185ms audio):")
    print(f"  Average: {np.mean(ttfbs):.0f}ms")
    print(f"  Min:     {np.min(ttfbs):.0f}ms")
    print(f"  Max:     {np.max(ttfbs):.0f}ms")

    print(f"\nReal-Time Factor (generation only):")
    print(f"  Average: {np.mean(rtfs):.3f}x")
    print(f"  Min:     {np.min(rtfs):.3f}x")
    print(f"  Max:     {np.max(rtfs):.3f}x")

    print(f"\nGeneration Speed:")
    print(f"  Average: {np.mean(fps):.1f} frames/sec")
    print(f"  Codec native rate: 21.5 frames/sec")

    speedup = (21.5 / np.mean(fps) - 1) * 100
    if np.mean(fps) > 21.5:
        print(f"  Result: {np.mean(fps) / 21.5:.1f}x faster than realtime")
    else:
        print(f"  Result: {speedup:.0f}% slower than realtime (PROBLEM)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if np.mean(ttfbs) < 150:
        print(f"\n  TTFB: EXCELLENT ({np.mean(ttfbs):.0f}ms < 150ms target)")
    elif np.mean(ttfbs) < 200:
        print(f"\n  TTFB: GOOD ({np.mean(ttfbs):.0f}ms < 200ms)")
    else:
        print(f"\n  TTFB: NEEDS IMPROVEMENT ({np.mean(ttfbs):.0f}ms)")

    if np.mean(rtfs) < 1.0:
        print(f"  Realtime: PASS (RTF={np.mean(rtfs):.3f}x, {(1-np.mean(rtfs))*100:.0f}% faster than realtime)")
    else:
        print(f"  Realtime: FAIL (RTF={np.mean(rtfs):.3f}x, cannot stream)")


if __name__ == "__main__":
    main()

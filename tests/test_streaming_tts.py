#!/usr/bin/env python3
"""
Test script for streaming Magpie TTS inference.

This tests the StreamingMagpieTTS implementation to validate:
1. Streaming produces audio incrementally
2. TTFB is significantly lower than batch mode
3. Audio quality is comparable to batch mode

Prerequisites:
    Run inside the nemotron-asr:cuda13-full container with NeMo installed.

Usage:
    docker run --rm -it --gpus all --ipc=host \
        -v $(pwd):/workspace \
        -e HUGGINGFACE_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
        nemotron-asr:cuda13-full \
        python /workspace/tests/test_streaming_tts.py

    # Compare with batch mode:
    python /workspace/tests/test_streaming_tts.py --compare-batch

    # Run quality comparison across presets:
    python /workspace/tests/test_streaming_tts.py --quality-test

    # Test specific preset:
    python /workspace/tests/test_streaming_tts.py --preset aggressive
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def compute_snr(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio between reference and test audio.

    Higher SNR = more similar to reference.
    >30dB = essentially identical
    20-30dB = minor differences
    <20dB = noticeable differences
    """
    # Align lengths
    min_len = min(len(reference), len(test))
    ref = reference[:min_len].astype(np.float64)
    tst = test[:min_len].astype(np.float64)

    # Normalize
    ref = ref / (np.abs(ref).max() + 1e-8)
    tst = tst / (np.abs(tst).max() + 1e-8)

    # Compute SNR
    noise = ref - tst
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 100.0  # Essentially identical

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_correlation(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between audio signals.

    1.0 = identical, 0.0 = uncorrelated, -1.0 = inverted
    """
    min_len = min(len(reference), len(test))
    ref = reference[:min_len].astype(np.float64)
    tst = test[:min_len].astype(np.float64)

    # Normalize
    ref = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
    tst = (tst - np.mean(tst)) / (np.std(tst) + 1e-8)

    correlation = np.mean(ref * tst)
    return correlation


def find_best_alignment(reference: np.ndarray, test: np.ndarray, max_shift: int = 2048) -> tuple:
    """
    Find the best alignment between two audio signals by cross-correlation.
    Returns (shift, aligned_test) where shift is the sample offset.
    """
    from scipy import signal as scipy_signal

    # Use shorter segment for efficiency
    seg_len = min(len(reference), len(test), 22000)  # 1 second max
    ref_seg = reference[:seg_len].astype(np.float64)
    test_seg = test[:seg_len].astype(np.float64)

    # Cross-correlate
    correlation = scipy_signal.correlate(ref_seg, test_seg, mode='full')

    # Find peak
    mid = len(correlation) // 2
    search_range = slice(mid - max_shift, mid + max_shift)
    peak_idx = np.argmax(np.abs(correlation[search_range])) + mid - max_shift

    shift = peak_idx - (len(test_seg) - 1)

    # Apply shift
    if shift > 0:
        aligned = np.pad(test, (shift, 0))[:len(reference)]
    elif shift < 0:
        aligned = test[-shift:]
        aligned = np.pad(aligned, (0, max(0, len(reference) - len(aligned))))
    else:
        aligned = test

    return shift, aligned[:len(reference)]


def setup_environment():
    """Set up HuggingFace token and check environment."""
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        print("ERROR: Set HUGGINGFACE_TOKEN or HUGGINGFACE_ACCESS_TOKEN environment variable")
        sys.exit(1)
    os.environ["HF_TOKEN"] = token
    return token


def load_model():
    """Load Magpie TTS model."""
    import torch
    from nemo.collections.tts.models import MagpieTTSModel

    print("Loading Magpie TTS model...")
    start = time.time()
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model = model.cuda().eval()
    print(f"Model loaded in {time.time() - start:.1f}s")
    return model


def test_streaming_inference(model, text: str, compare_batch: bool = False, preset: str = "aggressive"):
    """Test streaming inference and measure TTFB."""
    import torch

    # Add source path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from nemotron_speech.streaming_tts import StreamingConfig, StreamingMagpieTTS, STREAMING_PRESETS

    print("\n" + "=" * 60)
    print("Testing Streaming TTS Inference")
    print("=" * 60)
    print(f"Text: \"{text}\"")
    print(f"Preset: {preset}")

    # Get config from preset
    config = STREAMING_PRESETS.get(preset, STREAMING_PRESETS["aggressive"])
    print(f"Config: first_chunk={config.min_first_chunk_frames} frames (~{config.min_first_chunk_frames * 46.4:.0f}ms)")

    streamer = StreamingMagpieTTS(model, config)

    # Measure streaming inference
    print("\n[Streaming Mode]")
    chunks = []
    chunk_times = []
    start_time = time.time()

    for i, chunk in enumerate(streamer.synthesize_streaming(text, language="en", speaker_index=2)):
        chunk_time = time.time() - start_time
        chunk_times.append(chunk_time)
        chunks.append(chunk)
        chunk_samples = len(chunk) // 2  # int16 = 2 bytes per sample
        chunk_duration_ms = chunk_samples / 22000 * 1000
        print(f"  Chunk {i+1}: {len(chunk):,} bytes ({chunk_duration_ms:.0f}ms audio) at t={chunk_time*1000:.0f}ms")

    total_streaming_time = time.time() - start_time

    # Combine chunks
    all_audio_bytes = b''.join(chunks)
    total_samples = len(all_audio_bytes) // 2
    total_duration_sec = total_samples / 22000

    print(f"\n  TTFB: {chunk_times[0]*1000:.0f}ms")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total audio: {total_duration_sec:.2f}s ({total_samples:,} samples)")
    print(f"  Total time: {total_streaming_time*1000:.0f}ms")
    print(f"  RTF: {total_streaming_time/total_duration_sec:.3f}x")

    # Compare with batch mode
    if compare_batch:
        print("\n[Batch Mode (for comparison)]")
        torch.cuda.synchronize()
        start_batch = time.time()

        with torch.no_grad():
            audio, audio_len = model.do_tts(text, language="en", speaker_index=2)

        torch.cuda.synchronize()
        batch_time = time.time() - start_batch

        batch_samples = audio.shape[-1] if audio.dim() > 1 else audio.shape[0]
        batch_duration_sec = batch_samples / 22000

        print(f"  Latency: {batch_time*1000:.0f}ms (all-or-nothing)")
        print(f"  Audio: {batch_duration_sec:.2f}s ({batch_samples:,} samples)")
        print(f"  RTF: {batch_time/batch_duration_sec:.3f}x")

        # Compare TTFB improvement
        ttfb_improvement = (batch_time - chunk_times[0]) / batch_time * 100
        print(f"\n[Comparison]")
        print(f"  TTFB improvement: {ttfb_improvement:.0f}%")
        print(f"  ({batch_time*1000:.0f}ms batch → {chunk_times[0]*1000:.0f}ms streaming)")

        # Compare audio length (should be similar)
        length_diff = abs(total_samples - batch_samples) / batch_samples * 100
        print(f"  Audio length difference: {length_diff:.1f}%")

    # Save streaming output
    output_path = Path("test_streaming_output.wav")
    save_audio(all_audio_bytes, output_path)
    print(f"\n  Saved streaming output to: {output_path}")

    return chunk_times[0], total_streaming_time, total_duration_sec


def save_audio(audio_bytes: bytes, path: Path, sample_rate: int = 22000):
    """Save raw PCM bytes to WAV file."""
    import soundfile as sf

    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    sf.write(str(path), audio_np, sample_rate)


def run_quality_comparison(model, text: str = "Hello, this is a quality comparison test for streaming text to speech."):
    """
    Compare chunked codec decoding vs batch codec decoding of the SAME tokens.

    This tests whether chunked decoding introduces artifacts, NOT whether
    streaming produces the same tokens (TTS is stochastic by design).
    """
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from nemotron_speech.streaming_tts import StreamingMagpieTTS, STREAMING_PRESETS

    print("\n" + "=" * 60)
    print("Quality Comparison: Chunked vs Batch Codec Decoding")
    print("=" * 60)
    print(f"Text: \"{text}\"")
    print("\nThis test compares chunked decoding vs batch decoding of the SAME tokens.")
    print("(TTS is stochastic, so we can't compare different generations directly.)\n")

    # First, generate tokens using batch mode and save them
    print("[Step 1: Generate tokens with batch inference]")
    torch.cuda.synchronize()
    start = time.time()

    # We need to access infer_batch to get the tokens
    from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

    # Prepare batch
    tokenizer_name = "english_phoneme"
    tokens = model.tokenizer.encode(text=text, tokenizer_name=tokenizer_name)
    tokens = tokens + [model.eos_id]
    text_tensor = torch.tensor([tokens], device=model.device, dtype=torch.long)
    text_lens = torch.tensor([len(tokens)], device=model.device, dtype=torch.long)
    batch = {'text': text_tensor, 'text_lens': text_lens, 'speaker_indices': 2}

    with torch.no_grad():
        output = model.infer_batch(batch, temperature=0.7, topk=80, use_cfg=True, cfg_scale=2.5)

    batch_gen_time = time.time() - start
    predicted_codes = output.predicted_codes  # (B, num_codebooks, T)
    predicted_codes_lens = output.predicted_codes_lens

    print(f"  Token generation: {batch_gen_time*1000:.0f}ms")
    print(f"  Tokens shape: {predicted_codes.shape}")
    print(f"  Token length: {predicted_codes_lens.item()} frames")

    # Decode tokens with batch mode (reference)
    print("\n[Step 2: Batch decode all tokens at once (reference)]")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        ref_audio, ref_audio_len = model.codes_to_audio(predicted_codes, predicted_codes_lens)
    torch.cuda.synchronize()
    batch_decode_time = time.time() - start

    ref_np = ref_audio.cpu().float().numpy().squeeze()
    ref_samples = len(ref_np)
    ref_duration = ref_samples / 22000

    print(f"  Decode time: {batch_decode_time*1000:.0f}ms")
    print(f"  Audio: {ref_duration:.2f}s ({ref_samples:,} samples)")

    save_audio((ref_np * 32767).astype(np.int16).tobytes(), Path("quality_test_batch_decode.wav"))

    # Now test chunked decoding of the SAME tokens
    print("\n[Step 3: Chunked decode same tokens with different chunk sizes]")

    results = []
    chunk_configs = [
        ("8f, overlap=4", 8, 4),
        ("8f, overlap=6", 8, 6),
        ("12f, overlap=6", 12, 6),
        ("12f, overlap=8", 12, 8),
        ("16f, overlap=8", 16, 8),
        ("16f, overlap=12", 16, 12),
        ("20f, overlap=10", 20, 10),
    ]

    num_frames = predicted_codes.size(-1)

    for name, chunk_size, overlap in chunk_configs:
        print(f"\n  [{name}] (overlap={overlap})")

        chunks_audio = []
        overlap_buffer = None
        samples_per_frame = 1024

        torch.cuda.synchronize()
        start = time.time()
        ttfb = None

        with torch.no_grad():
            frame_idx = 0
            while frame_idx < num_frames:
                # Get chunk of tokens
                end_idx = min(frame_idx + chunk_size, num_frames)

                if overlap_buffer is not None:
                    # Include overlap from previous chunk
                    chunk_tokens = torch.cat([overlap_buffer, predicted_codes[:, :, frame_idx:end_idx]], dim=-1)
                    overlap_samples = overlap * samples_per_frame
                else:
                    chunk_tokens = predicted_codes[:, :, frame_idx:end_idx]
                    overlap_samples = 0

                chunk_lens = torch.tensor([chunk_tokens.size(-1)], device=chunk_tokens.device).long()

                # Decode chunk
                chunk_audio, _ = model.codes_to_audio(chunk_tokens, chunk_lens)

                if ttfb is None:
                    torch.cuda.synchronize()
                    ttfb = time.time() - start

                # Extract new audio (skip overlap)
                chunk_np = chunk_audio.cpu().float().numpy().squeeze()
                if overlap_samples > 0 and len(chunk_np) > overlap_samples:
                    new_audio = chunk_np[overlap_samples:]
                else:
                    new_audio = chunk_np

                chunks_audio.append(new_audio)

                # Save overlap buffer for next iteration
                if end_idx < num_frames and (end_idx - frame_idx) >= overlap:
                    overlap_buffer = predicted_codes[:, :, end_idx - overlap:end_idx]
                else:
                    overlap_buffer = None

                frame_idx = end_idx

        torch.cuda.synchronize()
        total_time = time.time() - start

        # Combine chunks
        chunked_np = np.concatenate(chunks_audio)
        chunked_samples = len(chunked_np)

        # Compare with reference
        min_len = min(len(ref_np), len(chunked_np))
        snr = compute_snr(ref_np[:min_len], chunked_np[:min_len])
        corr = compute_correlation(ref_np[:min_len], chunked_np[:min_len])
        len_diff = abs(chunked_samples - ref_samples) / ref_samples * 100

        print(f"    TTFB: {ttfb*1000:.0f}ms")
        print(f"    Total decode: {total_time*1000:.0f}ms")
        print(f"    SNR vs batch: {snr:.1f}dB")
        print(f"    Correlation: {corr:.4f}")
        print(f"    Length diff: {len_diff:.1f}%")

        save_audio((chunked_np * 32767).astype(np.int16).tobytes(), Path(f"quality_test_chunked_{chunk_size}f.wav"))

        results.append({
            "name": name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "ttfb_ms": ttfb * 1000,
            "total_ms": total_time * 1000,
            "snr_db": snr,
            "correlation": corr,
            "len_diff_pct": len_diff,
        })

    # Summary
    print("\n" + "=" * 60)
    print("CHUNKED DECODING QUALITY SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<20} {'TTFB':>8} {'SNR':>10} {'Corr':>8} {'Quality':>12}")
    print("-" * 62)

    for r in results:
        quality = "Excellent" if r["snr_db"] > 30 else \
                  "Good" if r["snr_db"] > 20 else \
                  "Fair" if r["snr_db"] > 10 else "Poor"
        print(f"{r['name']:<20} {r['ttfb_ms']:>6.0f}ms {r['snr_db']:>8.1f}dB {r['correlation']:>8.4f} {quality:>12}")

    print("-" * 62)
    print(f"{'Batch decode (ref)':<20} {batch_decode_time*1000:>6.0f}ms {'inf':>10} {'1.0000':>8} {'Reference':>12}")

    print("\n[Interpretation]")
    print("  SNR > 30dB: Essentially identical to batch decode")
    print("  SNR 20-30dB: Minor differences, likely inaudible")
    print("  SNR 10-20dB: Audible differences at chunk boundaries")
    print("  SNR < 10dB: Significant degradation")

    return results


def run_latency_benchmark(model, preset: str = "aggressive"):
    """Run latency benchmark with multiple sentences."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from nemotron_speech.streaming_tts import StreamingMagpieTTS, STREAMING_PRESETS

    test_sentences = [
        "Hello, how can I help you today?",
        "The quick brown fox jumps over the lazy dog.",
        "I'm running streaming TTS inference on an NVIDIA DGX Spark.",
        "This demonstrates low latency text to speech with chunked decoding.",
    ]

    config = STREAMING_PRESETS.get(preset, STREAMING_PRESETS["aggressive"])
    streamer = StreamingMagpieTTS(model, config)

    print("\n" + "=" * 60)
    print("Streaming TTS Latency Benchmark")
    print("=" * 60)

    results = []

    # Warm-up
    print("\n[Warm-up]")
    for chunk in streamer.synthesize_streaming("Warm up.", language="en", speaker_index=2):
        pass
    print("  Done")

    # Benchmark
    for i, text in enumerate(test_sentences, 1):
        print(f"\n[{i}/{len(test_sentences)}] \"{text[:40]}...\"")

        chunks = []
        start = time.time()

        for chunk in streamer.synthesize_streaming(text, language="en", speaker_index=2):
            if not chunks:
                ttfb = time.time() - start
            chunks.append(chunk)

        total_time = time.time() - start
        total_samples = sum(len(c) // 2 for c in chunks)
        duration_sec = total_samples / 22000

        results.append({
            "text": text,
            "ttfb_ms": ttfb * 1000,
            "total_ms": total_time * 1000,
            "audio_sec": duration_sec,
            "rtf": total_time / duration_sec,
            "chunks": len(chunks),
        })

        print(f"  TTFB: {ttfb*1000:.0f}ms | Total: {total_time*1000:.0f}ms | Audio: {duration_sec:.2f}s | Chunks: {len(chunks)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ttfbs = [r["ttfb_ms"] for r in results]
    rtfs = [r["rtf"] for r in results]

    print(f"\n  TTFB (ms):")
    print(f"    Average: {np.mean(ttfbs):.0f}ms")
    print(f"    Min: {np.min(ttfbs):.0f}ms")
    print(f"    Max: {np.max(ttfbs):.0f}ms")

    print(f"\n  Real-time Factor:")
    print(f"    Average: {np.mean(rtfs):.3f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test streaming Magpie TTS inference")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of streaming text to speech synthesis.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--compare-batch",
        action="store_true",
        help="Compare streaming with batch mode",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark with multiple sentences",
    )
    parser.add_argument(
        "--quality-test",
        action="store_true",
        help="Run quality comparison across all presets",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="aggressive",
        choices=["aggressive", "balanced", "conservative"],
        help="Streaming preset to use (default: aggressive for lowest TTFB)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Streaming Magpie TTS Test")
    print("=" * 60)

    # Setup
    setup_environment()

    # Check for required imports
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: PyTorch not available")
        sys.exit(1)

    try:
        from nemo.collections.tts.models import MagpieTTSModel
        print("NeMo MagpieTTSModel: available")
    except ImportError as e:
        print(f"ERROR: NeMo not available: {e}")
        print("Run inside the nemotron-asr:cuda13-full container")
        sys.exit(1)

    # Load model
    model = load_model()

    # Run tests
    if args.quality_test:
        run_quality_comparison(model, args.text)
    elif args.benchmark:
        run_latency_benchmark(model, preset=args.preset)
    else:
        test_streaming_inference(model, args.text, compare_batch=args.compare_batch, preset=args.preset)

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark offline inference speed on Blackwell GPU."""
import warnings
warnings.filterwarnings('ignore')
import time
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

MODEL_PATH = '/workspace/models/Parakeet_Reatime_En_600M.nemo'
AUDIO_PATH = '/workspace/tests/fixtures/harvard_16k.wav'

print("=" * 60)
print("Offline Inference Benchmark on Blackwell GPU")
print("=" * 60)

# Load model
print("\nLoading model...")
t0 = time.perf_counter()
model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location='cuda:0')
model.change_decoding_strategy(
    decoding_cfg=OmegaConf.create({
        'strategy': 'greedy',
        'greedy': {'max_symbols': 10}
    })
)
model.eval()
model_load_time = time.perf_counter() - t0
print(f"  Model load time: {model_load_time:.2f}s")

# Load audio
audio_data, sr = sf.read(AUDIO_PATH, dtype='float32')
audio_duration = len(audio_data) / sr
print(f"\nAudio: {audio_duration:.2f}s @ {sr}Hz")

# Prepare tensors
audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).cuda()
audio_len = torch.tensor([len(audio_data)]).cuda()

# Warm-up run
print("\nWarm-up run...")
with torch.no_grad():
    processed, processed_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)
    encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
    _ = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_len)
torch.cuda.synchronize()

# Benchmark runs
print("\nBenchmarking (5 runs)...")
times = []
text = ""
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        processed, processed_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)
        encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
        result = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_len)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    times.append(total_time)
    print(f"  Run {i+1}: {total_time*1000:.1f}ms")

    if i == 0:
        if isinstance(result, tuple):
            best_hyp = result[0]
        else:
            best_hyp = result
        if best_hyp and best_hyp[0]:
            hyp = best_hyp[0]
            text = hyp.text if hasattr(hyp, 'text') else str(hyp)

avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Audio duration:     {audio_duration:.2f}s")
print(f"Inference time:     {avg_time*1000:.1f}ms (avg), {min_time*1000:.1f}ms (min), {max_time*1000:.1f}ms (max)")
print(f"Real-time factor:   {avg_time/audio_duration:.3f}x (< 1.0 = faster than real-time)")
print(f"Throughput:         {audio_duration/avg_time:.1f}x real-time")
print(f"\nTranscription ({len(text)} chars):")
print(f'"{text}"')

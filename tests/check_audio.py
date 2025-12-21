#!/usr/bin/env python3
"""Check audio file quality."""
import soundfile as sf
import numpy as np

data, sr = sf.read('/workspace/tests/fixtures/test_speech.wav')
print(f'Sample rate: {sr}')
print(f'Duration: {len(data)/sr:.2f}s')
print(f'Min: {data.min():.4f}, Max: {data.max():.4f}')
print(f'RMS: {np.sqrt(np.mean(data**2)):.4f}')
nonzero = np.sum(data != 0)
print(f'Non-zero samples: {nonzero} / {len(data)}')
print(f'First 20 samples: {data[:20]}')

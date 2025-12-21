#!/bin/bash
# tests/test_environment.sh
# Reproducible environment tests for Nemotron-Speech-ASR on DGX Spark
set -e

echo "=== TEST 1: CUDA Driver API ==="
python3 -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so')
result = libcuda.cuInit(0)
print(f'cuInit result: {result} (0 = success)')
device_count = ctypes.c_int()
libcuda.cuDeviceGetCount(ctypes.byref(device_count))
print(f'Device count: {device_count.value}')
device = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(device), 0)
name = ctypes.create_string_buffer(256)
libcuda.cuDeviceGetName(name, 256, device)
print(f'Device name: {name.value.decode()}')
"

echo ""
echo "=== TEST 2: NVIDIA-SMI ==="
nvidia-smi

echo ""
echo "=== TEST 3: System Architecture ==="
uname -a
lscpu | grep -E "Architecture|Model name|CPU\(s\):"

echo ""
echo "=== TEST 4: PyTorch in vLLM Container ==="
# Using 25.11-py3 (latest ARM64-compatible version as of 2025-12-20)
docker run --rm --gpus all nvcr.io/nvidia/vllm:25.11-py3 python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== TEST 5: NeMo ASR with GPU ==="
# NeMo is NOT in vLLM container - we install it (takes ~3-5 min)
docker run --rm --gpus all --ipc=host nvcr.io/nvidia/vllm:25.11-py3 bash -c "pip install -q nemo_toolkit[asr] && python3 -c \"import nemo.collections.asr as nemo_asr; import torch; print('NeMo ASR: OK'); print(f'Device: {torch.cuda.get_device_name(0)}')\""

echo ""
echo "=== All tests passed ==="

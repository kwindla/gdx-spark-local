#!/bin/bash
# tests/test_streaming_asr.sh
# Test cache-aware streaming ASR inference inside the Docker container
#
# Prerequisites:
#   1. Build the container: docker build -f Dockerfile.asr -t nemotron-asr:latest .
#   2. Model weights at: models/Parakeet_Reatime_En_600M.nemo
#
# Usage:
#   ./tests/test_streaming_asr.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Nemotron ASR Streaming Test ===${NC}"
echo ""

# Check if container exists
if ! docker image inspect nemotron-asr:latest &>/dev/null; then
    echo -e "${RED}Error: nemotron-asr:latest image not found${NC}"
    echo "Build it first: docker build -f Dockerfile.asr -t nemotron-asr:latest ."
    exit 1
fi

# Check if model exists
MODEL_PATH="$PROJECT_DIR/models/Parakeet_Reatime_En_600M.nemo"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model found: $(basename $MODEL_PATH)${NC}"

# Create test fixtures directory
FIXTURES_DIR="$PROJECT_DIR/tests/fixtures"
mkdir -p "$FIXTURES_DIR"

# Download LibriSpeech test sample if not present
TEST_AUDIO="$FIXTURES_DIR/test_speech.wav"
if [ ! -f "$TEST_AUDIO" ]; then
    echo "Downloading LibriSpeech test sample..."
    # This is a short sample from LibriSpeech test-clean
    curl -sL "https://www.openslr.org/resources/12/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac" -o "$FIXTURES_DIR/sample.flac" 2>/dev/null || {
        # Fallback: generate test audio with TTS inside container
        echo "LibriSpeech download failed, generating synthetic test audio..."
        docker run --rm \
            -v "$FIXTURES_DIR:/fixtures" \
            nemotron-asr:latest \
            python3 -c "
import numpy as np
import soundfile as sf

# Generate 3 seconds of audio with speech-like characteristics
# (sine waves at speech frequencies to test the pipeline)
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Mix of frequencies typical in speech (100-3000 Hz)
audio = 0.3 * np.sin(2 * np.pi * 200 * t)
audio += 0.2 * np.sin(2 * np.pi * 500 * t)
audio += 0.1 * np.sin(2 * np.pi * 1000 * t)

# Add amplitude envelope
envelope = np.exp(-0.5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
audio = audio * envelope

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.8

sf.write('/fixtures/test_speech.wav', audio.astype(np.float32), sample_rate)
print('Generated synthetic test audio')
"
    }

    # Convert FLAC to WAV if downloaded
    if [ -f "$FIXTURES_DIR/sample.flac" ]; then
        docker run --rm \
            -v "$FIXTURES_DIR:/fixtures" \
            nemotron-asr:latest \
            ffmpeg -y -i /fixtures/sample.flac -ar 16000 -ac 1 /fixtures/test_speech.wav 2>/dev/null
        rm -f "$FIXTURES_DIR/sample.flac"
    fi
fi

if [ ! -f "$TEST_AUDIO" ]; then
    echo -e "${RED}Error: Could not create test audio${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Test audio ready: $(basename $TEST_AUDIO)${NC}"

# Create manifest file
MANIFEST_FILE="$FIXTURES_DIR/test_manifest.json"
cat > "$MANIFEST_FILE" << 'EOF'
{"audio_filepath": "/workspace/tests/fixtures/test_speech.wav", "duration": 3.0, "text": "test transcription"}
EOF
echo -e "${GREEN}✓ Manifest created${NC}"

echo ""
echo -e "${YELLOW}=== Running Cache-Aware Streaming Inference ===${NC}"
echo ""

# Run the NeMo streaming inference script
docker run --rm --gpus all --ipc=host \
    -v /usr/local/cuda/lib64/libcufft.so.12.0.0.61:/usr/local/cuda/lib64/libcufft.so.12:ro \
    -v /usr/local/cuda/lib64/libcufftw.so.12.0.0.61:/usr/local/cuda/lib64/libcufftw.so.12:ro \
    -v "$PROJECT_DIR:/workspace" \
    nemotron-asr:latest \
    python3 /opt/nemo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
        model_path=/workspace/models/Parakeet_Reatime_En_600M.nemo \
        dataset_manifest=/workspace/tests/fixtures/test_manifest.json \
        output_path=/workspace/tests/fixtures/output \
        batch_size=1 \
        compare_vs_offline=true \
        debug_mode=true

echo ""
echo -e "${GREEN}=== Test Complete ===${NC}"

# Show output
if [ -d "$FIXTURES_DIR/output" ]; then
    echo ""
    echo "Output files:"
    ls -la "$FIXTURES_DIR/output/"
    echo ""
    echo "Transcription results:"
    cat "$FIXTURES_DIR/output/"*.json 2>/dev/null || echo "(no JSON output found)"
fi

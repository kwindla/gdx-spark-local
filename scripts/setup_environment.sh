#!/bin/bash
# Setup script for Nemotron-Speech-ASR on DGX Spark
# Uses uv for package management

set -e

echo "=== Nemotron-Speech-ASR Environment Setup ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Check CUDA
echo "Checking CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create virtual environment if not exists
VENV_PATH="${VENV_PATH:-.venv}"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    uv venv "$VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Install dependencies
echo "Installing dependencies with uv..."
uv pip install nemo_toolkit[asr] pipecat-ai gdown websockets

# Install the project
echo "Installing nemotron-speech..."
uv pip install -e .

# Check NeMo
echo "Checking NeMo..."
python3 -c "import nemo.collections.asr as nemo_asr; print('NeMo ASR: OK')"

# Download model if not present
MODEL_PATH="${MODEL_PATH:-models/nemotron_speech_asr.nemo}"
MODEL_DIR=$(dirname "$MODEL_PATH")

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading Nemotron-Speech-ASR model..."
    mkdir -p "$MODEL_DIR"

    # Google Drive file ID from the PDF
    FILE_ID="14Q5HhxFGEz0MfylAU3RJFow4qdvF_9sG"

    python3 -c "import gdown; gdown.download(id='$FILE_ID', output='$MODEL_PATH', quiet=False)"

    echo "Model downloaded to: $MODEL_PATH"
else
    echo "Model already exists at: $MODEL_PATH"
fi

# Verify model
echo "Verifying model..."
python3 -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.restore_from('$MODEL_PATH')
print(f'Model type: {type(model).__name__}')
print(f'Model loaded successfully!')
"

echo "=== Setup Complete ==="
echo "Activate the environment with: source $VENV_PATH/bin/activate"

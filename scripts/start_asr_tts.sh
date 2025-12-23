#!/bin/bash
# Start both ASR and TTS servers in the same container.
# This allows them to share the same CUDA context and GPU memory.

set -e

echo "Starting ASR server on port 8080..."
python -m nemotron_speech.server --port 8080 &
ASR_PID=$!

echo "Starting TTS server on port 8001..."
python -m nemotron_speech.tts_server --port 8001 &
TTS_PID=$!

echo "Both servers started. ASR PID=$ASR_PID, TTS PID=$TTS_PID"

# Wait for either to exit
wait -n $ASR_PID $TTS_PID
EXIT_CODE=$?

echo "A server exited with code $EXIT_CODE. Stopping remaining servers..."
kill $ASR_PID $TTS_PID 2>/dev/null || true
exit $EXIT_CODE

"""HTTP server for local Magpie TTS inference.

Exposes an OpenAI-compatible /v1/audio/speech endpoint for TTS synthesis.
Runs inside the ASR container where NeMo/PyTorch are available.

Usage:
    python -m nemotron_speech.tts_server --port 8001

Environment variables:
    HUGGINGFACE_ACCESS_TOKEN or HF_TOKEN - Required for model download
"""

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel

# Magpie TTS outputs at 22kHz
MAGPIE_SAMPLE_RATE = 22000

# Available speakers
SPEAKERS = {
    "john": 0,
    "sofia": 1,
    "aria": 2,
    "jason": 3,
    "leo": 4,
}

# Supported languages
LANGUAGES = ["en", "es", "de", "fr", "vi", "it", "zh"]

# Global model instance
_model = None
_model_lock = asyncio.Lock()


def get_model():
    """Get the loaded model instance."""
    global _model
    return _model


async def load_model(model_id: str = "nvidia/magpie_tts_multilingual_357m"):
    """Load Magpie TTS model."""
    global _model

    async with _model_lock:
        if _model is not None:
            return _model

        logger.info(f"Loading Magpie TTS model: {model_id}")

        def _load():
            from nemo.collections.tts.models import MagpieTTSModel

            # Ensure HuggingFace token is set
            hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            model = MagpieTTSModel.from_pretrained(model_id)
            model = model.cuda()
            model.eval()
            return model

        start = time.time()
        _model = await asyncio.to_thread(_load)
        elapsed = time.time() - start
        logger.info(f"Magpie TTS model loaded in {elapsed:.1f}s")

        return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    await load_model()
    yield


app = FastAPI(
    title="Magpie TTS Server",
    description="Local NVIDIA Magpie TTS inference server",
    version="1.0.0",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    """Request body for /v1/audio/speech endpoint."""

    input: str
    voice: str = "aria"
    language: str = "en"
    response_format: str = "pcm"  # pcm or wav
    speed: float = 1.0  # Not used, for OpenAI compatibility


class TTSConfig(BaseModel):
    """TTS configuration response."""

    sample_rate: int = MAGPIE_SAMPLE_RATE
    channels: int = 1
    encoding: str = "pcm_s16le"
    voices: list[str] = list(SPEAKERS.keys())
    languages: list[str] = LANGUAGES


@app.get("/health")
async def health():
    """Health check endpoint."""
    model = get_model()
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
    }


@app.get("/v1/audio/config")
async def get_config():
    """Get TTS configuration (sample rate, voices, etc.)."""
    return TTSConfig()


@app.post("/v1/audio/speech")
async def synthesize_speech(request: SpeechRequest):
    """Synthesize speech from text.

    OpenAI-compatible endpoint that returns raw audio bytes.

    Args:
        request: Speech synthesis request with text and voice parameters.

    Returns:
        Raw PCM audio bytes (16-bit signed, mono, 22kHz).
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate voice
    voice = request.voice.lower()
    if voice not in SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{request.voice}'. Available: {list(SPEAKERS.keys())}",
        )
    speaker_idx = SPEAKERS[voice]

    # Validate language
    language = request.language.lower()
    if language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language '{request.language}'. Available: {LANGUAGES}",
        )

    # Normalize unicode characters
    text = request.input
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("\u2014", "-")  # EM DASH
    text = text.replace("\u2013", "-")  # EN DASH

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    logger.debug(f"TTS request: voice={voice}, language={language}, text=[{text[:50]}...]")

    def _synthesize():
        with torch.no_grad():
            audio, audio_len = model.do_tts(
                text,
                language=language,
                speaker_index=speaker_idx,
                apply_TN=False,
            )

        # Convert to numpy
        audio_np = audio.cpu().float().numpy()

        # Handle tensor shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        # Normalize and convert to int16
        if np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()

        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    start = time.time()
    audio_bytes = await asyncio.to_thread(_synthesize)
    elapsed = time.time() - start

    duration_ms = len(audio_bytes) / (MAGPIE_SAMPLE_RATE * 2) * 1000
    logger.info(
        f"TTS: {len(audio_bytes)} bytes, {duration_ms:.0f}ms audio, "
        f"latency={elapsed*1000:.0f}ms, RTF={elapsed/(duration_ms/1000):.2f}x"
    )

    # Return raw PCM bytes
    media_type = "audio/pcm" if request.response_format == "pcm" else "audio/wav"
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-Sample-Rate": str(MAGPIE_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Encoding": "pcm_s16le",
            "X-Duration-Ms": str(int(duration_ms)),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Magpie TTS HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument(
        "--model",
        default="nvidia/magpie_tts_multilingual_357m",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    logger.info(f"Starting Magpie TTS server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

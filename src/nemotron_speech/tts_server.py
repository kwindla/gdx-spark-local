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
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from nemotron_speech.adaptive_stream import (
    StreamManager,
    StreamState,
    TTSStream,
    get_stream_manager,
)
from nemotron_speech.audio_boundary import AudioBoundaryHandler, ChunkedAudioBuffer

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
    """Load model and warm up on startup."""
    model = await load_model()

    # Warm up both batch and streaming paths to JIT compile CUDA kernels
    logger.info("Warming up TTS model (batch + streaming paths)...")

    def _warmup():
        import torch
        from nemotron_speech.streaming_tts import StreamingMagpieTTS, StreamingConfig

        with torch.no_grad():
            # Warm up batch path
            logger.info("  Warming up batch inference...")
            _, _ = model.do_tts("Warm up.", language="en", speaker_index=2, apply_TN=False)
            torch.cuda.synchronize()

            # Warm up streaming path (with CFG enabled for quality)
            logger.info("  Warming up streaming inference...")
            config = StreamingConfig(
                min_first_chunk_frames=8,
                chunk_size_frames=16,
                overlap_frames=12,
                use_cfg=True,
            )
            streamer = StreamingMagpieTTS(model, config)
            for _ in streamer.synthesize_streaming("Warm up.", language="en", speaker_index=2):
                pass
            torch.cuda.synchronize()

    warmup_start = time.time()
    await asyncio.to_thread(_warmup)
    warmup_time = time.time() - warmup_start
    logger.info(f"TTS warm-up complete in {warmup_time:.1f}s")

    # Start stream manager for adaptive streaming
    stream_manager = get_stream_manager()
    await stream_manager.start()

    yield

    # Cleanup
    await stream_manager.stop()


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


class StreamingSpeechRequest(BaseModel):
    """Request body for /v1/audio/speech/stream endpoint."""

    input: str
    voice: str = "aria"
    language: str = "en"
    # Streaming parameters
    first_chunk_frames: int = 8  # ~140ms TTFB
    chunk_size_frames: int = 16  # ~740ms chunks
    overlap_frames: int = 12  # For quality at chunk boundaries


class TTSConfig(BaseModel):
    """TTS configuration response."""

    sample_rate: int = MAGPIE_SAMPLE_RATE
    channels: int = 1
    encoding: str = "pcm_s16le"
    voices: list[str] = list(SPEAKERS.keys())
    languages: list[str] = LANGUAGES


# Adaptive streaming request/response models
class CreateStreamRequest(BaseModel):
    """Request body for creating an adaptive TTS stream."""

    voice: str = "aria"
    language: str = "en"


class CreateStreamResponse(BaseModel):
    """Response for stream creation."""

    stream_id: str
    audio_url: str


class AppendTextRequest(BaseModel):
    """Request body for appending text to a stream."""

    text: str


class StreamStatusResponse(BaseModel):
    """Response with stream status information."""

    stream_id: str
    state: str
    segments_generated: int
    audio_ms: float
    buffer_ms: float


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


@app.post("/v1/audio/speech/stream")
async def synthesize_speech_stream(request: StreamingSpeechRequest):
    """Streaming speech synthesis endpoint.

    Returns audio chunks as they become available, minimizing TTFB.
    Uses chunked transfer encoding to stream PCM audio bytes.

    Headers in response:
        X-Sample-Rate: 22000
        X-Channels: 1
        X-Encoding: pcm_s16le

    Args:
        request: Streaming speech synthesis request.

    Returns:
        StreamingResponse with chunked PCM audio.
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
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201C", '"')
    text = text.replace("\u201D", '"')
    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    logger.debug(
        f"Streaming TTS request: voice={voice}, language={language}, "
        f"first_chunk={request.first_chunk_frames}, text=[{text[:50]}...]"
    )

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        """Generate audio using frame-by-frame streaming for low TTFB.

        Uses StreamingMagpieTTS for incremental generation.
        """
        from nemotron_speech.streaming_tts import StreamingMagpieTTS, StreamingConfig

        start_time = time.time()
        first_chunk_sent = False
        total_bytes = 0
        chunk_count = 0

        # Configure streaming with CFG enabled for quality
        config = StreamingConfig(
            min_first_chunk_frames=request.first_chunk_frames,
            chunk_size_frames=request.chunk_size_frames,
            overlap_frames=request.overlap_frames,
            use_cfg=True,  # Keep CFG enabled for quality (RTF still < 1.0)
        )

        def _run_streaming():
            """Run frame-by-frame streaming TTS synchronously."""
            streamer = StreamingMagpieTTS(model, config)
            for chunk in streamer.synthesize_streaming(
                text,
                language=language,
                speaker_index=speaker_idx,
                apply_tn=False,
            ):
                yield chunk

        # Use a queue to bridge sync generator to async
        import queue
        import threading

        chunk_queue = queue.Queue()
        generation_done = False
        generation_error = None

        def run_generation():
            nonlocal generation_done, generation_error
            try:
                for chunk in _run_streaming():
                    chunk_queue.put(chunk)
            except Exception as e:
                import traceback
                logger.error(f"Streaming generation error: {e}\n{traceback.format_exc()}")
                generation_error = e
            finally:
                chunk_queue.put(None)  # Signal completion
                generation_done = True

        # Start generation in background thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Yield chunks with minimal latency
        while True:
            try:
                chunk = chunk_queue.get(timeout=0.001)
                if chunk is None:  # Completion signal
                    break
                if not first_chunk_sent:
                    ttfb = (time.time() - start_time) * 1000
                    logger.info(f"Streaming TTS TTFB: {ttfb:.0f}ms (frame-by-frame, CFG enabled)")
                    first_chunk_sent = True
                total_bytes += len(chunk)
                chunk_count += 1
                yield chunk
            except queue.Empty:
                if generation_done:
                    # Drain any remaining items
                    while True:
                        try:
                            chunk = chunk_queue.get_nowait()
                            if chunk is None:
                                break
                            total_bytes += len(chunk)
                            chunk_count += 1
                            yield chunk
                        except queue.Empty:
                            break
                    break
                # Brief yield to event loop
                await asyncio.sleep(0)

        gen_thread.join(timeout=10.0)

        if generation_error:
            logger.error(f"Streaming TTS error: {generation_error}")

        elapsed = time.time() - start_time
        duration_ms = total_bytes / (MAGPIE_SAMPLE_RATE * 2) * 1000
        logger.info(
            f"Streaming TTS complete: {total_bytes} bytes ({chunk_count} chunks), "
            f"{duration_ms:.0f}ms audio, total_time={elapsed*1000:.0f}ms, RTF={elapsed/(duration_ms/1000):.2f}x"
        )

    return StreamingResponse(
        audio_generator(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(MAGPIE_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Encoding": "pcm_s16le",
        },
    )


# =============================================================================
# Adaptive Streaming Endpoints
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize unicode characters in text."""
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("\u2014", "-")  # EM DASH
    text = text.replace("\u2013", "-")  # EN DASH
    return text


@app.post("/v1/tts/stream", response_model=CreateStreamResponse)
async def create_tts_stream(request: CreateStreamRequest):
    """Create a new adaptive TTS stream.

    Creates a stream that can receive text incrementally and returns
    audio as it's generated. The stream automatically switches between
    streaming mode (fast TTFB) and batch mode (higher quality) based
    on buffer status.

    Returns:
        stream_id: Unique identifier for this stream
        audio_url: URL to connect for receiving audio
    """
    # Validate voice
    voice = request.voice.lower()
    if voice not in SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{request.voice}'. Available: {list(SPEAKERS.keys())}",
        )

    # Validate language
    language = request.language.lower()
    if language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language '{request.language}'. Available: {LANGUAGES}",
        )

    stream_manager = get_stream_manager()
    stream = await stream_manager.create_stream(voice=voice, language=language)

    return CreateStreamResponse(
        stream_id=stream.stream_id,
        audio_url=f"/v1/tts/stream/{stream.stream_id}/audio",
    )


@app.post("/v1/tts/stream/{stream_id}/append", status_code=202)
async def append_text_to_stream(stream_id: str, request: AppendTextRequest):
    """Append text to an existing stream.

    Text is buffered briefly (~100ms) to allow efficient batching,
    then flushed on timeout or sentence boundary.

    Args:
        stream_id: The stream to append to
        request: Contains the text to append
    """
    stream_manager = get_stream_manager()
    stream = await stream_manager.get_stream(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    if not stream.is_active:
        raise HTTPException(
            status_code=400,
            detail=f"Stream {stream_id} is not active (state={stream.state.value})",
        )

    text = normalize_text(request.text)
    if text.strip():
        stream.append_text(text)

    return {"status": "accepted"}


@app.post("/v1/tts/stream/{stream_id}/close")
async def close_tts_stream(stream_id: str):
    """Signal that no more text will be appended.

    The stream will continue generating audio for any pending text,
    then complete.

    Args:
        stream_id: The stream to close
    """
    stream_manager = get_stream_manager()
    stream = await stream_manager.get_stream(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    stream.close()
    return {"status": "closed"}


@app.delete("/v1/tts/stream/{stream_id}")
async def cancel_tts_stream(stream_id: str):
    """Cancel a stream immediately.

    Stops any ongoing generation and cleans up resources.

    Args:
        stream_id: The stream to cancel
    """
    stream_manager = get_stream_manager()
    stream = await stream_manager.get_stream(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    stream.cancel()
    await stream_manager.remove_stream(stream_id)
    return {"status": "cancelled"}


@app.get("/v1/tts/stream/{stream_id}/audio")
async def receive_tts_audio(stream_id: str):
    """Receive audio from an adaptive TTS stream.

    Returns a chunked streaming response with PCM audio bytes.
    The stream uses adaptive mode selection:
    - First segment: streaming mode (~150ms TTFB)
    - Subsequent segments: batch mode when buffer > 500ms

    Headers:
        X-Sample-Rate: 22000
        X-Channels: 1
        X-Encoding: pcm_s16le

    On error, returns JSON with error details instead of audio.
    """
    stream_manager = get_stream_manager()
    stream = await stream_manager.get_stream(stream_id)

    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    speaker_idx = SPEAKERS[stream.voice]

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        """Generate audio adaptively based on buffer status."""
        from nemotron_speech.streaming_tts import StreamingMagpieTTS, StreamingConfig

        boundary_handler = AudioBoundaryHandler(MAGPIE_SAMPLE_RATE)
        chunk_buffer = ChunkedAudioBuffer(chunk_size_bytes=4096)

        stream.state = StreamState.GENERATING

        try:
            while stream.is_active or stream.has_pending_text():
                # Wait for text if none available
                if not stream.has_pending_text():
                    if stream.state == StreamState.CLOSED:
                        break  # No more text coming

                    # Wait for text with timeout
                    stream.text_available.clear()
                    try:
                        await asyncio.wait_for(
                            stream.text_available.wait(),
                            timeout=0.1,
                        )
                    except asyncio.TimeoutError:
                        # Flush text buffer on timeout
                        stream.flush_text_buffer()
                        continue

                # Get next segment
                text = stream.get_next_segment()
                if text is None:
                    continue

                logger.info(
                    f"[{stream.stream_id[:8]}] Generating: '{text[:50]}...' "
                    f"mode={stream.generation_mode.value} buffer={stream.buffer_ms:.0f}ms"
                )

                # Generate audio based on mode
                if stream.should_use_batch:
                    # Batch mode: higher quality, full generation
                    audio_bytes = await _generate_batch(
                        model, text, stream.language, speaker_idx
                    )

                    # Process through boundary handler
                    is_final = (
                        stream.state == StreamState.CLOSED
                        and not stream.has_pending_text()
                    )
                    processed = boundary_handler.process_segment(audio_bytes, is_final)
                    stream.record_audio_generated(len(processed))

                    # Yield in chunks
                    for chunk in chunk_buffer.add(processed):
                        yield chunk

                else:
                    # Streaming mode: fast TTFB, frame-by-frame
                    async for audio_chunk in _generate_streaming(
                        model, text, stream.language, speaker_idx
                    ):
                        # For streaming, we yield immediately for low latency
                        # Boundary handling is simpler - just record the audio
                        stream.record_audio_generated(len(audio_chunk))
                        yield audio_chunk

                stream.mark_segment_complete()

            # Flush remaining audio
            remaining = chunk_buffer.flush()
            if remaining:
                yield remaining

            # Flush boundary handler tail
            tail = boundary_handler.flush()
            if tail:
                yield tail

            stream.complete()
            # Only remove on successful completion
            await stream_manager.remove_stream(stream.stream_id)

        except Exception as e:
            import traceback

            logger.error(f"[{stream.stream_id[:8]}] Generation error: {e}\n{traceback.format_exc()}")
            stream.set_error(str(e))
            # Remove on error too
            await stream_manager.remove_stream(stream.stream_id)
            raise

    return StreamingResponse(
        audio_generator(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(MAGPIE_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Encoding": "pcm_s16le",
        },
    )


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================


@app.websocket("/ws/tts/stream")
async def websocket_tts_stream(websocket: WebSocket):
    """WebSocket endpoint for adaptive TTS streaming.

    Provides full-duplex communication for text-to-speech:
    - Client sends JSON messages with text to synthesize
    - Server sends binary audio chunks back
    - Single connection handles both directions simultaneously

    Client -> Server messages (JSON):
        {"type": "init", "voice": "aria", "language": "en"} - Initialize stream
        {"type": "text", "text": "Hello world"} - Append text
        {"type": "close"} - Signal end of text input
        {"type": "cancel"} - Abort immediately
        {"type": "ping"} - Keepalive

    Server -> Client messages:
        {"type": "stream_created", "stream_id": "..."} - Stream ready
        {"type": "metadata", "buffer_ms": 1250, "mode": "batch"} - Status update
        {"type": "done", "total_audio_ms": 5432} - Stream complete
        {"type": "error", "message": "...", "fatal": false} - Error
        {"type": "pong"} - Keepalive response
        Binary frames: Raw PCM audio (16-bit, 22kHz, mono)
    """
    await websocket.accept()

    stream_manager = get_stream_manager()
    stream: Optional[TTSStream] = None
    audio_task: Optional[asyncio.Task] = None

    # Default configuration
    voice = "aria"
    language = "en"

    async def send_audio():
        """Background task to generate and send audio via WebSocket.

        Simple loop:
        1. Try to flush buffer if enough audio accumulated
        2. Generate audio for pending segments (batch mode)
        3. Exit when closed and nothing left
        4. Sleep briefly to yield to receive loop
        """
        nonlocal stream
        if stream is None:
            return

        model = get_model()
        if model is None:
            await websocket.send_json({
                "type": "error",
                "message": "Model not loaded",
                "fatal": True,
            })
            return

        speaker_idx = SPEAKERS[stream.voice]
        stream.state = StreamState.GENERATING
        first_audio_time = None

        try:
            while True:
                # Try to flush buffer → pending queue
                flushed = stream.flush_text_buffer()
                if flushed:
                    words = len(flushed.split())
                    logger.info(f"[{stream.stream_id[:8]}] Flushed {words} words: '{flushed[:40]}...'")

                # Generate audio for pending segments
                if stream.pending_text:
                    text = stream.pending_text.pop(0)
                    logger.info(f"[{stream.stream_id[:8]}] Generating: '{text[:50]}...'")

                    # Generate audio (batch mode)
                    audio_bytes = await _generate_batch(
                        model, text, stream.language, speaker_idx
                    )

                    # Track first audio TTFB
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        ttfb = (first_audio_time - stream.created_at) * 1000
                        logger.info(f"[{stream.stream_id[:8]}] First audio, TTFB: {ttfb:.0f}ms")

                    stream.record_audio_generated(len(audio_bytes))
                    stream.mark_segment_complete()

                    # Send audio in chunks via WebSocket
                    chunk_size = 4096
                    for i in range(0, len(audio_bytes), chunk_size):
                        try:
                            await websocket.send_bytes(audio_bytes[i : i + chunk_size])
                        except Exception:
                            return  # WebSocket closed

                    continue  # Check for more pending text immediately

                # Exit when closed AND buffer empty AND queue empty
                if stream.state == StreamState.CLOSED and not stream.text_buffer:
                    break

                # Yield to let receive loop run
                await asyncio.sleep(0.01)

            stream.complete()

            # Send completion message
            try:
                await websocket.send_json({
                    "type": "done",
                    "total_audio_ms": stream.generated_audio_ms,
                    "segments_generated": stream.segments_generated,
                })
            except Exception:
                pass

            logger.info(
                f"[{stream.stream_id[:8]}] WS stream complete: "
                f"{stream.generated_audio_ms:.0f}ms audio, {stream.segments_generated} segments"
            )

        except asyncio.CancelledError:
            logger.debug(f"[{stream.stream_id[:8]}] WS audio task cancelled")
            raise
        except Exception as e:
            logger.error(f"[{stream.stream_id[:8]}] WS audio generation error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "fatal": True,
                })
            except Exception:
                pass
            stream.set_error(str(e))

    try:
        # DEBUG: Test with stream creation + text + audio task
        msg_count = 0
        loop_start = time.time()

        while True:
            wait_start = time.time()
            message = await websocket.receive_text()
            now = time.time()
            wait_time = (now - wait_start) * 1000
            elapsed = (now - loop_start) * 1000
            msg_count += 1

            data = json.loads(message)
            msg_type = data.get("type")

            # Log every message with timing
            if msg_type == "text":
                text_preview = data.get("text", "")[:20]
                logger.info(f"WS #{msg_count} @{elapsed:.0f}ms (wait:{wait_time:.0f}ms): text='{text_preview}'")
            else:
                logger.info(f"WS #{msg_count} @{elapsed:.0f}ms (wait:{wait_time:.0f}ms): type={msg_type}")

            if msg_type == "init":
                voice = data.get("voice", "aria").lower()
                language = data.get("language", "en").lower()
                if stream is None:
                    stream = await stream_manager.create_stream(voice=voice, language=language)
                    await websocket.send_json({"type": "stream_created", "stream_id": stream.stream_id})
                    logger.info(f"[{stream.stream_id[:8]}] Stream created, starting audio task...")
                    # Start audio task - this should NOT block
                    audio_task = asyncio.create_task(send_audio())
                    logger.info(f"[{stream.stream_id[:8]}] Audio task started @{(time.time() - loop_start)*1000:.0f}ms")

            elif msg_type == "text":
                text = normalize_text(data.get("text", ""))
                if stream is None:
                    stream = await stream_manager.create_stream(voice=voice, language=language)
                    await websocket.send_json({"type": "stream_created", "stream_id": stream.stream_id})
                    audio_task = asyncio.create_task(send_audio())
                if text.strip():
                    stream.append_text(text)

            elif msg_type == "close":
                if stream:
                    stream.close()
                # Wait for audio task to complete
                if audio_task:
                    try:
                        await asyncio.wait_for(audio_task, timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning("Audio task timeout on close")
                        audio_task.cancel()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected" + (f" [{stream.stream_id[:8]}]" if stream else ""))
        if stream is not None:
            stream.cancel()
            if audio_task:
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if stream is not None:
            stream.cancel()

    finally:
        # Cleanup
        if stream is not None:
            await stream_manager.remove_stream(stream.stream_id)


async def _generate_batch(
    model, text: str, language: str, speaker_idx: int
) -> bytes:
    """Generate audio using batch mode (full generation)."""

    def _synthesize():
        with torch.no_grad():
            audio, audio_len = model.do_tts(
                text,
                language=language,
                speaker_index=speaker_idx,
                apply_TN=False,
            )

        audio_np = audio.cpu().float().numpy()

        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        if np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()

        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    return await asyncio.to_thread(_synthesize)


async def _generate_streaming(
    model, text: str, language: str, speaker_idx: int
) -> AsyncGenerator[bytes, None]:
    """Generate audio using streaming mode (frame-by-frame)."""
    from nemotron_speech.streaming_tts import StreamingMagpieTTS, StreamingConfig
    import queue
    import threading

    config = StreamingConfig(
        min_first_chunk_frames=8,  # 8 frames for smoother audio quality
        chunk_size_frames=16,
        overlap_frames=12,
        use_cfg=True,
    )

    chunk_queue: queue.Queue = queue.Queue()
    generation_done = False
    generation_error = None

    def run_generation():
        nonlocal generation_done, generation_error
        try:
            streamer = StreamingMagpieTTS(model, config)
            for chunk in streamer.synthesize_streaming(
                text,
                language=language,
                speaker_index=speaker_idx,
                apply_tn=False,
            ):
                chunk_queue.put(chunk)
        except Exception as e:
            generation_error = e
        finally:
            chunk_queue.put(None)
            generation_done = True

    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()

    while True:
        try:
            chunk = chunk_queue.get(timeout=0.001)
            if chunk is None:
                break
            yield chunk
        except queue.Empty:
            if generation_done:
                while True:
                    try:
                        chunk = chunk_queue.get_nowait()
                        if chunk is None:
                            break
                        yield chunk
                    except queue.Empty:
                        break
                break
            await asyncio.sleep(0)

    gen_thread.join(timeout=10.0)

    if generation_error:
        raise generation_error


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

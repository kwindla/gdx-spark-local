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
import re
import threading
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


# Regex pattern to match emoji characters
# Covers: emoticons, dingbats, symbols, flags, skin tones, etc.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc symbols and pictographs
    "\U0001F680-\U0001F6FF"  # Transport and map symbols
    "\U0001F700-\U0001F77F"  # Alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric shapes extended
    "\U0001F800-\U0001F8FF"  # Supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]+",
    flags=re.UNICODE
)


def normalize_text(text: str) -> str:
    """Normalize unicode characters in text.

    Handles:
    - Smart quotes -> ASCII quotes
    - Em/en dashes -> hyphens
    - Emoji characters -> removed (causes tokenizer crash)
    """
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("\u2014", "-")  # EM DASH
    text = text.replace("\u2013", "-")  # EN DASH
    # Remove emoji characters that crash the tokenizer
    text = _EMOJI_PATTERN.sub("", text)
    return text


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
        {"type": "init", "voice": "aria", "language": "en", "default_mode": "batch"}
            - Initialize stream
            - default_mode: "batch" (default) or "stream"

        {"type": "text", "text": "Hello world", "mode": "stream", "preset": "conservative"}
            - Append text segment for synthesis
            - mode: "batch" (default) or "stream"
              - "batch": Full generation, higher quality
              - "stream": Frame-by-frame, faster TTFB
            - preset: Only for stream mode - "aggressive", "balanced", "conservative" (default)
              - aggressive: ~185ms TTFB
              - balanced: ~280ms TTFB
              - conservative: ~370ms TTFB

        {"type": "close"} - Signal end of text input
        {"type": "cancel"} - Abort immediately
        {"type": "ping"} - Keepalive

    Server -> Client messages:
        {"type": "stream_created", "stream_id": "..."} - Stream ready
        {"type": "segment_complete", "segment": 1, "audio_ms": 1234} - Segment audio sent
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
    default_mode = "batch"

    # Segment queue: list of (text, mode, preset) tuples
    segment_queue: list[tuple[str, str, Optional[str]]] = []
    queue_lock = asyncio.Lock()
    queue_event = asyncio.Event()  # Signal when new segments added

    async def send_audio():
        """Background task to generate and send audio via WebSocket.

        Processes segments from queue, using streaming or batch mode as specified.
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
                # Get next segment from queue
                segment = None
                async with queue_lock:
                    if segment_queue:
                        segment = segment_queue.pop(0)

                if segment:
                    text, mode, preset = segment
                    logger.info(f"[{stream.stream_id[:8]}] Generating: '{text[:50]}...' mode={mode}")
                    segment_bytes = 0

                    if mode == "stream":
                        # Streaming mode: frame-by-frame for fast TTFB
                        chunk_count = 0
                        # Cancel event for early termination on interruption
                        segment_cancel = threading.Event()
                        async for audio_chunk in _generate_streaming_with_preset(
                            model, text, stream.language, speaker_idx, preset or "conservative",
                            cancel_event=segment_cancel
                        ):
                            # Check if stream was cancelled (interruption)
                            if stream.state == StreamState.CANCELLED:
                                segment_cancel.set()  # Signal thread to stop
                                logger.debug(f"[{stream.stream_id[:8]}] Streaming cancelled mid-generation")
                                break

                            # Track first audio TTFB
                            if first_audio_time is None:
                                first_audio_time = time.time()
                                ttfb = (first_audio_time - stream.created_at) * 1000
                                logger.info(f"[{stream.stream_id[:8]}] First audio (streaming), TTFB: {ttfb:.0f}ms")

                            stream.record_audio_generated(len(audio_chunk))
                            segment_bytes += len(audio_chunk)
                            chunk_count += 1

                            # Send immediately for lowest latency
                            try:
                                await websocket.send_bytes(audio_chunk)
                            except Exception:
                                return  # WebSocket closed

                        logger.debug(f"[{stream.stream_id[:8]}] Streaming segment complete: {chunk_count} chunks")

                    else:
                        # Batch mode: full generation, higher quality
                        audio_bytes = await _generate_batch(
                            model, text, stream.language, speaker_idx
                        )
                        segment_bytes = len(audio_bytes)

                        # Track first audio TTFB
                        if first_audio_time is None:
                            first_audio_time = time.time()
                            ttfb = (first_audio_time - stream.created_at) * 1000
                            logger.info(f"[{stream.stream_id[:8]}] First audio (batch), TTFB: {ttfb:.0f}ms")

                        stream.record_audio_generated(segment_bytes)

                        # Send audio in chunks via WebSocket
                        chunk_size = 4096
                        for i in range(0, segment_bytes, chunk_size):
                            try:
                                await websocket.send_bytes(audio_bytes[i : i + chunk_size])
                            except Exception:
                                return  # WebSocket closed

                    # Signal segment complete to client
                    segment_audio_ms = segment_bytes / (MAGPIE_SAMPLE_RATE * 2) * 1000
                    try:
                        await websocket.send_json({
                            "type": "segment_complete",
                            "segment": stream.segments_generated + 1,
                            "audio_ms": segment_audio_ms,
                        })
                    except Exception:
                        return  # WebSocket closed

                    stream.mark_segment_complete()
                    continue  # Check for more segments immediately

                # Exit when closed/cancelled AND queue empty
                if stream.state in (StreamState.CLOSED, StreamState.CANCELLED):
                    async with queue_lock:
                        if not segment_queue:
                            break

                # Wait for new segments or check periodically
                queue_event.clear()
                try:
                    await asyncio.wait_for(queue_event.wait(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass

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
            import traceback
            logger.error(f"[{stream.stream_id[:8]}] WS audio generation error: {e}\n{traceback.format_exc()}")
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
                mode = data.get("mode", default_mode)
                logger.info(f"WS #{msg_count} @{elapsed:.0f}ms (wait:{wait_time:.0f}ms): text='{text_preview}' mode={mode}")
            else:
                logger.info(f"WS #{msg_count} @{elapsed:.0f}ms (wait:{wait_time:.0f}ms): type={msg_type}")

            if msg_type == "init":
                voice = data.get("voice", "aria").lower()
                language = data.get("language", "en").lower()
                default_mode = data.get("default_mode", "batch")

                # Clean up any old stream/task from previous response
                if audio_task is not None and not audio_task.done():
                    logger.debug("Cancelling old audio_task for new init")
                    audio_task.cancel()
                    try:
                        await asyncio.wait_for(audio_task, timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                if stream is not None:
                    await stream_manager.remove_stream(stream.stream_id)
                async with queue_lock:
                    segment_queue.clear()

                # Create fresh stream
                stream = await stream_manager.create_stream(voice=voice, language=language)
                await websocket.send_json({"type": "stream_created", "stream_id": stream.stream_id})
                logger.info(f"[{stream.stream_id[:8]}] Stream created (default_mode={default_mode}), starting audio task...")
                audio_task = asyncio.create_task(send_audio())
                logger.info(f"[{stream.stream_id[:8]}] Audio task started @{(time.time() - loop_start)*1000:.0f}ms")

            elif msg_type == "text":
                text = normalize_text(data.get("text", ""))
                mode = data.get("mode", default_mode)
                preset = data.get("preset")  # Only used for stream mode

                # Check if we need a new stream (first text, or previous stream closed/done)
                need_new_stream = (
                    stream is None or
                    stream.state in (StreamState.CLOSED, StreamState.COMPLETED, StreamState.CANCELLED, StreamState.ERROR)
                )

                if need_new_stream:
                    # Clean up any old stream/task
                    if audio_task is not None and not audio_task.done():
                        logger.debug("Cancelling old audio_task for new text")
                        audio_task.cancel()
                        try:
                            await asyncio.wait_for(audio_task, timeout=0.5)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                    if stream is not None:
                        await stream_manager.remove_stream(stream.stream_id)
                    async with queue_lock:
                        segment_queue.clear()

                    # Create fresh stream
                    stream = await stream_manager.create_stream(voice=voice, language=language)
                    await websocket.send_json({"type": "stream_created", "stream_id": stream.stream_id})
                    audio_task = asyncio.create_task(send_audio())

                if text.strip():
                    async with queue_lock:
                        segment_queue.append((text, mode, preset))
                    queue_event.set()

            elif msg_type == "close":
                if stream:
                    stream.close()

                # Signal the queue so audio_task can process remaining text and exit
                queue_event.set()

                # DON'T BLOCK waiting for audio_task here!
                # Let it finish in the background. We'll clean up when:
                # - audio_task completes naturally (sends "done", removes stream)
                # - A new stream starts (we'll cancel old task if still running)
                logger.debug(f"[{stream.stream_id[:8] if stream else 'none'}] Close received, audio continuing in background")

            elif msg_type == "cancel":
                # Immediate cancellation (for interruption handling)
                # Unlike "close", this stops generation immediately
                logger.debug(f"[{stream.stream_id[:8] if stream else 'none'}] Cancel received")

                if stream:
                    stream.cancel()  # Sets state to CANCELLED

                if audio_task:
                    audio_task.cancel()
                    try:
                        # Brief wait - task exits quickly via CANCELLED state check
                        await asyncio.wait_for(audio_task, timeout=0.1)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass  # Task will clean up on its own

                # Clear queue and reset state
                async with queue_lock:
                    segment_queue.clear()
                if stream:
                    await stream_manager.remove_stream(stream.stream_id)
                stream = None
                audio_task = None
                queue_event.clear()

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

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


def _generate_fade_out_tail(last_chunk: bytes, fade_ms: int = 20, sample_rate: int = MAGPIE_SAMPLE_RATE) -> bytes:
    """Generate a fade-out tail based on the last chunk's ending amplitude.

    Instead of modifying the last chunk (which would require buffering),
    we generate a short fade-out that smoothly transitions from the
    last sample value to silence.

    Args:
        last_chunk: The last audio chunk (16-bit PCM)
        fade_ms: Fade-out duration in milliseconds
        sample_rate: Audio sample rate

    Returns:
        A short fade-out audio tail
    """
    if not last_chunk or len(last_chunk) < 4:
        return b""

    # Get the last few samples from the chunk to determine ending amplitude
    audio = np.frombuffer(last_chunk, dtype=np.int16).astype(np.float32)
    if len(audio) < 4:
        return b""

    # Use average of last few samples as starting amplitude
    end_amplitude = np.mean(audio[-4:])

    # Generate fade-out from end_amplitude to 0
    fade_samples = int(sample_rate * fade_ms / 1000)
    fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_audio = end_amplitude * fade_curve

    return np.clip(fade_audio, -32768, 32767).astype(np.int16).tobytes()


def _apply_fade_out(audio_bytes: bytes, fade_ms: int = 20, sample_rate: int = MAGPIE_SAMPLE_RATE) -> bytes:
    """Apply fade-out to the end of audio to mask HiFiGAN artifacts.

    Args:
        audio_bytes: Raw PCM audio (16-bit signed, mono)
        fade_ms: Fade-out duration in milliseconds
        sample_rate: Audio sample rate

    Returns:
        Audio bytes with fade-out applied
    """
    if not audio_bytes:
        return audio_bytes

    fade_samples = int(sample_rate * fade_ms / 1000)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    if len(audio) < fade_samples:
        return audio_bytes

    # Apply fade-out to last fade_ms
    fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio[-fade_samples:] *= fade_curve

    return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


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

    audio_bytes = await asyncio.to_thread(_synthesize)
    # Apply fade-out to mask end-of-generation artifact
    return _apply_fade_out(audio_bytes)


async def _generate_streaming_with_preset(
    model, text: str, language: str, speaker_idx: int, preset: str = "conservative",
    cancel_event: Optional[threading.Event] = None
) -> AsyncGenerator[bytes, None]:
    """Generate audio using streaming mode with configurable preset.

    Args:
        preset: "aggressive" (~185ms TTFB), "balanced" (~280ms), "conservative" (~370ms, default)
        cancel_event: If set, generation stops early (for interruption handling)
    """
    from nemotron_speech.streaming_tts import StreamingMagpieTTS, STREAMING_PRESETS
    import queue

    # Get preset config, default to conservative
    # Make a copy to avoid mutating the shared preset
    from dataclasses import replace
    base_config = STREAMING_PRESETS.get(preset, STREAMING_PRESETS["conservative"])
    config = replace(base_config,
        use_cfg=True,       # Enable CFG for quality
        use_crossfade=False # Disable crossfade - causes audio artifacts at chunk boundaries
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
                # Check for cancellation between chunks (~46ms granularity)
                if cancel_event and cancel_event.is_set():
                    logger.debug("Streaming generation cancelled")
                    break
                chunk_queue.put(chunk)
        except Exception as e:
            generation_error = e
        finally:
            chunk_queue.put(None)
            generation_done = True

    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()

    # Track last chunk to apply fade-out at the end
    # We yield chunks immediately (no buffering) for low TTFB,
    # then send a fade-out tail after the last chunk
    last_chunk: bytes | None = None

    while True:
        try:
            chunk = chunk_queue.get(timeout=0.001)
            if chunk is None:
                # Generation done - send fade-out tail based on last chunk
                if last_chunk:
                    yield _generate_fade_out_tail(last_chunk)
                break
            yield chunk
            last_chunk = chunk
        except queue.Empty:
            if generation_done:
                while True:
                    try:
                        chunk = chunk_queue.get_nowait()
                        if chunk is None:
                            # Send fade-out tail based on last chunk
                            if last_chunk:
                                yield _generate_fade_out_tail(last_chunk)
                            break
                        yield chunk
                        last_chunk = chunk
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

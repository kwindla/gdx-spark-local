"""Modal deployment for Magpie TTS server.

Deploy to Modal with GPU support for fast TTS inference.

Usage:
    # Deploy to Modal
    modal deploy tts_server_modal.py

    # Run locally with Modal (for testing)
    modal serve tts_server_modal.py

Environment:
    Requires HUGGINGFACE_ACCESS_TOKEN secret in Modal
    Set up with: modal secret create huggingface HUGGINGFACE_ACCESS_TOKEN=hf_...
"""

import asyncio
import json
import re
import threading
import time
from typing import Optional
from loguru import logger

import modal
import numpy as np

# Modal app definition
app = modal.App("magpie-tts-server")

model_cache = modal.Volume.from_name("magpie-tts-model-cache", create_if_missing=True)
CACHE_PATH = "/tts-model"

# Define the container image with all dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
            # "CXX": "g++",
            # "CC": "g++",
            "TORCH_HOME": CACHE_PATH,
        }
    )
    .apt_install("git", "libsndfile1", "ffmpeg","cmake","clang")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "cuda-python==13.0.1",
        "fastapi[standard]",
        "pydantic",
        "loguru",
        "numpy<2.0.0",
        # "cdifflib",
        # Additional NeMo dependencies
        # "Cython",
        # "webdataset",
        "omegaconf",
        "hydra-core",
        # "pytorch-lightning",
        # "wandb",
        # "nemo_toolkit[tts]==2.6.0",
        # "cdifflib==1.2.9",
    ).uv_pip_install(
        "nemo_toolkit[tts]@git+https://github.com/NVIDIA/NeMo.git@644201898480ec8c8d0a637f0c773825509ac4dc",
        extra_options="--no-cache",
    )
)

# Constants
MAGPIE_SAMPLE_RATE = 22000
SPEAKERS = {
    "john": 0,
    "sofia": 1,
    "aria": 2,
    "jason": 3,
    "leo": 4,
}
LANGUAGES = ["en", "es", "de", "fr", "vi", "it", "zh"]

# Emoji pattern for text normalization
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
    """Normalize unicode characters in text."""
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("\u2014", "-")  # EM DASH
    text = text.replace("\u2013", "-")  # EN DASH
    text = _EMOJI_PATTERN.sub("", text)
    return text


def _apply_fade_out(audio_bytes: bytes, fade_ms: int = 20, sample_rate: int = MAGPIE_SAMPLE_RATE) -> bytes:
    """Apply fade-out to mask end-of-generation artifacts."""
    
    if not audio_bytes:
        return audio_bytes

    fade_samples = int(sample_rate * fade_ms / 1000)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    if len(audio) < fade_samples:
        return audio_bytes

    fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio[-fade_samples:] *= fade_curve

    return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


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


def _crossfade_to_silence(audio_bytes: bytes, crossfade_ms: int = 40, sample_rate: int = MAGPIE_SAMPLE_RATE) -> bytes:
    """Crossfade audio into silence, removing decoder artifacts.

    The Magpie decoder sometimes generates a "whoosh" artifact after speech ends -
    a burst of energy that appears after a period of near-silence. This function:
    1. Detects if there's a silence-then-artifact pattern
    2. Truncates at the silence point if artifact is found
    3. Applies raised cosine fade to reach exactly zero

    Args:
        audio_bytes: Raw audio bytes (int16)
        crossfade_ms: Crossfade duration in milliseconds
        sample_rate: Audio sample rate

    Returns:
        Audio bytes with smooth fade to silence (ends with zeros)
    """
    if not audio_bytes:
        return audio_bytes

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    if len(audio) < 2:
        return audio_bytes

    # Detect silence-then-artifact pattern (whoosh)
    # The decoder sometimes generates a burst of energy after the speech ends.
    # Strategy: Find last sustained silence, check if there's energy after it.
    window_ms = 5
    window_samples = int(sample_rate * window_ms / 1000)
    silence_threshold = 25  # RMS below this is considered silence

    # Analyze last 80ms (the buffer size)
    analysis_samples = min(len(audio), int(sample_rate * 0.080))
    start_pos = len(audio) - analysis_samples

    # Compute RMS for each window
    window_rms = []
    for i in range(start_pos, len(audio) - window_samples + 1, window_samples):
        window = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        window_rms.append((i, rms))

    # Find the last silence point (before any trailing artifact)
    # Look for pattern: silence (low RMS) followed by bump (high RMS)
    last_silence_pos = None
    found_artifact = False

    for i in range(len(window_rms) - 1, -1, -1):
        pos, rms = window_rms[i]
        if rms < silence_threshold:
            # Check if there's significant energy after this silence
            max_rms_after = max([r for _, r in window_rms[i+1:]], default=0)
            if max_rms_after > 60:  # Energy spike after silence = artifact
                last_silence_pos = pos
                found_artifact = True
                break

    truncate_point = None
    if found_artifact and last_silence_pos is not None:
        truncate_point = last_silence_pos
        audio = audio[:truncate_point]

    if len(audio) < 2:
        return b'\x00' * 10  # Return minimal silence

    # Apply fade to the entire remaining audio (or just the last crossfade_ms)
    crossfade_samples = min(int(sample_rate * crossfade_ms / 1000), len(audio))

    # Use raised cosine fade: 0.5 * (1 + cos(π*t)) goes from 1.0 → 0.0 exactly
    t = np.arange(crossfade_samples, dtype=np.float32) / crossfade_samples
    fade_curve = 0.5 * (1.0 + np.cos(np.pi * t))

    # Apply fade to the end of audio
    audio[-crossfade_samples:] *= fade_curve

    # Ensure the last few samples are exactly zero
    zero_samples = min(5, len(audio))
    audio[-zero_samples:] = 0

    return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


# Overlap duration for crossfade between streaming chunks (80ms = 1760 samples = 3520 bytes at 22kHz)
# Increased from 40ms to 80ms to provide smoother transitions when the non-causal HiFi-GAN
# vocoder produces uncorrelated waveforms at chunk boundaries due to different future context
STREAMING_OVERLAP_MS = 80
STREAMING_OVERLAP_BYTES = int(MAGPIE_SAMPLE_RATE * STREAMING_OVERLAP_MS / 1000) * 2


def _overlap_add(chunk1_tail: bytes, chunk2_head: bytes) -> bytes:
    """Blend overlapping audio regions using adaptive crossfade.

    The HiFi-GAN vocoder produces different waveforms for the same time period
    depending on future context. When correlation is high, audio represents the
    same signal and Hann overlap-add works. When correlation is low/negative,
    the waveforms are different and we use a simple crossfade instead.

    Args:
        chunk1_tail: Tail of previous chunk (overlap region)
        chunk2_head: Head of current chunk (overlapping time period)

    Returns:
        Blended audio (same length as inputs)
    """
    if len(chunk1_tail) != len(chunk2_head):
        raise ValueError(f"Overlap regions must match: {len(chunk1_tail)} vs {len(chunk2_head)}")

    if not chunk1_tail:
        return b""

    # Convert to float for processing
    a1 = np.frombuffer(chunk1_tail, dtype=np.int16).astype(np.float32)
    a2 = np.frombuffer(chunk2_head, dtype=np.int16).astype(np.float32)

    # Measure correlation and amplitude to determine blending strategy
    corr = np.corrcoef(a1, a2)[0, 1] if len(a1) > 1 else 0

    n = len(a1)
    t = np.arange(n, dtype=np.float32) / n

    # Adaptive blending based on correlation:
    # - High correlation (>0.5): Hann overlap-add (COLA) - audio is similar
    # - Low correlation: Equal-power crossfade - maintains constant energy during transition
    if corr > 0.5:
        # Hann window halves - sum to 1.0 at every point (COLA constraint)
        w1 = 0.5 * (1.0 + np.cos(np.pi * t))  # 1.0 → 0.0
        w2 = 0.5 * (1.0 - np.cos(np.pi * t))  # 0.0 → 1.0
    else:
        # Equal-power crossfade for uncorrelated audio (w1² + w2² = 1)
        # This maintains constant energy during the transition, reducing perceived "pop"
        # For uncorrelated signals, linear crossfade causes a 3dB dip at the midpoint
        w1 = np.cos(np.pi * t / 2)   # 1.0 → 0.0 (smooth)
        w2 = np.sin(np.pi * t / 2)   # 0.0 → 1.0 (smooth)

    blended = a1 * w1 + a2 * w2

    return np.clip(blended, -32768, 32767).astype(np.int16).tobytes()


# Modal class for TTS inference
@app.cls(
    image=image,
    volumes = {
        CACHE_PATH: model_cache,
    },
    gpu="A100",  # Use A100 GPU for fast inference
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF token for model download
    timeout=3600,  # 1 hour timeout for long-running requests
    # min_containers = 1,
)
class MagpieTTSModel:
    """Modal class for Magpie TTS inference."""

    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        import sys
        import torch
        from loguru import logger
        from nemo.collections.tts.models import MagpieTTSModel
        
        # Add local module to path
        sys.path.insert(0, "/root")
        
        from ..streaming_tts import StreamingMagpieTTS, StreamingConfig

        logger.info("Loading Magpie TTS model...")
        model_id = "nvidia/magpie_tts_multilingual_357m"

        # Load model
        self.model = MagpieTTSModel.from_pretrained(model_id)
        self.model = self.model.cuda()
        self.model.eval()
        logger.info("Model loaded successfully")

        # Warm up both batch and streaming paths to JIT compile CUDA kernels
        logger.info("Warming up TTS model (batch + streaming paths)...")

        # Use a warmup text matching the longest expected input to pre-allocate GPU memory.
        # Too short = OOM during long inference; too long = can't load other models.
        # ~180 chars matches the longest test utterances.
        warmup_text = (
            "I just finished reading a fascinating book about the history of computing. "
            "It discussed how early computers filled entire rooms. "
            "Tell me more about the evolution of computer hardware."
        )

        with torch.no_grad():
            # Warm up batch path with long text to allocate peak memory
            logger.info(f"  Warming up batch inference ({len(warmup_text)} chars)...")
            _, _ = self.model.do_tts(warmup_text, language="en", speaker_index=2, apply_TN=False)
            torch.cuda.synchronize()

            # Warm up streaming path (with CFG enabled for quality)
            logger.info("  Warming up streaming inference...")
            config = StreamingConfig(
                min_first_chunk_frames=8,
                chunk_size_frames=16,
                overlap_frames=12,
                use_cfg=True,
            )
            streamer = StreamingMagpieTTS(self.model, config)
            for _ in streamer.synthesize_streaming(warmup_text, language="en", speaker_index=2):
                pass
            torch.cuda.synchronize()

            # Release cached memory. The warmup pre-allocated peak memory for TTS inference;
            # now we free the cached intermediates while keeping the model weights loaded.
            torch.cuda.empty_cache()
            logger.info("  Released CUDA cache after warmup")

        logger.info("Warm-up complete")

    def _synthesize_batch(self, text: str, voice: str = "aria", language: str = "en") -> bytes:
        """Internal: Synthesize speech in batch mode (full generation)."""
        import torch

        text = normalize_text(text)
        speaker_idx = SPEAKERS.get(voice.lower(), 2)

        with torch.no_grad():
            audio, audio_len = self.model.do_tts(
                text,
                language=language,
                speaker_index=speaker_idx,
                apply_TN=False,
            )

        # Convert to bytes
        audio_np = audio.cpu().float().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        if np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()

        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Apply fade-out
        return _apply_fade_out(audio_bytes)

    async def _generate_streaming_with_preset(
        self, text: str, language: str, speaker_idx: int, preset: str = "conservative",
        cancel_event: Optional[threading.Event] = None
    ):
        """Generate audio using streaming mode with COLA overlap-add (AsyncGenerator).

        Uses overlap-add to seamlessly blend chunk boundaries:
        1. StreamingMagpieTTS preserves overlap at chunk heads
        2. We blend overlapping regions using adaptive Hann/equal-power windows
        3. Apply crossfade to silence at the end to eliminate artifacts

        This is an async generator that yields chunks as they're produced.

        Args:
            text: Text to synthesize (already normalized)
            language: Language code
            speaker_idx: Speaker index
            preset: "aggressive" (~185ms TTFB), "balanced" (~280ms), "conservative" (~370ms)
            cancel_event: If set, generation stops early (for interruption handling)

        Yields:
            bytes: Audio chunks ready for streaming
        """
        import queue
        
        logger.debug(f"Importing StreamingMagpieTTS from ..streaming_tts")
        try:
            from ..streaming_tts import StreamingMagpieTTS, STREAMING_PRESETS
            logger.debug(f"Successfully imported StreamingMagpieTTS and STREAMING_PRESETS")
        except Exception as e:
            logger.error(f"Failed to import streaming_tts: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
        from dataclasses import replace

        # Get preset config
        base_config = STREAMING_PRESETS.get(preset, STREAMING_PRESETS["conservative"])
        config = replace(base_config,
            use_cfg=True,       # Enable CFG for quality
            use_crossfade=False # Crossfade handled here in post-vocoder audio buffer
        )

        chunk_queue = queue.Queue()
        generation_done = False
        generation_error = None

        def run_generation():
            nonlocal generation_done, generation_error
            try:
                logger.debug(f"Creating StreamingMagpieTTS with model={self.model}, config={config}")
                streamer = StreamingMagpieTTS(self.model, config)
                logger.debug(f"Starting synthesize_streaming for text: {text[:50]}...")
                chunk_count = 0
                for chunk in streamer.synthesize_streaming(
                    text,
                    language=language,
                    speaker_index=speaker_idx,
                    apply_tn=False,
                ):
                    # Check for cancellation between chunks (~46ms granularity)
                    if cancel_event and cancel_event.is_set():
                        logger.debug(f"Cancellation requested after {chunk_count} chunks")
                        break
                    chunk_queue.put(chunk)
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.debug(f"First chunk generated: {len(chunk)} bytes")
                logger.debug(f"Generation complete: {chunk_count} chunks total")
            except Exception as e:
                logger.error(f"Error in run_generation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                generation_error = e
            finally:
                chunk_queue.put(None)
                generation_done = True

        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Audio buffer for overlap-add at chunk boundaries
        audio_buffer = b""
        overlap_bytes = STREAMING_OVERLAP_BYTES

        def process_chunk(chunk: bytes) -> Optional[bytes]:
            """Process a chunk with COLA overlap-add. Returns bytes to yield or None."""
            nonlocal audio_buffer

            if not chunk:
                return None

            if not audio_buffer:
                # First chunk: no overlap to blend, just buffer the tail for next chunk
                if len(chunk) > overlap_bytes:
                    audio_buffer = chunk[-overlap_bytes:]
                    return chunk[:-overlap_bytes]
                else:
                    # Tiny first chunk - buffer entirely
                    audio_buffer = chunk
                    return None
            else:
                # Subsequent chunk: chunk[:overlap_bytes] overlaps with audio_buffer
                # Both represent the same time period - blend using adaptive overlap-add
                effective_overlap = min(overlap_bytes, len(audio_buffer), len(chunk) // 2)

                if effective_overlap > 0 and len(chunk) >= 2 * effective_overlap:
                    # Normal path: overlap-add, yield middle, buffer tail
                    blended = _overlap_add(
                        audio_buffer[-effective_overlap:],
                        chunk[:effective_overlap]
                    )
                    # Yield: excess buffer (if any) + blended overlap + middle of chunk
                    excess = audio_buffer[:-effective_overlap] if len(audio_buffer) > effective_overlap else b""
                    middle = chunk[effective_overlap:-overlap_bytes]
                    audio_buffer = chunk[-overlap_bytes:]
                    return excess + blended + middle
                else:
                    # Edge case: chunk too small - concatenate and re-buffer
                    combined = audio_buffer + chunk
                    if len(combined) > overlap_bytes:
                        audio_buffer = combined[-overlap_bytes:]
                        return combined[:-overlap_bytes]
                    else:
                        audio_buffer = combined
                        return None

        def yield_final_buffer():
            """Yield the final audio buffer with fade-out applied."""
            if not audio_buffer:
                return None
            # Check if buffer is already near-silent (no need to fade)
            buf_arr = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
            buf_rms = np.sqrt(np.mean(buf_arr ** 2))
            if buf_rms < 50:
                return audio_buffer
            # Apply crossfade to silence for natural ending
            return _crossfade_to_silence(audio_buffer, crossfade_ms=40)

        # Yield chunks as they arrive
        while True:
            try:
                chunk = chunk_queue.get(timeout=0.001)
                if chunk is None:
                    # Generation done - yield final buffer
                    final = yield_final_buffer()
                    if final:
                        yield final
                    break
                to_yield = process_chunk(chunk)
                if to_yield:
                    yield to_yield
            except queue.Empty:
                if generation_done:
                    # Drain any remaining chunks from queue
                    while True:
                        try:
                            chunk = chunk_queue.get_nowait()
                            if chunk is None:
                                final = yield_final_buffer()
                                if final:
                                    yield final
                                break
                            to_yield = process_chunk(chunk)
                            if to_yield:
                                yield to_yield
                        except queue.Empty:
                            break
                    break
                await asyncio.sleep(0)

        gen_thread.join(timeout=10.0)

        if generation_error:
            raise generation_error

    @modal.asgi_app()
    def api(self):
        """FastAPI app with all TTS endpoints."""
        import sys
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
        from fastapi.responses import Response
        from pydantic import BaseModel
        from loguru import logger
        
        # Add local module to path
        sys.path.insert(0, "/root")
        
        from ..adaptive_stream import get_stream_manager, StreamState, TTSStream

        # Pydantic models for request validation
        class SpeechRequest(BaseModel):
            input: str
            voice: str = "aria"
            language: str = "en"
            response_format: str = "pcm"
            speed: float = 1.0

        web_app = FastAPI(
            title="Magpie TTS Server (Modal)",
            description="Modal-deployed NVIDIA Magpie TTS inference server",
            version="1.0.0",
        )

        # Initialize stream manager (shared across all websocket connections)
        stream_manager = get_stream_manager()
        stream_manager_started = False

        async def ensure_stream_manager():
            nonlocal stream_manager_started
            if not stream_manager_started:
                await stream_manager.start()
                stream_manager_started = True

        @web_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
            }

        @web_app.get("/v1/audio/config")
        async def config():
            """Get TTS configuration."""
            return {
                "sample_rate": MAGPIE_SAMPLE_RATE,
                "channels": 1,
                "encoding": "pcm_s16le",
                "voices": list(SPEAKERS.keys()),
                "languages": LANGUAGES,
            }

        @web_app.post("/v1/audio/speech")
        async def speech(request: SpeechRequest):
            """OpenAI-compatible speech synthesis endpoint."""
            # Validate
            voice = request.voice.lower()
            if voice not in SPEAKERS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown voice '{voice}'. Available: {list(SPEAKERS.keys())}",
                )

            language = request.language.lower()
            if language not in LANGUAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown language '{language}'. Available: {LANGUAGES}",
                )

            text = normalize_text(request.input)
            if not text.strip():
                raise HTTPException(status_code=400, detail="Empty input text")

            logger.info(f"TTS request: voice={voice}, language={language}, text=[{text[:50]}...]")

            # Generate audio
            start = time.time()
            audio_bytes = await asyncio.to_thread(self._synthesize_batch, text, voice, language)
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

        @web_app.websocket("/ws/tts/stream")
        async def websocket_tts_stream(websocket: WebSocket):
            """WebSocket endpoint for adaptive TTS streaming.
            
            Provides full-duplex communication for text-to-speech.
            See original tts_server.py for full protocol documentation.
            """
            await ensure_stream_manager()
            await websocket.accept()
            
            stream: Optional[TTSStream] = None
            audio_task: Optional[asyncio.Task] = None
            
            # Default configuration
            voice = "aria"
            language = "en"
            default_mode = "batch"
            
            # Segment queue
            segment_queue: list[tuple[str, str, Optional[str]]] = []
            queue_lock = asyncio.Lock()
            queue_event = asyncio.Event()
            
            async def send_audio():
                """Background task to generate and send audio."""
                nonlocal stream
                if stream is None:
                    logger.warning("send_audio called with no stream")
                    return
                
                speaker_idx = SPEAKERS[stream.voice]
                stream.state = StreamState.GENERATING
                first_audio_time = None
                
                logger.debug(f"[{stream.stream_id[:8]}] send_audio started, waiting for segments...")
                
                try:
                    while True:
                        # Get next segment
                        segment = None
                        async with queue_lock:
                            if segment_queue:
                                segment = segment_queue.pop(0)
                        
                        if segment:
                            text, mode, preset = segment
                            logger.info(f"[{stream.stream_id[:8]}] Generating: '{text[:50]}...' mode={mode}")
                            segment_bytes = 0
                            
                            if mode == "stream":
                                # Streaming mode - use async generator
                                logger.info(f"[{stream.stream_id[:8]}] Starting streaming generation...")
                                try:
                                    # Create cancel event for this segment
                                    segment_cancel = threading.Event()
                                    
                                    # Iterate through chunks as they're generated
                                    async for audio_chunk in self._generate_streaming_with_preset(
                                        text, stream.language, speaker_idx, preset or "conservative",
                                        cancel_event=segment_cancel
                                    ):
                                        # Check if stream was cancelled (interruption)
                                        if stream.state == StreamState.CANCELLED:
                                            segment_cancel.set()  # Signal thread to stop
                                            logger.debug(f"[{stream.stream_id[:8]}] Streaming cancelled mid-generation")
                                            break
                                        
                                        if first_audio_time is None:
                                            first_audio_time = time.time()
                                            ttfb = (first_audio_time - stream.created_at) * 1000
                                            logger.info(f"[{stream.stream_id[:8]}] First audio (streaming), TTFB: {ttfb:.0f}ms")
                                        
                                        stream.record_audio_generated(len(audio_chunk))
                                        segment_bytes += len(audio_chunk)
                                        
                                        try:
                                            await websocket.send_bytes(audio_chunk)
                                        except Exception:
                                            return
                                    
                                    logger.debug(f"[{stream.stream_id[:8]}] Streaming segment complete")
                                except Exception as e:
                                    logger.error(f"[{stream.stream_id[:8]}] Error in streaming generation: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    raise
                            
                            else:
                                # Batch mode
                                audio_bytes = await asyncio.to_thread(
                                    self._synthesize_batch,
                                    text, stream.voice, stream.language
                                )
                                segment_bytes = len(audio_bytes)
                                
                                if first_audio_time is None:
                                    first_audio_time = time.time()
                                    ttfb = (first_audio_time - stream.created_at) * 1000
                                    logger.info(f"[{stream.stream_id[:8]}] First audio (batch), TTFB: {ttfb:.0f}ms")
                                
                                stream.record_audio_generated(segment_bytes)
                                
                                # Send in chunks
                                chunk_size = 4096
                                for i in range(0, segment_bytes, chunk_size):
                                    try:
                                        await websocket.send_bytes(audio_bytes[i : i + chunk_size])
                                    except Exception:
                                        return
                            
                            # Signal segment complete
                            segment_audio_ms = segment_bytes / (MAGPIE_SAMPLE_RATE * 2) * 1000
                            try:
                                await websocket.send_json({
                                    "type": "segment_complete",
                                    "segment": stream.segments_generated + 1,
                                    "audio_ms": segment_audio_ms,
                                })
                            except Exception:
                                return
                            
                            stream.mark_segment_complete()
                            continue
                        
                        # Exit when closed/cancelled and queue empty
                        if stream.state in (StreamState.CLOSED, StreamState.CANCELLED):
                            async with queue_lock:
                                if not segment_queue:
                                    logger.debug(f"[{stream.stream_id[:8]}] State is {stream.state}, queue empty, exiting send_audio loop")
                                    break
                        
                        # Wait for new segments
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
                
                logger.info("WebSocket message loop starting")
                
                while True:
                    message = await websocket.receive_text()
                    now = time.time()
                    msg_count += 1
                    
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    logger.info(f"WS #{msg_count}: type={msg_type}, data={data}")
                    
                    if msg_type == "init":
                        voice = data.get("voice", "aria").lower()
                        language = data.get("language", "en").lower()
                        default_mode = data.get("default_mode", "batch")
                        
                        # Clean up old stream
                        if audio_task is not None and not audio_task.done():
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
                        logger.info(f"[{stream.stream_id[:8]}] Stream created (default_mode={default_mode})")
                        audio_task = asyncio.create_task(send_audio())
                    
                    elif msg_type == "text":
                        text = normalize_text(data.get("text", ""))
                        mode = data.get("mode", default_mode)
                        preset = data.get("preset")
                        
                        logger.info(f"[{stream.stream_id[:8] if stream else 'none'}] Received text: '{text[:50]}...' mode={mode}")
                        
                        # Check if we need a new stream
                        need_new_stream = (
                            stream is None or
                            stream.state in (StreamState.CLOSED, StreamState.COMPLETED, StreamState.CANCELLED, StreamState.ERROR)
                        )
                        
                        if need_new_stream:
                            logger.info(f"Creating new stream (need_new_stream={need_new_stream}, current_state={stream.state if stream else 'None'})")
                            # Clean up and create new stream
                            if audio_task is not None and not audio_task.done():
                                audio_task.cancel()
                                try:
                                    await asyncio.wait_for(audio_task, timeout=0.5)
                                except (asyncio.TimeoutError, asyncio.CancelledError):
                                    pass
                            if stream is not None:
                                await stream_manager.remove_stream(stream.stream_id)
                            async with queue_lock:
                                segment_queue.clear()
                            
                            stream = await stream_manager.create_stream(voice=voice, language=language)
                            await websocket.send_json({"type": "stream_created", "stream_id": stream.stream_id})
                            logger.info(f"[{stream.stream_id[:8]}] New stream created, starting audio task")
                            audio_task = asyncio.create_task(send_audio())
                        
                        if text.strip():
                            async with queue_lock:
                                segment_queue.append((text, mode, preset))
                                queue_len = len(segment_queue)
                            logger.info(f"[{stream.stream_id[:8]}] Added segment to queue (queue_len={queue_len}), signaling event")
                            queue_event.set()
                    
                    elif msg_type == "close":
                        logger.info(f"[{stream.stream_id[:8] if stream else 'none'}] Close message received")
                        if stream:
                            async with queue_lock:
                                queue_len = len(segment_queue)
                            logger.info(f"[{stream.stream_id[:8]}] Before close: stream_state={stream.state}, queue_len={queue_len}")
                            stream.close()
                            logger.info(f"[{stream.stream_id[:8]}] After close: stream_state={stream.state}")
                        queue_event.set()
                        logger.debug(f"[{stream.stream_id[:8] if stream else 'none'}] Close signaled to audio task")
                    
                    elif msg_type == "cancel":
                        logger.debug(f"[{stream.stream_id[:8] if stream else 'none'}] Cancel received")
                        
                        if stream:
                            stream.cancel()
                        
                        if audio_task:
                            audio_task.cancel()
                            try:
                                await asyncio.wait_for(audio_task, timeout=0.1)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                        
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

        return web_app


# Local development entrypoint
# @app.local_entrypoint()
if __name__ == "__main__":
    """Local entrypoint for testing the deployed TTS service."""
    import json
    import requests
    from pathlib import Path
    
    print("Magpie TTS Server - Modal Deployment Test")
    print("==========================================")
    print()
    
    # Get the deployed API URL
    print("Getting deployed API URL...")
    MagpieClass = modal.Cls.from_name("magpie-tts-server", "MagpieTTSModel")
    api_url = MagpieClass().api.web_url
    print(f"API URL: {api_url}")
    print()
    
    # Test the /v1/audio/speech endpoint
    print("Testing /v1/audio/speech endpoint...")
    test_text = "Hello from Modal! This is a test of the Magpie TTS server."
    
    payload = {
        "input": test_text,
        "voice": "aria",
        "language": "en",
        "response_format": "pcm",
    }
    
    print(f"Text: '{test_text}'")
    print(f"Voice: {payload['voice']}")
    print(f"Language: {payload['language']}")
    print()
    
    response = requests.post(
        f"{api_url}/v1/audio/speech",
        json=payload,
        timeout=60,
    )
    
    if response.status_code == 200:
        # Get audio metadata from headers
        sample_rate = int(response.headers.get("X-Sample-Rate", "22000"))
        duration_ms = response.headers.get("X-Duration-Ms", "unknown")
        channels = int(response.headers.get("X-Channels", "1"))
        
        # Save audio as WAV file
        import wave
        output_file = Path("modal_test_output.wav")
        
        with wave.open(str(output_file), 'wb') as wav_file:
            wav_file.setnchannels(channels)  # mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(response.content)
        
        print("✅ Success!")
        print(f"   Generated: {len(response.content)} bytes")
        print(f"   Duration: {duration_ms}ms")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Channels: {channels}")
        print(f"   Saved to: {output_file.absolute()}")
        print()
        print(f"Play with: ffplay {output_file}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    
    print()
    print("=" * 60)
    print()
    
    # Test WebSocket streaming endpoint
    print("Testing WebSocket streaming endpoint...")
    import asyncio
    
    async def test_websocket():
        try:
            import websockets
        except ImportError:
            print("⚠️  websockets library not installed. Skipping WebSocket test.")
            print("   Install with: pip install websockets")
            return
        
        ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_endpoint = f"{ws_url}/ws/tts/stream"
        
        test_streaming_text = "Testing streaming synthesis with Modal deployment. This should arrive as chunks."
        print(f"Text: '{test_streaming_text}'")
        print(f"Mode: stream")
        print(f"Preset: conservative")
        print()
        
        audio_chunks = []
        chunk_times = []  # Track timing for each chunk
        
        try:
            async with websockets.connect(ws_endpoint, max_size=10_000_000) as websocket:
                # Send init message
                print("📤 Sending init message...")
                await websocket.send(json.dumps({
                    "type": "init",
                    "voice": "aria",
                    "language": "en",
                    "default_mode": "stream"
                }))
                print("✓ Init message sent")
                
                # Wait for stream_created
                print("⏳ Waiting for stream_created response...")
                msg = await websocket.recv()
                print(f"📥 Received message type: {type(msg)}, length: {len(msg) if isinstance(msg, (str, bytes)) else 'N/A'}")
                
                if isinstance(msg, bytes):
                    print(f"⚠️  Received binary message instead of JSON: {msg[:100]}")
                    # Try to receive the actual JSON message
                    msg = await websocket.recv()
                    print(f"📥 Second message type: {type(msg)}")
                
                data = json.loads(msg)
                print(f"📡 {data.get('type')}: stream_id={data.get('stream_id', 'N/A')[:8]}...")
                
                # Send text to synthesize and start timing
                import time
                request_start = time.time()
                print(f"📤 Sending text message: '{test_streaming_text[:50]}...'")
                await websocket.send(json.dumps({
                    "type": "text",
                    "text": test_streaming_text,
                    "mode": "stream",
                    "preset": "conservative"
                }))
                print(f"✓ Sent text for streaming synthesis at t=0.000s")
                print()
                
                # Receive audio chunks
                chunk_count = 0
                done = False
                first_chunk_time = None
                last_chunk_time = request_start
                
                while not done:
                    msg = await websocket.recv()
                    current_time = time.time()
                    
                    if isinstance(msg, bytes):
                        # Binary audio chunk
                        audio_chunks.append(msg)
                        chunk_count += 1
                        
                        # Calculate timing
                        time_since_request = (current_time - request_start) * 1000  # ms
                        
                        if first_chunk_time is None:
                            # First chunk - this is TTFB
                            first_chunk_time = current_time
                            ttfb = time_since_request
                            print(f"🎵 Chunk #{chunk_count}: {len(msg):6d} bytes | TTFB: {ttfb:7.1f}ms")
                        else:
                            # Subsequent chunks - show delta from last chunk
                            time_since_last = (current_time - last_chunk_time) * 1000  # ms
                            print(f"🎵 Chunk #{chunk_count}: {len(msg):6d} bytes | +{time_since_last:6.1f}ms | Total: {time_since_request:7.1f}ms")
                        
                        chunk_times.append({
                            'chunk': chunk_count,
                            'size': len(msg),
                            'time_since_request': time_since_request,
                            'time_since_last': (current_time - last_chunk_time) * 1000 if chunk_count > 1 else ttfb
                        })
                        last_chunk_time = current_time
                        
                    else:
                        # JSON message
                        data = json.loads(msg)
                        msg_type = data.get("type")
                        
                        if msg_type == "segment_complete":
                            elapsed = (current_time - request_start) * 1000
                            print(f"✓  Segment complete: {data.get('audio_ms', 0):.0f}ms audio (elapsed: {elapsed:.0f}ms)")
                        elif msg_type == "done":
                            done = True
                            total_elapsed = (current_time - request_start) * 1000
                            print()
                            print(f"✅ Stream complete!")
                            print(f"   Total audio: {data.get('total_audio_ms', 0):.0f}ms")
                            print(f"   Segments: {data.get('segments_generated', 0)}")
                            print(f"   Total time: {total_elapsed:.0f}ms")
                            print(f"   Chunks: {chunk_count}")
                            if chunk_times:
                                avg_chunk_time = sum(t['time_since_last'] for t in chunk_times[1:]) / max(len(chunk_times) - 1, 1)
                                print(f"   Avg inter-chunk latency: {avg_chunk_time:.1f}ms")
                        elif msg_type == "error":
                            print(f"❌ Error: {data.get('message')}")
                            done = True
                
                # Send close
                await websocket.send(json.dumps({"type": "close"}))
        
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            return
        
        # Save streamed audio as WAV
        if audio_chunks:
            import wave
            output_file = Path("modal_streaming_output.wav")
            
            # Combine all chunks
            combined_audio = b''.join(audio_chunks)
            
            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(22000)  # Magpie sample rate
                wav_file.writeframes(combined_audio)
            
            print()
            print(f"💾 Saved {len(combined_audio)} bytes ({chunk_count} chunks) to: {output_file.absolute()}")
            print(f"   Play with: ffplay {output_file}")
        else:
            print("⚠️  No audio chunks received")
    
    asyncio.run(test_websocket())
    
    print()
    print("=" * 60)
    print()
    print("Deployment Info:")
    print("================")
    print(f"API Base URL: {api_url}")
    print()
    print("Available Endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /v1/audio/config  - Get TTS configuration")
    print("  POST /v1/audio/speech  - Synthesize speech (OpenAI-compatible)")
    print("  WS   /ws/tts/stream    - WebSocket streaming endpoint")

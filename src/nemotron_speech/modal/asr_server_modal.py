"""Modal deployment for Nemotron Streaming ASR server.

Deploy to Modal with GPU support for real-time speech recognition.

Usage:
    # Deploy to Modal
    modal deploy src/nemotron_speech/modal/asr_server_modal.py

    # Test locally
    python src/nemotron_speech/modal/asr_server_modal.py

Environment:
    Model weights expected in Modal volume at /model/Parakeet_Reatime_En_600M.nemo
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional, Any

import modal
import numpy as np

# Modal app definition
app = modal.App("nemotron-asr-server")

# Model cache volume
model_cache = modal.Volume.from_name("nemotron-speech", create_if_missing=True)
CACHE_PATH = "/model"
MODEL_PATH = "/model/Parakeet_Reatime_En_600M.nemo"

# Define the container image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .apt_install("git", "libsndfile1", "ffmpeg")
    .uv_pip_install(
        "numpy<2.0.0",
        "torch",
        "aiohttp",
        "loguru",
        "omegaconf",
        "Cython",
        "webdataset",
        "hydra-core",
        "fastapi[standard]",
        "websockets",
    ).uv_pip_install(
        "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git@644201898480ec8c8d0a637f0c773825509ac4dc",
        extra_options="--no-cache",
    )
)

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = False

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]


@dataclass
class ASRSession:
    """Per-connection session state with caches for true incremental streaming."""
    
    id: str
    websocket: Any
    
    # Accumulated audio buffer (all audio received so far)
    accumulated_audio: Optional[np.ndarray] = None
    
    # Number of mel frames already emitted to encoder
    emitted_frames: int = 0
    
    # Encoder cache state
    cache_last_channel: Optional[Any] = None
    cache_last_time: Optional[Any] = None
    cache_last_channel_len: Optional[Any] = None
    
    # Decoder state
    previous_hypotheses: Any = None
    pred_out_stream: Any = None
    
    # Current transcription (model's cumulative output)
    current_text: str = ""
    
    # Last text emitted to client on hard reset (for server-side deduplication)
    # We only send the delta (new portion) to avoid downstream duplication
    last_emitted_text: str = ""
    
    # Audio overlap buffer for mid-utterance reset continuity
    # This preserves the last N ms of audio to provide encoder left-context
    # when a new segment starts after a reset
    overlap_buffer: Optional[np.ndarray] = None

RIGHT_CONTEXT = 1


# Modal class for ASR inference
@app.cls(
    image=image,
    volumes={
        CACHE_PATH: model_cache,
    },
    gpu="L40S",  
    timeout=3600,
    # min_containers=1,  # Keep warm for low latency
)
class NemotronASRModel:
    """Modal class for Nemotron ASR inference."""
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        import torch
        from loguru import logger
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf
        
        logger.info(f"Loading ASR model from {MODEL_PATH}...")
        
        self.model = nemo_asr.models.ASRModel.restore_from(
            MODEL_PATH, map_location='cpu'
        )
        self.model = self.model.cuda()
        
        # Configure attention context for streaming
        logger.info(f"Setting att_context_size=[70, {RIGHT_CONTEXT}] ({RIGHT_CONTEXT_OPTIONS.get(RIGHT_CONTEXT, 'custom')})")
        self.model.encoder.set_default_att_context_size([70, RIGHT_CONTEXT])
        
        # Configure greedy decoding
        logger.info("Configuring greedy decoding...")
        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create({
                'strategy': 'greedy',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': False,
                    'use_cuda_graph_decoder': False,
                }
            })
        )
        self.model.eval()
        
        # Disable dither for deterministic preprocessing
        self.model.preprocessor.featurizer.dither = 0.0
        
        # Get streaming config
        scfg = self.model.encoder.streaming_cfg
        logger.info(f"Streaming config: chunk_size={scfg.chunk_size}, shift_size={scfg.shift_size}")
        
        # Calculate parameters
        preprocessor_cfg = self.model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get('window_stride', 0.01)
        self.sample_rate = 16000
        self.hop_samples = int(hop_length_sec * self.sample_rate)
        
        # shift_size[1] = 16 frames for 160ms chunks
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size
        
        # pre_encode_cache_size[1] = 9 frames
        pre_cache = scfg.pre_encode_cache_size
        self.pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache
        
        # drop_extra_pre_encoded for non-first chunks
        self.drop_extra = scfg.drop_extra_pre_encoded
        
        # Calculate silence padding for final chunk:
        # - right_context chunks for encoder lookahead
        # - 1 additional chunk for decoder finalization
        # With right_context=1, this is (1+1)*160ms = 320ms
        self.final_padding_frames = (RIGHT_CONTEXT + 1) * self.shift_frames
        padding_ms = self.final_padding_frames * hop_length_sec * 1000
        
        # Calculate audio overlap for mid-utterance reset continuity
        # Use pre_encode_cache_size frames = 90ms of left-context
        # This allows the encoder to have proper context when starting a new segment
        self.overlap_samples = self.pre_encode_cache_size * self.hop_samples
        overlap_ms = self.overlap_samples * 1000 / self.sample_rate
        
        shift_ms = self.shift_frames * hop_length_sec * 1000
        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Shift size: {shift_ms:.0f}ms ({self.shift_frames} frames)")
        logger.info(f"Pre-encode cache: {self.pre_encode_cache_size} frames")
        logger.info(f"Final chunk padding: {padding_ms:.0f}ms ({self.final_padding_frames} frames)")
        logger.info(f"Audio overlap for resets: {overlap_ms:.0f}ms ({self.overlap_samples} samples)")
        
        # Warmup inference
        self._warmup()
        
        # Inference lock for thread safety
        self.inference_lock = asyncio.Lock()
        
        # Active sessions
        self.sessions = {}
    
    def _warmup(self):
        """Run warmup inference using streaming API to claim GPU memory."""
        import torch
        from loguru import logger
        
        logger.info("Running warmup inference (streaming API) to claim GPU memory...")
        start = time.perf_counter()
        
        # Generate 1 second of silence plus padding for warmup
        warmup_samples = self.sample_rate + (self.final_padding_frames * self.hop_samples)
        warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
        
        # Run streaming inference to force all CUDA kernels to compile
        with torch.inference_mode():
            audio_tensor = torch.from_numpy(warmup_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(warmup_audio)], device='cuda')
            
            # Preprocess
            mel, mel_len = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)
            
            # Get initial cache
            cache = self.model.encoder.get_initial_cache_state(batch_size=1)
            
            # Run streaming step
            _ = self.model.conformer_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache[0],
                cache_last_time=cache[1],
                cache_last_channel_len=cache[2],
                keep_all_outputs=True,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Warmup complete in {elapsed:.0f}ms - GPU memory claimed")
    
    def _init_session(self, session: ASRSession):
        """Initialize a fresh session.
        
        If an overlap_buffer is present from a previous segment, it will be
        prepended to the accumulated audio to provide encoder left-context.
        This enables seamless transcription across mid-utterance resets.
        """
        from loguru import logger
        
        # Initialize encoder cache
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]
        
        # Reset audio buffer and frame counter
        # If overlap buffer exists, use it as the starting audio
        if session.overlap_buffer is not None and len(session.overlap_buffer) > 0:
            session.accumulated_audio = session.overlap_buffer.copy()
            overlap_ms = len(session.overlap_buffer) * 1000 / self.sample_rate
            logger.debug(
                f"Session {session.id}: prepending {len(session.overlap_buffer)} samples "
                f"({overlap_ms:.0f}ms) of overlap audio"
            )
            session.overlap_buffer = None  # Clear after use
        else:
            session.accumulated_audio = np.array([], dtype=np.float32)
        
        session.emitted_frames = 0
        
        # Reset decoder state
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""
    
    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
        from loguru import logger
        
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}")
        
        session.accumulated_audio = np.concatenate([session.accumulated_audio, audio_np])
        
        # Process if we have enough audio for new frames
        # We need shift_frames worth of new mel frames (after skipping edge frame)
        min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples
        
        while len(session.accumulated_audio) >= min_audio_for_chunk:
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_chunk, session
                )
            
            if text is not None and text != session.current_text:
                session.current_text = text
                logger.debug(f"Session {session.id} interim: {text[-50:] if len(text) > 50 else text}")
                await session.websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "is_final": False
                })
            
            # Update minimum for next iteration
            min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples
    
    def _process_chunk(self, session: ASRSession) -> Optional[str]:
        """Process accumulated audio, extract new mel frames, run streaming inference."""
        import torch
        from loguru import logger
        
        try:
            # Preprocess ALL accumulated audio
            audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(session.accumulated_audio)], device='cuda')
            
            if DEBUG_ASR:
                audio_hash = _hash_audio(session.accumulated_audio)
                logger.debug(f"Session {session.id}: process audio={len(session.accumulated_audio)} hash={audio_hash}")
            
            with torch.inference_mode():
                mel, mel_len = self.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_len
                )
                
                # Available frames (excluding last edge frame)
                available_frames = mel.shape[-1] - 1
                new_frame_count = available_frames - session.emitted_frames
                
                if new_frame_count < self.shift_frames:
                    return session.current_text  # Not enough new frames
                
                # Extract chunk with pre-encode cache
                if session.emitted_frames == 0:
                    # First chunk: just shift_frames, no cache
                    chunk_start = 0
                    chunk_end = self.shift_frames
                    drop_extra = 0
                else:
                    # Subsequent chunks: include pre_encode_cache frames before
                    chunk_start = session.emitted_frames - self.pre_encode_cache_size
                    chunk_end = session.emitted_frames + self.shift_frames
                    drop_extra = self.drop_extra
                
                chunk_mel = mel[:, :, chunk_start:chunk_end]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')
                
                # Run streaming inference
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
                
                # Update emitted frame count
                session.emitted_frames += self.shift_frames
                
                # Extract text
                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, 'text'):
                        return hyp.text
                    elif isinstance(hyp, str):
                        return hyp
                    else:
                        return str(hyp)
                
                return session.current_text
        
        except Exception as e:
            from loguru import logger
            logger.error(f"Session {session.id} chunk processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def _reset_session(self, session: ASRSession, finalize: bool = True):
        """Handle reset with soft or hard finalization.
        
        Args:
            finalize: If True (hard reset), add padding and use keep_all_outputs=True
                      to capture trailing words, then reset decoder state.
                      If False (soft reset), just return current cumulative text
                      without forcing decoder output.
        
        Soft reset (finalize=False):
        - Returns current_text as is_final (model's streaming output)
        - No audio processing, no decoder finalization
        - Decoder state preserved (no corruption)
        - Used on VADUserStoppedSpeakingFrame for fast response
        
        Hard reset (finalize=True):
        - Adds padding and processes with keep_all_outputs=True
        - Captures trailing words at segment boundaries
        - Resets decoder state to prevent corruption from multiple hard resets
        - Preserves encoder cache for acoustic context
        - Used on UserStoppedSpeakingFrame for complete transcription
        """
        from loguru import logger
        import time
        
        logger.info(f"Session {session.id} _reset_session START: finalize={finalize}")
        
        # Log audio state at reset for diagnostics
        audio_samples = len(session.accumulated_audio) if session.accumulated_audio is not None else 0
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        logger.debug(
            f"Session {session.id} {'hard' if finalize else 'soft'} reset: "
            f"accumulated={audio_samples} samples ({audio_duration_ms}ms), "
            f"emitted={session.emitted_frames} frames"
        )
        
        if not finalize:
            # SOFT RESET: Return current text without processing
            # This is fast (~0ms) and doesn't corrupt decoder state.
            # The model's current_text is already cumulative (contains all text
            # from session start), so we just return it directly.
            # We don't concatenate with cumulative_text to avoid duplication.
            text = session.current_text
            
            logger.info(f"Session {session.id} soft reset: sending response")
            await session.websocket.send_json({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False  # Tell client this was soft reset
            })
            logger.info(f"Session {session.id} soft reset: response sent")
            
            logger.debug(f"Session {session.id} soft reset: '{text[-50:] if len(text) > 50 else text}'")
            logger.info(f"Session {session.id} _reset_session END: soft reset complete")
            # Keep all state intact - decoder, encoder, audio buffer
            return
        
        # HARD RESET: Full finalization with padding
        # Save original audio length before adding padding
        original_audio_length = len(session.accumulated_audio) if session.accumulated_audio is not None else 0
        
        # Pad with silence to ensure the model has enough trailing context
        # to finalize the last word. Padding = (right_context + 1) * shift_frames.
        if original_audio_length > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.accumulated_audio = np.concatenate([session.accumulated_audio, silence_padding])
        
        # Process all remaining audio with keep_all_outputs=True
        final_text = session.current_text
        if session.accumulated_audio is not None and len(session.accumulated_audio) > 0:
            start_time = time.perf_counter()
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_final_chunk, session
                )
                if text is not None:
                    final_text = text
                    session.current_text = text  # Update current_text for next soft reset
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Session {session.id} final chunk processed in {elapsed_ms:.1f}ms: '{final_text[-50:] if len(final_text) > 50 else final_text}'")
        
        # Server-side deduplication: only send the delta (new portion)
        # This avoids downstream duplication when aggregators concatenate transcripts
        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text):].lstrip()
        else:
            # ASR corrected earlier text - send full text
            # (This is rare but can happen with model corrections)
            delta_text = final_text
            logger.debug(
                f"Session {session.id}: ASR correction detected, "
                f"last='{session.last_emitted_text[-30:]}', new='{final_text[-30:]}'"
            )
        
        # Update tracking state before sending
        session.last_emitted_text = final_text
        
        # Send only the delta to client
        logger.info(f"Session {session.id} hard reset: sending response with delta='{delta_text}'")
        await session.websocket.send_json({
            "type": "transcript",
            "text": delta_text,
            "is_final": True,
            "finalize": True  # Tell client this was hard reset
        })
        logger.info(f"Session {session.id} hard reset: response sent")
        
        logger.debug(
            f"Session {session.id} hard reset: delta='{delta_text}' "
            f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}')"
        )
        
        # Remove padding (restore to original audio length)
        if original_audio_length > 0:
            session.accumulated_audio = session.accumulated_audio[:original_audio_length]
        else:
            session.accumulated_audio = np.array([], dtype=np.float32)
        
        # CONTINUOUS SESSION: Keep ALL state intact
        # The decoder state is preserved to maintain context for subsequent audio.
        # Server-side deduplication via last_emitted_text ensures clients receive
        # only new text portions, avoiding downstream duplication in aggregators.
        #
        # Note: We considered resetting decoder to prevent corruption from multiple
        # keep_all_outputs=True calls, but this breaks transcription continuity.
        # The soft/hard reset distinction helps by limiting keep_all_outputs=True
        # to only UserStoppedSpeakingFrame events.
        
        logger.debug(
            f"Session {session.id} hard reset complete, state preserved: "
            f"{len(session.accumulated_audio)} samples, {session.emitted_frames} frames"
        )
        
        logger.info(f"Session {session.id} _reset_session END: finalize={finalize}")
    
    def _process_final_chunk(self, session: ASRSession) -> Optional[str]:
        """Process all remaining audio with keep_all_outputs=True."""
        import torch
        from loguru import logger
        
        try:
            if len(session.accumulated_audio) == 0:
                return session.current_text
            
            # Preprocess ALL accumulated audio
            audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(session.accumulated_audio)], device='cuda')
            
            with torch.inference_mode():
                mel, mel_len = self.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_len
                )
                
                # For final chunk, use ALL remaining frames (including edge)
                total_mel_frames = mel.shape[-1]
                remaining_frames = total_mel_frames - session.emitted_frames
                
                logger.debug(
                    f"Session {session.id} final chunk: "
                    f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
                    f"remaining={remaining_frames}"
                )
                
                if remaining_frames <= 0:
                    logger.warning(f"Session {session.id}: No remaining frames to process!")
                    return session.current_text
                
                # Extract final chunk with pre-encode cache
                if session.emitted_frames == 0:
                    chunk_start = 0
                    drop_extra = 0
                else:
                    chunk_start = session.emitted_frames - self.pre_encode_cache_size
                    drop_extra = self.drop_extra
                
                chunk_mel = mel[:, :, chunk_start:]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')
                
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=True,  # Final chunk - output all remaining
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
                
                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, 'text'):
                        final_text = hyp.text
                    elif isinstance(hyp, str):
                        final_text = hyp
                    else:
                        final_text = str(hyp)
                    logger.debug(
                        f"Session {session.id} final chunk output: '{final_text[-50:] if len(final_text) > 50 else final_text}' "
                        f"(was: '{session.current_text[-30:] if len(session.current_text) > 30 else session.current_text}')"
                    )
                    return final_text
                
                logger.debug(f"Session {session.id} final chunk: no new text from model")
                return session.current_text
        
        except Exception as e:
            from loguru import logger
            logger.error(f"Session {session.id} final chunk error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @modal.asgi_app()
    def api(self):
        """FastAPI app with ASR WebSocket endpoint."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from loguru import logger
        import uuid
        
        web_app = FastAPI(
            title="Nemotron ASR Server (Modal)",
            description="Modal-deployed Nemotron streaming ASR server",
            version="1.0.0",
        )
        
        @web_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "sample_rate": self.sample_rate,
            }
        
        @web_app.websocket("/")
        async def websocket_handler(websocket: WebSocket):
            """Handle WebSocket ASR streaming connection."""
            await websocket.accept()
            
            session_id = str(uuid.uuid4())[:8]
            session = ASRSession(id=session_id, websocket=websocket)
            self.sessions[session_id] = session
            
            logger.info(f"Client {session_id} connected")
            
            try:
                # Initialize session
                async with self.inference_lock:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._init_session, session
                    )
                
                await websocket.send_json({"type": "ready"})
                logger.debug(f"Client {session_id}: sent ready")
                
                while True:
                    # Receive message
                    message = await websocket.receive()
                    
                    # FastAPI WebSocket messages have a 'type' field indicating the message type
                    # 'websocket.receive' = binary data, 'websocket.disconnect' = connection closed
                    msg_type = message.get("type", "")
                    
                    if msg_type == "websocket.disconnect":
                        break
                    
                    # Check if there's text data (JSON control messages)
                    if "text" in message and message["text"]:
                        # JSON message
                        try:
                            data = json.loads(message["text"])
                            data_type = data.get("type")
                            logger.info(f"Client {session_id}: received text message type={data_type}, data={data}")
                            
                            if data_type == "reset" or data_type == "end":
                                # finalize=True (default): hard reset with padding + keep_all_outputs
                                # finalize=False: soft reset, just return current text
                                finalize = data.get("finalize", True)
                                logger.info(f"Client {session_id}: calling _reset_session(finalize={finalize})")
                                try:
                                    await self._reset_session(session, finalize=finalize)
                                    logger.info(f"Client {session_id}: _reset_session completed successfully")
                                except Exception as e:
                                    logger.error(f"Client {session_id}: _reset_session failed: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    raise
                            else:
                                logger.warning(f"Client {session_id}: unknown message type: {data_type}")
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Client {session_id}: invalid JSON")
                    
                    # Check if there's binary data (audio)
                    if "bytes" in message and message["bytes"]:
                        audio_bytes = message["bytes"]
                        await self._handle_audio(session, audio_bytes)
            
            except WebSocketDisconnect:
                logger.info(f"Client {session_id} disconnected")
            
            except Exception as e:
                logger.error(f"Client {session_id} error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    pass
            
            finally:
                if session_id in self.sessions:
                    del self.sessions[session_id]
        
        return web_app


# Local development entrypoint
if __name__ == "__main__":
    """Local entrypoint for testing the deployed ASR service."""
    import json
    from pathlib import Path
    
    print("Nemotron ASR Server - Modal Deployment Test")
    print("============================================")
    print()
    
    # Get the deployed API URL
    print("Getting deployed API URL...")
    ASRClass = modal.Cls.from_name("nemotron-asr-server", "NemotronASRModel")
    api_url = ASRClass().api.web_url
    print(f"API URL: {api_url}")
    print()
    
    # Test WebSocket streaming endpoint
    print("Testing WebSocket streaming endpoint...")
    
    async def test_websocket():
        try:
            import websockets
            import wave
        except ImportError:
            print("⚠️  websockets library not installed.")
            print("   Install with: pip install websockets")
            return
        
        # Load test audio file
        test_audio_file = Path("tests/fixtures/harvard_16k.wav")
        if not test_audio_file.exists():
            print(f"⚠️  Test audio file not found: {test_audio_file}")
            print("   Please provide a 16kHz WAV file for testing")
            return
        
        print(f"Loading test audio: {test_audio_file}")
        with wave.open(str(test_audio_file), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
        
        duration_sec = n_frames / sample_rate
        print(f"Audio: {n_frames} frames, {sample_rate}Hz, {n_channels}ch, {duration_sec:.1f}s")
        print()
        
        # Convert to WebSocket URL
        ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as websocket:
                print("📡 Connected to WebSocket")
                
                # Wait for ready message
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"✓  Received: {data}")
                print()
                
                # Send audio in chunks (simulate streaming)
                chunk_size = 3200  # 100ms at 16kHz (16000 samples/sec * 0.1s * 2 bytes)
                chunk_count = 0
                start_time = time.time()
                
                print("📤 Sending audio chunks...")
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await websocket.send(chunk)
                    chunk_count += 1
                    
                    # Receive interim transcripts
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(msg)
                        if data.get("type") == "transcript" and not data.get("is_final"):
                            print(f"   Interim #{chunk_count}: {data['text']}")
                    except asyncio.TimeoutError:
                        pass
                    
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.01)
                
                # Send end signal
                await websocket.send(json.dumps({"type": "end"}))
                print()
                print("📥 Sent end signal, waiting for final transcript...")
                
                # Keep receiving until we get the final transcript
                # (there may be interim transcripts in flight before the final one)
                final_text = None
                timeout_count = 0
                max_timeout = 50  # 5 seconds total (50 * 0.1s)
                
                while timeout_count < max_timeout:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(msg)
                        
                        if data.get("type") == "transcript":
                            if data.get("is_final"):
                                # Got the final transcript
                                final_text = data["text"]
                                print(f"   ✓ Received final transcript (is_final={data.get('is_final')}, finalize={data.get('finalize')})")
                                break
                            else:
                                # Interim transcript still being processed
                                print(f"   (Draining interim: {data['text'][-50:] if len(data['text']) > 50 else data['text']})")
                        else:
                            print(f"⚠️  Unexpected message type: {data.get('type')}")
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        if timeout_count % 10 == 0:
                            print(f"   (Still waiting... {timeout_count * 0.1:.1f}s)")
                        continue
                
                if timeout_count >= max_timeout:
                    print("⚠️  Timeout waiting for final transcript")
                
                elapsed = time.time() - start_time
                
                if final_text is not None:
                    print()
                    print("✅ Final transcript received!")
                    print(f"   Text: '{final_text}'")
                    print(f"   Chunks sent: {chunk_count}")
                    print(f"   Total time: {elapsed:.2f}s")
                    print(f"   RTF: {elapsed/duration_sec:.2f}x")
                else:
                    print("⚠️  No final transcript received")
        
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_websocket())
    
    print()
    print("=" * 60)
    print()
    print("Deployment Info:")
    print("================")
    print(f"API Base URL: {api_url}")
    print()
    print("Available Endpoints:")
    print("  GET /health  - Health check")
    print("  WS  /        - WebSocket streaming ASR")

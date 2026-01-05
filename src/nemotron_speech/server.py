"""WebSocket ASR server for Nemotron-Speech with true incremental streaming."""

import asyncio
import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiohttp import web, WSMsgType
from loguru import logger

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = os.environ.get("DEBUG_ASR", "0") == "1"


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]

# Model path
DEFAULT_MODEL_PATH = "/workspace/models/Parakeet_Reatime_En_600M.nemo"

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}


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
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

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


class ASRServer:
    """WebSocket server for streaming ASR with true incremental processing."""

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        right_context: int = 1,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.right_context = right_context
        self.model = None
        self.sample_rate = 16000

        # Inference lock
        self.inference_lock = asyncio.Lock()

        # Active sessions
        self.sessions: dict[str, ASRSession] = {}

        # Model loaded flag for health check
        self.model_loaded = False

        # Streaming parameters (calculated from model config)
        self.shift_frames = None
        self.pre_encode_cache_size = None
        self.hop_samples = None

        # Audio overlap for mid-utterance reset continuity (calculated in load_model)
        self.overlap_samples = None

    def load_model(self):
        """Load the NeMo ASR model with streaming configuration."""
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf

        logger.info(f"Loading model from {self.model_path}...")

        self.model = nemo_asr.models.ASRModel.restore_from(
            self.model_path, map_location='cpu'
        )
        self.model = self.model.cuda()

        # Configure attention context for streaming
        logger.info(f"Setting att_context_size=[70, {self.right_context}] ({RIGHT_CONTEXT_OPTIONS.get(self.right_context, 'custom')})")
        self.model.encoder.set_default_att_context_size([70, self.right_context])

        # Configure greedy decoding (required for Blackwell GPU)
        logger.info("Configuring greedy decoding for Blackwell compatibility...")
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
        self.final_padding_frames = (self.right_context + 1) * self.shift_frames
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

        # Warmup inference to ensure model is fully loaded on GPU
        # This prevents GPU memory issues when LLM starts later
        self._warmup()

    def _warmup(self):
        """Run warmup inference using streaming API to claim GPU memory.

        IMPORTANT: We use the streaming API (conformer_stream_step) for warmup,
        NOT the batch API (model.transcribe). The batch API corrupts internal
        model state and causes subsequent streaming inference to become
        non-deterministic. See docs/asr-determinism-investigation.md.
        """
        import time

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

            # Run streaming step (processes entire mel as one chunk)
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

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket client connection."""
        import uuid

        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)

        session_id = str(uuid.uuid4())[:8]
        session = ASRSession(id=session_id, websocket=ws)
        self.sessions[session_id] = session

        logger.info(f"Client {session_id} connected")

        try:
            async with self.inference_lock:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._init_session, session
                )

            await ws.send_str(json.dumps({"type": "ready"}))
            logger.debug(f"Client {session_id}: sent ready")

            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await self._handle_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "reset" or msg_type == "end":
                            # finalize=True (default): hard reset with padding + keep_all_outputs
                            # finalize=False: soft reset, just return current text
                            finalize = data.get("finalize", True)
                            await self._reset_session(session, finalize=finalize)
                        else:
                            logger.warning(f"Client {session_id}: unknown message type: {msg_type}")

                    except json.JSONDecodeError:
                        logger.warning(f"Client {session_id}: invalid JSON")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"Client {session_id} WebSocket error: {ws.exception()}")
                    break

            logger.info(f"Client {session_id} disconnected")

        except Exception as e:
            logger.error(f"Client {session_id} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await ws.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except:
                pass
        finally:
            if session_id in self.sessions:
                del self.sessions[session_id]

        return ws

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
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
                await session.websocket.send_str(json.dumps({
                    "type": "transcript",
                    "text": text,
                    "is_final": False
                }))

            # Update minimum for next iteration
            min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples

    def _process_chunk(self, session: ASRSession) -> Optional[str]:
        """Process accumulated audio, extract new mel frames, run streaming inference."""
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

                if DEBUG_ASR:
                    mel_hash = hashlib.md5(mel.cpu().numpy().tobytes()).hexdigest()[:8]
                    logger.debug(f"Session {session.id}: mel shape={mel.shape[-1]} hash={mel_hash}")

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
        import time

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

            await session.websocket.send_str(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False  # Tell client this was soft reset
            }))

            logger.debug(f"Session {session.id} soft reset: '{text[-50:] if len(text) > 50 else text}'")
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
        await session.websocket.send_str(json.dumps({
            "type": "transcript",
            "text": delta_text,
            "is_final": True,
            "finalize": True  # Tell client this was hard reset
        }))

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

    def _process_final_chunk(self, session: ASRSession) -> Optional[str]:
        """Process all remaining audio with keep_all_outputs=True."""
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
            logger.error(f"Session {session.id} final chunk error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy" if self.model_loaded else "loading",
            "model_loaded": self.model_loaded,
        })

    async def start(self):
        """Start the HTTP + WebSocket server."""
        self.load_model()
        self.model_loaded = True

        logger.info(f"Starting streaming ASR server on ws://{self.host}:{self.port}")

        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/", self.websocket_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"ASR server listening on ws://{self.host}:{self.port}")
        logger.info(f"Health check available at http://{self.host}:{self.port}/health")
        await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="Nemotron Streaming ASR WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to NeMo model")
    parser.add_argument(
        "--right-context",
        type=int,
        default=1,
        choices=[0, 1, 6, 13],
        help="Right context frames: 0=80ms, 1=160ms, 6=560ms, 13=1.12s latency"
    )
    args = parser.parse_args()

    server = ASRServer(
        model_path=args.model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()

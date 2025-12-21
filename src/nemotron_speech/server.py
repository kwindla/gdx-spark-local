"""WebSocket ASR server for Nemotron-Speech with true incremental streaming."""

import asyncio
import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import websockets
from loguru import logger

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

    # Current transcription
    current_text: str = ""


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

        # Streaming parameters (calculated from model config)
        self.shift_frames = None
        self.pre_encode_cache_size = None
        self.hop_samples = None

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

        shift_ms = self.shift_frames * hop_length_sec * 1000
        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Shift size: {shift_ms:.0f}ms ({self.shift_frames} frames)")
        logger.info(f"Pre-encode cache: {self.pre_encode_cache_size} frames")

    def _init_session(self, session: ASRSession):
        """Initialize a fresh session."""
        # Initialize encoder cache
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]

        # Reset audio buffer and frame counter
        session.accumulated_audio = np.array([], dtype=np.float32)
        session.emitted_frames = 0

        # Reset decoder state
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""

    async def handle_client(self, websocket):
        """Handle a WebSocket client connection."""
        import uuid

        session_id = str(uuid.uuid4())[:8]
        session = ASRSession(id=session_id, websocket=websocket)
        self.sessions[session_id] = session

        logger.info(f"Client {session_id} connected")

        try:
            async with self.inference_lock:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._init_session, session
                )

            await websocket.send(json.dumps({"type": "ready"}))
            logger.debug(f"Client {session_id}: sent ready")

            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_audio(session, message)
                else:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "reset" or msg_type == "end":
                            await self._reset_session(session)
                        else:
                            logger.warning(f"Client {session_id}: unknown message type: {msg_type}")

                    except json.JSONDecodeError:
                        logger.warning(f"Client {session_id}: invalid JSON")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {session_id} disconnected")
        except Exception as e:
            logger.error(f"Client {session_id} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except:
                pass
        finally:
            del self.sessions[session_id]

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
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
                await session.websocket.send(json.dumps({
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
            logger.error(f"Session {session.id} chunk processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def _reset_session(self, session: ASRSession):
        """Finalize and reset session."""
        # Process all remaining audio with keep_all_outputs=True
        if len(session.accumulated_audio) > 0:
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_final_chunk, session
                )
                if text is not None:
                    session.current_text = text

        # Send final transcript
        await session.websocket.send(json.dumps({
            "type": "transcript",
            "text": session.current_text,
            "is_final": True
        }))

        logger.debug(f"Session {session.id} reset, emitted {session.emitted_frames} frames, final: {session.current_text[:50]}...")

        # Reinitialize
        async with self.inference_lock:
            await asyncio.get_event_loop().run_in_executor(
                None, self._init_session, session
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
                remaining_frames = mel.shape[-1] - session.emitted_frames

                if remaining_frames <= 0:
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
                        return hyp.text
                    elif isinstance(hyp, str):
                        return hyp
                    else:
                        return str(hyp)

                return session.current_text

        except Exception as e:
            logger.error(f"Session {session.id} final chunk error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def start(self):
        """Start the WebSocket server."""
        self.load_model()

        logger.info(f"Starting streaming ASR server on ws://{self.host}:{self.port}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,
        ):
            logger.info(f"ASR server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()


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

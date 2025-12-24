"""
Streaming TTS inference for Magpie TTS model.

This module provides streaming inference capabilities for the Magpie TTS model,
yielding audio chunks as soon as they're available to minimize TTFB.

Two streaming approaches are available:

1. Sentence-level streaming (RECOMMENDED):
   - Splits text into sentences
   - Generates each sentence using fast batch mode (~0.27x RTF)
   - Streams sentences as they complete
   - TTFB: ~200ms (first sentence), smooth playback

2. Frame-by-frame streaming (DEPRECATED - too slow):
   - Generates tokens one at a time
   - Very slow RTF (~2.8x), causes audio gaps
   - Only useful for very short utterances

Usage:
    # Inside container with NeMo/PyTorch available
    from streaming_tts import SentenceStreamingTTS

    streamer = SentenceStreamingTTS(model)
    for audio_chunk in streamer.synthesize_streaming(
        text="Hello! How can I help you today?",
        language="en",
        speaker_index=2,  # aria
    ):
        # audio_chunk is bytes (int16 PCM, 22kHz, mono)
        websocket.send(audio_chunk)

Performance:
    - TTFB: ~150-250ms (first sentence generation)
    - RTF: ~0.27x (batch mode, faster than real-time)
    - No audio gaps between chunks
"""

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import torch


@dataclass
class StreamingConfig:
    """Configuration for streaming TTS inference."""

    # Number of frames to accumulate before decoding
    # 1 frame = 1024 samples = ~46.4ms at 22kHz
    chunk_size_frames: int = 10  # ~465ms chunks

    # Overlap frames for smooth boundaries with non-causal decoder
    overlap_frames: int = 2  # ~93ms overlap

    # Minimum frames before first audio output (aggressive for low TTFB)
    min_first_chunk_frames: int = 4  # ~185ms TTFB target

    # Enable crossfade at chunk boundaries
    use_crossfade: bool = True
    crossfade_samples: int = 512  # ~23ms crossfade

    # Maximum decoder steps (same as batch mode)
    max_decoder_steps: int = 500

    # Generation parameters
    temperature: float = 0.7
    topk: int = 80
    use_cfg: bool = True
    cfg_scale: float = 2.5


# Preset configurations for different use cases
STREAMING_PRESETS = {
    # Aggressive: lowest TTFB, may have subtle artifacts
    "aggressive": StreamingConfig(
        min_first_chunk_frames=4,   # ~185ms TTFB
        chunk_size_frames=8,        # ~370ms chunks
        overlap_frames=2,
    ),
    # Balanced: good TTFB with reliable quality
    "balanced": StreamingConfig(
        min_first_chunk_frames=6,   # ~280ms TTFB
        chunk_size_frames=10,       # ~465ms chunks
        overlap_frames=2,
    ),
    # Conservative: prioritize quality over TTFB
    "conservative": StreamingConfig(
        min_first_chunk_frames=8,   # ~370ms TTFB
        chunk_size_frames=12,       # ~560ms chunks
        overlap_frames=3,
    ),
}


class StreamingMagpieTTS:
    """Streaming inference wrapper for MagpieTTSModel."""

    SAMPLE_RATE = 22000
    SAMPLES_PER_FRAME = 1024

    def __init__(self, model, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming TTS with a loaded MagpieTTSModel.

        Args:
            model: Loaded MagpieTTSModel instance (on GPU)
            config: Streaming configuration
        """
        self.model = model
        self.config = config or StreamingConfig()

        # Verify model is ready
        if not hasattr(model, "do_tts"):
            raise ValueError("Model must be a MagpieTTSModel with do_tts method")

    def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        speaker_index: int = 2,  # aria
        apply_tn: bool = False,
    ) -> Generator[bytes, None, None]:
        """
        Synthesize speech with streaming output.

        Yields audio chunks as soon as they're available, minimizing TTFB.

        Args:
            text: Text to synthesize
            language: Language code (en, es, de, fr, vi, it, zh)
            speaker_index: Speaker voice index (0-4)
            apply_tn: Apply text normalization

        Yields:
            bytes: PCM audio data (int16, mono, 22kHz)
        """
        # Import NeMo utilities (only available in container)
        from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

        cfg = self.config
        model = self.model

        with torch.no_grad():
            # Prepare batch (same as do_tts)
            batch = self._prepare_batch(text, language, speaker_index, apply_tn)

            # Initialize decoder state
            model.decoder.reset_cache(use_cache=model.use_kv_cache_for_inference)
            context_tensors = model.prepare_context_tensors(batch)

            # Initialize token generation
            text = context_tensors.text
            audio_codes_bos = torch.full(
                (text.size(0), model.num_audio_codebooks, model.frame_stacking_factor),
                model.audio_bos_id,
                device=text.device,
            ).long()

            audio_codes_lens = torch.full((text.size(0),), 1, device=text.device).long()
            audio_codes_input = audio_codes_bos
            audio_codes_mask = get_mask_from_lengths(audio_codes_lens)

            # Prepare CFG if enabled
            if cfg.use_cfg:
                dummy_cond, dummy_cond_mask, dummy_add_dec_input, dummy_add_dec_mask, _ = (
                    model.prepare_dummy_cond_for_cfg(
                        context_tensors.cond,
                        context_tensors.cond_mask,
                        context_tensors.additional_decoder_input,
                        context_tensors.additional_decoder_mask,
                    )
                )

            # Streaming state
            all_predictions = []
            pending_tokens = []
            overlap_buffer = None
            end_indices = {}
            first_chunk_yielded = False

            # Main generation loop
            for idx in range(cfg.max_decoder_steps // model.frame_stacking_factor):
                # Generate next token(s)
                audio_codes_embedded = model.embed_audio_tokens(audio_codes_input)

                if context_tensors.additional_decoder_input is not None:
                    _audio_codes_embedded = torch.cat(
                        [context_tensors.additional_decoder_input, audio_codes_embedded], dim=1
                    )
                    _audio_codes_mask = torch.cat(
                        [context_tensors.additional_decoder_mask, audio_codes_mask], dim=1
                    )
                else:
                    _audio_codes_embedded = audio_codes_embedded
                    _audio_codes_mask = audio_codes_mask

                # Forward pass with optional CFG
                if cfg.use_cfg:
                    batch_size = audio_codes_embedded.size(0)
                    cfg_cond = torch.cat([context_tensors.cond, dummy_cond], dim=0)
                    cfg_cond_mask = torch.cat([context_tensors.cond_mask, dummy_cond_mask], dim=0)
                    cfg_audio_codes_embedded = torch.cat(
                        [_audio_codes_embedded, _audio_codes_embedded], dim=0
                    )
                    cfg_audio_codes_mask = torch.cat([_audio_codes_mask, _audio_codes_mask], dim=0)

                    combined_logits, _, dec_out = model.forward(
                        dec_input_embedded=cfg_audio_codes_embedded,
                        dec_input_mask=cfg_audio_codes_mask,
                        cond=cfg_cond,
                        cond_mask=cfg_cond_mask,
                        attn_prior=None,
                        multi_encoder_mapping=context_tensors.multi_encoder_mapping,
                    )

                    cond_logits = combined_logits[:batch_size]
                    uncond_logits = combined_logits[batch_size:]
                    all_code_logits = (1 - cfg.cfg_scale) * uncond_logits + cfg.cfg_scale * cond_logits
                else:
                    all_code_logits, _, dec_out = model.forward(
                        dec_input_embedded=_audio_codes_embedded,
                        dec_input_mask=_audio_codes_mask,
                        cond=context_tensors.cond,
                        cond_mask=context_tensors.cond_mask,
                        attn_prior=None,
                        multi_encoder_mapping=context_tensors.multi_encoder_mapping,
                    )
                    batch_size = audio_codes_embedded.size(0)

                # Sample next tokens
                forbid_eos = idx * model.frame_stacking_factor < 4
                all_code_logits_t = all_code_logits[:, -1, :]
                audio_codes_next = model.sample_codes_from_logits(
                    all_code_logits_t,
                    temperature=cfg.temperature,
                    topk=cfg.topk,
                    unfinished_items={},
                    finished_items={},
                    forbid_audio_eos=forbid_eos,
                )

                # Check for EOS
                all_codes_next_argmax = model.sample_codes_from_logits(
                    all_code_logits_t, temperature=0.01, topk=1,
                    unfinished_items={}, finished_items={}, forbid_audio_eos=forbid_eos,
                )

                # Import EOS detection enum
                from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod
                eos_method = EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices:
                        end_frame_index = model.detect_eos(
                            audio_codes_next[item_idx],
                            all_codes_next_argmax[item_idx],
                            eos_method,
                        )
                        if end_frame_index != float('inf'):
                            end_indices[item_idx] = idx * model.frame_stacking_factor + end_frame_index

                # Accumulate predictions
                all_predictions.append(audio_codes_next)
                pending_tokens.append(audio_codes_next)
                audio_codes_input = torch.cat([audio_codes_input, audio_codes_next], dim=-1)
                audio_codes_lens = audio_codes_lens + 1
                audio_codes_mask = get_mask_from_lengths(audio_codes_lens)

                # Check if we should decode and yield a chunk
                total_pending_frames = len(pending_tokens) * model.frame_stacking_factor

                should_yield = False
                if not first_chunk_yielded:
                    # First chunk: yield once we have enough for min TTFB
                    if total_pending_frames >= cfg.min_first_chunk_frames:
                        should_yield = True
                        first_chunk_yielded = True
                else:
                    # Subsequent chunks: yield at chunk_size intervals
                    if total_pending_frames >= cfg.chunk_size_frames:
                        should_yield = True

                # Also yield on EOS
                if len(end_indices) > 0:
                    should_yield = True

                if should_yield and pending_tokens:
                    # Decode accumulated tokens
                    pending_codes = torch.cat(pending_tokens, dim=-1)

                    # Include overlap buffer for context
                    if overlap_buffer is not None:
                        decode_codes = torch.cat([overlap_buffer, pending_codes], dim=-1)
                        overlap_samples = cfg.overlap_frames * self.SAMPLES_PER_FRAME
                    else:
                        decode_codes = pending_codes
                        overlap_samples = 0

                    # Decode to audio
                    decode_lens = torch.tensor(
                        [decode_codes.size(-1)], device=decode_codes.device
                    ).long()
                    audio, audio_len = model.codes_to_audio(decode_codes, decode_lens)

                    # Extract new audio (skip overlap region)
                    if overlap_samples > 0:
                        new_audio = audio[..., overlap_samples:]
                        if cfg.use_crossfade and hasattr(self, '_prev_tail'):
                            new_audio = self._apply_crossfade(
                                self._prev_tail, new_audio, cfg.crossfade_samples
                            )
                    else:
                        new_audio = audio

                    # Save tail for next crossfade
                    if cfg.use_crossfade:
                        self._prev_tail = new_audio[..., -cfg.crossfade_samples:].clone()

                    # Update overlap buffer for next chunk
                    if pending_codes.size(-1) >= cfg.overlap_frames:
                        overlap_buffer = pending_codes[..., -cfg.overlap_frames:]
                    else:
                        overlap_buffer = pending_codes.clone()

                    # Convert to bytes and yield
                    audio_bytes = self._audio_to_bytes(new_audio)
                    yield audio_bytes

                    # Clear pending
                    pending_tokens = []

                # Check termination
                if len(end_indices) == text.size(0) and len(all_predictions) >= 4:
                    break

            # Yield any remaining audio
            if pending_tokens:
                pending_codes = torch.cat(pending_tokens, dim=-1)
                if overlap_buffer is not None:
                    decode_codes = torch.cat([overlap_buffer, pending_codes], dim=-1)
                    overlap_samples = cfg.overlap_frames * self.SAMPLES_PER_FRAME
                else:
                    decode_codes = pending_codes
                    overlap_samples = 0

                decode_lens = torch.tensor(
                    [decode_codes.size(-1)], device=decode_codes.device
                ).long()
                audio, _ = model.codes_to_audio(decode_codes, decode_lens)

                if overlap_samples > 0:
                    new_audio = audio[..., overlap_samples:]
                else:
                    new_audio = audio

                audio_bytes = self._audio_to_bytes(new_audio)
                yield audio_bytes

            # Cleanup
            if hasattr(self, '_prev_tail'):
                del self._prev_tail

    def _prepare_batch(
        self, text: str, language: str, speaker_index: int, apply_tn: bool
    ) -> dict:
        """Prepare input batch (mirrors do_tts logic)."""
        model = self.model

        # Apply text normalization if requested
        if apply_tn:
            normalized_text = model._get_normalized_text(transcript=text, language=language)
        else:
            normalized_text = text

        # Determine tokenizer
        tokenizer_name = None
        available_tokenizers = list(model.tokenizer.tokenizers.keys())
        language_tokenizer_map = {
            "en": ["english_phoneme", "english"],
            "de": ["german_phoneme", "german"],
            "es": ["spanish_phoneme", "spanish"],
            "fr": ["french_chartokenizer", "french"],
            "it": ["italian_phoneme", "italian"],
            "vi": ["vietnamese_phoneme", "vietnamese"],
            "zh": ["mandarin_phoneme", "mandarin", "chinese"],
        }

        if language in language_tokenizer_map:
            for candidate in language_tokenizer_map[language]:
                if candidate in available_tokenizers:
                    tokenizer_name = candidate
                    break

        if tokenizer_name is None:
            tokenizer_name = available_tokenizers[0]

        # Tokenize
        tokens = model.tokenizer.encode(text=normalized_text, tokenizer_name=tokenizer_name)
        tokens = tokens + [model.eos_id]
        text_tensor = torch.tensor([tokens], device=model.device, dtype=torch.long)
        text_lens = torch.tensor([len(tokens)], device=model.device, dtype=torch.long)

        return {
            'text': text_tensor,
            'text_lens': text_lens,
            'speaker_indices': speaker_index,
        }

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to PCM bytes."""
        audio_np = audio.cpu().float().numpy()

        # Handle tensor shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        # Handle empty audio
        if audio_np.size == 0:
            return b""

        # Normalize and convert to int16
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _apply_crossfade(
        self, prev_tail: torch.Tensor, new_audio: torch.Tensor, crossfade_samples: int
    ) -> torch.Tensor:
        """Apply crossfade between previous chunk tail and new audio."""
        if prev_tail.size(-1) < crossfade_samples or new_audio.size(-1) < crossfade_samples:
            return new_audio

        # Create fade curves
        fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=new_audio.device)
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=new_audio.device)

        # Apply crossfade
        new_audio_start = new_audio[..., :crossfade_samples]
        prev_end = prev_tail[..., -crossfade_samples:]

        crossfaded = prev_end * fade_out + new_audio_start * fade_in

        # Replace start of new audio with crossfaded portion
        result = new_audio.clone()
        result[..., :crossfade_samples] = crossfaded

        return result


class SentenceStreamingTTS:
    """Sentence-level streaming TTS using fast batch mode.

    This approach splits text into sentences and generates each using the
    efficient batch mode (do_tts), streaming sentences as they complete.

    This achieves:
    - Low TTFB (~150-250ms for first sentence)
    - No audio gaps (batch mode is faster than real-time)
    - High quality (no frame-by-frame artifacts)
    """

    SAMPLE_RATE = 22000

    def __init__(self, model):
        """Initialize with a loaded MagpieTTSModel."""
        self.model = model

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming.

        Handles common sentence endings while preserving punctuation.
        """
        import re

        # Split on sentence-ending punctuation followed by space or end
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Filter empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentence breaks found, return the whole text
        if not sentences:
            return [text.strip()] if text.strip() else []

        return sentences

    def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        speaker_index: int = 2,
        apply_tn: bool = False,
    ) -> Generator[bytes, None, None]:
        """Synthesize speech with sentence-level streaming.

        Splits text into sentences and generates each using fast batch mode.
        Yields audio as soon as each sentence is ready.

        Args:
            text: Text to synthesize
            language: Language code (en, es, de, fr, vi, it, zh)
            speaker_index: Speaker voice index (0-4)
            apply_tn: Apply text normalization

        Yields:
            bytes: PCM audio data (int16, mono, 22kHz)
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return

        with torch.no_grad():
            for sentence in sentences:
                # Generate using fast batch mode
                audio, audio_len = self.model.do_tts(
                    sentence,
                    language=language,
                    speaker_index=speaker_index,
                    apply_TN=apply_tn,
                )

                # Convert to bytes
                audio_bytes = self._audio_to_bytes(audio)
                yield audio_bytes

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to PCM bytes."""
        audio_np = audio.cpu().float().numpy()

        # Handle tensor shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        # Handle empty audio
        if audio_np.size == 0:
            return b""

        # Normalize and convert to int16
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()


# Simple test
if __name__ == "__main__":
    print("Streaming TTS module loaded.")
    print("Run inside the nemotron-asr container with NeMo available.")
    print()
    print("RECOMMENDED: Use SentenceStreamingTTS for low-latency streaming:")
    print("  from nemo.collections.tts.models import MagpieTTSModel")
    print("  model = MagpieTTSModel.from_pretrained('nvidia/magpie_tts_multilingual_357m')")
    print("  model = model.cuda().eval()")
    print()
    print("  from streaming_tts import SentenceStreamingTTS")
    print("  streamer = SentenceStreamingTTS(model)")
    print("  for chunk in streamer.synthesize_streaming('Hello! How are you?'):")
    print("      print(f'Received {len(chunk)} bytes')")

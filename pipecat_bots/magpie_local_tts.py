"""Local NVIDIA Magpie TTS service for Pipecat.

Uses the open-source Magpie TTS model via NeMo for local GPU inference.
No network calls - runs entirely on DGX Spark GPU.

Requirements:
    - NeMo main branch: uv pip install 'nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git@main'
    - kaldialign: uv pip install kaldialign
    - HuggingFace token with access to nvidia/magpie_tts_multilingual_357m

Usage:
    tts = MagpieLocalTTSService(speaker="Aria", language="en")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

import asyncio
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

# Magpie TTS outputs at 22kHz (from nemo-nano-codec-22khz)
MAGPIE_SAMPLE_RATE = 22000

# Available speakers in the model
SPEAKERS = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4,
}

# Supported languages
LANGUAGES = ["en", "es", "de", "fr", "vi", "it", "zh"]


class MagpieLocalTTSService(TTSService):
    """Local NVIDIA Magpie TTS service using NeMo.

    Runs Magpie TTS model locally on GPU for low-latency speech synthesis.
    No network calls - all inference happens on-device.
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie TTS."""

        speaker: str = "Aria"
        language: str = "en"
        apply_text_normalization: bool = False

    def __init__(
        self,
        *,
        model_id: str = "nvidia/magpie_tts_multilingual_357m",
        speaker: str = "Aria",
        language: str = "en",
        apply_text_normalization: bool = False,
        sample_rate: int = MAGPIE_SAMPLE_RATE,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Magpie TTS service.

        Args:
            model_id: HuggingFace model ID for Magpie TTS.
            speaker: Speaker voice (John, Sofia, Aria, Jason, Leo).
            language: Language code (en, es, de, fr, vi, it, zh).
            apply_text_normalization: Whether to apply text normalization.
            sample_rate: Output sample rate (default 22000Hz, native rate).
            params: Additional TTS parameters.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or MagpieLocalTTSService.InputParams()

        self._model_id = model_id
        self._speaker = speaker
        self._speaker_idx = SPEAKERS.get(speaker, SPEAKERS["Aria"])
        self._language = language
        self._apply_tn = apply_text_normalization
        self._output_sample_rate = sample_rate

        # Model will be loaded on start
        self._model = None
        self._model_loaded = False
        self._load_lock = asyncio.Lock()

        self.set_model_name("magpie-local")
        self.set_voice(speaker)

        logger.info(
            f"MagpieLocalTTS initialized: speaker={speaker}, "
            f"language={language}, sample_rate={sample_rate}Hz"
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Load the model on pipeline start."""
        await super().start(frame)
        await self._ensure_model_loaded()

    async def _ensure_model_loaded(self):
        """Load the model if not already loaded."""
        if self._model_loaded:
            return

        async with self._load_lock:
            if self._model_loaded:
                return

            logger.info(f"Loading Magpie TTS model: {self._model_id}")

            def load_model():
                import os
                from nemo.collections.tts.models import MagpieTTSModel

                # Ensure HuggingFace token is set for model download
                hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token

                model = MagpieTTSModel.from_pretrained(self._model_id)
                model = model.cuda()
                model.eval()
                return model

            try:
                self._model = await asyncio.to_thread(load_model)
                self._model_loaded = True
                logger.info("Magpie TTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Magpie TTS model: {e}")
                raise

    def _generate_sync(self, text: str) -> np.ndarray:
        """Synchronous TTS generation (runs in thread pool).

        Args:
            text: Text to synthesize.

        Returns:
            Audio as numpy array (float32, mono).
        """
        with torch.no_grad():
            audio, audio_len = self._model.do_tts(
                text,
                language=self._language,
                speaker_index=self._speaker_idx,
                apply_TN=self._apply_tn,
            )

        # Convert to numpy
        audio_np = audio.cpu().float().numpy()

        # Handle tensor shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)  # Remove batch dimension
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        return audio_np

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using local Magpie TTS.

        Args:
            text: The text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame
        """
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Ensure model is loaded
        await self._ensure_model_loaded()

        if not self._model:
            logger.error("Magpie TTS model not loaded")
            yield ErrorFrame(error="Model not loaded")
            yield TTSStoppedFrame()
            return

        # Normalize unicode characters that may cause issues
        text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
        text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
        text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
        text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
        text = text.replace("\u2014", "-")  # EM DASH
        text = text.replace("\u2013", "-")  # EN DASH

        logger.debug(f"MagpieLocalTTS: Generating [{text}]")

        try:
            # Run GPU inference in thread pool (non-blocking)
            audio_np = await asyncio.to_thread(self._generate_sync, text)

            await self.stop_ttfb_metrics()

            # Convert to 16-bit PCM bytes
            # Normalize to [-1, 1] range if needed
            if np.abs(audio_np).max() > 1.0:
                audio_np = audio_np / np.abs(audio_np).max()

            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            duration_ms = len(audio_np) / self._output_sample_rate * 1000
            logger.info(
                f"MagpieLocalTTS: Generated {len(audio_bytes)} bytes, "
                f"duration={duration_ms:.0f}ms at {self._output_sample_rate}Hz"
            )

            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self._output_sample_rate,
                num_channels=1,
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"MagpieLocalTTS error: {e}")
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()

    async def cancel(self, frame: CancelFrame):
        """Handle cancellation."""
        await super().cancel(frame)
        # Note: GPU inference cannot be interrupted mid-generation

    async def stop(self, frame: EndFrame):
        """Clean up on stop."""
        await super().stop(frame)
        # Model stays loaded for potential reuse

    def set_speaker(self, speaker: str):
        """Change the speaker voice.

        Args:
            speaker: Speaker name (John, Sofia, Aria, Jason, Leo).
        """
        if speaker not in SPEAKERS:
            logger.warning(f"Unknown speaker '{speaker}', using Aria")
            speaker = "Aria"

        self._speaker = speaker
        self._speaker_idx = SPEAKERS[speaker]
        self.set_voice(speaker)
        logger.info(f"MagpieLocalTTS: Speaker changed to {speaker}")

    def set_language(self, language: str):
        """Change the language.

        Args:
            language: Language code (en, es, de, fr, vi, it, zh).
        """
        if language not in LANGUAGES:
            logger.warning(f"Unknown language '{language}', using en")
            language = "en"

        self._language = language
        logger.info(f"MagpieLocalTTS: Language changed to {language}")

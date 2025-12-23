#!/usr/bin/env python3
#
# Pipecat bot using local NVIDIA ASR/LLM/TTS.
#
# Based on pipecat/examples/foundational/07-interruptible.py (v0.0.98)
#
# Usage:
#   uv run pipecat_bots/bot.py
#   uv run pipecat_bots/bot.py -t daily
#   uv run pipecat_bots/bot.py -t webrtc
#
# Environment variables (loaded from .env):
#   NVIDIA_ASR_URL - ASR WebSocket URL (default: ws://localhost:8080)
#   NVIDIA_LLM_URL - LLM API URL (default: http://localhost:8000/v1)
#   NVIDIA_LLM_MODEL - LLM model name (default: /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
#   USE_LOCAL_TTS - Use local Magpie TTS instead of Cartesia (default: true)
#   CARTESIA_API_KEY - Cartesia TTS API key (required if USE_LOCAL_TTS=false)
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService


class AudioFrameLogger(FrameProcessor):
    """Log TTSAudioRawFrame details for debugging sample rate issues."""

    def __init__(self, name: str = "AudioLogger"):
        super().__init__()
        self._name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            duration_ms = len(frame.audio) / (frame.sample_rate * 2) * 1000
            logger.info(
                f"[{self._name}] TTSAudioRawFrame: "
                f"{len(frame.audio)} bytes, "
                f"sample_rate={frame.sample_rate}Hz, "
                f"channels={frame.num_channels}, "
                f"duration={duration_ms:.0f}ms"
            )

        await self.push_frame(frame, direction)

# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from magpie_http_tts import MagpieHTTPTTSService  # HTTP client for Magpie TTS server
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# Configuration from environment
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_LLM_URL = os.getenv("NVIDIA_LLM_URL", "http://localhost:8000/v1")
NVIDIA_LLM_MODEL = os.getenv(
    "NVIDIA_LLM_MODEL",
    "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
)
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# TTS configuration
# Set USE_LOCAL_TTS=false to use Cartesia cloud TTS instead
USE_LOCAL_TTS = os.getenv("USE_LOCAL_TTS", "true").lower() in ("true", "1", "yes")

# Transport configurations with VAD and turn analyzer
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    tts_type = f"Magpie ({NVIDIA_TTS_URL})" if USE_LOCAL_TTS else "Cartesia (cloud)"
    logger.info(f"Starting bot (local ASR/LLM, {tts_type} TTS)")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL}")
    logger.info(f"  LLM URL: {NVIDIA_LLM_URL}")
    logger.info(f"  LLM Model: {NVIDIA_LLM_MODEL}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL if USE_LOCAL_TTS else 'Cartesia cloud'}")
    logger.info(f"  Transport type: {type(transport).__name__}")

    # NVIDIA Parakeet ASR via WebSocket
    stt = NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=16000,
    )

    # TTS service selection
    if USE_LOCAL_TTS:
        # Magpie TTS via HTTP server (runs in container, client runs on host)
        tts = MagpieHTTPTTSService(
            server_url=NVIDIA_TTS_URL,
            voice="aria",
            language="en",
        )
    else:
        # Cartesia TTS (cloud) - fallback option
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

    # NVIDIA Nemotron-3-Nano LLM via vLLM (OpenAI-compatible API)
    llm = OpenAILLMService(
        api_key=os.getenv("NVIDIA_LLM_API_KEY", "not-needed"),
        base_url=NVIDIA_LLM_URL,
        model=NVIDIA_LLM_MODEL,
        params=OpenAILLMService.InputParams(
            extra={
                # extra_body passes vLLM-specific params in the request body
                "extra_body": {
                    # Disable reasoning mode for faster responses
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant running on an NVIDIA DGX Spark. "
                "Your goal is to have a natural conversation with the user. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Avoid special characters that can't easily be spoken, such as emojis or bullet points."
            ),
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Audio frame logger to debug sample rate issues
    audio_logger = AudioFrameLogger("TTS->Transport")

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,  # RTVI processor (early in chain for client messages)
            stt,
            context_aggregator.user(),
            llm,
            tts,
            audio_logger,  # Log audio frames between TTS and transport
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("RTVI client ready")
        await rtvi.set_bot_ready()
        # Kick off the conversation
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

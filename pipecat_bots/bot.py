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
#   NVIDIA_LLM_URL - LLM API URL for OpenAI-compat (default: http://localhost:8000/v1)
#   NVIDIA_LLAMA_CPP_URL - llama.cpp native API URL (default: http://localhost:8000)
#   NVIDIA_LLM_MODEL - LLM model name (default: /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
#   NVIDIA_TTS_URL - Magpie TTS server URL (default: http://localhost:8001)
#   USE_LOCAL_TTS - Use local Magpie TTS instead of Cartesia (default: true)
#   USE_WEBSOCKET_TTS - Use WebSocket adaptive TTS (default: false)
#   USE_CHUNKED_LLM - Use chunked LLM for sentence-boundary streaming (default: false)
#   CARTESIA_API_KEY - Cartesia TTS API key (required if USE_LOCAL_TTS=false)
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame
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


# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from magpie_http_tts import MagpieHTTPTTSService  # HTTP client for Magpie TTS server (batch)
from magpie_websocket_tts import MagpieWebSocketTTSService  # WebSocket adaptive TTS
from llama_cpp_chunked_llm import LlamaCppChunkedLLMService  # Direct llama.cpp chunked LLM
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
# For LlamaCppChunkedLLMService - direct llama.cpp HTTP API (not /v1)
NVIDIA_LLAMA_CPP_URL = os.getenv("NVIDIA_LLAMA_CPP_URL", "http://localhost:8000")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# TTS configuration
# Set USE_LOCAL_TTS=false to use Cartesia cloud TTS instead
USE_LOCAL_TTS = os.getenv("USE_LOCAL_TTS", "true").lower() in ("true", "1", "yes")
# Set USE_WEBSOCKET_TTS=true for WebSocket-based adaptive streaming
USE_WEBSOCKET_TTS = os.getenv("USE_WEBSOCKET_TTS", "false").lower() in ("true", "1", "yes")
# Set USE_CHUNKED_LLM=true for sentence-boundary chunking (best with USE_WEBSOCKET_TTS)
USE_CHUNKED_LLM = os.getenv("USE_CHUNKED_LLM", "false").lower() in ("true", "1", "yes")

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
    # Determine TTS mode
    if USE_LOCAL_TTS:
        if USE_WEBSOCKET_TTS:
            tts_mode = "websocket"
        else:
            tts_mode = "batch"
        tts_type = f"Magpie {tts_mode} ({NVIDIA_TTS_URL})"
    else:
        tts_mode = "cloud"
        tts_type = "Cartesia (cloud)"

    # Determine LLM mode
    if USE_CHUNKED_LLM:
        llm_mode = "chunked"
        llm_url = NVIDIA_LLAMA_CPP_URL
    else:
        llm_mode = "standard"
        llm_url = NVIDIA_LLM_URL

    logger.info(f"Starting bot (local ASR, {llm_mode} LLM, {tts_type} TTS)")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL}")
    logger.info(f"  LLM Mode: {llm_mode}")
    logger.info(f"  LLM URL: {llm_url}")
    if not USE_CHUNKED_LLM:
        logger.info(f"  LLM Model: {NVIDIA_LLM_MODEL}")
    logger.info(f"  TTS: {tts_type}")
    logger.info(f"  Transport: {type(transport).__name__}")

    # NVIDIA Parakeet ASR via WebSocket
    stt = NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=16000,
    )

    # TTS service selection
    if USE_LOCAL_TTS:
        if USE_WEBSOCKET_TTS:
            # WebSocket Magpie TTS - full-duplex adaptive streaming
            # Enable adaptive mode when using chunked LLM for optimal latency
            tts = MagpieWebSocketTTSService(
                server_url=NVIDIA_TTS_URL,
                voice="aria",
                language="en",
                params=MagpieWebSocketTTSService.InputParams(
                    language="en",
                    streaming_preset="conservative",  # ~370ms TTFB for first chunk
                    use_adaptive_mode=USE_CHUNKED_LLM,  # Enable when using chunked LLM
                ),
            )
            mode_desc = "adaptive" if USE_CHUNKED_LLM else "batch-only"
            logger.info(f"Using WebSocket Magpie TTS (full-duplex, {mode_desc})")
        else:
            # Batch Magpie TTS via HTTP server
            tts = MagpieHTTPTTSService(
                server_url=NVIDIA_TTS_URL,
                voice="aria",
                language="en",
            )
            logger.info("Using batch Magpie TTS")
    else:
        # Cartesia TTS (cloud) - fallback option
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

    # LLM service selection
    if USE_CHUNKED_LLM:
        # Chunked LLM - sentence-boundary streaming direct to llama.cpp
        # Best paired with USE_WEBSOCKET_TTS for adaptive mode switching
        llm = LlamaCppChunkedLLMService(
            llama_url=NVIDIA_LLAMA_CPP_URL,
            params=LlamaCppChunkedLLMService.InputParams(
                first_chunk_min_tokens=10,
                first_chunk_max_tokens=24,
            ),
        )
        logger.info("Using LlamaCppChunkedLLMService (direct llama.cpp)")
    else:
        # Standard OpenAI-compatible LLM via vLLM
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
                "You are built with Nemotron Three Nano, a large language model developed by NVIDIA. "
                "Your goal is to have a natural conversation with the user. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Avoid special characters. Use only simple, plain text sentences. "
                "Always punctuate your responses using standard sentence punctuation: commas, periods, question marks, exclamation points, etc."
                "Always spell out numbers as words."
            ),
        },
        {
            "role": "user",
            "content": "Say hello and ask how you can help.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,  # RTVI processor (early in chain for client messages)
            stt,
            context_aggregator.user(),
            llm,
            tts,
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

#!/usr/bin/env python3
#
# Pipecat bot using vLLM for higher quality inference.
#
# Uses vLLM (OpenAI-compatible API) instead of llama.cpp. Requires more VRAM (~72GB)
# but provides higher quality inference with full BF16 weights.
#
# Environment variables:
#   NVIDIA_ASR_URL        ASR WebSocket URL (default: ws://localhost:8080)
#   NVIDIA_LLM_URL        vLLM API URL (default: http://localhost:8000/v1)
#   NVIDIA_LLM_MODEL      Model name/path (default: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
#   NVIDIA_LLM_API_KEY    API key for vLLM (default: not-needed)
#   NVIDIA_TTS_URL        Magpie TTS server URL (default: http://localhost:8001)
#
# Usage:
#   uv run pipecat_bots/bot_vllm.py
#   uv run pipecat_bots/bot_vllm.py -t daily
#   uv run pipecat_bots/bot_vllm.py -t webrtc
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
# Use our custom SentenceAggregator that flushes on LLMFullResponseEndFrame
from sentence_aggregator import SentenceAggregator
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from magpie_websocket_tts import MagpieWebSocketTTSService
from v2v_metrics import V2VMetricsProcessor

load_dotenv(override=True)

# Configuration from environment
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_LLM_URL = os.getenv("NVIDIA_LLM_URL", "http://localhost:8000/v1")
NVIDIA_LLM_MODEL = os.getenv(
    "NVIDIA_LLM_MODEL",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
)
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY", "not-needed")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# VAD configuration - used by both VAD analyzer and V2V metrics
VAD_STOP_SECS = 0.2

# Transport configurations with VAD and SmartTurn analyzer
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting vLLM bot")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL}")
    logger.info(f"  LLM URL: {NVIDIA_LLM_URL}")
    logger.info(f"  LLM Model: {NVIDIA_LLM_MODEL}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL}")
    logger.info(f"  Transport: {type(transport).__name__}")
    logger.info(f"  VAD stop_secs: {VAD_STOP_SECS}s")

    # NVIDIA Parakeet ASR via WebSocket
    stt = NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=16000,
    )

    # WebSocket Magpie TTS (batch-only mode - vLLM doesn't do sentence-boundary chunking)
    tts = MagpieWebSocketTTSService(
        server_url=NVIDIA_TTS_URL,
        voice="aria",
        language="en",
        params=MagpieWebSocketTTSService.InputParams(
            language="en",
            streaming_preset="conservative",
            use_adaptive_mode=True,
        ),
    )
    logger.info("Using WebSocket Magpie TTS (adaptive mode)")

    # vLLM via OpenAI-compatible API
    llm = OpenAILLMService(
        api_key=NVIDIA_LLM_API_KEY,
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
    logger.info("Using vLLM via OpenAILLMService (thinking disabled)")

    # Voice-to-voice response time metrics
    v2v_metrics = V2VMetricsProcessor(vad_stop_secs=VAD_STOP_SECS)

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
    sentence_aggregator = SentenceAggregator()

    # RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            sentence_aggregator,
            tts,
            v2v_metrics,
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
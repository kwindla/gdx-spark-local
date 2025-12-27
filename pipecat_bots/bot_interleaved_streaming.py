#!/usr/bin/env python3
#
# Pipecat bot with interleaved streaming for lowest latency.
#
# Uses chunked LLM (sentence-boundary streaming) with adaptive WebSocket TTS.
# SmartTurn analyzer for responsive turn-taking.
#
# Environment variables:
#   NVIDIA_ASR_URL        ASR WebSocket URL (default: ws://localhost:8080)
#   NVIDIA_LLAMA_CPP_URL  llama.cpp API URL (default: http://localhost:8000)
#   NVIDIA_TTS_URL        Magpie TTS server URL (default: http://localhost:8001)
#
# Usage:
#   uv run pipecat_bots/bot_interleaved_streaming.py
#   uv run pipecat_bots/bot_interleaved_streaming.py -t daily
#   uv run pipecat_bots/bot_interleaved_streaming.py -t webrtc
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
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from magpie_websocket_tts import MagpieWebSocketTTSService
from llama_cpp_chunked_llm import LlamaCppChunkedLLMService

load_dotenv(override=True)

# Configuration from environment
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_LLAMA_CPP_URL = os.getenv("NVIDIA_LLAMA_CPP_URL", "http://localhost:8000")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# Transport configurations with VAD and SmartTurn analyzer
# stop_secs=0.34 aligns with ASR model's trailing context requirements:
# - At 16kHz/20ms chunks, 340ms VAD silence = ~320ms at server (minus triggering chunk)
# - ASR needs (right_context+1)*160ms = 320ms trailing silence for finalization
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.34)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.34)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.34)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting interleaved streaming bot")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL}")
    logger.info(f"  LLM URL: {NVIDIA_LLAMA_CPP_URL}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL}")
    logger.info(f"  Transport: {type(transport).__name__}")

    # NVIDIA Parakeet ASR via WebSocket
    stt = NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=16000,
    )

    # WebSocket Magpie TTS with adaptive mode
    # Adaptive mode: streaming for first segment (~370ms TTFB), batch for subsequent (quality)
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

    # Chunked LLM - sentence-boundary streaming direct to llama.cpp
    llm = LlamaCppChunkedLLMService(
        llama_url=NVIDIA_LLAMA_CPP_URL,
        params=LlamaCppChunkedLLMService.InputParams(
            first_chunk_min_tokens=10,
            first_chunk_max_tokens=24,
        ),
    )
    logger.info("Using LlamaCppChunkedLLMService (sentence-boundary streaming)")

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
            rtvi,
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

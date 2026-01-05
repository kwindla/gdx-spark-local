#!/usr/bin/env python3
"""Test script for the Pipeline-based STT adapter.

Tests all 4 Pipecat services (Deepgram, Cartesia, ElevenLabs, NVIDIA Parakeet)
using the same audio sample.

Usage:
    python -m asr_eval.test_pipeline_adapter [--sample-id SAMPLE_ID]
"""

import argparse
import asyncio
import sys

from loguru import logger

from asr_eval.adapters.pipeline_stt_adapter import PipelineSTTAdapter
from asr_eval.storage.database import Database
from asr_eval.models import AudioSample, ServiceName

# Services to test via Pipecat Pipeline
PIPECAT_SERVICES = [
    ServiceName.DEEPGRAM,
    ServiceName.CARTESIA,
    ServiceName.ELEVENLABS,
    ServiceName.NVIDIA_PARAKEET,
]


async def test_service(sample: AudioSample, service_name: ServiceName) -> None:
    """Test a single STT service with the given sample."""
    logger.info(f"[{service_name.value}] Testing...")

    adapter = PipelineSTTAdapter(service_name=service_name)

    try:
        result = await adapter.transcribe(sample)

        if result.error:
            logger.error(f"[{service_name.value}] FAILED: {result.error}")
        else:
            text_preview = result.transcribed_text[:60] + "..." if len(result.transcribed_text) > 60 else result.transcribed_text
            logger.info(
                f"[{service_name.value}] SUCCESS: \"{text_preview}\" "
                f"({result.time_to_transcription_ms:.0f}ms, RTF={result.rtf:.2f})"
            )

    except Exception as e:
        logger.exception(f"[{service_name.value}] EXCEPTION: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Test Pipeline STT adapter")
    parser.add_argument(
        "--sample-id",
        type=str,
        help="Specific sample ID to test (default: first sample with ground truth)",
    )
    args = parser.parse_args()

    # Get a sample from the database
    db = Database()

    if args.sample_id:
        sample = await db.get_sample(args.sample_id)
        if not sample:
            logger.error(f"Sample not found: {args.sample_id}")
            sys.exit(1)
    else:
        # Get first sample with ground truth
        samples = await db.get_all_samples()
        sample = None
        for s in samples[:10]:
            gt = await db.get_ground_truth(s.sample_id)
            if gt:
                sample = s
                logger.info(f"Using sample: {s.sample_id}")
                logger.info(f"Ground truth: {gt.text[:80]}...")
                break

        if not sample:
            logger.error("No samples with ground truth found")
            sys.exit(1)

    logger.info(f"Audio: {sample.audio_path}")
    logger.info(f"Duration: {sample.duration_seconds:.2f}s")
    logger.info("")

    # Test each service
    for service_name in PIPECAT_SERVICES:
        await test_service(sample, service_name)
        logger.info("")


if __name__ == "__main__":
    asyncio.run(main())

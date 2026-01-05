"""Run batch STT transcription for a specific model.

Usage:
    python -m asr_eval.run_batch_transcription --service deepgram --model nova-3 --samples 1000
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from asr_eval.adapters.pipeline_stt_adapter import PipelineSTTAdapter
from asr_eval.models import ServiceName
from asr_eval.storage.database import Database


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch STT transcription")
    parser.add_argument("--service", required=True, choices=[s.value for s in ServiceName])
    parser.add_argument("--model", default=None, help="Model name (e.g., nova-3, whisper-large)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to transcribe")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples (for resuming)")
    parser.add_argument("--output", default=None, help="Output JSONL file path")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--vad-stop-ms", type=int, default=200, help="VAD stop detection ms")
    parser.add_argument("--post-silence-ms", type=int, default=2000, help="Post-audio silence ms")
    return parser.parse_args()


async def run_batch(args):
    service_name = ServiceName(args.service)
    model_label = args.model or "default"

    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("asr_eval_data/transcription_runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{service_name.value}_{model_label}_{timestamp}.jsonl"

    # Convert ms to seconds for VAD
    vad_stop_secs = args.vad_stop_ms / 1000.0

    logger.info(f"Starting batch transcription: {service_name.value} ({model_label})")
    logger.info(f"Output: {output_path}")
    logger.info(f"VAD config: vad_stop_secs={vad_stop_secs}, post_silence_ms={args.post_silence_ms}")

    # Initialize database
    db = Database()
    await db.initialize()

    # Get samples (sorted by dataset_index for repeatability)
    all_samples = await db.get_all_samples()
    # Apply offset and limit
    end_idx = args.offset + args.samples
    samples = all_samples[args.offset:end_idx]
    if args.offset > 0:
        logger.info(f"Resuming from sample {args.offset}, processing {len(samples)} samples")
    else:
        logger.info(f"Selected {len(samples)} samples for transcription")

    # Create adapter
    adapter = PipelineSTTAdapter(
        service_name=service_name,
        model=args.model,
        vad_stop_secs=vad_stop_secs,
        post_audio_silence_ms=args.post_silence_ms,
    )

    # Write header (skip if appending)
    if not args.append:
        with open(output_path, "w") as f:
            header = {
                "type": "header",
                "service": service_name.value,
                "model": model_label,
                "num_samples": len(samples),
                "vad_stop_ms": args.vad_stop_ms,
                "post_silence_ms": args.post_silence_ms,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(header) + "\n")
    else:
        logger.info(f"Appending to existing file: {output_path}")

    # Process samples
    successful = 0
    failed = 0
    total_latency = 0.0

    for i, sample in enumerate(samples):
        try:
            result = await adapter.transcribe(sample)

            # Write result
            with open(output_path, "a") as f:
                record = {
                    "type": "result",
                    "sample_id": sample.sample_id,
                    "dataset_index": sample.dataset_index,
                    "audio_duration_ms": sample.duration_seconds * 1000,
                    "transcribed_text": result.transcribed_text,
                    "latency_ms": result.time_to_transcription_ms,
                    "error": result.error,
                }
                f.write(json.dumps(record) + "\n")

            if result.error:
                failed += 1
                logger.warning(f"[{i+1}/{len(samples)}] {sample.sample_id}: ERROR - {result.error}")
            else:
                successful += 1
                total_latency += result.time_to_transcription_ms
                if (i + 1) % 50 == 0:
                    avg_latency = total_latency / successful if successful > 0 else 0
                    logger.info(f"[{i+1}/{len(samples)}] Progress: {successful} ok, {failed} failed, avg latency: {avg_latency:.0f}ms")

        except Exception as e:
            failed += 1
            logger.error(f"[{i+1}/{len(samples)}] {sample.sample_id}: EXCEPTION - {e}")
            with open(output_path, "a") as f:
                record = {
                    "type": "result",
                    "sample_id": sample.sample_id,
                    "dataset_index": sample.dataset_index,
                    "error": str(e),
                }
                f.write(json.dumps(record) + "\n")

    # Write summary (skip if appending - user should compute final summary)
    avg_latency = total_latency / successful if successful > 0 else 0
    if not args.append:
        with open(output_path, "a") as f:
            summary = {
                "type": "summary",
                "successful": successful,
                "failed": failed,
                "total": len(samples),
                "avg_latency_ms": avg_latency,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(summary) + "\n")

    logger.info(f"Completed: {successful}/{len(samples)} successful, avg latency: {avg_latency:.0f}ms")
    logger.info(f"Results saved to: {output_path}")


def main():
    args = parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()

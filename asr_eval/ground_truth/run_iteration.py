"""Run a ground truth transcription iteration.

Generates transcriptions for a repeatable set of samples and saves to JSONL
for longitudinal comparison as we iterate on the prompt.

Usage:
    uv run python -m asr_eval.ground_truth.run_iteration --samples 100
    uv run python -m asr_eval.ground_truth.run_iteration --samples 100 --clear
"""

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from asr_eval.config import get_config
from asr_eval.ground_truth.gemini_transcriber import (
    TRANSCRIPTION_PROMPT,
    GeminiTranscriber,
)
from asr_eval.models import AudioSample
from asr_eval.storage.database import Database


def get_prompt_hash(prompt: str) -> str:
    """Generate a short hash of the prompt for identification."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


async def get_first_n_samples(db: Database, n: int) -> list[AudioSample]:
    """Get first N samples ordered by dataset_index for repeatability."""
    async with db.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT * FROM samples
            ORDER BY dataset_index
            LIMIT ?
            """,
            (n,),
        )
        rows = await cursor.fetchall()
        return [
            AudioSample(
                sample_id=row["sample_id"],
                audio_path=row["audio_path"],
                duration_seconds=row["duration_seconds"],
                language=row["language"],
                synthetic=bool(row["synthetic"]),
                dataset_index=row["dataset_index"],
            )
            for row in rows
        ]


async def run_iteration(
    num_samples: int = 100,
    clear_existing: bool = False,
) -> Path:
    """Run a transcription iteration and save to JSONL.

    Args:
        num_samples: Number of samples to transcribe
        clear_existing: Whether to clear existing ground truths first

    Returns:
        Path to the output JSONL file
    """
    config = get_config()
    db = Database()
    await db.initialize()

    # Optionally clear existing ground truths
    if clear_existing:
        count = await db.clear_all_ground_truths()
        logger.info(f"Cleared {count} existing ground truth records")

    # Get samples
    samples = await get_first_n_samples(db, num_samples)
    if not samples:
        logger.error("No samples found in database")
        raise ValueError("No samples found")

    logger.info(f"Selected {len(samples)} samples for transcription")

    # Create output directory
    runs_dir = config.data_dir / "ground_truth_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID from timestamp
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = runs_dir / f"{run_id}.jsonl"

    # Create transcriber
    transcriber = GeminiTranscriber()

    # Write header
    header = {
        "type": "header",
        "run_id": run_id,
        "model": transcriber.model_name,
        "thinking_level": transcriber.thinking_level,
        "prompt_hash": get_prompt_hash(TRANSCRIPTION_PROMPT),
        "prompt_text": TRANSCRIPTION_PROMPT,
        "num_samples": len(samples),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(header) + "\n")

    # Transcribe samples and append to file
    completed = 0
    errors = 0

    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] Transcribing {sample.sample_id}...")

        try:
            gt = await transcriber.transcribe_sample(sample)

            if gt:
                record = {
                    "type": "sample",
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "duration_seconds": sample.duration_seconds,
                    "transcription": gt.text,
                    "generated_at": gt.generated_at.isoformat() + "Z",
                }
                completed += 1
            else:
                record = {
                    "type": "sample",
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "duration_seconds": sample.duration_seconds,
                    "transcription": None,
                    "error": "Empty response from Gemini",
                    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                errors += 1

        except Exception as e:
            logger.error(f"Error transcribing {sample.sample_id}: {e}")
            record = {
                "type": "sample",
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "duration_seconds": sample.duration_seconds,
                "transcription": None,
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            errors += 1

        # Append to file (streaming writes)
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Write footer with summary
    footer = {
        "type": "footer",
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_samples": len(samples),
        "successful": completed,
        "errors": errors,
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(footer) + "\n")

    logger.info(f"Run complete: {completed} successful, {errors} errors")
    logger.info(f"Output saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run a ground truth transcription iteration"
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples to transcribe (default: 100)",
    )
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Clear existing ground truths before running",
    )
    args = parser.parse_args()

    asyncio.run(run_iteration(num_samples=args.samples, clear_existing=args.clear))


if __name__ == "__main__":
    main()

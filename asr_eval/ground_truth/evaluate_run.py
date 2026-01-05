"""Interactive evaluation script for ground truth transcription runs.

Allows manual review of transcriptions with audio playback.

Usage:
    uv run python -m asr_eval.ground_truth.evaluate_run <run_id>
    uv run python -m asr_eval.ground_truth.evaluate_run 2026-01-03_14-30-00
    uv run python -m asr_eval.ground_truth.evaluate_run --list
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import readchar

from asr_eval.config import get_config


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def play_audio(audio_path: str) -> subprocess.Popen:
    """Play a PCM audio file using ffplay.

    Returns the subprocess so caller can wait or kill if needed.
    """
    cmd = [
        "ffplay",
        "-f", "s16le",
        "-ar", "16000",
        "-ch_layout", "mono",
        "-nodisp",
        "-autoexit",
        "-loglevel", "quiet",
        audio_path,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_run(run_path: Path) -> tuple[dict, list[dict], Optional[dict]]:
    """Load a run JSONL file.

    Returns:
        (header, samples, footer)
    """
    header = None
    samples = []
    footer = None

    with open(run_path) as f:
        for line in f:
            record = json.loads(line.strip())
            if record["type"] == "header":
                header = record
            elif record["type"] == "sample":
                samples.append(record)
            elif record["type"] == "footer":
                footer = record

    if not header:
        raise ValueError(f"No header found in {run_path}")

    return header, samples, footer


def load_existing_notes(notes_path: Path) -> dict[str, dict]:
    """Load existing notes if any, returning a dict keyed by sample_id."""
    notes = {}
    if notes_path.exists():
        with open(notes_path) as f:
            for line in f:
                record = json.loads(line.strip())
                if record["type"] == "review":
                    notes[record["sample_id"]] = record
    return notes


def save_review(notes_path: Path, sample_id: str, status: str, note: Optional[str]):
    """Append a review record to the notes file."""
    record = {
        "type": "review",
        "sample_id": sample_id,
        "status": status,
        "note": note,
        "reviewed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(notes_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def display_sample(
    index: int,
    total: int,
    sample: dict,
    stats: dict,
    existing_review: Optional[dict] = None,
):
    """Display a sample for review."""
    clear_screen()

    sample_id = sample["sample_id"]
    duration = sample["duration_seconds"]
    transcription = sample.get("transcription", "[ERROR: No transcription]")
    error = sample.get("error")

    print("=" * 70)
    print(f"Sample {index + 1}/{total}  [id: {sample_id[:20]}...]  Duration: {duration:.1f}s")
    print("=" * 70)
    print()

    if error:
        print(f"ERROR: {error}")
        print()

    print("Transcription:")
    print(f'"{transcription}"')
    print()

    if existing_review:
        status = existing_review["status"]
        note = existing_review.get("note", "")
        print(f"[Previous review: {status}]" + (f" Note: {note}" if note else ""))
        print()

    print("-" * 70)
    print("[p] Play  [r] Replay  [a] Approve  [n] Note  [Enter] Next  [q] Quit")
    print("-" * 70)
    print(f"Progress: {stats['approved']} approved, {stats['noted']} noted, {stats['skipped']} skipped")


def run_evaluation(run_path: Path):
    """Run the interactive evaluation for a run file."""
    # Load run data
    header, samples, footer = load_run(run_path)

    run_id = header["run_id"]
    print(f"Loaded run {run_id} with {len(samples)} samples")
    print(f"Model: {header['model']}, Thinking: {header['thinking_level']}")
    print()

    # Set up notes file
    notes_path = run_path.parent / f"{run_id}_notes.jsonl"

    # Load existing notes
    existing_notes = load_existing_notes(notes_path)
    if existing_notes:
        print(f"Found {len(existing_notes)} existing reviews")

    # Write header if new notes file
    if not notes_path.exists():
        header_record = {
            "type": "header",
            "run_id": run_id,
            "evaluator": "user",
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with open(notes_path, "w") as f:
            f.write(json.dumps(header_record) + "\n")

    # Stats
    stats = {"approved": 0, "noted": 0, "skipped": 0}

    # Count existing reviews
    for note in existing_notes.values():
        status = note.get("status", "skipped")
        if status in stats:
            stats[status] += 1

    # Find first unreviewed sample
    start_index = 0
    for i, sample in enumerate(samples):
        if sample["sample_id"] not in existing_notes:
            start_index = i
            break

    print(f"Starting from sample {start_index + 1}")
    input("Press Enter to begin...")

    current_process: Optional[subprocess.Popen] = None
    current_audio_path: Optional[str] = None

    try:
        i = start_index
        while i < len(samples):
            sample = samples[i]
            sample_id = sample["sample_id"]
            audio_path = sample["audio_path"]

            existing_review = existing_notes.get(sample_id)
            display_sample(i, len(samples), sample, stats, existing_review)

            while True:
                key = readchar.readkey()

                if key == "p" or key == "r":
                    # Kill any existing playback
                    if current_process and current_process.poll() is None:
                        current_process.kill()

                    # Play audio
                    if Path(audio_path).exists():
                        current_process = play_audio(audio_path)
                        current_audio_path = audio_path
                    else:
                        print(f"\nAudio file not found: {audio_path}")

                elif key == "a":
                    # Approve
                    save_review(notes_path, sample_id, "approved", None)
                    existing_notes[sample_id] = {"status": "approved", "note": None}
                    stats["approved"] += 1
                    i += 1
                    break

                elif key == "n":
                    # Add note
                    print("\nNote (press Enter when done): ", end="", flush=True)
                    note = input()
                    save_review(notes_path, sample_id, "noted", note)
                    existing_notes[sample_id] = {"status": "noted", "note": note}
                    stats["noted"] += 1
                    i += 1
                    break

                elif key == "\r" or key == "\n" or key == readchar.key.ENTER:
                    # Next (skip)
                    if sample_id not in existing_notes:
                        save_review(notes_path, sample_id, "skipped", None)
                        existing_notes[sample_id] = {"status": "skipped", "note": None}
                        stats["skipped"] += 1
                    i += 1
                    break

                elif key == "q":
                    # Quit
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        pass

    finally:
        # Kill any running audio
        if current_process and current_process.poll() is None:
            current_process.kill()

    # Write summary
    summary = {
        "type": "summary",
        "total": len(samples),
        "approved": stats["approved"],
        "noted": stats["noted"],
        "skipped": stats["skipped"],
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(notes_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    clear_screen()
    print("=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print()
    print(f"Total samples: {len(samples)}")
    print(f"Approved: {stats['approved']}")
    print(f"Noted: {stats['noted']}")
    print(f"Skipped: {stats['skipped']}")
    print()
    print(f"Notes saved to: {notes_path}")


def list_runs():
    """List all available runs."""
    config = get_config()
    runs_dir = config.data_dir / "ground_truth_runs"

    if not runs_dir.exists():
        print("No runs directory found")
        return

    jsonl_files = sorted(runs_dir.glob("*.jsonl"))
    run_files = [f for f in jsonl_files if not f.stem.endswith("_notes")]

    if not run_files:
        print("No runs found")
        return

    print("Available runs:")
    print("-" * 70)

    for run_path in run_files:
        try:
            header, samples, footer = load_run(run_path)
            run_id = header["run_id"]
            model = header.get("model", "?")
            thinking = header.get("thinking_level", "?")
            num_samples = len(samples)

            # Check for notes
            notes_path = run_path.parent / f"{run_id}_notes.jsonl"
            notes_info = ""
            if notes_path.exists():
                notes = load_existing_notes(notes_path)
                notes_info = f" [{len(notes)} reviewed]"

            print(f"  {run_id}  ({model}, {thinking})  {num_samples} samples{notes_info}")

        except Exception as e:
            print(f"  {run_path.name}  [error: {e}]")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive evaluation for ground truth transcription runs"
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        help="Run ID to evaluate (e.g., 2026-01-03_14-30-00)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available runs",
    )
    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    if not args.run_id:
        parser.print_help()
        print("\nUse --list to see available runs")
        sys.exit(1)

    config = get_config()
    runs_dir = config.data_dir / "ground_truth_runs"

    # Try to find the run file
    run_path = runs_dir / f"{args.run_id}.jsonl"
    if not run_path.exists():
        # Maybe they gave a full path
        run_path = Path(args.run_id)
        if not run_path.exists():
            print(f"Run not found: {args.run_id}")
            print("Use --list to see available runs")
            sys.exit(1)

    run_evaluation(run_path)


if __name__ == "__main__":
    main()

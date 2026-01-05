"""CLI for generating ground truth transcriptions using Gemini."""

import asyncio
import sys

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from asr_eval.config import get_config
from asr_eval.ground_truth.gemini_transcriber import GeminiTranscriber
from asr_eval.storage.database import Database

app = typer.Typer(help="Generate ground truth transcriptions using Gemini 2.5 Flash.")
console = Console()


@app.command()
def main(
    batch_size: int = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Number of samples to process (default: all remaining)",
    ),
    model: str = typer.Option(
        "gemini-3-flash-preview",
        "--model",
        "-m",
        help="Gemini model to use",
    ),
):
    """Generate ground truth transcriptions for audio samples.

    Uses Gemini 2.5 Flash to create literal transcriptions of audio samples
    that will be used as the reference for WER calculation.
    """
    config = get_config()

    console.print(f"\n[bold blue]Ground Truth Generation[/bold blue]\n")
    console.print(f"Model: {model}")

    # Check API key
    if not config.google_api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set in environment[/red]")
        sys.exit(1)

    # Get database stats
    db = Database()

    async def get_stats():
        await db.initialize()
        total_samples = await db.get_sample_count()
        gt_count = await db.get_ground_truth_count()
        pending = await db.get_samples_without_ground_truth()
        return total_samples, gt_count, pending

    total_samples, gt_count, pending_samples = asyncio.run(get_stats())

    console.print(f"Total samples: {total_samples}")
    console.print(f"Already transcribed: {gt_count}")
    console.print(f"Pending: {len(pending_samples)}")

    if not pending_samples:
        console.print("\n[green]All samples already have ground truth![/green]")
        return

    # Apply batch size limit
    if batch_size:
        pending_samples = pending_samples[:batch_size]
        console.print(f"Processing batch of {len(pending_samples)} samples")

    console.print(f"\nGenerating transcriptions for {len(pending_samples)} samples...\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(pending_samples))

        def progress_callback(current: int, total: int, sample_id: str):
            progress.update(task, completed=current)

        try:
            transcriber = GeminiTranscriber(model_name=model)
            results = asyncio.run(
                transcriber.transcribe_batch(
                    pending_samples,
                    progress_callback=progress_callback,
                )
            )

            progress.update(task, completed=len(pending_samples))

        except KeyboardInterrupt:
            console.print("\n[yellow]Generation interrupted by user[/yellow]")
            console.print("Progress has been saved. Run again to continue.")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

    # Summary
    successful = len([r for r in results if r is not None])
    failed = len(pending_samples) - successful

    console.print(f"\n[green]Ground truth generation complete![/green]")
    console.print(f"Successful: {successful}")
    if failed > 0:
        console.print(f"[yellow]Failed: {failed}[/yellow]")

    # Show updated stats
    new_total, new_gt, _ = asyncio.run(get_stats())
    console.print(f"\nTotal ground truth: {new_gt}/{new_total} samples")


if __name__ == "__main__":
    app()

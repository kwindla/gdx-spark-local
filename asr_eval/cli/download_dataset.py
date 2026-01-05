"""CLI for downloading and preparing the ASR evaluation dataset."""

import asyncio
import sys

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from asr_eval.config import get_config
from asr_eval.dataset.downloader import download_dataset

app = typer.Typer(help="Download and prepare the ASR evaluation dataset.")
console = Console()


@app.command()
def main(
    samples: int = typer.Option(
        1000,
        "--samples",
        "-n",
        help="Number of samples to download",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        "-o",
        help="Number of samples to skip (for incremental downloads)",
    ),
):
    """Download and prepare the smart-turn dataset for ASR evaluation.

    Downloads audio samples from the pipecat-ai/smart-turn-data-v3.1-train
    dataset, filtering for English non-synthetic samples.

    Use --offset to download additional samples without reprocessing existing ones.
    Example: To add 4000 more samples after already having 1000:
      python -m asr_eval.cli.download_dataset -n 4000 -o 1000
    """
    config = get_config()

    console.print(f"\n[bold blue]ASR Evaluation Dataset Downloader[/bold blue]\n")
    console.print(f"Dataset: {config.dataset_name}")
    console.print(f"Samples: {samples}")
    console.print(f"Offset: {offset}")
    console.print(f"Seed: {seed}")
    console.print(f"Output: {config.audio_dir}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)

        def progress_callback(current: int, total: int, message: str):
            progress.update(task, description=message)

        try:
            samples_list = asyncio.run(
                download_dataset(
                    num_samples=samples,
                    seed=seed,
                    offset=offset,
                    progress_callback=progress_callback,
                )
            )

            progress.update(task, description="Complete!")

        except KeyboardInterrupt:
            console.print("\n[yellow]Download interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

    console.print(f"\n[green]Successfully prepared {len(samples_list)} samples![/green]")
    console.print(f"Audio files saved to: {config.audio_dir}")
    console.print(f"Database: {config.results_db}\n")

    # Show sample statistics
    if samples_list:
        total_duration = sum(s.duration_seconds for s in samples_list)
        avg_duration = total_duration / len(samples_list)
        console.print(f"Total audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        console.print(f"Average sample duration: {avg_duration:.2f}s")
        console.print(f"Min duration: {min(s.duration_seconds for s in samples_list):.2f}s")
        console.print(f"Max duration: {max(s.duration_seconds for s in samples_list):.2f}s")


if __name__ == "__main__":
    app()

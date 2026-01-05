"""CLI for running ASR evaluation on STT services."""

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from asr_eval.config import get_config
from asr_eval.evaluation.agent_sdk_judge import AgentSDKJudge
from asr_eval.evaluation.wer_calculator import WERCalculator
from asr_eval.harness.service_wrapper import STTServiceHarness
from asr_eval.models import ServiceName
from asr_eval.services.cartesia_direct import CartesiaDirectAdapter
from asr_eval.services.deepgram_direct import DeepgramDirectAdapter
from asr_eval.services.elevenlabs_direct import ElevenLabsDirectAdapter
from asr_eval.services.faster_whisper_adapter import FasterWhisperAdapter
from asr_eval.services.nvidia_direct import NvidiaDirectAdapter
from asr_eval.storage.database import Database

app = typer.Typer(help="Run ASR evaluation on STT services.")
console = Console()

# Service name mapping
SERVICE_MAP = {
    "nvidia": ServiceName.NVIDIA_PARAKEET,
    "nvidia_parakeet": ServiceName.NVIDIA_PARAKEET,
    "deepgram": ServiceName.DEEPGRAM,
    "cartesia": ServiceName.CARTESIA,
    "elevenlabs": ServiceName.ELEVENLABS,
    "whisper": ServiceName.FASTER_WHISPER,
    "faster_whisper": ServiceName.FASTER_WHISPER,
}


def get_adapter(service_name: ServiceName):
    """Get the appropriate adapter for a service."""
    if service_name == ServiceName.NVIDIA_PARAKEET:
        return NvidiaDirectAdapter()
    elif service_name == ServiceName.DEEPGRAM:
        return DeepgramDirectAdapter()
    elif service_name == ServiceName.CARTESIA:
        return CartesiaDirectAdapter()
    elif service_name == ServiceName.ELEVENLABS:
        return ElevenLabsDirectAdapter()
    elif service_name == ServiceName.FASTER_WHISPER:
        return FasterWhisperAdapter()
    else:
        raise ValueError(f"Unknown service: {service_name}")


@app.command("transcribe")
def transcribe(
    services: str = typer.Option(
        "all",
        "--services",
        "-s",
        help="Comma-separated list of services to test (nvidia,deepgram,cartesia,elevenlabs,whisper) or 'all'",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of samples to process",
    ),
    simulate_realtime: bool = typer.Option(
        True,
        "--realtime/--no-realtime",
        help="Simulate real-time streaming (slower but more realistic)",
    ),
):
    """Run transcription on samples using specified STT services."""
    config = get_config()

    # Parse services
    if services.lower() == "all":
        service_list = list(ServiceName)
    else:
        service_list = []
        for s in services.split(","):
            s = s.strip().lower()
            if s in SERVICE_MAP:
                service_list.append(SERVICE_MAP[s])
            else:
                console.print(f"[red]Unknown service: {s}[/red]")
                console.print(f"Available: {', '.join(SERVICE_MAP.keys())}")
                sys.exit(1)

    console.print(f"\n[bold blue]ASR Evaluation - Transcription[/bold blue]\n")
    console.print(f"Services: {', '.join(s.value for s in service_list)}")
    console.print(f"Realtime simulation: {simulate_realtime}")

    async def run():
        db = Database()
        await db.initialize()

        # Get samples
        samples = await db.get_all_samples()
        if not samples:
            console.print("[red]No samples found. Run download_dataset first.[/red]")
            return

        if limit:
            samples = samples[:limit]

        console.print(f"Samples: {len(samples)}\n")

        for service_name in service_list:
            console.print(f"\n[bold]Processing {service_name.value}...[/bold]")

            try:
                # Check which samples need processing
                pending = await db.get_samples_without_transcription(service_name)
                if limit:
                    pending = [s for s in pending if s in samples]

                if not pending:
                    console.print(f"  All samples already transcribed for {service_name.value}")
                    continue

                console.print(f"  Pending samples: {len(pending)}")

                # Special handling for direct adapters (don't use Pipecat harness)
                if service_name in (ServiceName.FASTER_WHISPER, ServiceName.DEEPGRAM, ServiceName.ELEVENLABS, ServiceName.CARTESIA, ServiceName.NVIDIA_PARAKEET):
                    adapter = get_adapter(service_name)

                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Transcribing with {service_name.value}...",
                            total=len(pending),
                        )

                        def callback(current, total, sample_id):
                            progress.update(task, completed=current)

                        results = await adapter.transcribe_batch(
                            pending, progress_callback=callback
                        )

                        progress.update(task, completed=len(pending))

                    # Save results
                    for result in results:
                        await db.insert_transcription(result)

                else:
                    # Use Pipecat harness for other services
                    adapter = get_adapter(service_name)
                    harness = STTServiceHarness(
                        adapter=adapter,
                        simulate_realtime=simulate_realtime,
                    )

                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Transcribing with {service_name.value}...",
                            total=len(pending),
                        )

                        def callback(current, total, sample_id):
                            progress.update(task, completed=current)

                        results = await harness.run_batch(
                            pending, progress_callback=callback
                        )

                        progress.update(task, completed=len(pending))

                    # Save results
                    for result in results:
                        await db.insert_transcription(result)

                # Summary
                successful = len([r for r in results if not r.error])
                errors = len([r for r in results if r.error])
                console.print(f"  [green]Completed: {successful}[/green]")
                if errors > 0:
                    console.print(f"  [yellow]Errors: {errors}[/yellow]")

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                continue

    asyncio.run(run())


@app.command("wer")
def calculate_wer(
    services: str = typer.Option(
        "all",
        "--services",
        "-s",
        help="Comma-separated list of services or 'all'",
    ),
):
    """Calculate WER/CER metrics for transcribed samples."""
    console.print(f"\n[bold blue]ASR Evaluation - WER Calculation[/bold blue]\n")

    # Parse services
    if services.lower() == "all":
        service_list = list(ServiceName)
    else:
        service_list = []
        for s in services.split(","):
            s = s.strip().lower()
            if s in SERVICE_MAP:
                service_list.append(SERVICE_MAP[s])
            else:
                console.print(f"[red]Unknown service: {s}[/red]")
                sys.exit(1)

    async def run():
        calculator = WERCalculator()

        for service_name in service_list:
            console.print(f"\n[bold]Calculating WER for {service_name.value}...[/bold]")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Calculating...", total=None)

                def callback(current, total, sample_id):
                    progress.update(task, total=total, completed=current)

                results = await calculator.evaluate_all(
                    service_name, progress_callback=callback
                )

            if results:
                wer_values = [r.wer for r in results if r.wer < float("inf")]
                if wer_values:
                    avg_wer = sum(wer_values) / len(wer_values)
                    console.print(f"  Samples evaluated: {len(results)}")
                    console.print(f"  Average WER: {avg_wer:.2%}")

    asyncio.run(run())


@app.command("agent-sdk-wer")
def agent_sdk_wer_evaluation(
    services: str = typer.Option(
        "all",
        "--services",
        "-s",
        help="Comma-separated list of services or 'all'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-evaluate samples that already have agent_sdk_wer",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of samples to evaluate",
    ),
):
    """Calculate WER using Agent SDK multi-turn reasoning with Claude Opus."""
    console.print(f"\n[bold blue]ASR Evaluation - Agent SDK WER[/bold blue]\n")

    config = get_config()
    if not config.anthropic_api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
        sys.exit(1)

    # Parse services
    if services.lower() == "all":
        service_list = list(ServiceName)
    else:
        service_list = []
        for s in services.split(","):
            s = s.strip().lower()
            if s in SERVICE_MAP:
                service_list.append(SERVICE_MAP[s])
            else:
                console.print(f"[red]Unknown service: {s}[/red]")
                sys.exit(1)

    async def run():
        judge = AgentSDKJudge()

        for service_name in service_list:
            console.print(f"\n[bold]Agent SDK WER evaluation for {service_name.value}...[/bold]")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Evaluating...", total=None)

                def callback(current, total, sample_id):
                    progress.update(task, total=total, completed=current)

                results = await judge.evaluate_all(
                    service_name,
                    progress_callback=callback,
                    skip_existing=not force,
                    limit=limit,
                )

            if results:
                agent_sdk_wers = [r.agent_sdk_wer for r in results if r.agent_sdk_wer is not None]
                if agent_sdk_wers:
                    avg_agent_sdk_wer = sum(agent_sdk_wers) / len(agent_sdk_wers)
                    console.print(f"  Samples evaluated: {len(results)}")
                    console.print(f"  Average Agent SDK WER: {avg_agent_sdk_wer:.2%}")

                # Show pooled WER
                pooled = await judge.compute_pooled_wer(service_name)
                if pooled:
                    console.print(f"  Pooled WER: {pooled['pooled_wer']:.2%}")

    asyncio.run(run())


@app.command("full")
def full_evaluation(
    services: str = typer.Option(
        "all",
        "--services",
        "-s",
        help="Comma-separated list of services or 'all'",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of samples",
    ),
    skip_transcription: bool = typer.Option(
        False,
        "--skip-transcription",
        help="Skip transcription step (use existing)",
    ),
):
    """Run full evaluation pipeline: transcribe, WER, and Agent SDK semantic WER."""
    console.print(f"\n[bold blue]ASR Evaluation - Full Pipeline[/bold blue]\n")

    if not skip_transcription:
        console.print("[bold]Step 1: Transcription[/bold]")
        transcribe(services=services, limit=limit, simulate_realtime=True)

    console.print("\n[bold]Step 2: WER Calculation[/bold]")
    calculate_wer(services=services)

    console.print("\n[bold]Step 3: Agent SDK Semantic WER[/bold]")
    agent_sdk_wer_evaluation(services=services, force=False, limit=None)

    console.print("\n[bold green]Evaluation complete![/bold green]")
    console.print("Run 'python -m asr_eval.cli.analyze_results' to view results.")


if __name__ == "__main__":
    app()

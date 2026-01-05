"""CLI for analyzing ASR evaluation results."""

import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from asr_eval.models import AggregateMetrics, ServiceName
from asr_eval.storage.database import Database

app = typer.Typer(help="Analyze ASR evaluation results.")
console = Console()


async def compute_aggregate_metrics(
    service_name: ServiceName,
    db: Database,
) -> Optional[AggregateMetrics]:
    """Compute aggregate metrics for a service."""
    metrics_list = await db.get_all_metrics_for_service(service_name)

    if not metrics_list:
        return None

    # Filter out infinite WER values
    valid_metrics = [m for m in metrics_list if m.wer < float("inf")]

    if not valid_metrics:
        return None

    wer_values = [m.wer for m in valid_metrics]
    cer_values = [m.cer for m in valid_metrics if m.cer < float("inf")]
    semantic_wer_values = [
        m.semantic_wer for m in valid_metrics if m.semantic_wer is not None
    ]
    gemini_wer_values = [
        m.gemini_wer for m in valid_metrics if m.gemini_wer is not None
    ]

    # Get timing from transcriptions
    transcription_times = []
    for m in valid_metrics:
        tr = await db.get_transcription(m.sample_id, service_name)
        if tr and not tr.error:
            transcription_times.append(tr.time_to_transcription_ms)

    return AggregateMetrics(
        service_name=service_name,
        num_samples=len(metrics_list),
        num_errors=len(metrics_list) - len(valid_metrics),
        mean_wer=statistics.mean(wer_values),
        median_wer=statistics.median(wer_values),
        std_wer=statistics.stdev(wer_values) if len(wer_values) > 1 else 0,
        min_wer=min(wer_values),
        max_wer=max(wer_values),
        mean_cer=statistics.mean(cer_values) if cer_values else 0,
        median_cer=statistics.median(cer_values) if cer_values else 0,
        mean_semantic_wer=(
            statistics.mean(semantic_wer_values) if semantic_wer_values else None
        ),
        median_semantic_wer=(
            statistics.median(semantic_wer_values) if semantic_wer_values else None
        ),
        mean_gemini_wer=(
            statistics.mean(gemini_wer_values) if gemini_wer_values else None
        ),
        median_gemini_wer=(
            statistics.median(gemini_wer_values) if gemini_wer_values else None
        ),
        mean_time_to_transcription_ms=(
            statistics.mean(transcription_times) if transcription_times else 0
        ),
        median_time_to_transcription_ms=(
            statistics.median(transcription_times) if transcription_times else 0
        ),
        p95_time_to_transcription_ms=(
            sorted(transcription_times)[int(len(transcription_times) * 0.95)]
            if transcription_times
            else 0
        ),
        mean_rtf=(
            statistics.mean(
                [
                    t / (m.reference_words * 60 / 150)  # Approx 150 WPM
                    for t, m in zip(transcription_times, valid_metrics)
                    if m.reference_words > 0
                ]
            )
            if transcription_times
            else 0
        ),
    )


@app.command("summary")
def summary():
    """Show summary table of all services."""
    console.print(f"\n[bold blue]ASR Evaluation Results Summary[/bold blue]\n")

    async def run():
        db = Database()
        await db.initialize()

        # Get sample and ground truth counts
        sample_count = await db.get_sample_count()
        gt_count = await db.get_ground_truth_count()

        console.print(f"Total samples: {sample_count}")
        console.print(f"Ground truth available: {gt_count}\n")

        # Build results table
        table = Table(title="Service Comparison")
        table.add_column("Service", style="cyan")
        table.add_column("Samples", justify="right")
        table.add_column("WER", justify="right")
        table.add_column("CER", justify="right")
        table.add_column("Semantic WER", justify="right")
        table.add_column("Gemini WER", justify="right")
        table.add_column("Avg Time (ms)", justify="right")
        table.add_column("P95 Time (ms)", justify="right")

        for service_name in ServiceName:
            agg = await compute_aggregate_metrics(service_name, db)

            if agg is None:
                table.add_row(
                    service_name.value,
                    "0",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                )
            else:
                semantic_wer_str = (
                    f"{agg.mean_semantic_wer:.1%}"
                    if agg.mean_semantic_wer is not None
                    else "-"
                )
                gemini_wer_str = (
                    f"{agg.mean_gemini_wer:.1%}"
                    if agg.mean_gemini_wer is not None
                    else "-"
                )
                table.add_row(
                    service_name.value,
                    str(agg.num_samples),
                    f"{agg.mean_wer:.1%}",
                    f"{agg.mean_cer:.1%}",
                    semantic_wer_str,
                    gemini_wer_str,
                    f"{agg.mean_time_to_transcription_ms:.0f}",
                    f"{agg.p95_time_to_transcription_ms:.0f}",
                )

        console.print(table)

    asyncio.run(run())


@app.command("details")
def details(
    service: str = typer.Argument(
        ...,
        help="Service name (nvidia, deepgram, cartesia, elevenlabs, whisper)",
    ),
):
    """Show detailed metrics for a specific service."""
    service_map = {
        "nvidia": ServiceName.NVIDIA_PARAKEET,
        "nvidia_parakeet": ServiceName.NVIDIA_PARAKEET,
        "deepgram": ServiceName.DEEPGRAM,
        "cartesia": ServiceName.CARTESIA,
        "elevenlabs": ServiceName.ELEVENLABS,
        "whisper": ServiceName.FASTER_WHISPER,
        "faster_whisper": ServiceName.FASTER_WHISPER,
    }

    service_name = service_map.get(service.lower())
    if not service_name:
        console.print(f"[red]Unknown service: {service}[/red]")
        sys.exit(1)

    console.print(f"\n[bold blue]Detailed Results: {service_name.value}[/bold blue]\n")

    async def run():
        db = Database()
        await db.initialize()

        agg = await compute_aggregate_metrics(service_name, db)

        if agg is None:
            console.print(f"[yellow]No results found for {service_name.value}[/yellow]")
            return

        # WER Statistics
        console.print("[bold]Word Error Rate (WER)[/bold]")
        console.print(f"  Mean:   {agg.mean_wer:.2%}")
        console.print(f"  Median: {agg.median_wer:.2%}")
        console.print(f"  Std:    {agg.std_wer:.2%}")
        console.print(f"  Min:    {agg.min_wer:.2%}")
        console.print(f"  Max:    {agg.max_wer:.2%}")

        # CER Statistics
        console.print(f"\n[bold]Character Error Rate (CER)[/bold]")
        console.print(f"  Mean:   {agg.mean_cer:.2%}")
        console.print(f"  Median: {agg.median_cer:.2%}")

        # Semantic WER
        if agg.mean_semantic_wer is not None:
            console.print(f"\n[bold]Semantic WER (Claude 4.5 Opus Judge)[/bold]")
            console.print(f"  Mean:   {agg.mean_semantic_wer:.2%}")
            console.print(f"  Median: {agg.median_semantic_wer:.2%}")

        # Gemini WER
        if agg.mean_gemini_wer is not None:
            console.print(f"\n[bold]Gemini WER (Gemini Flash Normalized)[/bold]")
            console.print(f"  Mean:   {agg.mean_gemini_wer:.2%}")
            console.print(f"  Median: {agg.median_gemini_wer:.2%}")

        # Timing
        console.print(f"\n[bold]Time to Transcription[/bold]")
        console.print(f"  Mean:   {agg.mean_time_to_transcription_ms:.0f} ms")
        console.print(f"  Median: {agg.median_time_to_transcription_ms:.0f} ms")
        console.print(f"  P95:    {agg.p95_time_to_transcription_ms:.0f} ms")

        # Sample count
        console.print(f"\n[bold]Samples[/bold]")
        console.print(f"  Total:  {agg.num_samples}")
        console.print(f"  Errors: {agg.num_errors}")

    asyncio.run(run())


@app.command("export")
def export(
    output: Path = typer.Option(
        Path("asr_eval_results.json"),
        "--output",
        "-o",
        help="Output file path",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format (json, csv)",
    ),
):
    """Export results to a file."""
    console.print(f"\n[bold blue]Exporting Results[/bold blue]\n")

    async def run():
        db = Database()
        await db.initialize()

        results = {}

        for service_name in ServiceName:
            agg = await compute_aggregate_metrics(service_name, db)

            if agg:
                results[service_name.value] = agg.model_dump()

        if format.lower() == "json":
            output.write_text(json.dumps(results, indent=2, default=str))
            console.print(f"Results exported to: {output}")

        elif format.lower() == "csv":
            import csv

            with open(output, "w", newline="") as f:
                if results:
                    first = list(results.values())[0]
                    writer = csv.DictWriter(f, fieldnames=first.keys())
                    writer.writeheader()
                    for service_data in results.values():
                        writer.writerow(service_data)
            console.print(f"Results exported to: {output}")

        else:
            console.print(f"[red]Unknown format: {format}[/red]")
            sys.exit(1)

    asyncio.run(run())


@app.command("errors")
def show_errors(
    service: str = typer.Argument(
        ...,
        help="Service name",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of worst samples to show",
    ),
):
    """Show samples with highest error rates."""
    service_map = {
        "nvidia": ServiceName.NVIDIA_PARAKEET,
        "deepgram": ServiceName.DEEPGRAM,
        "cartesia": ServiceName.CARTESIA,
        "elevenlabs": ServiceName.ELEVENLABS,
        "whisper": ServiceName.FASTER_WHISPER,
    }

    service_name = service_map.get(service.lower())
    if not service_name:
        console.print(f"[red]Unknown service: {service}[/red]")
        sys.exit(1)

    console.print(f"\n[bold blue]Worst Performing Samples: {service_name.value}[/bold blue]\n")

    async def run():
        db = Database()
        await db.initialize()

        metrics_list = await db.get_all_metrics_for_service(service_name)

        if not metrics_list:
            console.print("[yellow]No results found[/yellow]")
            return

        # Sort by WER descending
        sorted_metrics = sorted(metrics_list, key=lambda m: m.wer, reverse=True)

        table = Table(title=f"Top {limit} Highest WER Samples")
        table.add_column("Sample ID", style="cyan", max_width=20)
        table.add_column("WER", justify="right")
        table.add_column("S/D/I", justify="right")
        table.add_column("Semantic Errors", max_width=40)

        for m in sorted_metrics[:limit]:
            semantic_summary = ""
            if m.semantic_errors:
                semantic_summary = ", ".join(
                    [
                        f"{e.error_type}: {e.reference_word or ''} -> {e.hypothesis_word or ''}"
                        for e in m.semantic_errors[:3]
                    ]
                )
                if len(m.semantic_errors) > 3:
                    semantic_summary += f" (+{len(m.semantic_errors) - 3} more)"

            table.add_row(
                m.sample_id[:20],
                f"{m.wer:.1%}",
                f"{m.substitutions}/{m.deletions}/{m.insertions}",
                semantic_summary or "-",
            )

        console.print(table)

    asyncio.run(run())


if __name__ == "__main__":
    app()

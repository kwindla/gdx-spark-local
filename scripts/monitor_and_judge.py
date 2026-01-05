#!/usr/bin/env python3
"""Monitor transcription jobs and start judge processes when they complete."""

import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

# Job configuration: PID -> (service_name, log_file)
JOBS = {
    2979203: ("deepgram", "asr_eval_data/logs/deepgram_nova3_2026-01-03_23-18.log"),
    2979456: ("faster_whisper", "asr_eval_data/logs/deepgram_whisper_2026-01-03_23-18.log"),
    2979679: ("cartesia", "asr_eval_data/logs/cartesia_2026-01-03_23-18.log"),
    2980015: ("elevenlabs", "asr_eval_data/logs/elevenlabs_realtime_2026-01-03_23-18.log"),
}

CHECK_INTERVAL = 600  # 10 minutes
PROJECT_DIR = "/home/khkramer/src/gdx-spark-local"
DOCS_FILE = f"{PROJECT_DIR}/docs/agent-sdk-judge.md"

completed_jobs = set()
judge_pids = {}


def is_process_running(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_job_progress(log_file: str) -> str:
    """Get the latest progress from a log file."""
    try:
        result = subprocess.run(
            ["grep", "Progress:", log_file],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        if lines and lines[-1]:
            # Extract just the progress part
            line = lines[-1]
            if "[" in line and "]" in line:
                start = line.index("[")
                end = line.index("]") + 1
                return line[start:end]
        return "unknown"
    except Exception:
        return "unknown"


def start_judge(service_name: str) -> tuple[int, str]:
    """Start the judge process for a service. Returns (pid, log_file)."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_file = f"asr_eval_data/judge_run_{service_name}_{date_str}.log"

    cmd = f'PYTHONPATH="{PROJECT_DIR}" nohup uv run --extra asr-eval python scripts/run_judge.py --service {service_name} > {log_file} 2>&1 & echo $!'

    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_DIR
    )
    pid = int(result.stdout.strip())
    return pid, log_file


def update_docs(service_name: str, pid: int, log_file: str):
    """Add a note to the docs file about the new judge run."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    results_file = f"asr_eval_data/judge_results_{service_name}_{date_str}.json"

    entry = f"""
**{service_name}** (started {date_str}):
```
PID: {pid}
Log: {log_file}
Results: {results_file}
```
"""

    with open(DOCS_FILE, 'r') as f:
        content = f.read()

    # Find the "Current Evaluation Runs" section and append after the nvidia_parakeet entry
    marker = "Monitor progress:"
    if marker in content:
        parts = content.split(marker)
        # Insert before "Monitor progress:"
        new_content = parts[0] + entry + "\n" + marker + parts[1]

        with open(DOCS_FILE, 'w') as f:
            f.write(new_content)
        print(f"[DOCS] Updated {DOCS_FILE} with {service_name} judge info")


def main():
    print(f"=== Transcription Job Monitor ===")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Monitoring {len(JOBS)} jobs, checking every {CHECK_INTERVAL}s")
    print()

    for pid, (service, log) in JOBS.items():
        print(f"  PID {pid}: {service} -> {log}")
    print(flush=True)

    while len(completed_jobs) < len(JOBS):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking job status...")

        for pid, (service_name, log_file) in JOBS.items():
            if pid in completed_jobs:
                continue

            if is_process_running(pid):
                progress = get_job_progress(log_file)
                print(f"  {service_name}: {progress} (running)")
            else:
                print(f"  {service_name}: COMPLETED")
                completed_jobs.add(pid)

                # Start judge process
                print(f"  -> Starting judge for {service_name}...")
                judge_pid, judge_log = start_judge(service_name)
                judge_pids[service_name] = (judge_pid, judge_log)
                print(f"  -> Judge started: PID {judge_pid}, Log: {judge_log}")

                # Update docs
                update_docs(service_name, judge_pid, judge_log)

        print(flush=True)

        if len(completed_jobs) < len(JOBS):
            time.sleep(CHECK_INTERVAL)

    print(f"\n=== All transcription jobs completed ===")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"\nJudge processes started:")
    for service, (pid, log) in judge_pids.items():
        print(f"  {service}: PID {pid}, Log: {log}")


if __name__ == '__main__':
    main()

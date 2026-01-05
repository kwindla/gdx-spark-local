#!/usr/bin/env python3
"""Run Agent SDK judge evaluation on ASR transcriptions.

Usage:
    python scripts/run_judge.py --service nvidia_parakeet
    python scripts/run_judge.py --service deepgram
    python scripts/run_judge.py --service elevenlabs
    python scripts/run_judge.py --service cartesia
    python scripts/run_judge.py --service faster_whisper
"""

import argparse
import asyncio
import aiosqlite
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure API key is set
if not os.environ.get('ANTHROPIC_API_KEY'):
    raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

from asr_eval.evaluation.agent_sdk_judge import AgentSDKJudge


VALID_SERVICES = [
    'nvidia_parakeet',
    'deepgram',
    'elevenlabs',
    'cartesia',
    'faster_whisper',
]

GROUND_TRUTH_FILE = 'asr_eval_data/ground_truth_runs/2026-01-03_17-00-06.jsonl'


async def run_evaluation(service_name: str):
    start_time = time.time()
    date_str = datetime.now().strftime('%Y-%m-%d')

    print(f"=== Agent SDK Judge Run ({service_name}) ===")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Model: claude-sonnet-4-5-20250929")
    print()

    # Load ground truth from JSONL
    gt_data = {}
    with open(GROUND_TRUTH_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'sample':
                gt_data[data['sample_id']] = data['transcription']

    print(f"Loaded {len(gt_data)} ground truth samples")

    # Find samples with ground truth and non-empty transcriptions
    async with aiosqlite.connect('asr_eval_data/results.db') as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute('''
            SELECT t.sample_id, t.transcribed_text
            FROM transcriptions t
            WHERE t.service_name = ?
            AND t.transcribed_text != ""
        ''', (service_name,))
        rows = await cursor.fetchall()

    # Filter to samples with ground truth
    samples = []
    for row in rows:
        if row['sample_id'] in gt_data:
            samples.append({
                'sample_id': row['sample_id'],
                'ground_truth': gt_data[row['sample_id']],
                'transcription': row['transcribed_text']
            })

    print(f"Found {len(samples)} {service_name} samples with ground truth")
    print(flush=True)

    # Initialize judge
    judge = AgentSDKJudge()

    # Evaluate all samples
    results = []
    errors_count = 0

    for i, sample in enumerate(samples):
        try:
            result, trace = await judge.evaluate(
                sample['ground_truth'],
                sample['transcription']
            )
            results.append({
                'sample_id': sample['sample_id'],
                'wer': result['wer'],
                'substitutions': result['substitutions'],
                'deletions': result['deletions'],
                'insertions': result['insertions'],
                'reference_words': result['reference_words'],
            })
            print(f"[{i+1}/{len(samples)}] {sample['sample_id']}: WER={result['wer']:.2%}", flush=True)
        except Exception as e:
            errors_count += 1
            print(f"[{i+1}/{len(samples)}] {sample['sample_id']}: ERROR - {e}", flush=True)

        # Progress update every 100 samples
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(samples) - i - 1) / rate
            print(f"--- Progress: {i+1}/{len(samples)} ({(i+1)/len(samples)*100:.1f}%) | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m ---", flush=True)

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=== COMPLETE ===")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Samples evaluated: {len(results)}")
    print(f"Errors: {errors_count}")

    if results:
        # Naive average WER (for reference only)
        avg_wer = sum(r['wer'] for r in results) / len(results)
        zero_wer = sum(1 for r in results if r['wer'] == 0)

        # Pooled WER (correct aggregate metric)
        total_errors = sum(r['substitutions'] + r['deletions'] + r['insertions'] for r in results)
        total_ref_words = sum(r['reference_words'] for r in results)
        pooled_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

        print(f"Pooled WER: {pooled_wer:.2%} ({total_errors}/{total_ref_words} words)")
        print(f"Naive Avg WER: {avg_wer:.2%} (for reference only)")
        print(f"Perfect matches (0% WER): {zero_wer} ({zero_wer/len(results)*100:.1f}%)")

    # Save results to JSON
    output_file = f'asr_eval_data/judge_results_{service_name}_{date_str}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'service': service_name,
            'run_date': datetime.now().isoformat(),
            'model': 'claude-sonnet-4-5-20250929',
            'samples_evaluated': len(results),
            'errors': errors_count,
            'results': results
        }, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run Agent SDK judge evaluation')
    parser.add_argument('--service', '-s', required=True, choices=VALID_SERVICES,
                        help='ASR service to evaluate')
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.service))


if __name__ == '__main__':
    main()

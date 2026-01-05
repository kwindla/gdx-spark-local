#!/usr/bin/env python3
"""Import transcriptions from JSONL files into the database."""

import argparse
import asyncio
import json
import aiosqlite
from datetime import datetime
from pathlib import Path


# Map JSONL (service, model) to DB service_name
SERVICE_MAP = {
    ("deepgram", "nova-3"): "deepgram",
    ("deepgram", "whisper-large"): "faster_whisper",
    ("cartesia", "default"): "cartesia",
    ("elevenlabs", "scribe_v2_realtime"): "elevenlabs",
    ("nvidia_parakeet", "default"): "nvidia_parakeet",
}


async def import_jsonl(jsonl_path: str, db_path: str = "asr_eval_data/results.db"):
    """Import transcriptions from a JSONL file into the database."""

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = json.loads(lines[0])
    if header.get('type') != 'header':
        raise ValueError(f"Expected header, got: {header.get('type')}")

    service = header['service']
    model = header['model']
    started_at = header['started_at']

    # Get DB service_name
    db_service = SERVICE_MAP.get((service, model))
    if not db_service:
        raise ValueError(f"Unknown service/model: {service}/{model}")

    print(f"Importing: {jsonl_path}")
    print(f"  Service: {service}, Model: {model} -> DB: {db_service}")
    print(f"  Started: {started_at}")

    # Parse results
    results = []
    for line in lines[1:]:
        data = json.loads(line)
        if data.get('type') == 'result':
            results.append(data)

    print(f"  Results: {len(results)}")

    async with aiosqlite.connect(db_path) as db:
        # Update or insert transcriptions
        updated = 0
        inserted = 0

        for r in results:
            sample_id = r['sample_id']
            transcribed_text = r.get('transcribed_text', '')
            latency_ms = r.get('latency_ms', 0)
            audio_duration_ms = r.get('audio_duration_ms', 0)
            error = r.get('error')

            # Calculate RTF
            rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            # Check if exists
            cursor = await db.execute(
                "SELECT id FROM transcriptions WHERE sample_id = ? AND service_name = ?",
                (sample_id, db_service)
            )
            existing = await cursor.fetchone()

            timestamp = datetime.now().isoformat()

            if existing:
                # Update
                await db.execute('''
                    UPDATE transcriptions
                    SET transcribed_text = ?, time_to_transcription_ms = ?,
                        audio_duration_ms = ?, rtf = ?, timestamp = ?, error = ?
                    WHERE sample_id = ? AND service_name = ?
                ''', (transcribed_text, latency_ms, audio_duration_ms, rtf,
                      timestamp, error, sample_id, db_service))
                updated += 1
            else:
                # Insert
                await db.execute('''
                    INSERT INTO transcriptions
                    (sample_id, service_name, transcribed_text, time_to_transcription_ms,
                     audio_duration_ms, rtf, timestamp, error, adapter_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (sample_id, db_service, transcribed_text, latency_ms,
                      audio_duration_ms, rtf, timestamp, error, 'pipecat'))
                inserted += 1

        await db.commit()

    print(f"  Updated: {updated}, Inserted: {inserted}")
    return db_service, len(results)


async def main():
    parser = argparse.ArgumentParser(description='Import transcriptions from JSONL')
    parser.add_argument('files', nargs='+', help='JSONL files to import')
    args = parser.parse_args()

    for f in args.files:
        await import_jsonl(f)
        print()


if __name__ == '__main__':
    asyncio.run(main())

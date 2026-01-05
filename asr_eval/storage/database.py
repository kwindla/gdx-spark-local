"""SQLite storage layer for ASR evaluation results."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiosqlite

from asr_eval.config import get_config
from asr_eval.models import (
    AdapterType,
    AgentSDKTrace,
    AudioSample,
    EvaluationMetrics,
    GroundTruth,
    HumanReview,
    ReviewStatus,
    SemanticError,
    ServiceName,
    TranscriptionResult,
)

# SQL schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id TEXT PRIMARY KEY,
    audio_path TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    language TEXT NOT NULL DEFAULT 'eng',
    synthetic INTEGER NOT NULL DEFAULT 0,
    dataset_index INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ground_truth (
    sample_id TEXT PRIMARY KEY REFERENCES samples(sample_id),
    text TEXT NOT NULL,
    model_used TEXT NOT NULL DEFAULT 'gemini-2.5-flash',
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(sample_id),
    service_name TEXT NOT NULL,
    transcribed_text TEXT NOT NULL,
    time_to_transcription_ms REAL NOT NULL,
    audio_duration_ms REAL NOT NULL,
    rtf REAL NOT NULL,
    timestamp TEXT NOT NULL,
    error TEXT,
    adapter_type TEXT NOT NULL DEFAULT 'direct',
    UNIQUE(sample_id, service_name, adapter_type)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(sample_id),
    service_name TEXT NOT NULL,
    wer REAL NOT NULL,
    cer REAL NOT NULL,
    substitutions INTEGER NOT NULL,
    deletions INTEGER NOT NULL,
    insertions INTEGER NOT NULL,
    reference_words INTEGER NOT NULL,
    semantic_wer REAL,
    semantic_errors TEXT,  -- JSON array of SemanticError
    gemini_wer REAL,  -- Gemini Flash normalized WER
    agent_sdk_wer REAL,  -- Agent SDK multi-turn WER
    timestamp TEXT NOT NULL,
    UNIQUE(sample_id, service_name)
);

-- Full reasoning traces from Agent SDK multi-turn evaluation
CREATE TABLE IF NOT EXISTS agent_sdk_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(sample_id),
    service_name TEXT NOT NULL,
    session_id TEXT NOT NULL,
    conversation_trace TEXT NOT NULL,  -- JSON: full message history
    tool_calls TEXT NOT NULL,          -- JSON: all tool invocations
    normalized_reference TEXT,
    normalized_hypothesis TEXT,
    alignment TEXT,                    -- JSON: word-level alignment
    agent_sdk_wer REAL NOT NULL,
    substitutions INTEGER NOT NULL,
    deletions INTEGER NOT NULL,
    insertions INTEGER NOT NULL,
    reference_words INTEGER NOT NULL,
    errors TEXT,                       -- JSON: identified errors
    total_cost_usd REAL,
    duration_ms INTEGER,
    num_turns INTEGER NOT NULL DEFAULT 1,
    model_used TEXT NOT NULL DEFAULT 'claude-opus-4-5-20251101',
    timestamp TEXT NOT NULL,
    UNIQUE(sample_id, service_name)
);

-- Human review tracking for samples with disagreement
CREATE TABLE IF NOT EXISTS human_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(sample_id),
    service_name TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    flagged_reason TEXT,
    agent_sdk_wer REAL,
    gemini_wer REAL,
    wer_disagreement REAL,
    human_approved_wer REAL,
    human_notes TEXT,
    reviewed_by TEXT,
    reviewed_at TEXT,
    code_version TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(sample_id, service_name)
);

CREATE INDEX IF NOT EXISTS idx_transcriptions_service ON transcriptions(service_name);
CREATE INDEX IF NOT EXISTS idx_metrics_service ON metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_agent_sdk_traces_service ON agent_sdk_traces(service_name);
CREATE INDEX IF NOT EXISTS idx_human_reviews_status ON human_reviews(review_status);
"""


class Database:
    """Async SQLite database for ASR evaluation results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_config().results_db

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a database connection."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def initialize(self) -> None:
        """Create database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self.connection() as db:
            await db.executescript(SCHEMA)
            await db.commit()

    # Sample operations
    async def insert_sample(self, sample: AudioSample) -> None:
        """Insert an audio sample."""
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO samples
                (sample_id, audio_path, duration_seconds, language, synthetic, dataset_index)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.sample_id,
                    sample.audio_path,
                    sample.duration_seconds,
                    sample.language,
                    int(sample.synthetic),
                    sample.dataset_index,
                ),
            )
            await db.commit()

    async def insert_samples_batch(self, samples: list[AudioSample]) -> None:
        """Insert multiple audio samples."""
        async with self.connection() as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO samples
                (sample_id, audio_path, duration_seconds, language, synthetic, dataset_index)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.sample_id,
                        s.audio_path,
                        s.duration_seconds,
                        s.language,
                        int(s.synthetic),
                        s.dataset_index,
                    )
                    for s in samples
                ],
            )
            await db.commit()

    async def get_sample(self, sample_id: str) -> Optional[AudioSample]:
        """Get a sample by ID."""
        async with self.connection() as db:
            cursor = await db.execute(
                "SELECT * FROM samples WHERE sample_id = ?", (sample_id,)
            )
            row = await cursor.fetchone()
            if row:
                return AudioSample(
                    sample_id=row["sample_id"],
                    audio_path=row["audio_path"],
                    duration_seconds=row["duration_seconds"],
                    language=row["language"],
                    synthetic=bool(row["synthetic"]),
                    dataset_index=row["dataset_index"],
                )
            return None

    async def get_all_samples(self) -> list[AudioSample]:
        """Get all samples."""
        async with self.connection() as db:
            cursor = await db.execute("SELECT * FROM samples ORDER BY dataset_index")
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

    async def get_sample_count(self) -> int:
        """Get total number of samples."""
        async with self.connection() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM samples")
            row = await cursor.fetchone()
            return row[0] if row else 0

    # Ground truth operations
    async def insert_ground_truth(self, gt: GroundTruth) -> None:
        """Insert a ground truth transcription."""
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO ground_truth
                (sample_id, text, model_used, generated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    gt.sample_id,
                    gt.text,
                    gt.model_used,
                    gt.generated_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_ground_truth(self, sample_id: str) -> Optional[GroundTruth]:
        """Get ground truth for a sample."""
        async with self.connection() as db:
            cursor = await db.execute(
                "SELECT * FROM ground_truth WHERE sample_id = ?", (sample_id,)
            )
            row = await cursor.fetchone()
            if row:
                from datetime import datetime

                return GroundTruth(
                    sample_id=row["sample_id"],
                    text=row["text"],
                    model_used=row["model_used"],
                    generated_at=datetime.fromisoformat(row["generated_at"]),
                )
            return None

    async def get_samples_without_ground_truth(self) -> list[AudioSample]:
        """Get samples that don't have ground truth yet."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                SELECT s.* FROM samples s
                LEFT JOIN ground_truth gt ON s.sample_id = gt.sample_id
                WHERE gt.sample_id IS NULL
                ORDER BY s.dataset_index
                """
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

    async def get_ground_truth_count(self) -> int:
        """Get number of ground truth entries."""
        async with self.connection() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM ground_truth")
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear_all_ground_truths(self) -> int:
        """Delete all ground truth records. Returns count of deleted records."""
        async with self.connection() as db:
            cursor = await db.execute("DELETE FROM ground_truth")
            await db.commit()
            return cursor.rowcount

    # Transcription operations
    async def insert_transcription(self, result: TranscriptionResult) -> None:
        """Insert a transcription result."""
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO transcriptions
                (sample_id, service_name, transcribed_text, time_to_transcription_ms,
                 audio_duration_ms, rtf, timestamp, error, adapter_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.sample_id,
                    result.service_name.value,
                    result.transcribed_text,
                    result.time_to_transcription_ms,
                    result.audio_duration_ms,
                    result.rtf,
                    result.timestamp.isoformat(),
                    result.error,
                    result.adapter_type.value,
                ),
            )
            await db.commit()

    async def get_transcription(
        self,
        sample_id: str,
        service_name: ServiceName,
        adapter_type: AdapterType = AdapterType.DIRECT,
    ) -> Optional[TranscriptionResult]:
        """Get transcription for a sample, service, and adapter type."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM transcriptions
                WHERE sample_id = ? AND service_name = ? AND adapter_type = ?
                """,
                (sample_id, service_name.value, adapter_type.value),
            )
            row = await cursor.fetchone()
            if row:
                from datetime import datetime

                # Handle legacy rows without adapter_type column
                row_adapter_type = (
                    row["adapter_type"] if "adapter_type" in row.keys() else "direct"
                )
                return TranscriptionResult(
                    sample_id=row["sample_id"],
                    service_name=ServiceName(row["service_name"]),
                    transcribed_text=row["transcribed_text"],
                    time_to_transcription_ms=row["time_to_transcription_ms"],
                    audio_duration_ms=row["audio_duration_ms"],
                    rtf=row["rtf"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    error=row["error"],
                    adapter_type=AdapterType(row_adapter_type),
                )
            return None

    async def get_samples_without_transcription(
        self,
        service_name: ServiceName,
        adapter_type: AdapterType = AdapterType.DIRECT,
    ) -> list[AudioSample]:
        """Get samples that haven't been transcribed by a service and adapter type."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                SELECT s.* FROM samples s
                LEFT JOIN transcriptions t ON s.sample_id = t.sample_id
                    AND t.service_name = ? AND t.adapter_type = ?
                WHERE t.sample_id IS NULL
                ORDER BY s.dataset_index
                """,
                (service_name.value, adapter_type.value),
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

    async def get_transcription_count(self, service_name: ServiceName) -> int:
        """Get number of transcriptions for a service."""
        async with self.connection() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM transcriptions WHERE service_name = ?",
                (service_name.value,),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    # Metrics operations
    async def insert_metrics(self, metrics: EvaluationMetrics) -> None:
        """Insert evaluation metrics."""
        semantic_errors_json = None
        if metrics.semantic_errors:
            semantic_errors_json = json.dumps(
                [e.model_dump() for e in metrics.semantic_errors]
            )

        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO metrics
                (sample_id, service_name, wer, cer, substitutions, deletions,
                 insertions, reference_words, semantic_wer, semantic_errors,
                 gemini_wer, agent_sdk_wer, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.sample_id,
                    metrics.service_name.value,
                    metrics.wer,
                    metrics.cer,
                    metrics.substitutions,
                    metrics.deletions,
                    metrics.insertions,
                    metrics.reference_words,
                    metrics.semantic_wer,
                    semantic_errors_json,
                    metrics.gemini_wer,
                    metrics.agent_sdk_wer,
                    metrics.timestamp.isoformat(),
                ),
            )
            await db.commit()

    async def get_metrics(
        self, sample_id: str, service_name: ServiceName
    ) -> Optional[EvaluationMetrics]:
        """Get metrics for a sample and service."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM metrics
                WHERE sample_id = ? AND service_name = ?
                """,
                (sample_id, service_name.value),
            )
            row = await cursor.fetchone()
            if row:
                from datetime import datetime

                semantic_errors = None
                if row["semantic_errors"]:
                    semantic_errors = [
                        SemanticError(**e) for e in json.loads(row["semantic_errors"])
                    ]

                # Handle legacy rows without gemini_wer or agent_sdk_wer columns
                gemini_wer = row["gemini_wer"] if "gemini_wer" in row.keys() else None
                agent_sdk_wer = row["agent_sdk_wer"] if "agent_sdk_wer" in row.keys() else None

                return EvaluationMetrics(
                    sample_id=row["sample_id"],
                    service_name=ServiceName(row["service_name"]),
                    wer=row["wer"],
                    cer=row["cer"],
                    substitutions=row["substitutions"],
                    deletions=row["deletions"],
                    insertions=row["insertions"],
                    reference_words=row["reference_words"],
                    semantic_wer=row["semantic_wer"],
                    semantic_errors=semantic_errors,
                    gemini_wer=gemini_wer,
                    agent_sdk_wer=agent_sdk_wer,
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
            return None

    async def get_all_metrics_for_service(
        self, service_name: ServiceName
    ) -> list[EvaluationMetrics]:
        """Get all metrics for a service."""
        async with self.connection() as db:
            cursor = await db.execute(
                "SELECT * FROM metrics WHERE service_name = ?",
                (service_name.value,),
            )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                from datetime import datetime

                semantic_errors = None
                if row["semantic_errors"]:
                    semantic_errors = [
                        SemanticError(**e) for e in json.loads(row["semantic_errors"])
                    ]

                # Handle legacy rows without gemini_wer or agent_sdk_wer columns
                gemini_wer = row["gemini_wer"] if "gemini_wer" in row.keys() else None
                agent_sdk_wer = row["agent_sdk_wer"] if "agent_sdk_wer" in row.keys() else None

                results.append(
                    EvaluationMetrics(
                        sample_id=row["sample_id"],
                        service_name=ServiceName(row["service_name"]),
                        wer=row["wer"],
                        cer=row["cer"],
                        substitutions=row["substitutions"],
                        deletions=row["deletions"],
                        insertions=row["insertions"],
                        reference_words=row["reference_words"],
                        semantic_wer=row["semantic_wer"],
                        semantic_errors=semantic_errors,
                        gemini_wer=gemini_wer,
                        agent_sdk_wer=agent_sdk_wer,
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )
            return results

    # Agent SDK trace operations
    async def insert_agent_sdk_trace(self, trace: AgentSDKTrace) -> None:
        """Insert an Agent SDK evaluation trace."""
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO agent_sdk_traces
                (sample_id, service_name, session_id, conversation_trace, tool_calls,
                 normalized_reference, normalized_hypothesis, alignment,
                 agent_sdk_wer, substitutions, deletions, insertions, reference_words,
                 errors, total_cost_usd, duration_ms, num_turns, model_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.sample_id,
                    trace.service_name.value,
                    trace.session_id,
                    json.dumps(trace.conversation_trace),
                    json.dumps(trace.tool_calls),
                    trace.normalized_reference,
                    trace.normalized_hypothesis,
                    json.dumps(trace.alignment) if trace.alignment else None,
                    trace.agent_sdk_wer,
                    trace.substitutions,
                    trace.deletions,
                    trace.insertions,
                    trace.reference_words,
                    json.dumps(trace.errors) if trace.errors else None,
                    trace.total_cost_usd,
                    trace.duration_ms,
                    trace.num_turns,
                    trace.model_used,
                    trace.timestamp.isoformat(),
                ),
            )
            await db.commit()

    async def get_agent_sdk_trace(
        self, sample_id: str, service_name: ServiceName
    ) -> Optional[AgentSDKTrace]:
        """Get Agent SDK trace for a sample and service."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM agent_sdk_traces
                WHERE sample_id = ? AND service_name = ?
                """,
                (sample_id, service_name.value),
            )
            row = await cursor.fetchone()
            if row:
                from datetime import datetime

                return AgentSDKTrace(
                    sample_id=row["sample_id"],
                    service_name=ServiceName(row["service_name"]),
                    session_id=row["session_id"],
                    conversation_trace=json.loads(row["conversation_trace"]),
                    tool_calls=json.loads(row["tool_calls"]),
                    normalized_reference=row["normalized_reference"],
                    normalized_hypothesis=row["normalized_hypothesis"],
                    alignment=json.loads(row["alignment"]) if row["alignment"] else None,
                    agent_sdk_wer=row["agent_sdk_wer"],
                    substitutions=row["substitutions"],
                    deletions=row["deletions"],
                    insertions=row["insertions"],
                    reference_words=row["reference_words"],
                    errors=json.loads(row["errors"]) if row["errors"] else None,
                    total_cost_usd=row["total_cost_usd"],
                    duration_ms=row["duration_ms"],
                    num_turns=row["num_turns"],
                    model_used=row["model_used"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
            return None

    # Human review operations
    async def insert_human_review(self, review: HumanReview) -> None:
        """Insert or update a human review record."""
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO human_reviews
                (sample_id, service_name, review_status, flagged_reason,
                 agent_sdk_wer, gemini_wer, wer_disagreement,
                 human_approved_wer, human_notes, reviewed_by, reviewed_at,
                 code_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.sample_id,
                    review.service_name.value,
                    review.review_status.value,
                    review.flagged_reason,
                    review.agent_sdk_wer,
                    review.gemini_wer,
                    review.wer_disagreement,
                    review.human_approved_wer,
                    review.human_notes,
                    review.reviewed_by,
                    review.reviewed_at.isoformat() if review.reviewed_at else None,
                    review.code_version,
                    review.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_human_review(
        self, sample_id: str, service_name: ServiceName
    ) -> Optional[HumanReview]:
        """Get human review for a sample and service."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM human_reviews
                WHERE sample_id = ? AND service_name = ?
                """,
                (sample_id, service_name.value),
            )
            row = await cursor.fetchone()
            if row:
                from datetime import datetime

                return HumanReview(
                    sample_id=row["sample_id"],
                    service_name=ServiceName(row["service_name"]),
                    review_status=ReviewStatus(row["review_status"]),
                    flagged_reason=row["flagged_reason"],
                    agent_sdk_wer=row["agent_sdk_wer"],
                    gemini_wer=row["gemini_wer"],
                    wer_disagreement=row["wer_disagreement"],
                    human_approved_wer=row["human_approved_wer"],
                    human_notes=row["human_notes"],
                    reviewed_by=row["reviewed_by"],
                    reviewed_at=(
                        datetime.fromisoformat(row["reviewed_at"])
                        if row["reviewed_at"]
                        else None
                    ),
                    code_version=row["code_version"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            return None

    async def get_pending_reviews(
        self, service_name: Optional[ServiceName] = None
    ) -> list[HumanReview]:
        """Get all pending human reviews, optionally filtered by service."""
        async with self.connection() as db:
            if service_name:
                cursor = await db.execute(
                    """
                    SELECT * FROM human_reviews
                    WHERE review_status = 'pending' AND service_name = ?
                    ORDER BY sample_id
                    """,
                    (service_name.value,),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM human_reviews
                    WHERE review_status = 'pending'
                    ORDER BY sample_id
                    """
                )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                from datetime import datetime

                results.append(
                    HumanReview(
                        sample_id=row["sample_id"],
                        service_name=ServiceName(row["service_name"]),
                        review_status=ReviewStatus(row["review_status"]),
                        flagged_reason=row["flagged_reason"],
                        agent_sdk_wer=row["agent_sdk_wer"],
                        gemini_wer=row["gemini_wer"],
                        wer_disagreement=row["wer_disagreement"],
                        human_approved_wer=row["human_approved_wer"],
                        human_notes=row["human_notes"],
                        reviewed_by=row["reviewed_by"],
                        reviewed_at=(
                            datetime.fromisoformat(row["reviewed_at"])
                            if row["reviewed_at"]
                            else None
                        ),
                        code_version=row["code_version"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return results

    async def reset_all_reviews(self) -> int:
        """Reset all human reviews to pending status. Returns count of reset reviews."""
        async with self.connection() as db:
            cursor = await db.execute(
                """
                UPDATE human_reviews
                SET review_status = 'pending',
                    human_approved_wer = NULL,
                    human_notes = NULL,
                    reviewed_by = NULL,
                    reviewed_at = NULL
                WHERE review_status != 'pending'
                """
            )
            await db.commit()
            return cursor.rowcount

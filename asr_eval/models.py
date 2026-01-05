"""Pydantic data models for ASR evaluation."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utcnow() -> datetime:
    """Return current UTC datetime (avoids deprecation warning)."""
    return datetime.now(timezone.utc)

from pydantic import BaseModel, Field


class ServiceName(str, Enum):
    """Supported STT services."""

    NVIDIA_PARAKEET = "nvidia_parakeet"
    DEEPGRAM = "deepgram"
    CARTESIA = "cartesia"
    ELEVENLABS = "elevenlabs"
    FASTER_WHISPER = "faster_whisper"
    SPEECHMATICS = "speechmatics"
    SONIOX = "soniox"


class AdapterType(str, Enum):
    """Type of adapter used for transcription."""

    PIPECAT_STREAMING = "pipecat_streaming"
    PIPECAT_BATCH = "pipecat_batch"
    DIRECT = "direct"


class AudioSample(BaseModel):
    """A single audio sample from the dataset."""

    sample_id: str = Field(description="Unique identifier for the sample")
    audio_path: str = Field(description="Local path to the PCM audio file")
    duration_seconds: float = Field(description="Audio duration in seconds")
    language: str = Field(default="eng", description="Language code")
    synthetic: bool = Field(default=False, description="Whether sample is synthetic")
    dataset_index: int = Field(description="Original index in HuggingFace dataset")


class GroundTruth(BaseModel):
    """Ground truth transcription from Gemini."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    text: str = Field(description="Ground truth transcription text")
    model_used: str = Field(
        default="gemini-2.5-flash", description="Model used for transcription"
    )
    generated_at: datetime = Field(default_factory=_utcnow)


class TranscriptionResult(BaseModel):
    """Result from an STT service transcription."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service used")
    transcribed_text: str = Field(description="Transcribed text from the service")
    time_to_transcription_ms: float = Field(
        description="Time from turn-finished signal to final TranscriptionFrame"
    )
    audio_duration_ms: float = Field(description="Duration of the audio in ms")
    timestamp: datetime = Field(default_factory=_utcnow)
    error: Optional[str] = Field(default=None, description="Error message if failed")
    adapter_type: AdapterType = Field(
        default=AdapterType.DIRECT,
        description="Type of adapter used: direct, pipecat_streaming, or pipecat_batch",
    )


class SemanticError(BaseModel):
    """A semantically meaningful transcription error."""

    error_type: str = Field(
        description="Type of error: substitution, deletion, insertion"
    )
    reference_word: Optional[str] = Field(
        default=None, description="Word from ground truth (for substitution/deletion)"
    )
    hypothesis_word: Optional[str] = Field(
        default=None, description="Word from transcription (for substitution/insertion)"
    )
    context: Optional[str] = Field(
        default=None, description="Surrounding context for the error"
    )
    severity: str = Field(
        default="minor",
        description="Severity: minor (typo), moderate (meaning change), major (critical error)",
    )


class EvaluationMetrics(BaseModel):
    """Computed metrics for a transcription."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service evaluated")

    # Raw WER/CER (from jiwer, no normalization)
    wer: float = Field(description="Word Error Rate (0-1+)")
    cer: float = Field(description="Character Error Rate (0-1+)")
    substitutions: int = Field(description="Number of word substitutions")
    deletions: int = Field(description="Number of word deletions")
    insertions: int = Field(description="Number of word insertions")
    reference_words: int = Field(description="Total words in reference")

    # Semantic evaluation (from Claude 4.5 Opus)
    semantic_wer: Optional[float] = Field(
        default=None, description="Semantic WER (meaningful errors only)"
    )
    semantic_errors: Optional[list[SemanticError]] = Field(
        default=None, description="List of semantic errors"
    )

    # Gemini Flash evaluation (natural language normalization)
    gemini_wer: Optional[float] = Field(
        default=None, description="Gemini Flash normalized WER"
    )

    # Agent SDK evaluation (multi-turn reasoning with verification)
    agent_sdk_wer: Optional[float] = Field(
        default=None, description="Agent SDK multi-turn WER with reasoning"
    )

    timestamp: datetime = Field(default_factory=_utcnow)


class AggregateMetrics(BaseModel):
    """Aggregate metrics for a service across all samples."""

    service_name: ServiceName
    num_samples: int
    num_errors: int = Field(description="Number of samples with errors")

    # WER statistics
    mean_wer: float
    median_wer: float
    std_wer: float
    min_wer: float
    max_wer: float

    # CER statistics
    mean_cer: float
    median_cer: float

    # Semantic WER statistics
    mean_semantic_wer: Optional[float] = None
    median_semantic_wer: Optional[float] = None

    # Gemini WER statistics
    mean_gemini_wer: Optional[float] = None
    median_gemini_wer: Optional[float] = None

    # Timing statistics
    mean_time_to_transcription_ms: float
    median_time_to_transcription_ms: float
    p95_time_to_transcription_ms: float

    # Agent SDK WER statistics
    mean_agent_sdk_wer: Optional[float] = None
    median_agent_sdk_wer: Optional[float] = None

    # Pooled WER (sum of errors / sum of reference words)
    pooled_wer: Optional[float] = None
    pooled_gemini_wer: Optional[float] = None
    pooled_agent_sdk_wer: Optional[float] = None


class ReviewStatus(str, Enum):
    """Status of human review."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AgentSDKTrace(BaseModel):
    """Full reasoning trace from Agent SDK multi-turn evaluation."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service evaluated")
    session_id: str = Field(description="Unique session identifier")

    # Full conversation trace
    conversation_trace: list[dict] = Field(
        description="Full message history from the evaluation"
    )
    tool_calls: list[dict] = Field(description="All tool invocations during evaluation")

    # Normalized texts (as determined by the LLM)
    normalized_reference: Optional[str] = Field(
        default=None, description="LLM-normalized reference text"
    )
    normalized_hypothesis: Optional[str] = Field(
        default=None, description="LLM-normalized hypothesis text"
    )
    alignment: Optional[list[dict]] = Field(
        default=None, description="Word-level alignment"
    )

    # WER calculation results
    agent_sdk_wer: float = Field(description="Calculated WER")
    substitutions: int = Field(description="Number of substitutions")
    deletions: int = Field(description="Number of deletions")
    insertions: int = Field(description="Number of insertions")
    reference_words: int = Field(description="Total reference words after normalization")
    errors: Optional[list[dict]] = Field(default=None, description="Identified errors")

    # Performance metrics
    total_cost_usd: Optional[float] = Field(default=None, description="API cost in USD")
    duration_ms: Optional[int] = Field(default=None, description="Total evaluation time")
    num_turns: int = Field(default=1, description="Number of conversation turns")
    model_used: str = Field(
        default="claude-opus-4-5-20251101", description="Model used for evaluation"
    )

    timestamp: datetime = Field(default_factory=_utcnow)


class HumanReview(BaseModel):
    """Human review tracking for samples with disagreement."""

    sample_id: str = Field(description="Reference to AudioSample.sample_id")
    service_name: ServiceName = Field(description="STT service")
    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING, description="Current review status"
    )

    # Reason for flagging
    flagged_reason: Optional[str] = Field(
        default=None, description="Why the sample was flagged"
    )

    # WER values at time of flagging
    agent_sdk_wer: Optional[float] = Field(default=None)
    gemini_wer: Optional[float] = Field(default=None)
    wer_disagreement: Optional[float] = Field(
        default=None, description="Absolute difference between WER values"
    )

    # Human decision
    human_approved_wer: Optional[float] = Field(
        default=None, description="Human-approved WER value"
    )
    human_notes: Optional[str] = Field(default=None, description="Reviewer notes")
    reviewed_by: Optional[str] = Field(default=None, description="Reviewer identifier")
    reviewed_at: Optional[datetime] = Field(default=None)

    # Version tracking for state reset detection
    code_version: Optional[str] = Field(
        default=None, description="Code version hash when flagged"
    )

    created_at: datetime = Field(default_factory=_utcnow)

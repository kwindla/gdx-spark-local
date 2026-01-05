"""Pipecat STT adapters for ASR evaluation."""

from asr_eval.adapters.pipeline_stt_adapter import (
    PipelineSTTAdapter,
    create_stt_service,
    transcribe_with_pipeline,
)
from asr_eval.adapters.synthetic_input_transport import SyntheticInputTransport
from asr_eval.adapters.transcription_collector import (
    CollectedResult,
    TranscriptionCollector,
    TranscriptionTiming,
)
from asr_eval.adapters.universal_pipecat import (
    EvaluationMode,
    PipecatServiceConfig,
    UniversalPipecatAdapter,
)

__all__ = [
    # New Pipeline-based adapter
    "PipelineSTTAdapter",
    "create_stt_service",
    "transcribe_with_pipeline",
    # Components
    "SyntheticInputTransport",
    "TranscriptionCollector",
    "TranscriptionTiming",
    "CollectedResult",
    # Legacy adapter (kept for reference)
    "EvaluationMode",
    "PipecatServiceConfig",
    "UniversalPipecatAdapter",
]

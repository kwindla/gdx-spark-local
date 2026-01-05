"""Configuration management for ASR evaluation system."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ASREvalConfig(BaseSettings):
    """Configuration for ASR evaluation system."""

    model_config = SettingsConfigDict(
        env_prefix="ASR_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("./asr_eval_data"))
    cache_dir: Path = Field(default=Path("./asr_eval_data/cache"))
    audio_dir: Path = Field(default=Path("./asr_eval_data/audio"))
    results_db: Path = Field(default=Path("./asr_eval_data/results.db"))

    # Dataset configuration
    dataset_name: str = "pipecat-ai/smart-turn-data-v3.1-train"
    num_samples: int = 1000
    seed: int = 42

    # API Keys (loaded from environment)
    deepgram_api_key: str = Field(default="", alias="DEEPGRAM_API_KEY")
    cartesia_api_key: str = Field(default="", alias="CARTESIA_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    speechmatics_api_key: str = Field(default="", alias="SPEECHMATICS_API_KEY")
    soniox_api_key: str = Field(default="", alias="SONIOX_API_KEY")

    # Service endpoints
    nvidia_asr_url: str = Field(
        default="ws://localhost:8080", alias="NVIDIA_ASR_URL"
    )

    # Audio configuration
    sample_rate: int = 16000
    chunk_duration_ms: int = 20
    simulate_realtime: bool = True

    # Rate limits
    gemini_requests_per_minute: int = 60
    deepgram_concurrent: int = 10
    deepgram_requests_per_minute: int = 100
    cartesia_concurrent: int = 5
    elevenlabs_concurrent: int = 5

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes for streaming."""
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        return samples_per_chunk * 2  # 16-bit audio = 2 bytes per sample


# Global configuration instance
_config: Optional[ASREvalConfig] = None


def get_config() -> ASREvalConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = ASREvalConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None

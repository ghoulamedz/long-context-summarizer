"""Type definitions and dataclasses for the long context summarizer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np


class Status(Enum):
    """Processing status for documents."""

    PENDING = "pending"
    LOADING = "loading"
    CHUNKING = "chunking"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentResult:
    """Result of processing a single document."""

    id: str
    status: Status
    document: str | None = None
    summary: str | None = None
    extractions: list[ExtractionResult] | None = None
    error: str | None = None
    chunks_used: int = 0
    tokens_used: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["end_time"] = self.end_time
        result["duration"] = (
            self.end_time - self.start_time
            if self.end_time and self.start_time
            else None
        )
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SummaryResult:
    """Result of summarizing a document."""

    document_id: str
    original_length: int
    summary_length: int
    ratio: float
    summary: str
    chunks: int
    time_seconds: float
    tokens_generated: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Result of extracting entities from a document."""

    document_id: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    time_seconds: float
    tokens_used: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingStats:
    """Aggregate statistics for a batch of processing."""

    total_documents: int
    successful: int
    failed: int
    total_time_seconds: float
    avg_time_per_doc: float
    total_tokens_used: int
    total_chunks_used: int
    avg_tokens_per_doc: float
    avg_chunks_per_doc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkInfo:
    """Information about a text chunk."""

    index: int
    content: str
    start_position: int
    end_position: int
    overlap: int = 0
    is_last: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for the underlying LLM model."""

    model_path: str
    context_limit: int = 60000
    n_predict: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    n_threads: int = 4
    n_gpu_layers: int = -1
    seed: int = 42
    n_ctx_window: int | None = None
    n_ctx_window_set: bool = False
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None
    rope_scaling: str | None = None
    rms_norm_eps: float = 1e-5
    causal_attn: bool = True
    flash_attn: bool = False
    low_vram: bool = False

    @property
    def effective_context(self) -> int:
        """Get the effective context limit (min of model and config)."""
        return min(self.context_limit, getattr(self, "_effective_context", self.context_limit))

    @property
    def effective_predict(self) -> int:
        """Get the effective prediction limit."""
        return min(self.n_predict, getattr(self, "_effective_predict", self.n_predict))

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Create a ModelConfig from environment variables."""
        return cls(
            model_path=str(Path.home() / ".cache" / "huggingface" / "hub" / "models" / "default.gguf"),
            context_limit=int(__import__("os").environ.get("PI_CONTEXT_LIMIT", "60000")),
            n_predict=int(__import__("os").environ.get("N_PREDICT", "4096")),
        )


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    max_documents: int = 100
    max_size_mb: int = 1000
    chunk_size: int = 2000
    chunk_overlap: int = 200
    max_chunks_per_doc: int = 100

    @property
    def max_chunks(self) -> int:
        """Calculate maximum chunks based on size limit."""
        max_chars = self.max_size_mb * 1024 * 1024
        return max(1, max_chars // (self.chunk_size - self.chunk_overlap))

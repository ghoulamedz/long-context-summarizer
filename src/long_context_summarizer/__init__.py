"""Long Context Summarizer - A 60k context LLM tool for document processing."""

from .core import LongContextEngine, Summarizer, Extractor
from .engine import build_engine
from .document import Document, DocumentBatch
from .strategies import (
    SummaryStrategy,
    ExtractStrategy,
    ChunkStrategy,
    MergeStrategy,
    ChainOfDistillationStrategy,
    MapReduceStrategy,
)
from .types import (
    DocumentResult,
    SummaryResult,
    ExtractionResult,
    ProcessingStats,
)

__version__ = "0.1.0"
__all__ = [
    "LongContextEngine",
    "build_engine",
    "Summarizer",
    "Extractor",
    "Document",
    "DocumentBatch",
    "SummaryStrategy",
    "ExtractStrategy",
    "ChunkStrategy",
    "MergeStrategy",
    "ChainOfDistillationStrategy",
    "MapReduceStrategy",
    "DocumentResult",
    "SummaryResult",
    "ExtractionResult",
    "ProcessingStats",
]

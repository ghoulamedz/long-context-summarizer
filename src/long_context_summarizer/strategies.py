"""Processing strategies for document summarization and extraction."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import LongContextEngine


@dataclass
class ProcessingResult:
    """Result of applying a strategy to a document."""

    success: bool
    output: str | dict | list | None = None
    error: str | None = None
    time_seconds: float = 0.0
    tokens_used: int = 0
    steps: list[dict] = field(default_factory=list)


class Strategy(ABC):
    """Base class for processing strategies."""

    @abstractmethod
    def apply(self, engine: LongContextEngine, content: str, **kwargs) -> ProcessingResult:
        """Apply the strategy to content."""
        pass


@dataclass
class SummaryStrategy:
    """
    Summarization strategy using direct prompting.

    Uses a single-shot or few-shot prompt to generate a summary.
    """

    max_length: int = 500
    format: str = "markdown"
    instructions: str = "Summarize the following text in a clear, concise manner. "
    "Focus on key points, arguments, and conclusions. "
    "Use markdown for structure."
    use_few_shot: bool = False
    few_shot_examples: list[str] = field(default_factory=list)

    def apply(self, engine: LongContextEngine, content: str, **kwargs) -> ProcessingResult:
        """Apply summarization."""
        result = ProcessingResult(success=True, time_seconds=time.time())

        try:
            prompt = self._build_prompt(content)
            result.output = engine._call_llm(prompt)
            result.tokens_used = len(prompt) + len(result.output) if result.output else 0
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.time_seconds = time.time() - result.time_seconds
        return result

    def _build_prompt(self, content: str) -> str:
        """Build the summary prompt."""
        examples = "\n".join(self.few_shot_examples) if self.few_shot_examples else ""

        if examples:
            prompt = (
                f"You are an expert summarizer.\n\n"
                f"Examples:\n{examples}\n\n"
                f"---\n\n"
                f"SUMMARIZE THIS:\n{content}"
            )
        else:
            prompt = (
                f"You are an expert summarizer.\n\n"
                f"INSTRUCTIONS:\n{self.instructions}\n\n"
                f"Max length: {self.max_length} tokens\n"
                f"Format: {self.format}\n\n"
                f"CONTENT:\n{content}\n\n"
                f"SUMMARY:"
            )
        return prompt


@dataclass
class ExtractStrategy:
    """
    Entity and relationship extraction strategy.

    Extracts named entities, their types, and relationships between them.
    """

    entity_types: list[str] = field(default_factory=lambda: [
        "person", "organization", "location", "date", "money",
        "product", "event", "concept"
    ])
    max_entities: int = 100
    instructions: str = "Extract all named entities and their relationships "
    "from the text. Return JSON format."

    def apply(self, engine: LongContextEngine, content: str, **kwargs) -> ProcessingResult:
        """Apply extraction."""
        result = ProcessingResult(success=True, time_seconds=time.time())

        try:
            prompt = self._build_prompt(content)
            response = engine._call_llm(prompt)

            # Parse the response
            try:
                import json
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    result.output = json.loads(response[start:end])
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

            # Fallback: parse as text
            result.output = response
            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)

        result.time_seconds = time.time() - result.time_seconds
        return result

    def _build_prompt(self, content: str) -> str:
        """Build the extraction prompt."""
        prompt = (
            f"You are an expert entity extractor.\n\n"
            f"INSTRUCTIONS:\n{self.instructions}\n\n"
            f"Entity types: {', '.join(self.entity_types)}\n\n"
            f"CONTENT:\n{content}\n\n"
            f"JSON:\n{{"
        )
        return prompt


@dataclass
class ChunkStrategy:
    """
    Chunking strategy for splitting long documents.

    Supports various chunking methods: fixed-size, semantic, recursive.
    """

    chunk_size: int = 2000
    chunk_overlap: int = 200
    method: str = "fixed"  # fixed, semantic, recursive
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", " ", ""
    ])

    def apply(self, engine: LongContextEngine, content: str, **kwargs) -> ProcessingResult:
        """Apply chunking strategy."""
        result = ProcessingResult(success=True, output=[], time_seconds=time.time())

        try:
            chunks = list(self._chunk(content))
            result.output = chunks
            result.success = True
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.time_seconds = time.time() - result.time_seconds
        return result

    def _chunk(self, content: str) -> list[str]:
        """Split content into chunks."""
        if self.method == "fixed":
            return self._fixed_chunk(content)
        elif self.method == "recursive":
            return self._recursive_chunk(content)
        else:
            return self._fixed_chunk(content)

    def _fixed_chunk(self, content: str) -> list[str]:
        """Split into fixed-size chunks."""
        chunks = []
        pos = 0
        while pos < len(content):
            end = min(pos + self.chunk_size, len(content))
            chunks.append(content[pos:end])
            pos = end - self.chunk_overlap
        return chunks

    def _recursive_chunk(self, content: str) -> list[str]:
        """Recursive character-based chunking."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', content)
        if not sentences:
            return [content]

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1
            if current_len + sentence_len <= self.chunk_size:
                current.append(sentence)
                current_len += sentence_len
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [sentence]
                current_len = sentence_len

        if current:
            chunks.append(" ".join(current))

        return chunks


@dataclass
class MergeStrategy:
    """
    Merge strategy for combining chunks back together.

    Supports: simple, overlap, semantic-aware.
    """

    method: str = "overlap"

    def apply(self, engine: LongContextEngine, chunks: list[str], **kwargs) -> ProcessingResult:
        """Apply merge strategy."""
        result = ProcessingResult(success=True, time_seconds=time.time())

        try:
            if self.method == "overlap":
                merged = self._overlap_merge(chunks)
            elif self.method == "simple":
                merged = "\n\n".join(chunks)
            elif self.method == "semantic":
                merged = self._semantic_merge(chunks)
            else:
                merged = "\n\n".join(chunks)

            result.output = merged
            result.success = True
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.time_seconds = time.time() - result.time_seconds
        return result

    def _overlap_merge(self, chunks: list[str], overlap: int = 200) -> str:
        """Merge chunks with overlap."""
        if not chunks:
            return ""

        merged = chunks[0]
        for chunk in chunks[1:]:
            merged += chunk[overlap:]
        return merged

    def _semantic_merge(self, chunks: list[str]) -> str:
        """Semantic-aware merge (placeholder for future implementation)."""
        return "\n\n".join(chunks)


@dataclass
class ChainOfDistillationStrategy:
    """
    Chain-of-distillation strategy.

    Uses a multi-pass approach:
    1. Extract key points from each chunk
    2. Merge key points
    3. Generate final summary
    """

    num_passes: int = 3
    pass_prompts: list[str] = field(default_factory=lambda: [
        "Extract the most important points from this text.",
        "Identify the main argument and supporting points.",
        "Write a concise summary.",
    ])

    def apply(self, engine: LongContextEngine, content: str, **kwargs) -> ProcessingResult:
        """Apply CoD strategy."""
        result = ProcessingResult(success=True, time_seconds=time.time())

        try:
            chunks = list(ChunkStrategy(chunk_size=2000, chunk_overlap=200).apply(
                engine, content
            ).output)

            # Pass 1: Extract key points
            points = []
            for i, chunk in enumerate(chunks):
                prompt = f"Extract key points from this chunk:\n{chunk}"
                points.append(engine._call_llm(prompt))

            # Pass 2: Merge and synthesize
            merged = "\n\n".join(points)
            prompt = f"Merge these key points into a coherent summary:\n{merged}"
            summary = engine._call_llm(prompt)

            # Pass 3: Refine
            prompt = f"Refine this summary to be clear and concise:\n{summary}"
            result.output = engine._call_llm(prompt)

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)

        result.time_seconds = time.time() - result.time_seconds
        return result


@dataclass
class MapReduceStrategy:
    """
    Map-Reduce strategy for long documents.

    1. Map: Summarize each chunk independently
    2. Reduce: Merge all summaries into one
    """

    map_prompt: str = "Summarize this text in 2-3 sentences."
    reduce_prompt: str = "Merge these summaries into one coherent summary:\n{summaries}"
    num_parallel: int = 4  # Number of chunks to process in parallel

    def apply(self, engine: LongContextEngine, content: str, **kwargs) -> ProcessingResult:
        """Apply Map-Reduce strategy."""
        result = ProcessingResult(success=True, time_seconds=time.time())

        try:
            chunks = list(ChunkStrategy(chunk_size=2000, chunk_overlap=200).apply(
                engine, content
            ).output)

            # Map phase
            map_results = []
            for i, chunk in enumerate(chunks[: self.num_parallel]):
                prompt = self.map_prompt + f"\n\n{chunk}"
                summary = engine._call_llm(prompt)
                map_results.append(summary)

            # Reduce phase
            merged = "\n\n".join(map_results)
            prompt = self.reduce_prompt.format(summaries=merged)
            result.output = engine._call_llm(prompt)

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)

        result.time_seconds = time.time() - result.time_seconds
        return result


class StrategyFactory:
    """Factory for creating strategies."""

    @staticmethod
    def create(name: str, **kwargs) -> Strategy:
        """Create a strategy by name."""
        strategies = {
            "summary": lambda: SummaryStrategy(**kwargs),
            "extract": lambda: ExtractStrategy(**kwargs),
            "chunk": lambda: ChunkStrategy(**kwargs),
            "merge": lambda: MergeStrategy(**kwargs),
            "cod": lambda: ChainOfDistillationStrategy(**kwargs),
            "mapreduce": lambda: MapReduceStrategy(**kwargs),
        }

        if name not in strategies:
            raise ValueError(f"Unknown strategy: {name}")

        return strategies[name]()

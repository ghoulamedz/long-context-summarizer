"""Core engine that orchestrates long-context processing with 60k context LLMs."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ModelConfig, BatchConfig, DocumentBatch, DocumentResult, Status
    from .document import Document


class LongContextEngine:
    """
    Main engine for long-context document processing.

    Supports:
    - 60k context window LLMs (Llama-3.1-70B-Instruct-60k, Qwen2.5-72B-Instruct-72k)
    - 4096 token generation (max_tokens)
    - Document chunking and merging
    - Multi-step summarization
    - Entity extraction
    - Q&A over long documents
    """

    def __init__(
        self,
        model_config: ModelConfig,
        batch_config: BatchConfig | None = None,
    ):
        self.model_config = model_config
        self.batch_config = batch_config or BatchConfig()
        self._llm = None  # type: ignore
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the engine and load the model."""
        if self._initialized:
            return

        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self.model_config.model_path,
                n_ctx=self.model_config.effective_context,
                n_ctx_window=self.model_config.n_ctx_window,
                n_ctx_window_set=self.model_config.n_ctx_window_set,
                n_predict=self.model_config.effective_predict,
                seed=self.model_config.seed,
                n_threads=self.model_config.n_threads,
                n_gpu_layers=self.model_config.n_gpu_layers,
                low_vram=self.model_config.low_vram,
                rope_freq_base=self.model_config.rope_freq_base,
                rope_freq_scale=self.model_config.rope_freq_scale,
                rope_scaling=self.model_config.rope_scaling,
                rms_norm_eps=self.model_config.rms_norm_eps,
                causal_attn=self.model_config.causal_attn,
                flash_attn=self.model_config.flash_attn,
                verbose=False,
            )
        except ImportError:
            # Fallback to transformers if llama-cpp-python is not available
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path,
                device_map="auto",
                torch_dtype="auto",
                max_position_embeddings=self.model_config.effective_context,
            )

        self._initialized = True

    def _get_llm(self):
        """Get the LLM instance (sync wrapper for async)."""
        if self._llm is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._llm

    async def _get_llm_async(self):
        """Get the LLM instance asynchronously."""
        if self._llm is None:
            await self.initialize()
        return self._llm

    async def process_document(
        self,
        document: Document,
        strategy: str = "summary",
        **kwargs,
    ) -> DocumentResult:
        """
        Process a single document using the specified strategy.

        Args:
            document: The document to process.
            strategy: Processing strategy ('summary', 'extract', 'qa').
            **kwargs: Additional arguments passed to the strategy.

        Returns:
            DocumentResult with processing results.
        """
        result = DocumentResult(
            id=document.id,
            status=Status.LOADING,
            document=document.content[:500] + "..." if len(document.content) > 500 else document.content,
        )

        try:
            result.status = Status.CHUNKING
            result.chunks_used = 0
            result.tokens_used = 0

            # Chunk the document
            chunks = list(document.get_chunks())
            result.chunks_used = len(chunks)

            # Merge chunks and process
            merged_content = self._merge_chunks(chunks)

            if len(merged_content) > self.model_config.effective_context:
                # Fallback: process with truncation or smaller context
                merged_content = merged_content[: self.model_config.effective_context]
                result.chunks_used = len(chunks)  # Use all available chunks

            result.status = Status.PROCESSING

            # Build prompt based on strategy
            if strategy == "summary":
                prompt = self._build_summary_prompt(merged_content, **kwargs)
            elif strategy == "extract":
                prompt = self._build_extract_prompt(merged_content, **kwargs)
            elif strategy == "qa":
                prompt = self._build_qa_prompt(merged_content, **kwargs)
            else:
                prompt = self._build_summary_prompt(merged_content, **kwargs)

            # Generate response
            response = await self._call_llm(prompt)

            if response:
                if strategy == "summary":
                    result.summary = response
                elif strategy == "extract":
                    result.extractions = self._parse_extraction(response)
                elif strategy == "qa":
                    result.summary = response  # Use response as answer

            result.status = Status.COMPLETED
            result.end_time = time.time()

        except Exception as e:
            result.error = str(e)
            result.status = Status.FAILED
            result.end_time = time.time()

        return result

    def _merge_chunks(self, chunks: list[ChunkInfo]) -> str:
        """Merge chunks into a single text with metadata markers."""
        lines = []
        for chunk in chunks:
            lines.append(f"--- CHUNK {chunk.index} ---")
            lines.append(chunk.content)
            lines.append("---")
        return "\n\n".join(lines)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        llm = await self._get_llm_async()

        if hasattr(self._llm, "create_chat_completion"):
            # Chat-based model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            response = llm.create_chat_completion(messages=messages)
            return response["choices"][0]["message"]["content"]
        else:
            # Completion-based model
            response = llm(prompt, max_tokens=self.model_config.effective_predict)
            return response["choices"][0]["text"]

    def _build_summary_prompt(self, content: str, **kwargs) -> str:
        """Build a summary prompt."""
        instructions = kwargs.get("instructions", "Summarize the following text concisely.")
        max_length = kwargs.get("max_length", 500)
        format = kwargs.get("format", "markdown")

        prompt = f"""You are an expert summarizer.

INSTRUCTIONS:
{instructions}

Max length: {max_length} tokens
Format: {format}

CONTENT:
{content}

SUMMARY:"""
        return prompt

    def _build_extract_prompt(self, content: str, **kwargs) -> str:
        """Build an extraction prompt."""
        instructions = kwargs.get("instructions", "Extract entities and relationships.")
        schema = kwargs.get("schema", {"entities": ["person", "org", "location", "date", "money"]})

        prompt = f"""You are an expert entity extractor.

INSTRUCTIONS:
{instructions}

Schema: {schema}

CONTENT:
{content}

EXTRactions:"""
        return prompt

    def _build_qa_prompt(self, content: str, question: str, **kwargs) -> str:
        """Build a Q&A prompt."""
        instructions = kwargs.get("instructions", "Answer the question based on the text.")
        format = kwargs.get("format", "json")

        prompt = f"""You are an expert question answerer.

INSTRUCTIONS:
{instructions}

Question: {question}

CONTENT:
{content}

ANSWER:"""
        return prompt

    def _parse_extraction(self, response: str) -> list[dict]:
        """Parse extraction results from LLM output."""
        import json

        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return data.get("entities", [])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse simple format
        entities = []
        lines = response.strip().split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    entities.append({"entity": parts[0].strip(), "type": parts[1].strip()})
        return entities


class EngineBuilder:
    """Builder pattern for creating configured engines."""

    @staticmethod
    def from_env() -> LongContextEngine:
        """Create an engine from environment variables."""
        import os

        model_path = os.environ.get("LLM_MODEL_PATH", "")
        if not model_path:
            model_path = os.environ.get("LLM_MODEL", "")

        if not model_path:
            model_path = "/path/to/model.gguf"  # default

        config = ModelConfig(
            model_path=model_path,
            context_limit=int(os.environ.get("PI_CONTEXT_LIMIT", "60000")),
            n_predict=int(os.environ.get("N_PREDICT", "4096")),
        )

        batch_config = BatchConfig(
            chunk_size=int(os.environ.get("CHUNK_SIZE", "2000")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
            max_size_mb=int(os.environ.get("MAX_SIZE_MB", "1000")),
        )

        return LongContextEngine(model_config=config, batch_config=batch_config)

    @staticmethod
    def from_model_path(model_path: str, context_limit: int = 60000, n_predict: int = 4096) -> LongContextEngine:
        """Create an engine from a model path."""
        config = ModelConfig(
            model_path=model_path,
            context_limit=context_limit,
            n_predict=n_predict,
        )
        return LongContextEngine(model_config=config)

    @staticmethod
    def from_config(config_path: str) -> LongContextEngine:
        """Create an engine from a config file."""
        import json
        import os

        with open(config_path) as f:
            data = json.load(f)

        config = ModelConfig(**data.get("model", {}))
        batch_config = BatchConfig(**data.get("batch", {}))

        return LongContextEngine(model_config=config, batch_config=batch_config)


def build_engine(
    model_path: str | None = None,
    context_limit: int = 60000,
    n_predict: int = 4096,
    **kwargs,
) -> LongContextEngine:
    """
    Build and return a configured LongContextEngine.

    Args:
        model_path: Path to the GGUF model file.
        context_limit: Maximum context size (default: 60000).
        n_predict: Maximum tokens to generate per call (default: 4096).
        **kwargs: Additional keyword arguments.

    Returns:
        A configured and ready-to-use LongContextEngine.
    """
    config = ModelConfig(
        model_path=model_path or "/path/to/model.gguf",
        context_limit=context_limit,
        n_predict=n_predict,
        **kwargs,
    )

    batch_config = BatchConfig(
        chunk_size=kwargs.get("chunk_size", 2000),
        chunk_overlap=kwargs.get("chunk_overlap", 200),
        max_size_mb=kwargs.get("max_size_mb", 1000),
    )

    return LongContextEngine(model_config=config, batch_config=batch_config)

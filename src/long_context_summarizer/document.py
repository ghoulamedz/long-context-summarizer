"""Document loading, chunking, and batching utilities."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .types import DocumentResult, Status, ChunkInfo, BatchConfig


@dataclass
class Document:
    """Represents a document with content and metadata."""

    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    source: str = ""
    title: str = ""
    file_path: str | None = None
    loaded_at: float = field(default_factory=time.time)

    @classmethod
    def from_text(cls, text: str, source: str = "text", title: str = "") -> Document:
        """Create a document from raw text."""
        return cls(
            id=cls._generate_id(text),
            content=text,
            source=source,
            title=title or source,
        )

    @classmethod
    def from_file(cls, file_path: str | Path, title: str = "") -> Document:
        """Load a document from a file path."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return cls(
            id=cls._generate_id(str(file_path)),
            content=content,
            source=f"file:{file_path.name}",
            title=title or file_path.stem,
            file_path=str(file_path),
        )

    @classmethod
    def from_url(cls, url: str, title: str = "") -> Document:
        """Create a document from a URL (placeholder for now)."""
        return cls(
            id=cls._generate_id(url),
            content=f"URL: {url}\n(Use web fetch to get content)",
            source="url",
            title=title or url,
        )

    @classmethod
    def _generate_id(cls, content: str) -> str:
        """Generate a unique ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"id": self.id, "content": self.content, "metadata": self.metadata}

    @property
    def length(self) -> int:
        """Get the length of the content in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get the word count."""
        return len(re.findall(r"\b\w+\b", self.content))

    @property
    def is_empty(self) -> bool:
        """Check if the document is empty."""
        return not self.content.strip()


@dataclass
class ChunkInfo:
    """Information about a text chunk."""

    index: int
    content: str
    start_position: int
    end_position: int
    overlap: int = 0
    is_last: bool = False
    chunk_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "overlap": self.overlap,
            "is_last": self.is_last,
            "chunk_id": self.chunk_id,
        }


@dataclass
class DocumentBatch:
    """A batch of documents ready for processing."""

    documents: list[Document]
    config: BatchConfig
    config_id: str = "default"

    def __post_init__(self):
        if not self.documents:
            self.documents = []

    @property
    def total_length(self) -> int:
        """Total length of all documents in the batch."""
        return sum(doc.length for doc in self.documents)

    @property
    def total_words(self) -> int:
        """Total word count."""
        return sum(doc.word_count for doc in self.documents)

    def add(self, doc: Document) -> None:
        """Add a document to the batch."""
        if len(self.documents) >= self.config.max_documents:
            raise ValueError("Batch is full")

        # Check if document fits within size limit
        current_total = sum(d.length for d in self.documents)
        max_chars = self.config.max_size_mb * 1024 * 1024
        if current_total + doc.length > max_chars:
            raise ValueError("Batch would exceed size limit")

        self.documents.append(doc)

    def get_chunks(self) -> Iterator[ChunkInfo]:
        """Yield chunks from all documents in the batch."""
        for doc in self.documents:
            chunks = list(self._chunk_document(doc))
            for chunk in chunks:
                yield chunk

    def _chunk_document(self, doc: Document) -> Iterator[ChunkInfo]:
        """Split a document into chunks with overlap."""
        content = doc.content
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        max_chunks = min(self.config.max_chunks_per_doc, self.config.max_chunks)

        if len(content) <= chunk_size:
            yield ChunkInfo(
                index=0,
                content=content,
                start_position=0,
                end_position=len(content),
                is_last=True,
            )
            return

        position = 0
        chunk_index = 0

        while position < len(content):
            chunk_end = min(position + chunk_size, len(content))
            chunk_content = content[position:chunk_end]

            is_last = (chunk_end >= len(content))

            yield ChunkInfo(
                index=chunk_index,
                content=chunk_content,
                start_position=position,
                end_position=chunk_end,
                overlap=overlap,
                is_last=is_last,
                chunk_id=f"{doc.id}-chunk-{chunk_index:04d}",
            )

            if is_last:
                break

            position += chunk_size - overlap
            chunk_index += 1

            if chunk_index >= max_chunks:
                break

    def to_dict(self) -> dict:
        """Convert batch to dictionary."""
        return {
            "config_id": self.config_id,
            "num_documents": len(self.documents),
            "total_length": self.total_length,
            "total_words": self.total_words,
            "documents": [d.to_dict() for d in self.documents],
            "chunks": [c.to_dict() for c in self.get_chunks()],
        }

    @property
    def chunk_count(self) -> int:
        """Total number of chunks across all documents."""
        return sum(1 for _ in self.get_chunks())

    def filter_empty(self) -> DocumentBatch:
        """Return a new batch with only non-empty documents."""
        non_empty = [d for d in self.documents if d.content.strip()]
        return DocumentBatch(documents=non_empty, config=self.config)


class DocumentLoader:
    """Utility class for loading documents from various sources."""

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()

    def from_text(self, text: str, source: str = "text") -> Document:
        """Load from text."""
        return Document.from_text(text, source=source)

    def from_file(self, path: str | Path) -> Document:
        """Load from a file."""
        return Document.from_file(path)

    def from_string_list(self, texts: list[str], source: str = "batch") -> list[Document]:
        """Load from a list of strings."""
        documents = []
        for i, text in enumerate(texts):
            doc = Document.from_text(text, source=f"{source}-text-{i}")
            documents.append(doc)
        return documents

    def load_markdown(self, text: str) -> Document:
        """Load a Markdown document with metadata extraction."""
        doc = Document.from_text(text, source="markdown")
        doc.metadata["format"] = "markdown"
        doc.metadata["type"] = "text/markdown"

        # Extract headings
        headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        doc.metadata["headings"] = headings
        return doc

    def load_html(self, text: str) -> Document:
        """Load an HTML document with metadata extraction."""
        doc = Document.from_text(text, source="html")
        doc.metadata["format"] = "html"
        doc.metadata["type"] = "text/html"
        return doc

    def load_json(self, text: str) -> Document:
        """Load a JSON document."""
        doc = Document.from_text(text, source="json")
        doc.metadata["format"] = "json"
        return doc

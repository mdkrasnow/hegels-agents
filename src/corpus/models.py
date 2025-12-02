"""
Corpus Data Models

This module defines the core data structures for corpus management,
including documents and chunks with their metadata.

Design Principles:
1. Immutable dataclasses for safety and clarity
2. Pydantic-compatible validation
3. Serialization support for caching
4. Progressive robustness fields prepared for future enhancement
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json
import hashlib


@dataclass
class CorpusDocument:
    """
    Represents a document in the corpus.

    A document is the top-level unit of the corpus, containing metadata
    about the source material. Documents are split into CorpusChunk
    objects for embedding and retrieval.

    Attributes:
        title: Document title or filename
        content: Full document text content
        external_id: Optional external identifier (filename, paper ID, etc.)
        metadata: Additional document metadata (author, date, source, etc.)
        id: Internal document ID (assigned after storage)
        created_at: Timestamp when document was added to corpus
    """
    title: str
    content: str
    external_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate document after initialization."""
        if not self.title:
            raise ValueError("Document title cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def content_hash(self) -> str:
        """Generate a hash of the document content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def word_count(self) -> int:
        """Approximate word count of the document."""
        return len(self.content.split())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "external_id": self.external_id,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusDocument":
        """Create document from dictionary."""
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        return cls(
            id=data.get("id"),
            title=data["title"],
            content=data["content"],
            external_id=data.get("external_id"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


@dataclass
class CorpusChunk:
    """
    Represents a chunk/section of a document for embedding and retrieval.

    Documents are split into chunks of appropriate size for embedding models.
    Each chunk maintains a reference to its parent document and position.

    Attributes:
        document_id: ID of the parent document
        section_index: Position of this chunk within the document
        content: Text content of the chunk
        embedding: Vector embedding of the content (computed separately)
        token_count: Approximate token count for the chunk
        metadata: Additional chunk metadata (e.g., section title, page number)

    Progressive Robustness Fields (optional):
        confidence_score: Content quality/reliability score
        bias_flags: Detected potential biases
        validation_metadata: Validation and provenance information
    """
    document_id: int
    section_index: int
    content: str
    embedding: Optional[list[float]] = None
    token_count: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    # Progressive robustness fields
    confidence_score: Optional[float] = None
    bias_flags: Optional[list[str]] = None
    validation_metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate chunk after initialization."""
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if self.section_index < 0:
            raise ValueError("Section index must be non-negative")
        # Estimate token count if not provided (rough approximation)
        if self.token_count is None:
            self.token_count = len(self.content.split()) * 4 // 3  # ~1.33 tokens/word

    @property
    def content_hash(self) -> str:
        """Generate a hash of the chunk content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def has_embedding(self) -> bool:
        """Check if embedding has been computed."""
        return self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "document_id": self.document_id,
            "section_index": self.section_index,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }
        # Only include embedding if present (can be large)
        if self.embedding is not None:
            result["embedding"] = self.embedding
        # Include robustness fields if present
        if self.confidence_score is not None:
            result["confidence_score"] = self.confidence_score
        if self.bias_flags is not None:
            result["bias_flags"] = self.bias_flags
        if self.validation_metadata is not None:
            result["validation_metadata"] = self.validation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusChunk":
        """Create chunk from dictionary."""
        return cls(
            id=data.get("id"),
            document_id=data["document_id"],
            section_index=data["section_index"],
            content=data["content"],
            embedding=data.get("embedding"),
            token_count=data.get("token_count"),
            metadata=data.get("metadata", {}),
            confidence_score=data.get("confidence_score"),
            bias_flags=data.get("bias_flags"),
            validation_metadata=data.get("validation_metadata"),
        )


def chunk_document(
    document: CorpusDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "\n\n"
) -> list[CorpusChunk]:
    """
    Split a document into chunks suitable for embedding.

    Uses a simple text splitting strategy with configurable chunk size
    and overlap. Attempts to split at natural boundaries (paragraphs, sentences).

    Args:
        document: The document to chunk
        chunk_size: Target number of words per chunk (default: 512)
        chunk_overlap: Number of words to overlap between chunks (default: 64)
        separator: Preferred split boundary (default: paragraph break)

    Returns:
        List of CorpusChunk objects

    Raises:
        ValueError: If document has no ID assigned
    """
    if document.id is None:
        raise ValueError("Document must have an ID before chunking")

    content = document.content.strip()
    if not content:
        return []

    chunks: list[CorpusChunk] = []

    # First, try to split by separator (paragraphs)
    paragraphs = content.split(separator)

    current_chunk_words: list[str] = []
    section_index = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        words = paragraph.split()

        # If adding this paragraph would exceed chunk size, save current chunk
        if current_chunk_words and len(current_chunk_words) + len(words) > chunk_size:
            chunk_content = " ".join(current_chunk_words)
            chunks.append(CorpusChunk(
                document_id=document.id,
                section_index=section_index,
                content=chunk_content,
                metadata={
                    "document_title": document.title,
                    "source": document.external_id,
                }
            ))
            section_index += 1

            # Keep overlap words for next chunk
            if chunk_overlap > 0 and len(current_chunk_words) > chunk_overlap:
                current_chunk_words = current_chunk_words[-chunk_overlap:]
            else:
                current_chunk_words = []

        current_chunk_words.extend(words)

        # Handle very long paragraphs that exceed chunk size
        while len(current_chunk_words) > chunk_size:
            chunk_content = " ".join(current_chunk_words[:chunk_size])
            chunks.append(CorpusChunk(
                document_id=document.id,
                section_index=section_index,
                content=chunk_content,
                metadata={
                    "document_title": document.title,
                    "source": document.external_id,
                }
            ))
            section_index += 1

            # Keep overlap
            if chunk_overlap > 0:
                current_chunk_words = current_chunk_words[chunk_size - chunk_overlap:]
            else:
                current_chunk_words = current_chunk_words[chunk_size:]

    # Don't forget the last chunk
    if current_chunk_words:
        chunk_content = " ".join(current_chunk_words)
        chunks.append(CorpusChunk(
            document_id=document.id,
            section_index=section_index,
            content=chunk_content,
            metadata={
                "document_title": document.title,
                "source": document.external_id,
            }
        ))

    return chunks

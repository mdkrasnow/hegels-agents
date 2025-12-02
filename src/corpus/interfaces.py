"""
Abstract Interfaces for Corpus Retrieval

This module defines the core interfaces for corpus retrieval, designed for
progressive enhancement from file-based to production database implementations.

Key Design Principles:
1. Clean interface abstraction - same API for file-based and database retrieval
2. Standard return format for consistent downstream processing
3. Extension points for future robustness features (confidence, bias flags)
4. Type-safe dataclass returns for better IDE support and validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievalResult:
    """
    Standard return format for corpus retrieval results.

    This dataclass provides a consistent structure for retrieval results
    across different retriever implementations (file-based, Supabase, etc.).

    Attributes:
        content: The retrieved text content
        similarity: Cosine similarity score [0.0, 1.0]
        metadata: Additional metadata about the source (document_id, section_index, etc.)

    Progressive Robustness Fields (optional, for future enhancement):
        confidence_score: Model confidence in this retrieval's relevance
        bias_flags: Detected potential biases in the content
        validation_metadata: Additional validation information
    """
    content: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # Progressive robustness fields (optional, None by default)
    confidence_score: Optional[float] = None
    bias_flags: Optional[list[str]] = None
    validation_metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate retrieval result after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("similarity must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "content": self.content,
            "similarity": self.similarity,
            "metadata": self.metadata,
        }
        if self.confidence_score is not None:
            result["confidence_score"] = self.confidence_score
        if self.bias_flags is not None:
            result["bias_flags"] = self.bias_flags
        if self.validation_metadata is not None:
            result["validation_metadata"] = self.validation_metadata
        return result


class ICorpusRetriever(ABC):
    """
    Abstract interface for corpus retrieval.

    This interface defines the contract for all corpus retriever implementations,
    enabling seamless switching between file-based validation and production
    database retrieval.

    Implementations:
        - FileCorpusRetriever: File-based implementation for validation (Phase 1a)
        - SupabaseCorpusRetriever: Production pgvector implementation (Phase 1b)

    Usage:
        retriever: ICorpusRetriever = FileCorpusRetriever(corpus_path="./data/corpus")
        results = retriever.retrieve("What causes climate change?", k=5, threshold=0.7)
        for result in results:
            print(f"[{result.similarity:.2f}] {result.content[:100]}...")
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 8,
        threshold: float = 0.7
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant corpus sections for a query.

        Args:
            query: Natural language query to search the corpus
            k: Maximum number of sections to retrieve (default: 8)
            threshold: Minimum cosine similarity [0, 1] for results (default: 0.7)

        Returns:
            List of RetrievalResult objects sorted by similarity (descending)

        Raises:
            ValueError: If query is empty or k/threshold are invalid
            RuntimeError: If retrieval fails due to system errors
        """
        pass

    @abstractmethod
    def get_corpus_stats(self) -> dict[str, Any]:
        """
        Get statistics about the loaded corpus.

        Returns:
            Dictionary containing corpus statistics:
                - document_count: Number of documents
                - chunk_count: Number of chunks/sections
                - total_tokens: Approximate total token count
                - embedding_dimension: Dimension of embeddings
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the retriever is ready for queries.

        Returns:
            True if corpus is loaded and embeddings are available
        """
        pass


class IEmbeddingProvider(ABC):
    """
    Abstract interface for embedding providers.

    Allows swapping between different embedding models/services.
    """

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """
        Embed multiple texts for storage/retrieval.

        Args:
            texts: List of texts to embed
            task_type: Embedding task type (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY)

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query for retrieval.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        pass

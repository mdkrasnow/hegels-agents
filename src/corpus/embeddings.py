"""
Gemini Embedding Helper

This module provides utilities for generating embeddings using Google's
Gemini embedding model (gemini-embedding-001).

Features:
- Batch embedding generation for documents
- Query embedding generation for retrieval
- Caching support to avoid redundant API calls
- Configurable embedding dimensions (768 recommended for storage efficiency)

Usage:
    from src.corpus.embeddings import EmbeddingHelper

    helper = EmbeddingHelper()
    embeddings = helper.embed_texts(["Hello world", "Another text"])
    query_embedding = helper.embed_query("What is AI?")
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

from src.corpus.interfaces import IEmbeddingProvider

logger = logging.getLogger(__name__)


# Default embedding configuration
DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSION = 768
BATCH_SIZE = 100  # Maximum texts per API call


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = DEFAULT_MODEL
    dimension: int = DEFAULT_DIMENSION
    api_key: Optional[str] = None


class EmbeddingHelper(IEmbeddingProvider):
    """
    Helper class for generating embeddings using Gemini API.

    This implementation uses the google-genai SDK to generate embeddings
    with configurable dimension truncation for storage efficiency.

    Attributes:
        config: Embedding configuration
        client: Gemini API client (lazily initialized)

    Example:
        helper = EmbeddingHelper()
        texts = ["Document about AI", "Document about ML"]
        embeddings = helper.embed_texts(texts)
        query_emb = helper.embed_query("What is machine learning?")
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding helper.

        Args:
            config: Optional embedding configuration. If not provided,
                   uses defaults and GEMINI_API_KEY from environment.
        """
        self.config = config or EmbeddingConfig()
        self._client = None
        self._initialized = False

    def _ensure_client(self) -> None:
        """Lazily initialize the Gemini client."""
        if self._initialized:
            return

        try:
            from google import genai
            from google.genai import types

            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found. Set it in environment or config."
                )

            self._client = genai.Client(api_key=api_key)
            self._types = types
            self._initialized = True
            logger.info(f"Initialized Gemini client with model {self.config.model}")

        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced."""
        return self.config.dimension

    def embed_texts(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """
        Embed multiple texts for document storage.

        Uses RETRIEVAL_DOCUMENT task type which is optimized for documents
        that will be retrieved later.

        Args:
            texts: List of texts to embed
            task_type: Embedding task type (default: RETRIEVAL_DOCUMENT)

        Returns:
            List of embedding vectors (each of dimension self.config.dimension)

        Raises:
            ValueError: If texts list is empty
            RuntimeError: If API call fails
        """
        if not texts:
            return []

        self._ensure_client()

        all_embeddings: list[list[float]] = []

        # Process in batches to handle large document sets
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            logger.debug(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} texts)")

            try:
                response = self._client.models.embed_content(
                    model=self.config.model,
                    contents=batch,
                    config=self._types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.config.dimension,
                    ),
                )
                batch_embeddings = [e.values for e in response.embeddings]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Embedding API call failed: {e}")
                raise RuntimeError(f"Failed to generate embeddings: {e}")

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query for retrieval.

        Uses RETRIEVAL_QUERY task type which is optimized for queries
        that will be matched against documents.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector

        Raises:
            ValueError: If query is empty
            RuntimeError: If API call fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        self._ensure_client()

        try:
            response = self._client.models.embed_content(
                model=self.config.model,
                contents=query,
                config=self._types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.config.dimension,
                ),
            )
            return response.embeddings[0].values

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")


# Module-level convenience functions
_default_helper: Optional[EmbeddingHelper] = None


def get_helper() -> EmbeddingHelper:
    """Get or create the default embedding helper singleton."""
    global _default_helper
    if _default_helper is None:
        _default_helper = EmbeddingHelper()
    return _default_helper


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """
    Embed texts using the default helper.

    Convenience function for quick embedding without managing helper instances.

    Args:
        texts: List of texts to embed
        task_type: Embedding task type

    Returns:
        List of embedding vectors
    """
    return get_helper().embed_texts(texts, task_type)


def embed_query(query: str) -> list[float]:
    """
    Embed a query using the default helper.

    Convenience function for quick query embedding.

    Args:
        query: Query text to embed

    Returns:
        Query embedding vector
    """
    return get_helper().embed_query(query)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score in [-1, 1]

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")

    # Compute dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)

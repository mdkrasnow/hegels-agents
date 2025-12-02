"""
File-Based Corpus Retriever

This module implements the ICorpusRetriever interface using file-based storage.
It's designed for Phase 1a validation before transitioning to production
database infrastructure.

Features:
- Load text files from a directory structure
- Chunk documents and generate embeddings
- Cache embeddings to disk (JSON/pickle) for fast reload
- Cosine similarity search for retrieval
- Thread-safe operations

Usage:
    from src.corpus.file_retriever import FileCorpusRetriever

    retriever = FileCorpusRetriever(corpus_path="./data/corpus")
    retriever.load_corpus()
    results = retriever.retrieve("What is machine learning?", k=5)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

from src.corpus.interfaces import ICorpusRetriever, RetrievalResult
from src.corpus.models import CorpusDocument, CorpusChunk, chunk_document
from src.corpus.embeddings import (
    EmbeddingHelper,
    EmbeddingConfig,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


@dataclass
class FileRetrieverConfig:
    """Configuration for file-based corpus retriever."""
    corpus_path: str
    cache_path: Optional[str] = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_dimension: int = 768
    file_extensions: tuple[str, ...] = (".txt", ".md", ".text")
    use_cache: bool = True


class FileCorpusRetriever(ICorpusRetriever):
    """
    File-based implementation of corpus retrieval.

    This retriever loads text files from a directory, chunks them,
    generates embeddings, and provides similarity-based retrieval.

    Attributes:
        config: Retriever configuration
        documents: Loaded corpus documents
        chunks: Chunked document sections with embeddings
        embedding_helper: Helper for generating embeddings

    Example:
        retriever = FileCorpusRetriever(corpus_path="./data/corpus")
        retriever.load_corpus()

        results = retriever.retrieve("climate change effects", k=5, threshold=0.6)
        for result in results:
            print(f"[{result.similarity:.3f}] {result.content[:100]}...")
    """

    def __init__(
        self,
        corpus_path: str,
        config: Optional[FileRetrieverConfig] = None,
        embedding_helper: Optional[EmbeddingHelper] = None,
    ):
        """
        Initialize the file-based retriever.

        Args:
            corpus_path: Path to directory containing corpus files
            config: Optional configuration (defaults used if not provided)
            embedding_helper: Optional embedding helper (created if not provided)
        """
        self.config = config or FileRetrieverConfig(corpus_path=corpus_path)
        self.config.corpus_path = corpus_path  # Ensure path is set

        # Set default cache path if not specified
        if self.config.cache_path is None:
            self.config.cache_path = str(
                Path(corpus_path).parent / ".corpus_cache"
            )

        self.documents: list[CorpusDocument] = []
        self.chunks: list[CorpusChunk] = []

        # Initialize embedding helper
        self.embedding_helper = embedding_helper or EmbeddingHelper(
            config=EmbeddingConfig(dimension=self.config.embedding_dimension)
        )

        self._ready = False

    def load_corpus(self, force_reload: bool = False) -> None:
        """
        Load corpus from files or cache.

        Args:
            force_reload: If True, ignore cache and reload from files
        """
        cache_path = Path(self.config.cache_path)

        # Try loading from cache first
        if self.config.use_cache and not force_reload and cache_path.exists():
            if self._load_from_cache():
                logger.info("Loaded corpus from cache")
                self._ready = True
                return

        # Load from files
        logger.info(f"Loading corpus from {self.config.corpus_path}")
        self._load_from_files()

        # Generate embeddings
        self._generate_embeddings()

        # Save to cache
        if self.config.use_cache:
            self._save_to_cache()

        self._ready = True
        logger.info(
            f"Corpus loaded: {len(self.documents)} documents, {len(self.chunks)} chunks"
        )

    def _load_from_files(self) -> None:
        """Load all text files from the corpus directory."""
        corpus_path = Path(self.config.corpus_path)

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        self.documents = []
        self.chunks = []
        doc_id = 0

        # Find all text files
        for ext in self.config.file_extensions:
            for file_path in corpus_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if not content.strip():
                        logger.warning(f"Skipping empty file: {file_path}")
                        continue

                    doc = CorpusDocument(
                        id=doc_id,
                        title=file_path.stem,
                        content=content,
                        external_id=str(file_path.relative_to(corpus_path)),
                        metadata={
                            "file_path": str(file_path),
                            "file_size": file_path.stat().st_size,
                        }
                    )
                    self.documents.append(doc)

                    # Chunk the document
                    doc_chunks = chunk_document(
                        doc,
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                    self.chunks.extend(doc_chunks)

                    doc_id += 1
                    logger.debug(f"Loaded: {file_path.name} ({len(doc_chunks)} chunks)")

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        if not self.documents:
            logger.warning(f"No documents found in {corpus_path}")

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all chunks."""
        if not self.chunks:
            return

        logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")

        # Extract chunk contents
        chunk_texts = [chunk.content for chunk in self.chunks]

        # Generate embeddings in batch
        embeddings = self.embedding_helper.embed_texts(chunk_texts)

        # Assign embeddings to chunks
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding

        logger.info("Embeddings generated successfully")

    def _load_from_cache(self) -> bool:
        """Load corpus from cache if available and valid."""
        cache_path = Path(self.config.cache_path)
        docs_cache = cache_path / "documents.json"
        chunks_cache = cache_path / "chunks.pkl"

        if not docs_cache.exists() or not chunks_cache.exists():
            return False

        try:
            # Load documents
            with open(docs_cache, "r") as f:
                docs_data = json.load(f)
                self.documents = [
                    CorpusDocument.from_dict(d) for d in docs_data
                ]

            # Load chunks with embeddings (pickle for efficiency)
            with open(chunks_cache, "rb") as f:
                chunks_data = pickle.load(f)
                self.chunks = [
                    CorpusChunk.from_dict(c) for c in chunks_data
                ]

            return True

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save corpus to cache for faster reload."""
        cache_path = Path(self.config.cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save documents (JSON for readability)
            docs_cache = cache_path / "documents.json"
            with open(docs_cache, "w") as f:
                json.dump([d.to_dict() for d in self.documents], f, indent=2)

            # Save chunks with embeddings (pickle for size/speed)
            chunks_cache = cache_path / "chunks.pkl"
            with open(chunks_cache, "wb") as f:
                pickle.dump([c.to_dict() for c in self.chunks], f)

            logger.info(f"Corpus cached to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

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
            RuntimeError: If corpus not loaded
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if not self._ready:
            raise RuntimeError("Corpus not loaded. Call load_corpus() first.")

        # Generate query embedding
        query_embedding = self.embedding_helper.embed_query(query)

        # Calculate similarity for all chunks
        similarities: list[tuple[int, float]] = []
        for i, chunk in enumerate(self.chunks):
            if chunk.embedding is None:
                continue
            sim = cosine_similarity(query_embedding, chunk.embedding)
            if sim >= threshold:
                similarities.append((i, sim))

        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # Build results
        results: list[RetrievalResult] = []
        for chunk_idx, similarity in top_k:
            chunk = self.chunks[chunk_idx]

            # Find parent document for additional metadata
            parent_doc = next(
                (d for d in self.documents if d.id == chunk.document_id),
                None
            )

            result = RetrievalResult(
                content=chunk.content,
                similarity=similarity,
                metadata={
                    "document_id": chunk.document_id,
                    "section_index": chunk.section_index,
                    "document_title": parent_doc.title if parent_doc else None,
                    "source": chunk.metadata.get("source"),
                    **chunk.metadata,
                },
                confidence_score=chunk.confidence_score,
                bias_flags=chunk.bias_flags,
            )
            results.append(result)

        return results

    def get_corpus_stats(self) -> dict[str, Any]:
        """
        Get statistics about the loaded corpus.

        Returns:
            Dictionary containing corpus statistics
        """
        total_tokens = sum(c.token_count or 0 for c in self.chunks)
        chunks_with_embeddings = sum(1 for c in self.chunks if c.has_embedding())

        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "chunks_with_embeddings": chunks_with_embeddings,
            "total_tokens": total_tokens,
            "embedding_dimension": self.config.embedding_dimension,
            "corpus_path": self.config.corpus_path,
            "cache_path": self.config.cache_path,
            "is_ready": self._ready,
        }

    def is_ready(self) -> bool:
        """Check if the retriever is ready for queries."""
        return self._ready

    def clear_cache(self) -> None:
        """Clear the corpus cache."""
        cache_path = Path(self.config.cache_path)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"Cache cleared: {cache_path}")

    def search_keyword(self, keyword: str, case_sensitive: bool = False) -> list[RetrievalResult]:
        """
        Simple keyword-based search (fallback when embeddings unavailable).

        Args:
            keyword: Keyword to search for
            case_sensitive: Whether to do case-sensitive search

        Returns:
            List of RetrievalResult objects containing the keyword
        """
        if not keyword:
            return []

        results: list[RetrievalResult] = []
        search_term = keyword if case_sensitive else keyword.lower()

        for chunk in self.chunks:
            content = chunk.content if case_sensitive else chunk.content.lower()
            if search_term in content:
                # Use a simple relevance score based on occurrence frequency
                occurrences = content.count(search_term)
                # Normalize to [0, 1] range (more occurrences = higher score)
                similarity = min(1.0, occurrences * 0.1 + 0.5)

                result = RetrievalResult(
                    content=chunk.content,
                    similarity=similarity,
                    metadata={
                        "document_id": chunk.document_id,
                        "section_index": chunk.section_index,
                        "match_type": "keyword",
                        "occurrences": occurrences,
                    }
                )
                results.append(result)

        # Sort by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results

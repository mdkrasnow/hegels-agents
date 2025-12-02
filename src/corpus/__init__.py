"""
Corpus Management Module for Hegels Agents

This module provides infrastructure for corpus storage, retrieval, and embedding
management. It follows a progressive enhancement architecture:

- Phase 1a: File-based retrieval validation
- Phase 1b: Production Supabase/pgvector scaling

The module exposes a clean ICorpusRetriever interface that abstracts the underlying
storage mechanism, enabling seamless transition from file-based to database-backed
retrieval.
"""

from src.corpus.interfaces import ICorpusRetriever, RetrievalResult
from src.corpus.models import CorpusDocument, CorpusChunk
from src.corpus.embeddings import EmbeddingHelper, embed_texts, embed_query
from src.corpus.file_retriever import FileCorpusRetriever

__all__ = [
    # Interfaces
    "ICorpusRetriever",
    "RetrievalResult",
    # Models
    "CorpusDocument",
    "CorpusChunk",
    # Embedding utilities
    "EmbeddingHelper",
    "embed_texts",
    "embed_query",
    # Retrievers
    "FileCorpusRetriever",
]

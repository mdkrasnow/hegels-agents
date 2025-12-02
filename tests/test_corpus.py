"""
Tests for Corpus Module

This module contains tests for the corpus retrieval system including:
- Data models (CorpusDocument, CorpusChunk)
- Interface definitions (ICorpusRetriever)
- File-based retrieval (FileCorpusRetriever)
- Embedding utilities

Note: Tests that require the Gemini API are marked with @pytest.mark.integration
and are skipped by default unless GEMINI_API_KEY is set.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.corpus.models import CorpusDocument, CorpusChunk, chunk_document
from src.corpus.interfaces import RetrievalResult, ICorpusRetriever
from src.corpus.embeddings import cosine_similarity, EmbeddingHelper, EmbeddingConfig
from src.corpus.file_retriever import FileCorpusRetriever, FileRetrieverConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_document() -> CorpusDocument:
    """Create a sample document for testing."""
    return CorpusDocument(
        id=1,
        title="Test Document",
        content="This is a test document. It contains multiple sentences. " * 50,
        external_id="test.txt",
        metadata={"author": "Test Author"},
    )


@pytest.fixture
def sample_chunks(sample_document: CorpusDocument) -> list[CorpusChunk]:
    """Create sample chunks from a document."""
    return chunk_document(sample_document, chunk_size=100, chunk_overlap=10)


@pytest.fixture
def temp_corpus_dir() -> Path:
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus"
        corpus_path.mkdir()

        # Create test documents
        (corpus_path / "doc1.txt").write_text(
            "Machine learning is a subset of artificial intelligence. "
            "It enables computers to learn from data without being explicitly programmed. "
            "Deep learning is a type of machine learning using neural networks."
        )
        (corpus_path / "doc2.txt").write_text(
            "Climate change is affecting global temperatures. "
            "The greenhouse effect traps heat in the atmosphere. "
            "Renewable energy can help reduce carbon emissions."
        )
        (corpus_path / "doc3.txt").write_text(
            "The human brain contains approximately 86 billion neurons. "
            "Memory formation involves changes in synaptic connections. "
            "Cognitive psychology studies mental processes like attention and perception."
        )

        yield corpus_path


# ============================================================================
# CorpusDocument Tests
# ============================================================================

class TestCorpusDocument:
    """Tests for CorpusDocument dataclass."""

    def test_document_creation(self) -> None:
        """Test basic document creation."""
        doc = CorpusDocument(
            title="Test",
            content="Test content",
        )
        assert doc.title == "Test"
        assert doc.content == "Test content"
        assert doc.id is None
        assert doc.created_at is not None

    def test_document_validation_empty_title(self) -> None:
        """Test that empty title raises error."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            CorpusDocument(title="", content="Content")

    def test_document_validation_empty_content(self) -> None:
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            CorpusDocument(title="Title", content="")

    def test_document_content_hash(self) -> None:
        """Test content hash generation."""
        doc = CorpusDocument(title="Test", content="Test content")
        assert len(doc.content_hash) == 16  # SHA256 truncated to 16 chars
        # Same content should produce same hash
        doc2 = CorpusDocument(title="Different", content="Test content")
        assert doc.content_hash == doc2.content_hash

    def test_document_word_count(self) -> None:
        """Test word count calculation."""
        doc = CorpusDocument(
            title="Test",
            content="One two three four five",
        )
        assert doc.word_count == 5

    def test_document_serialization(self, sample_document: CorpusDocument) -> None:
        """Test document to_dict and from_dict."""
        data = sample_document.to_dict()
        assert data["title"] == sample_document.title
        assert data["content"] == sample_document.content

        restored = CorpusDocument.from_dict(data)
        assert restored.title == sample_document.title
        assert restored.content == sample_document.content


# ============================================================================
# CorpusChunk Tests
# ============================================================================

class TestCorpusChunk:
    """Tests for CorpusChunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test basic chunk creation."""
        chunk = CorpusChunk(
            document_id=1,
            section_index=0,
            content="Test content",
        )
        assert chunk.document_id == 1
        assert chunk.section_index == 0
        assert chunk.content == "Test content"
        assert chunk.token_count is not None  # Auto-calculated

    def test_chunk_validation_empty_content(self) -> None:
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            CorpusChunk(document_id=1, section_index=0, content="")

    def test_chunk_validation_negative_index(self) -> None:
        """Test that negative section index raises error."""
        with pytest.raises(ValueError, match="Section index must be non-negative"):
            CorpusChunk(document_id=1, section_index=-1, content="Test")

    def test_chunk_has_embedding(self) -> None:
        """Test has_embedding method."""
        chunk = CorpusChunk(document_id=1, section_index=0, content="Test")
        assert not chunk.has_embedding()

        chunk.embedding = [0.1, 0.2, 0.3]
        assert chunk.has_embedding()

    def test_chunk_serialization(self) -> None:
        """Test chunk to_dict and from_dict."""
        chunk = CorpusChunk(
            document_id=1,
            section_index=0,
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
        )
        data = chunk.to_dict()

        restored = CorpusChunk.from_dict(data)
        assert restored.document_id == chunk.document_id
        assert restored.content == chunk.content
        assert restored.embedding == chunk.embedding


# ============================================================================
# chunk_document Tests
# ============================================================================

class TestChunkDocument:
    """Tests for the chunk_document function."""

    def test_chunk_document_basic(self, sample_document: CorpusDocument) -> None:
        """Test basic document chunking."""
        chunks = chunk_document(sample_document, chunk_size=50, chunk_overlap=5)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == sample_document.id
            assert chunk.content

    def test_chunk_document_requires_id(self) -> None:
        """Test that chunking requires document ID."""
        doc = CorpusDocument(title="Test", content="Content")
        with pytest.raises(ValueError, match="must have an ID"):
            chunk_document(doc)

    def test_chunk_document_metadata(self, sample_document: CorpusDocument) -> None:
        """Test that chunks inherit document metadata."""
        chunks = chunk_document(sample_document)
        for chunk in chunks:
            assert "document_title" in chunk.metadata
            assert chunk.metadata["document_title"] == sample_document.title


# ============================================================================
# RetrievalResult Tests
# ============================================================================

class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_result_creation(self) -> None:
        """Test basic result creation."""
        result = RetrievalResult(
            content="Test content",
            similarity=0.85,
            metadata={"source": "test.txt"},
        )
        assert result.content == "Test content"
        assert result.similarity == 0.85
        assert result.metadata["source"] == "test.txt"

    def test_result_similarity_validation(self) -> None:
        """Test similarity validation."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RetrievalResult(content="Test", similarity=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RetrievalResult(content="Test", similarity=-0.1)

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        result = RetrievalResult(
            content="Test",
            similarity=0.9,
            metadata={"key": "value"},
            confidence_score=0.8,
        )
        data = result.to_dict()
        assert data["content"] == "Test"
        assert data["similarity"] == 0.9
        assert data["confidence_score"] == 0.8


# ============================================================================
# Cosine Similarity Tests
# ============================================================================

class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2)) < 0.001

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(vec1, vec2) - (-1.0)) < 0.001

    def test_dimension_mismatch(self) -> None:
        """Test that mismatched dimensions raise error."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(vec1, vec2)


# ============================================================================
# FileCorpusRetriever Tests (Unit Tests - No API)
# ============================================================================

class TestFileCorpusRetrieverUnit:
    """Unit tests for FileCorpusRetriever (no API calls)."""

    def test_retriever_creation(self, temp_corpus_dir: Path) -> None:
        """Test retriever initialization."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        assert not retriever.is_ready()

    def test_retriever_config(self, temp_corpus_dir: Path) -> None:
        """Test retriever configuration."""
        config = FileRetrieverConfig(
            corpus_path=str(temp_corpus_dir),
            chunk_size=256,
            chunk_overlap=32,
            embedding_dimension=512,
        )
        retriever = FileCorpusRetriever(
            corpus_path=str(temp_corpus_dir),
            config=config,
        )
        assert retriever.config.chunk_size == 256
        assert retriever.config.chunk_overlap == 32
        assert retriever.config.embedding_dimension == 512

    def test_retriever_not_ready_raises(self, temp_corpus_dir: Path) -> None:
        """Test that retrieve raises when not ready."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        with pytest.raises(RuntimeError, match="not loaded"):
            retriever.retrieve("test query")

    def test_retriever_invalid_query_raises(self, temp_corpus_dir: Path) -> None:
        """Test that empty query raises error."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        retriever._ready = True  # Bypass loading for test
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("")

    def test_retriever_invalid_k_raises(self, temp_corpus_dir: Path) -> None:
        """Test that invalid k raises error."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        retriever._ready = True
        with pytest.raises(ValueError, match="must be positive"):
            retriever.retrieve("test", k=0)

    def test_retriever_invalid_threshold_raises(self, temp_corpus_dir: Path) -> None:
        """Test that invalid threshold raises error."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        retriever._ready = True
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            retriever.retrieve("test", threshold=1.5)

    def test_keyword_search(self, temp_corpus_dir: Path) -> None:
        """Test keyword-based search (no embeddings needed)."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))

        # Manually load documents without embeddings
        retriever._load_from_files()

        # Search for keyword
        results = retriever.search_keyword("machine learning")
        assert len(results) > 0
        assert all("machine learning" in r.content.lower() for r in results)

    def test_stats_empty_retriever(self, temp_corpus_dir: Path) -> None:
        """Test stats for empty retriever."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        stats = retriever.get_corpus_stats()
        assert stats["document_count"] == 0
        assert stats["is_ready"] is False


# ============================================================================
# Integration Tests (Require API Key)
# ============================================================================

@pytest.mark.integration
class TestFileCorpusRetrieverIntegration:
    """Integration tests for FileCorpusRetriever (require API key)."""

    @pytest.fixture(autouse=True)
    def check_api_key(self) -> None:
        """Skip tests if API key not available."""
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_full_retrieval_flow(self, temp_corpus_dir: Path) -> None:
        """Test complete retrieval flow with real embeddings."""
        retriever = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        retriever.load_corpus()

        assert retriever.is_ready()

        # Query about machine learning
        results = retriever.retrieve(
            "What is machine learning?",
            k=3,
            threshold=0.3,
        )

        assert len(results) > 0
        # Should find the AI document as most relevant
        assert any("machine learning" in r.content.lower() for r in results)

    def test_retrieval_with_cache(self, temp_corpus_dir: Path) -> None:
        """Test that caching works correctly."""
        retriever1 = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        retriever1.load_corpus()

        # Create new retriever and load from cache
        retriever2 = FileCorpusRetriever(corpus_path=str(temp_corpus_dir))
        retriever2.load_corpus()  # Should load from cache

        # Both should give same results
        results1 = retriever1.retrieve("climate", k=2, threshold=0.3)
        results2 = retriever2.retrieve("climate", k=2, threshold=0.3)

        assert len(results1) == len(results2)


# ============================================================================
# Mock Tests for Embedding Helper
# ============================================================================

class TestEmbeddingHelperMock:
    """Tests for EmbeddingHelper using mocks."""

    def test_embed_texts_empty(self) -> None:
        """Test embedding empty list returns empty."""
        helper = EmbeddingHelper()
        # This should return empty without calling API
        result = helper.embed_texts([])
        assert result == []

    def test_embed_query_empty_raises(self) -> None:
        """Test that empty query raises error."""
        helper = EmbeddingHelper()
        with pytest.raises(ValueError, match="cannot be empty"):
            helper.embed_query("")

    def test_embedding_dimension(self) -> None:
        """Test embedding dimension property."""
        config = EmbeddingConfig(dimension=512)
        helper = EmbeddingHelper(config=config)
        assert helper.embedding_dimension == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

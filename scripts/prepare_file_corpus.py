#!/usr/bin/env python3
"""
Prepare File Corpus Script

This script prepares a file-based corpus for use with the FileCorpusRetriever.
It loads documents from a directory, chunks them, generates embeddings,
and caches the results for fast loading.

Usage:
    python scripts/prepare_file_corpus.py --corpus-path ./data/corpus --force

Options:
    --corpus-path: Path to the corpus directory (default: ./data/corpus)
    --cache-path: Path to store cache (default: auto-generated)
    --chunk-size: Target chunk size in words (default: 512)
    --chunk-overlap: Overlap between chunks in words (default: 64)
    --force: Force regeneration of embeddings even if cache exists
    --dry-run: Show what would be processed without making changes
    --stats-only: Only show corpus statistics
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.corpus.file_retriever import FileCorpusRetriever, FileRetrieverConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare file-based corpus for RAG retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--corpus-path",
        type=str,
        default="./data/corpus",
        help="Path to the corpus directory containing text files",
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to store the embedding cache (default: auto-generated)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in words (default: 512)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between chunks in words (default: 64)",
    )

    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of embeddings even if cache exists",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show corpus statistics",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Run a test query after loading the corpus",
    )

    return parser.parse_args()


def show_dry_run_info(corpus_path: Path, config: FileRetrieverConfig) -> None:
    """Show information about what would be processed in a dry run."""
    print("\n=== DRY RUN ===")
    print(f"Corpus path: {corpus_path}")
    print(f"Cache path: {config.cache_path}")
    print(f"Chunk size: {config.chunk_size} words")
    print(f"Chunk overlap: {config.chunk_overlap} words")
    print(f"Embedding dimension: {config.embedding_dimension}")
    print(f"File extensions: {config.file_extensions}")

    # Count files
    file_count = 0
    total_size = 0
    for ext in config.file_extensions:
        for file_path in corpus_path.rglob(f"*{ext}"):
            file_count += 1
            total_size += file_path.stat().st_size

    print(f"\nFiles found: {file_count}")
    print(f"Total size: {total_size / 1024:.2f} KB")
    print("\nNo changes made (dry run)")


def show_corpus_stats(retriever: FileCorpusRetriever) -> None:
    """Display corpus statistics."""
    stats = retriever.get_corpus_stats()

    print("\n=== CORPUS STATISTICS ===")
    print(f"Documents: {stats['document_count']}")
    print(f"Chunks: {stats['chunk_count']}")
    print(f"Chunks with embeddings: {stats['chunks_with_embeddings']}")
    print(f"Total tokens (approx): {stats['total_tokens']:,}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Corpus path: {stats['corpus_path']}")
    print(f"Cache path: {stats['cache_path']}")
    print(f"Ready: {stats['is_ready']}")

    # Document details
    if retriever.documents:
        print("\n=== DOCUMENTS ===")
        for doc in retriever.documents:
            chunk_count = len([c for c in retriever.chunks if c.document_id == doc.id])
            print(f"  [{doc.id}] {doc.title}: {doc.word_count} words, {chunk_count} chunks")


def run_test_query(retriever: FileCorpusRetriever, query: str) -> None:
    """Run a test query and display results."""
    print(f"\n=== TEST QUERY ===")
    print(f"Query: {query}")
    print()

    try:
        results = retriever.retrieve(query, k=5, threshold=0.5)

        if not results:
            print("No results found above threshold")
            return

        print(f"Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"--- Result {i} (similarity: {result.similarity:.3f}) ---")
            print(f"Source: {result.metadata.get('document_title', 'Unknown')}")
            # Show first 200 chars of content
            content_preview = result.content[:200].replace("\n", " ")
            if len(result.content) > 200:
                content_preview += "..."
            print(f"Content: {content_preview}")
            print()

    except Exception as e:
        print(f"Error running query: {e}")


def main() -> int:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Resolve corpus path
    corpus_path = Path(args.corpus_path).resolve()
    if not corpus_path.exists():
        logger.error(f"Corpus path does not exist: {corpus_path}")
        return 1

    # Create configuration
    config = FileRetrieverConfig(
        corpus_path=str(corpus_path),
        cache_path=args.cache_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_dimension=args.embedding_dimension,
    )

    # Dry run mode
    if args.dry_run:
        show_dry_run_info(corpus_path, config)
        return 0

    # Create retriever
    retriever = FileCorpusRetriever(
        corpus_path=str(corpus_path),
        config=config,
    )

    # Clear cache if force flag is set
    if args.force:
        logger.info("Force flag set, clearing cache...")
        retriever.clear_cache()

    # Load corpus
    try:
        print(f"\nLoading corpus from: {corpus_path}")
        retriever.load_corpus(force_reload=args.force)
        print("Corpus loaded successfully!")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Show statistics
    show_corpus_stats(retriever)

    # Stats only mode
    if args.stats_only:
        return 0

    # Run test query if provided
    if args.test_query:
        run_test_query(retriever, args.test_query)

    return 0


if __name__ == "__main__":
    sys.exit(main())

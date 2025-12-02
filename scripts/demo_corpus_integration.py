#!/usr/bin/env python3
"""
Corpus Integration Demonstration

This script demonstrates the file-based corpus system working with the agent
framework, showing retrieval capabilities and integration points.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from corpus.file_retriever import FileCorpusRetriever
from corpus.utils import format_search_results


def demo_corpus_functionality():
    """Demonstrate corpus loading, indexing, and search capabilities."""
    
    print("=" * 60)
    print("CORPUS INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Setup corpus
    corpus_dir = project_root / "corpus_data"
    print(f"Loading corpus from: {corpus_dir}")
    
    retriever = FileCorpusRetriever(
        corpus_dir=str(corpus_dir),
        chunk_size=800,
        chunk_overlap=150,
        search_method="hybrid"
    )
    
    # Load and index
    print("\n1. Loading corpus files...")
    loading_stats = retriever.load_corpus()
    print(f"   ✓ Loaded {loading_stats['files_loaded']} files")
    print(f"   ✓ Created {loading_stats['chunks_created']} chunks")
    
    print("\n2. Building search index...")
    indexing_stats = retriever.build_search_index()
    print(f"   ✓ Indexed {indexing_stats['indexed_chunks']} chunks")
    print(f"   ✓ Vocabulary size: {indexing_stats['vocabulary_size']} terms")
    
    # Demonstrate different search methods
    print("\n3. Demonstrating search capabilities...")
    
    test_queries = [
        "quantum mechanics wave particle duality",
        "natural selection evolution Darwin",
        "World War II causes",
        "ethics moral philosophy Kant",
        "sorting algorithms complexity"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: \"{query}\"")
        
        # Test different search methods
        for method in ['keyword', 'tfidf', 'hybrid']:
            results = retriever.search(query, max_results=3, method=method)
            
            if results:
                top_result = results[0]
                source_file = Path(top_result.chunk.source_file).name
                print(f"   {method:>8}: {len(results)} results, top: {source_file} (score: {top_result.score:.3f})")
            else:
                print(f"   {method:>8}: No results")
    
    # Demonstrate agent integration format
    print("\n4. Agent integration demonstration...")
    
    agent_questions = [
        "What is the uncertainty principle in quantum mechanics?",
        "How does natural selection work in evolution?",
        "What caused World War II?",
        "What is the difference between deontological and consequentialist ethics?"
    ]
    
    for question in agent_questions:
        print(f"\n   Question: {question}")
        
        # Get formatted context for agent
        context = retriever.retrieve_for_question(question, max_results=2)
        
        if context:
            print(f"   Context length: {len(context)} characters")
            print("   Context preview:")
            lines = context.split('\n')
            for line in lines[:4]:  # Show first few lines
                if line.strip():
                    print(f"     {line}")
            if len(lines) > 4:
                print("     ...")
        else:
            print("   No relevant context found")
    
    # Show corpus statistics
    print("\n5. Corpus statistics...")
    stats = retriever.get_statistics()
    
    print(f"   Total files: {stats['files']['total']}")
    print(f"   Total chunks: {stats['chunks']['total']}")
    print(f"   Average chunk size: {stats['chunks']['avg_size']:.0f} characters")
    print(f"   Total content: {stats['content']['total_length']:,} characters")
    print(f"   Search vocabulary: {stats['search']['vocabulary_size']:,} unique terms")
    
    file_types = stats['files']['by_extension']
    print(f"   File types: {dict(file_types)}")
    
    # Demonstrate chunk context retrieval
    print("\n6. Chunk context demonstration...")
    if retriever.chunks:
        # Get a chunk from quantum mechanics
        quantum_chunks = [c for c in retriever.chunks if 'quantum' in c.content.lower()]
        if quantum_chunks:
            target_chunk = quantum_chunks[0]
            context_chunks = retriever.get_context_around_chunk(
                target_chunk.chunk_id, before=1, after=1
            )
            
            print(f"   Target chunk: {target_chunk.chunk_id}")
            print(f"   Context chunks: {len(context_chunks)} chunks")
            print(f"   Context preview: {context_chunks[0].content[:100]}...")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Summary of capabilities
    print("\nCorpus capabilities demonstrated:")
    print("✓ File loading and text processing")
    print("✓ Intelligent text chunking with overlap")
    print("✓ Multiple search methods (keyword, TF-IDF, hybrid)")
    print("✓ Relevance scoring and ranking")
    print("✓ Agent-ready context formatting")
    print("✓ Chunk context retrieval")
    print("✓ Performance monitoring and statistics")
    
    return True


def demo_search_comparison():
    """Demonstrate differences between search methods."""
    
    print("\n" + "=" * 60)
    print("SEARCH METHOD COMPARISON")
    print("=" * 60)
    
    corpus_dir = project_root / "corpus_data"
    retriever = FileCorpusRetriever(str(corpus_dir))
    retriever.load_corpus()
    retriever.build_search_index()
    
    test_query = "quantum mechanics uncertainty principle Heisenberg"
    print(f"Query: \"{test_query}\"")
    
    methods = ['keyword', 'tfidf', 'hybrid']
    
    for method in methods:
        print(f"\n{method.upper()} Search Results:")
        results = retriever.search(test_query, max_results=5, method=method)
        
        for i, result in enumerate(results, 1):
            source = Path(result.chunk.source_file).name
            content_preview = result.chunk.content[:100].replace('\n', ' ')
            
            print(f"{i}. {source}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Content: {content_preview}...")
            
            # Show match details for keyword search
            if method == 'keyword':
                matched_words = result.match_details.get('matched_words', [])
                if matched_words:
                    print(f"   Matched words: {', '.join(matched_words)}")


if __name__ == "__main__":
    try:
        # Run main demonstration
        demo_corpus_functionality()
        
        # Run search comparison
        demo_search_comparison()
        
        print(f"\nAll demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
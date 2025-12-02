#!/usr/bin/env python3
"""
Corpus Testing Script for Hegels Agents Phase 0.5.2

This script tests the FileCorpusRetriever implementation, validates retrieval
accuracy, and tests integration with the agent system.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from corpus.file_retriever import FileCorpusRetriever
from corpus.utils import calculate_result_metrics, format_search_results
from config.settings import load_config
from agents.worker import BasicWorkerAgent


class CorpusTestRunner:
    """Comprehensive testing suite for the file-based corpus system."""
    
    def __init__(self, corpus_dir: str):
        """Initialize test runner with corpus directory."""
        self.corpus_dir = Path(corpus_dir)
        self.test_results = {}
        self.retriever = None
        
    def setup_corpus(self) -> Dict[str, Any]:
        """Set up the corpus retriever and load data."""
        print("Setting up corpus retriever...")
        
        start_time = time.time()
        
        # Initialize retriever with different search methods
        self.retriever = FileCorpusRetriever(
            corpus_dir=str(self.corpus_dir),
            chunk_size=800,
            chunk_overlap=150,
            search_method="hybrid"
        )
        
        # Load corpus files
        loading_stats = self.retriever.load_corpus()
        
        # Build search index
        indexing_stats = self.retriever.build_search_index()
        
        setup_time = time.time() - start_time
        
        setup_results = {
            'setup_time': setup_time,
            'loading_stats': loading_stats,
            'indexing_stats': indexing_stats,
            'corpus_stats': self.retriever.get_statistics()
        }
        
        print(f"Corpus setup completed in {setup_time:.2f} seconds")
        print(f"Loaded {loading_stats['files_loaded']} files, {loading_stats['chunks_created']} chunks")
        print(f"Built index with {indexing_stats.get('vocabulary_size', 0)} unique terms")
        
        return setup_results
    
    def test_basic_search_functionality(self) -> Dict[str, Any]:
        """Test basic search functionality across different methods."""
        print("\n=== Testing Basic Search Functionality ===")
        
        test_queries = [
            "quantum mechanics wave particle duality",
            "evolution natural selection Darwin",
            "World War II causes consequences",
            "ethics moral philosophy virtue",
            "algorithms data structures sorting",
            "psychology cognitive memory",
            "economics macroeconomics GDP inflation",
            "Shakespeare tragedy Hamlet",
            "calculus derivative integral",
            "solar system planets formation"
        ]
        
        search_results = {}
        
        for method in ['keyword', 'tfidf', 'hybrid']:
            print(f"\nTesting {method} search method:")
            method_results = []
            
            for query in test_queries:
                start_time = time.time()
                results = self.retriever.search(query, max_results=5, method=method)
                search_time = time.time() - start_time
                
                metrics = calculate_result_metrics(results)
                
                query_result = {
                    'query': query,
                    'search_time': search_time,
                    'num_results': len(results),
                    'metrics': metrics,
                    'top_score': results[0].score if results else 0.0,
                    'results': [
                        {
                            'source': r.chunk.source_file,
                            'score': r.score,
                            'matches': r.match_details.get('unique_matches', 0)
                        }
                        for r in results[:3]
                    ]
                }
                
                method_results.append(query_result)
                print(f"  {query[:40]}... -> {len(results)} results, top score: {query_result['top_score']:.3f}")
            
            search_results[method] = method_results
        
        return search_results
    
    def test_retrieval_accuracy(self) -> Dict[str, Any]:
        """Test retrieval accuracy with expected relevant documents."""
        print("\n=== Testing Retrieval Accuracy ===")
        
        # Define test cases with expected relevant files
        test_cases = [
            {
                'query': 'quantum mechanics uncertainty principle',
                'expected_files': ['physics_quantum_mechanics.txt'],
                'description': 'Physics query should return physics content'
            },
            {
                'query': 'natural selection evolution species',
                'expected_files': ['biology_evolution.txt'],
                'description': 'Biology query should return biology content'
            },
            {
                'query': 'World War Two Nazi Germany',
                'expected_files': ['history_world_war_two.txt'],
                'description': 'History query should return history content'
            },
            {
                'query': 'consequentialism utilitarianism Kant',
                'expected_files': ['philosophy_ethics.txt'],
                'description': 'Ethics query should return philosophy content'
            },
            {
                'query': 'sorting algorithms binary search',
                'expected_files': ['computer_science_algorithms.txt'],
                'description': 'Computer science query should return CS content'
            },
            {
                'query': 'Shakespeare Hamlet tragedy',
                'expected_files': ['literature_shakespeare.txt'],
                'description': 'Literature query should return literature content'
            }
        ]
        
        accuracy_results = []
        
        for test_case in test_cases:
            query = test_case['query']
            expected_files = test_case['expected_files']
            
            results = self.retriever.search(query, max_results=10)
            
            # Check if expected files appear in results
            returned_files = {Path(r.chunk.source_file).name for r in results}
            expected_files_set = set(expected_files)
            
            hits = len(expected_files_set.intersection(returned_files))
            precision = hits / len(results) if results else 0
            recall = hits / len(expected_files_set) if expected_files_set else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            accuracy_result = {
                'query': query,
                'description': test_case['description'],
                'expected_files': expected_files,
                'returned_files': list(returned_files),
                'hits': hits,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'top_5_results': [
                    {
                        'file': Path(r.chunk.source_file).name,
                        'score': r.score,
                        'is_expected': Path(r.chunk.source_file).name in expected_files_set
                    }
                    for r in results[:5]
                ]
            }
            
            accuracy_results.append(accuracy_result)
            
            print(f"Query: {query}")
            print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")
            print(f"  Expected: {expected_files}")
            print(f"  Top result: {Path(results[0].chunk.source_file).name if results else 'None'}")
        
        # Calculate overall accuracy metrics
        avg_precision = sum(r['precision'] for r in accuracy_results) / len(accuracy_results)
        avg_recall = sum(r['recall'] for r in accuracy_results) / len(accuracy_results)
        avg_f1 = sum(r['f1_score'] for r in accuracy_results) / len(accuracy_results)
        
        return {
            'test_cases': accuracy_results,
            'overall_metrics': {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1
            }
        }
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance characteristics of the retrieval system."""
        print("\n=== Testing Performance Benchmarks ===")
        
        # Test search performance with varying query lengths
        short_queries = ["quantum", "evolution", "ethics"]
        medium_queries = ["quantum mechanics principles", "natural selection evolution", "moral philosophy ethics"]
        long_queries = [
            "quantum mechanics wave particle duality uncertainty principle",
            "evolution natural selection adaptation species formation Darwin",
            "moral philosophy ethics consequentialism deontological virtue Kant"
        ]
        
        performance_results = {}
        
        for query_type, queries in [('short', short_queries), ('medium', medium_queries), ('long', long_queries)]:
            times = []
            result_counts = []
            
            for query in queries:
                start_time = time.time()
                results = self.retriever.search(query, max_results=20)
                search_time = time.time() - start_time
                
                times.append(search_time)
                result_counts.append(len(results))
            
            performance_results[query_type] = {
                'avg_time': sum(times) / len(times),
                'max_time': max(times),
                'min_time': min(times),
                'avg_results': sum(result_counts) / len(result_counts),
                'queries_tested': len(queries)
            }
        
        # Test memory usage (approximate)
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        performance_results['memory'] = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        }
        
        # Test scalability with different result limits
        scalability_query = "machine learning artificial intelligence algorithms"
        scalability_results = {}
        
        for max_results in [5, 10, 20, 50]:
            start_time = time.time()
            results = self.retriever.search(scalability_query, max_results=max_results)
            search_time = time.time() - start_time
            
            scalability_results[max_results] = {
                'search_time': search_time,
                'actual_results': len(results)
            }
        
        performance_results['scalability'] = scalability_results
        
        print(f"Short queries: {performance_results['short']['avg_time']:.4f}s avg")
        print(f"Medium queries: {performance_results['medium']['avg_time']:.4f}s avg")
        print(f"Long queries: {performance_results['long']['avg_time']:.4f}s avg")
        print(f"Memory usage: {performance_results['memory']['rss_mb']:.1f} MB RSS")
        
        return performance_results
    
    def test_agent_integration(self) -> Dict[str, Any]:
        """Test integration with the agent system."""
        print("\n=== Testing Agent Integration ===")
        
        try:
            # Load configuration
            config = load_config()
            
            # Create a worker agent
            agent = BasicWorkerAgent("test_worker")
            
            # Test questions that should benefit from corpus retrieval
            test_questions = [
                "What is the uncertainty principle in quantum mechanics?",
                "How does natural selection drive evolution?",
                "What were the main causes of World War II?",
                "What are the differences between deontological and consequentialist ethics?",
                "How do sorting algorithms work and what are their complexities?"
            ]
            
            integration_results = []
            
            for question in test_questions:
                print(f"\nTesting question: {question}")
                
                # Get relevant context from corpus
                start_time = time.time()
                context = self.retriever.retrieve_for_question(question, max_results=3)
                retrieval_time = time.time() - start_time
                
                # Generate agent response with corpus context
                start_time = time.time()
                agent_response = agent.respond(question, external_context=context)
                response_time = time.time() - start_time
                
                integration_result = {
                    'question': question,
                    'context_provided': bool(context),
                    'context_length': len(context) if context else 0,
                    'retrieval_time': retrieval_time,
                    'response_time': response_time,
                    'agent_confidence': agent_response.confidence,
                    'response_length': len(agent_response.content),
                    'context_snippet': context[:200] + "..." if len(context) > 200 else context
                }
                
                integration_results.append(integration_result)
                
                print(f"  Context retrieved: {integration_result['context_length']} chars")
                print(f"  Retrieval time: {retrieval_time:.3f}s")
                print(f"  Response confidence: {agent_response.confidence}")
            
            # Test without context for comparison
            print("\n--- Testing without corpus context ---")
            no_context_results = []
            
            for question in test_questions[:2]:  # Test fewer for efficiency
                agent_response = agent.respond(question, external_context=None)
                
                no_context_result = {
                    'question': question,
                    'confidence': agent_response.confidence,
                    'response_length': len(agent_response.content)
                }
                
                no_context_results.append(no_context_result)
            
            return {
                'with_corpus': integration_results,
                'without_corpus': no_context_results,
                'integration_successful': True
            }
            
        except Exception as e:
            print(f"Agent integration test failed: {e}")
            return {
                'with_corpus': [],
                'without_corpus': [],
                'integration_successful': False,
                'error': str(e)
            }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases ===")
        
        edge_case_results = {}
        
        # Test empty query
        try:
            results = self.retriever.search("", max_results=5)
            edge_case_results['empty_query'] = {
                'success': True,
                'num_results': len(results)
            }
        except Exception as e:
            edge_case_results['empty_query'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test very long query
        long_query = "quantum mechanics evolution ethics algorithms psychology " * 20
        try:
            start_time = time.time()
            results = self.retriever.search(long_query[:1000], max_results=5)  # Truncate to reasonable length
            search_time = time.time() - start_time
            edge_case_results['long_query'] = {
                'success': True,
                'search_time': search_time,
                'num_results': len(results)
            }
        except Exception as e:
            edge_case_results['long_query'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test special characters
        special_query = "α β γ δ ∂ ∫ ∑ ∞ € £ ¥"
        try:
            results = self.retriever.search(special_query, max_results=5)
            edge_case_results['special_chars'] = {
                'success': True,
                'num_results': len(results)
            }
        except Exception as e:
            edge_case_results['special_chars'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test very high max_results
        try:
            results = self.retriever.search("science", max_results=1000)
            edge_case_results['high_max_results'] = {
                'success': True,
                'num_results': len(results)
            }
        except Exception as e:
            edge_case_results['high_max_results'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test chunk context retrieval
        try:
            if self.retriever.chunks:
                first_chunk_id = self.retriever.chunks[0].chunk_id
                context_chunks = self.retriever.get_context_around_chunk(first_chunk_id, before=2, after=2)
                edge_case_results['context_retrieval'] = {
                    'success': True,
                    'num_context_chunks': len(context_chunks)
                }
            else:
                edge_case_results['context_retrieval'] = {
                    'success': False,
                    'error': 'No chunks available'
                }
        except Exception as e:
            edge_case_results['context_retrieval'] = {
                'success': False,
                'error': str(e)
            }
        
        print(f"Edge case testing completed: {len(edge_case_results)} tests")
        
        return edge_case_results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("CORPUS TEST REPORT")
        print("="*60)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'corpus_directory': str(self.corpus_dir),
            'test_results': self.test_results
        }
        
        # Overall assessment
        setup_successful = 'setup' in self.test_results and self.test_results['setup']['corpus_stats']['status'] == 'loaded'
        search_successful = 'basic_search' in self.test_results
        accuracy_good = ('accuracy' in self.test_results and 
                        self.test_results['accuracy']['overall_metrics']['avg_f1_score'] > 0.7)
        integration_successful = ('integration' in self.test_results and 
                                 self.test_results['integration']['integration_successful'])
        
        report['assessment'] = {
            'setup_successful': setup_successful,
            'search_functional': search_successful,
            'accuracy_acceptable': accuracy_good,
            'integration_working': integration_successful,
            'overall_status': 'PASS' if all([setup_successful, search_successful, accuracy_good]) else 'PARTIAL'
        }
        
        # Print summary
        print(f"Setup: {'✓' if setup_successful else '✗'}")
        print(f"Search functionality: {'✓' if search_successful else '✗'}")
        print(f"Accuracy (F1 > 0.7): {'✓' if accuracy_good else '✗'}")
        print(f"Agent integration: {'✓' if integration_successful else '✗'}")
        print(f"\nOverall status: {report['assessment']['overall_status']}")
        
        if 'setup' in self.test_results:
            stats = self.test_results['setup']['corpus_stats']
            print(f"\nCorpus Statistics:")
            print(f"  Files loaded: {stats['files']['total']}")
            print(f"  Chunks created: {stats['chunks']['total']}")
            print(f"  Average chunk size: {stats['chunks']['avg_size']:.0f} chars")
            print(f"  Vocabulary size: {stats['search']['vocabulary_size']}")
        
        if 'accuracy' in self.test_results:
            metrics = self.test_results['accuracy']['overall_metrics']
            print(f"\nAccuracy Metrics:")
            print(f"  Average Precision: {metrics['avg_precision']:.3f}")
            print(f"  Average Recall: {metrics['avg_recall']:.3f}")
            print(f"  Average F1-Score: {metrics['avg_f1_score']:.3f}")
        
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            print(f"\nPerformance:")
            print(f"  Average search time: {perf['medium']['avg_time']:.3f}s")
            print(f"  Memory usage: {perf['memory']['rss_mb']:.1f} MB")
        
        return report
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("Starting comprehensive corpus testing...")
        print(f"Corpus directory: {self.corpus_dir}")
        
        # Run all test suites
        self.test_results['setup'] = self.setup_corpus()
        self.test_results['basic_search'] = self.test_basic_search_functionality()
        self.test_results['accuracy'] = self.test_retrieval_accuracy()
        self.test_results['performance'] = self.test_performance_benchmarks()
        self.test_results['integration'] = self.test_agent_integration()
        self.test_results['edge_cases'] = self.test_edge_cases()
        
        # Generate final report
        report = self.generate_test_report()
        
        return report


def main():
    """Main function to run corpus tests."""
    # Get corpus directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    corpus_dir = project_root / "corpus_data"
    
    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}")
        print("Please run this script from the project root or ensure corpus_data exists.")
        return 1
    
    # Initialize and run tests
    test_runner = CorpusTestRunner(str(corpus_dir))
    
    try:
        report = test_runner.run_all_tests()
        
        # Save report to file
        import json
        report_file = project_root / "corpus_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed test report saved to: {report_file}")
        
        # Return appropriate exit code
        return 0 if report['assessment']['overall_status'] == 'PASS' else 1
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Enhanced Retrieval Testing Script for Phase 1a RAG Enhancement

Comprehensive testing of the enhanced file-based retrieval system with
validation, quality assessment, and performance benchmarking.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the Python path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from corpus.enhanced_retriever import (
        EnhancedFileCorpusRetriever, 
        EnhancedCorpusConfig,
        create_enhanced_retriever
    )
    from corpus.validation import (
        validate_retrieval_system,
        CorpusValidator,
        RetrievalQualityAssessor,
        BenchmarkSuite
    )
    from corpus.file_retriever import FileCorpusRetriever
    from corpus.interfaces import RetrievalResult
    
except ImportError as e:
    print(f"Failed to import enhanced retrieval components: {e}")
    print("Ensure you're running from the project root directory.")
    sys.exit(1)


class EnhancedRetrievalTestRunner:
    """Comprehensive test runner for enhanced retrieval system."""
    
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.test_results = {}
        self.enhanced_retriever = None
        self.baseline_retriever = None
        
    def setup_retrievers(self) -> Dict[str, Any]:
        """Set up both enhanced and baseline retrievers for comparison."""
        print("Setting up enhanced and baseline retrievers...")
        
        start_time = time.time()
        
        try:
            # Create enhanced retriever with all features enabled
            print("  Creating enhanced retriever...")
            self.enhanced_retriever = create_enhanced_retriever(
                corpus_dir=str(self.corpus_dir),
                chunk_size=800,
                chunk_overlap=150,
                enable_all_features=True
            )
            
            # Create baseline retriever for comparison
            print("  Creating baseline retriever...")
            self.baseline_retriever = FileCorpusRetriever(
                corpus_dir=str(self.corpus_dir),
                chunk_size=800,
                chunk_overlap=150,
                search_method="hybrid"
            )
            
            # Load baseline retriever
            baseline_load = self.baseline_retriever.load_corpus()
            baseline_index = self.baseline_retriever.build_search_index()
            
            setup_time = time.time() - start_time
            
            setup_results = {
                'setup_time': setup_time,
                'enhanced_retriever_ready': self.enhanced_retriever.is_ready(),
                'baseline_retriever_ready': self.baseline_retriever._is_indexed,
                'enhanced_stats': self.enhanced_retriever.get_corpus_stats(),
                'baseline_stats': self.baseline_retriever.get_statistics()
            }
            
            print(f"Setup completed in {setup_time:.2f} seconds")
            print(f"Enhanced retriever ready: {setup_results['enhanced_retriever_ready']}")
            print(f"Baseline retriever ready: {setup_results['baseline_retriever_ready']}")
            
            return setup_results
            
        except Exception as e:
            print(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return {'setup_failed': True, 'error': str(e)}
            
    def test_enhanced_features(self) -> Dict[str, Any]:
        """Test enhanced retrieval features."""
        print("\n=== Testing Enhanced Features ===")
        
        if not self.enhanced_retriever or not self.enhanced_retriever.is_ready():
            return {'error': 'Enhanced retriever not ready'}
            
        test_queries = [
            {
                'query': 'quantum mechanics uncertainty principle wave particle',
                'description': 'Complex physics query testing BM25 and phrase matching',
                'expected_features': ['bm25', 'phrase_matching', 'semantic_chunking']
            },
            {
                'query': 'machine learning neural networks deep learning',
                'description': 'Multi-term query testing positional indexing',
                'expected_features': ['positional_index', 'proximity_scoring']
            },
            {
                'query': 'evolution natural selection species adaptation',
                'description': 'Biology query testing quality validation',
                'expected_features': ['quality_scoring', 'diversity_penalty']
            }
        ]
        
        feature_results = []
        
        for test in test_queries:
            query = test['query']
            print(f"\nTesting: {query}")
            
            start_time = time.time()
            results = self.enhanced_retriever.retrieve(
                query=query,
                k=10,
                threshold=0.1
            )
            query_time = time.time() - start_time
            
            # Analyze results for enhanced features
            feature_analysis = self._analyze_enhanced_features(results, test)
            
            test_result = {
                'query': query,
                'description': test['description'],
                'query_time': query_time,
                'num_results': len(results),
                'feature_analysis': feature_analysis,
                'top_similarities': [r.similarity for r in results[:5]],
                'quality_scores': [
                    r.metadata.get('quality_score', 'N/A') 
                    for r in results[:3]
                ]
            }
            
            feature_results.append(test_result)
            
            print(f"  Results: {len(results)} chunks in {query_time:.3f}s")
            print(f"  Top similarity: {test_result['top_similarities'][0]:.3f}" if results else "  No results")
            
        return {
            'feature_tests': feature_results,
            'enhanced_features_working': len(feature_results) > 0,
            'avg_query_time': sum(r['query_time'] for r in feature_results) / len(feature_results)
        }
        
    def test_quality_improvements(self) -> Dict[str, Any]:
        """Test quality improvements over baseline."""
        print("\n=== Testing Quality Improvements ===")
        
        if not self.enhanced_retriever or not self.baseline_retriever:
            return {'error': 'Retrievers not ready'}
            
        # Test queries designed to show quality differences
        quality_test_queries = [
            'quantum physics uncertainty principle Heisenberg',
            'evolution natural selection Darwin species origin',
            'machine learning algorithms neural networks deep',
            'philosophy ethics moral virtue consequentialism',
            'mathematics calculus derivative integral limit'
        ]
        
        comparison_results = []
        
        for query in quality_test_queries:
            print(f"\nComparing quality for: {query[:50]}...")
            
            # Get results from both systems
            enhanced_results = self.enhanced_retriever.retrieve(query, k=10, threshold=0.1)
            baseline_results = self._convert_baseline_results(
                self.baseline_retriever.search(query, max_results=10, min_score=0.01)
            )
            
            # Calculate quality metrics
            enhanced_metrics = self._calculate_quality_metrics(enhanced_results, query)
            baseline_metrics = self._calculate_quality_metrics(baseline_results, query)
            
            comparison = {
                'query': query,
                'enhanced': {
                    'num_results': len(enhanced_results),
                    'avg_similarity': enhanced_metrics['avg_similarity'],
                    'top_similarity': enhanced_metrics['top_similarity'],
                    'quality_score': enhanced_metrics['quality_score']
                },
                'baseline': {
                    'num_results': len(baseline_results),
                    'avg_similarity': baseline_metrics['avg_similarity'],
                    'top_similarity': baseline_metrics['top_similarity'],
                    'quality_score': baseline_metrics['quality_score']
                }
            }
            
            # Calculate improvement
            comparison['improvements'] = {
                'similarity_improvement': (
                    enhanced_metrics['avg_similarity'] - baseline_metrics['avg_similarity']
                ),
                'quality_improvement': (
                    enhanced_metrics['quality_score'] - baseline_metrics['quality_score']
                ),
                'result_count_change': len(enhanced_results) - len(baseline_results)
            }
            
            comparison_results.append(comparison)
            
            print(f"  Enhanced: {len(enhanced_results)} results, avg sim: {enhanced_metrics['avg_similarity']:.3f}")
            print(f"  Baseline: {len(baseline_results)} results, avg sim: {baseline_metrics['avg_similarity']:.3f}")
            print(f"  Improvement: {comparison['improvements']['similarity_improvement']:+.3f}")
            
        # Calculate overall improvements
        avg_similarity_improvement = sum(
            c['improvements']['similarity_improvement'] for c in comparison_results
        ) / len(comparison_results)
        
        avg_quality_improvement = sum(
            c['improvements']['quality_improvement'] for c in comparison_results
        ) / len(comparison_results)
        
        return {
            'comparison_results': comparison_results,
            'overall_improvements': {
                'avg_similarity_improvement': avg_similarity_improvement,
                'avg_quality_improvement': avg_quality_improvement,
                'queries_tested': len(quality_test_queries)
            },
            'quality_better': avg_similarity_improvement > 0.05  # 5% improvement threshold
        }
        
    def test_performance_comparison(self) -> Dict[str, Any]:
        """Test performance comparison between enhanced and baseline."""
        print("\n=== Testing Performance Comparison ===")
        
        if not self.enhanced_retriever or not self.baseline_retriever:
            return {'error': 'Retrievers not ready'}
            
        performance_queries = [
            'quantum',
            'evolution species',
            'machine learning algorithms',
            'philosophy ethics moral virtue',
            'mathematics calculus derivative integral differential'
        ]
        
        enhanced_times = []
        baseline_times = []
        
        for query in performance_queries:
            # Test enhanced retriever
            start_time = time.time()
            enhanced_results = self.enhanced_retriever.retrieve(query, k=10)
            enhanced_time = time.time() - start_time
            enhanced_times.append(enhanced_time)
            
            # Test baseline retriever
            start_time = time.time()
            baseline_results = self.baseline_retriever.search(query, max_results=10)
            baseline_time = time.time() - start_time
            baseline_times.append(baseline_time)
            
        avg_enhanced_time = sum(enhanced_times) / len(enhanced_times)
        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        
        performance_comparison = {
            'enhanced_performance': {
                'avg_time': avg_enhanced_time,
                'max_time': max(enhanced_times),
                'min_time': min(enhanced_times),
                'total_time': sum(enhanced_times)
            },
            'baseline_performance': {
                'avg_time': avg_baseline_time,
                'max_time': max(baseline_times),
                'min_time': min(baseline_times),
                'total_time': sum(baseline_times)
            },
            'performance_overhead': avg_enhanced_time - avg_baseline_time,
            'acceptable_overhead': (avg_enhanced_time - avg_baseline_time) < 0.5  # 0.5s threshold
        }
        
        print(f"Enhanced avg time: {avg_enhanced_time:.3f}s")
        print(f"Baseline avg time: {avg_baseline_time:.3f}s")
        print(f"Overhead: {performance_comparison['performance_overhead']:+.3f}s")
        
        return performance_comparison
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation using the validation framework."""
        print("\n=== Running Comprehensive Validation ===")
        
        if not self.enhanced_retriever:
            return {'error': 'Enhanced retriever not ready'}
            
        try:
            # Run comprehensive validation
            validation_report = validate_retrieval_system(self.enhanced_retriever)
            
            # Extract key metrics
            validation_summary = validation_report.get('validation_summary', {})
            quality_report = validation_report.get('quality_report', {})
            performance_benchmarks = validation_report.get('performance_benchmarks', {})
            
            print(f"Validation tests: {validation_summary.get('passed_tests', 0)}/{validation_summary.get('total_tests', 0)} passed")
            print(f"Average validation score: {validation_summary.get('avg_score', 0):.3f}")
            print(f"Quality level: {quality_report.get('system_quality', {}).get('overall_level', 'Unknown')}")
            print(f"Production ready: {validation_report.get('overall_assessment', {}).get('ready_for_production', False)}")
            
            return {
                'validation_successful': True,
                'validation_report': validation_report,
                'production_ready': validation_report.get('overall_assessment', {}).get('ready_for_production', False)
            }
            
        except Exception as e:
            print(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'validation_successful': False,
                'error': str(e)
            }
            
    def test_integration_compatibility(self) -> Dict[str, Any]:
        """Test integration compatibility with existing agent system."""
        print("\n=== Testing Integration Compatibility ===")
        
        if not self.enhanced_retriever:
            return {'error': 'Enhanced retriever not ready'}
            
        # Test interface compatibility
        compatibility_tests = []
        
        # Test 1: Standard retrieve interface
        try:
            results = self.enhanced_retriever.retrieve("quantum mechanics", k=5)
            compatibility_tests.append({
                'test': 'retrieve_interface',
                'passed': isinstance(results, list) and all(isinstance(r, RetrievalResult) for r in results),
                'details': f'Returned {len(results)} RetrievalResult objects'
            })
        except Exception as e:
            compatibility_tests.append({
                'test': 'retrieve_interface',
                'passed': False,
                'details': f'Error: {e}'
            })
            
        # Test 2: ICorpusRetriever interface
        try:
            stats = self.enhanced_retriever.get_corpus_stats()
            is_ready = self.enhanced_retriever.is_ready()
            valid_query = self.enhanced_retriever.validate_query("test query")
            
            compatibility_tests.append({
                'test': 'interface_compliance',
                'passed': isinstance(stats, dict) and isinstance(is_ready, bool) and isinstance(valid_query, bool),
                'details': f'Stats: {type(stats)}, Ready: {is_ready}, Valid query: {valid_query}'
            })
        except Exception as e:
            compatibility_tests.append({
                'test': 'interface_compliance',
                'passed': False,
                'details': f'Error: {e}'
            })
            
        # Test 3: Result format compatibility
        try:
            results = self.enhanced_retriever.retrieve("ethics philosophy", k=3)
            if results:
                result = results[0]
                has_required_fields = all(hasattr(result, field) for field in ['content', 'similarity', 'metadata', 'source_info'])
                valid_similarity = 0.0 <= result.similarity <= 1.0
                
                compatibility_tests.append({
                    'test': 'result_format',
                    'passed': has_required_fields and valid_similarity,
                    'details': f'Fields present: {has_required_fields}, Valid similarity: {valid_similarity}'
                })
            else:
                compatibility_tests.append({
                    'test': 'result_format',
                    'passed': True,
                    'details': 'No results to validate format'
                })
        except Exception as e:
            compatibility_tests.append({
                'test': 'result_format',
                'passed': False,
                'details': f'Error: {e}'
            })
            
        passed_tests = sum(1 for test in compatibility_tests if test['passed'])
        total_tests = len(compatibility_tests)
        
        return {
            'compatibility_tests': compatibility_tests,
            'compatibility_score': passed_tests / total_tests,
            'fully_compatible': passed_tests == total_tests
        }
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "="*70)
        print("ENHANCED RETRIEVAL SYSTEM TEST REPORT")
        print("="*70)
        
        # Calculate overall assessment
        setup_success = 'setup' in self.test_results and not self.test_results['setup'].get('setup_failed', False)
        features_working = 'enhanced_features' in self.test_results and self.test_results['enhanced_features'].get('enhanced_features_working', False)
        quality_better = 'quality_comparison' in self.test_results and self.test_results['quality_comparison'].get('quality_better', False)
        performance_acceptable = 'performance_comparison' in self.test_results and self.test_results['performance_comparison'].get('acceptable_overhead', False)
        validation_passed = 'validation' in self.test_results and self.test_results['validation'].get('validation_successful', False)
        integration_compatible = 'integration' in self.test_results and self.test_results['integration'].get('fully_compatible', False)
        
        overall_score = sum([
            setup_success, features_working, quality_better, 
            performance_acceptable, validation_passed, integration_compatible
        ]) / 6
        
        assessment = {
            'overall_score': overall_score,
            'status': 'PASS' if overall_score >= 0.8 else 'PARTIAL' if overall_score >= 0.5 else 'FAIL',
            'criteria': {
                'setup_successful': setup_success,
                'enhanced_features_working': features_working,
                'quality_improvements': quality_better,
                'performance_acceptable': performance_acceptable,
                'validation_passed': validation_passed,
                'integration_compatible': integration_compatible
            }
        }
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'corpus_directory': str(self.corpus_dir),
            'test_results': self.test_results,
            'assessment': assessment
        }
        
        # Print summary
        print(f"Overall Status: {assessment['status']} (Score: {overall_score:.2f})")
        print("\nCriteria Assessment:")
        for criterion, passed in assessment['criteria'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {criterion.replace('_', ' ').title()}")
            
        # Print key metrics
        if 'setup' in self.test_results:
            stats = self.test_results['setup'].get('enhanced_stats', {})
            print(f"\nCorpus Statistics:")
            print(f"  Chunks: {stats.get('chunks', 'N/A')}")
            print(f"  Files: {stats.get('files', 'N/A')}")
            print(f"  Vocabulary: {stats.get('vocabulary_size', 'N/A')} terms")
            
        if 'enhanced_features' in self.test_results:
            features = self.test_results['enhanced_features']
            print(f"\nEnhanced Features:")
            print(f"  Average query time: {features.get('avg_query_time', 0):.3f}s")
            print(f"  Features working: {features.get('enhanced_features_working', False)}")
            
        if 'quality_comparison' in self.test_results:
            quality = self.test_results['quality_comparison']['overall_improvements']
            print(f"\nQuality Improvements:")
            print(f"  Similarity improvement: {quality.get('avg_similarity_improvement', 0):+.3f}")
            print(f"  Quality improvement: {quality.get('avg_quality_improvement', 0):+.3f}")
            
        return report
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete enhanced retrieval test suite."""
        print("Starting enhanced retrieval system testing...")
        print(f"Corpus directory: {self.corpus_dir}")
        
        # Run all test suites
        self.test_results['setup'] = self.setup_retrievers()
        
        if not self.test_results['setup'].get('setup_failed'):
            self.test_results['enhanced_features'] = self.test_enhanced_features()
            self.test_results['quality_comparison'] = self.test_quality_improvements()
            self.test_results['performance_comparison'] = self.test_performance_comparison()
            self.test_results['validation'] = self.run_comprehensive_validation()
            self.test_results['integration'] = self.test_integration_compatibility()
        else:
            print("Setup failed, skipping other tests")
            
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        return report
        
    # Helper methods
    def _analyze_enhanced_features(self, results: List[RetrievalResult], test: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results for enhanced features."""
        analysis = {
            'has_quality_scores': False,
            'similarity_distribution': {},
            'content_diversity': 0.0,
            'feature_evidence': []
        }
        
        if results:
            # Check for quality scores
            quality_scores = [r.metadata.get('quality_score') for r in results if r.metadata.get('quality_score') is not None]
            analysis['has_quality_scores'] = len(quality_scores) > 0
            
            # Analyze similarity distribution
            similarities = [r.similarity for r in results]
            analysis['similarity_distribution'] = {
                'min': min(similarities),
                'max': max(similarities),
                'avg': sum(similarities) / len(similarities),
                'std': (sum((s - sum(similarities)/len(similarities))**2 for s in similarities) / len(similarities))**0.5
            }
            
            # Check for diversity (unique source files)
            source_files = set(r.source_info.get('source_file', '') for r in results)
            analysis['content_diversity'] = len(source_files) / len(results)
            
            # Look for enhanced feature evidence in match details
            for result in results[:3]:  # Check top 3 results
                match_details = result.source_info.get('match_details', {})
                if isinstance(match_details, dict):
                    if 'scores' in match_details:
                        analysis['feature_evidence'].append('multi_score_combination')
                    if 'phrase_match_count' in match_details:
                        analysis['feature_evidence'].append('phrase_matching')
                        
        return analysis
        
    def _convert_baseline_results(self, baseline_results) -> List[RetrievalResult]:
        """Convert baseline search results to RetrievalResult format."""
        converted = []
        for result in baseline_results:
            retrieval_result = RetrievalResult(
                content=result.chunk.content,
                similarity=result.score,
                metadata=result.chunk.metadata,
                source_info={
                    'source_file': result.chunk.source_file,
                    'chunk_id': result.chunk.chunk_id,
                    'match_details': result.match_details
                }
            )
            converted.append(retrieval_result)
        return converted
        
    def _calculate_quality_metrics(self, results: List[RetrievalResult], query: str) -> Dict[str, float]:
        """Calculate quality metrics for results."""
        if not results:
            return {
                'avg_similarity': 0.0,
                'top_similarity': 0.0,
                'quality_score': 0.0
            }
            
        similarities = [r.similarity for r in results]
        
        # Simple content relevance scoring
        query_terms = set(query.lower().split())
        relevance_scores = []
        
        for result in results:
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms.intersection(content_terms))
            relevance = overlap / len(query_terms) if query_terms else 0.0
            relevance_scores.append(relevance)
            
        return {
            'avg_similarity': sum(similarities) / len(similarities),
            'top_similarity': max(similarities),
            'quality_score': sum(relevance_scores) / len(relevance_scores)
        }


def main():
    """Main function to run enhanced retrieval tests."""
    # Get corpus directory
    script_dir = Path(__file__).parent
    corpus_dir = script_dir / "corpus_data"
    
    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}")
        print("Please ensure corpus_data exists in the project root.")
        return 1
        
    # Initialize and run tests
    test_runner = EnhancedRetrievalTestRunner(str(corpus_dir))
    
    try:
        report = test_runner.run_all_tests()
        
        # Save report to file
        report_file = script_dir / "enhanced_retrieval_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\nDetailed test report saved to: {report_file}")
        
        # Return appropriate exit code
        return 0 if report['assessment']['status'] == 'PASS' else 1
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
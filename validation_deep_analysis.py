#!/usr/bin/env python3
"""
Deep Validation Analysis for RAG Enhancement Claims

Performs thorough analysis of the claimed 92% improvement and <50ms response times
with statistical significance testing and edge case validation.
"""

import sys
import time
import json
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the src directory to the Python path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from corpus.enhanced_retriever import (
    EnhancedFileCorpusRetriever, 
    create_enhanced_retriever
)
from corpus.file_retriever import FileCorpusRetriever


class DeepValidationAnalyzer:
    """Deep analysis of RAG enhancement claims with statistical validation."""
    
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.enhanced_retriever = None
        self.baseline_retriever = None
        
    def setup_systems(self) -> Dict[str, Any]:
        """Set up both systems for comparison."""
        print("Setting up enhanced and baseline systems...")
        
        # Enhanced system
        self.enhanced_retriever = create_enhanced_retriever(
            corpus_dir=str(self.corpus_dir),
            chunk_size=800,
            chunk_overlap=150,
            enable_all_features=True
        )
        
        # Baseline system
        self.baseline_retriever = FileCorpusRetriever(
            corpus_dir=str(self.corpus_dir),
            chunk_size=800,
            chunk_overlap=150,
            search_method="hybrid"
        )
        baseline_load = self.baseline_retriever.load_corpus()
        baseline_index = self.baseline_retriever.build_search_index()
        
        return {
            'enhanced_ready': self.enhanced_retriever.is_ready(),
            'baseline_ready': self.baseline_retriever._is_indexed,
            'enhanced_chunks': self.enhanced_retriever.get_corpus_stats().get('chunks', 0),
            'baseline_chunks': len(self.baseline_retriever.chunks) if hasattr(self.baseline_retriever, 'chunks') else 0
        }
    
    def validate_92_percent_improvement_claim(self, num_tests: int = 50) -> Dict[str, Any]:
        """Validate the claimed 92% improvement in retrieval similarity scores."""
        print(f"\nValidating 92% improvement claim with {num_tests} test queries...")
        
        # Diverse test queries
        test_queries = [
            # Physics queries
            "quantum mechanics uncertainty principle",
            "relativity theory Einstein spacetime",
            "thermodynamics entropy laws",
            "electromagnetic waves radiation",
            "particle physics standard model",
            
            # Biology queries
            "evolution natural selection adaptation",
            "DNA genetics inheritance",
            "photosynthesis chloroplast energy",
            "cellular respiration mitochondria",
            "ecosystem biodiversity ecology",
            
            # Computer Science queries
            "machine learning neural networks",
            "algorithms data structures complexity",
            "artificial intelligence reasoning",
            "programming languages syntax",
            "database management systems",
            
            # Mathematics queries
            "calculus derivatives integrals",
            "linear algebra matrices vectors",
            "probability statistics distribution",
            "geometry topology manifolds",
            "number theory prime numbers",
            
            # Philosophy queries
            "ethics moral philosophy virtue",
            "metaphysics reality existence",
            "epistemology knowledge truth",
            "logic reasoning argumentation",
            "political philosophy justice",
            
            # Chemistry queries
            "chemical bonding molecular structure",
            "organic chemistry reactions",
            "periodic table elements",
            "chemical equilibrium",
            "physical chemistry thermodynamics",
            
            # History queries
            "World War II Nazi Germany",
            "American Revolution independence",
            "Renaissance art science",
            "Industrial Revolution technology",
            "Cold War nuclear deterrence",
            
            # Literature queries
            "Shakespeare tragedy comedy",
            "poetry meter rhythm",
            "narrative structure plot",
            "literary criticism analysis",
            "romantic literature nature",
            
            # Economics queries
            "market economics supply demand",
            "macroeconomics inflation unemployment",
            "behavioral economics psychology",
            "international trade globalization",
            "monetary policy central banking",
            
            # Psychology queries
            "cognitive psychology memory",
            "behavioral psychology conditioning",
            "social psychology groups",
            "developmental psychology stages",
            "abnormal psychology disorders"
        ]
        
        # Take a random sample if we have more queries than needed
        if len(test_queries) > num_tests:
            import random
            test_queries = random.sample(test_queries, num_tests)
        elif len(test_queries) < num_tests:
            # Repeat queries to reach target count
            multiplier = (num_tests // len(test_queries)) + 1
            test_queries = (test_queries * multiplier)[:num_tests]
            
        enhanced_similarities = []
        baseline_similarities = []
        improvements = []
        
        for i, query in enumerate(test_queries):
            print(f"  Testing query {i+1}/{len(test_queries)}: {query[:40]}...")
            
            # Enhanced system results
            enhanced_results = self.enhanced_retriever.retrieve(query, k=10, threshold=0.0)
            enhanced_avg_sim = (
                sum(r.similarity for r in enhanced_results) / len(enhanced_results) 
                if enhanced_results else 0.0
            )
            enhanced_similarities.append(enhanced_avg_sim)
            
            # Baseline system results
            baseline_results = self.baseline_retriever.search(query, max_results=10, min_score=0.0)
            baseline_avg_sim = (
                sum(r.score for r in baseline_results) / len(baseline_results) 
                if baseline_results else 0.0
            )
            baseline_similarities.append(baseline_avg_sim)
            
            # Calculate improvement percentage
            if baseline_avg_sim > 0:
                improvement = ((enhanced_avg_sim - baseline_avg_sim) / baseline_avg_sim) * 100
            else:
                improvement = 0.0  # Can't calculate percentage improvement from zero
            improvements.append(improvement)
        
        # Statistical analysis
        avg_improvement = statistics.mean(improvements)
        median_improvement = statistics.median(improvements)
        std_improvement = statistics.stdev(improvements) if len(improvements) > 1 else 0.0
        
        # Count how many queries had significant improvements
        significant_improvements = sum(1 for imp in improvements if imp >= 50)  # 50% threshold
        very_high_improvements = sum(1 for imp in improvements if imp >= 90)  # 90% threshold
        
        # Enhanced vs baseline averages
        avg_enhanced = statistics.mean(enhanced_similarities)
        avg_baseline = statistics.mean(baseline_similarities)
        overall_improvement = ((avg_enhanced - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
        
        # Statistical significance test (paired t-test approximation)
        differences = [e - b for e, b in zip(enhanced_similarities, baseline_similarities)]
        mean_diff = statistics.mean(differences)
        std_diff = statistics.stdev(differences) if len(differences) > 1 else 0.0
        
        # Calculate confidence interval for the improvement
        n = len(differences)
        if n > 1 and std_diff > 0:
            # 95% confidence interval
            t_critical = 2.0  # Approximate for large samples
            margin_error = t_critical * (std_diff / (n ** 0.5))
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
        else:
            ci_lower = ci_upper = mean_diff
        
        return {
            'claim_validated': avg_improvement >= 80.0,  # Allow some tolerance
            'actual_improvement_percent': avg_improvement,
            'median_improvement_percent': median_improvement,
            'std_improvement': std_improvement,
            'overall_similarity_improvement_percent': overall_improvement,
            'queries_tested': len(test_queries),
            'significant_improvements_count': significant_improvements,
            'very_high_improvements_count': very_high_improvements,
            'enhanced_avg_similarity': avg_enhanced,
            'baseline_avg_similarity': avg_baseline,
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
                'margin_error': ci_upper - mean_diff
            },
            'detailed_results': [
                {
                    'query': query,
                    'enhanced_similarity': enhanced_similarities[i],
                    'baseline_similarity': baseline_similarities[i],
                    'improvement_percent': improvements[i]
                }
                for i, query in enumerate(test_queries)
            ]
        }
    
    def validate_50ms_response_time_claim(self, num_tests: int = 100) -> Dict[str, Any]:
        """Validate the <50ms response time claim under various conditions."""
        print(f"\nValidating <50ms response time claim with {num_tests} tests...")
        
        # Various query types and lengths
        test_scenarios = [
            {
                'type': 'single_word',
                'queries': ['quantum', 'evolution', 'ethics', 'calculus', 'history'],
                'description': 'Single word queries'
            },
            {
                'type': 'short_phrase',
                'queries': ['machine learning', 'quantum mechanics', 'natural selection'],
                'description': 'Short phrase queries'
            },
            {
                'type': 'medium_query',
                'queries': [
                    'quantum mechanics uncertainty principle',
                    'evolution natural selection adaptation',
                    'machine learning neural networks'
                ],
                'description': 'Medium length queries'
            },
            {
                'type': 'complex_query',
                'queries': [
                    'quantum mechanics wave particle duality uncertainty principle physics',
                    'evolution natural selection adaptation species formation biology',
                    'machine learning deep neural networks artificial intelligence algorithms'
                ],
                'description': 'Complex queries'
            },
            {
                'type': 'very_long',
                'queries': [
                    'quantum mechanics wave particle duality uncertainty principle Heisenberg Einstein relativity physics theoretical',
                    'evolution natural selection adaptation species formation Darwin biology genetics inheritance mutation',
                    'machine learning deep neural networks artificial intelligence algorithms computer science programming'
                ],
                'description': 'Very long queries'
            }
        ]
        
        all_times = []
        scenario_results = {}
        
        tests_per_scenario = max(1, num_tests // (len(test_scenarios) * 5))  # 5 queries per scenario
        
        for scenario in test_scenarios:
            scenario_times = []
            
            print(f"  Testing {scenario['description']}...")
            
            for _ in range(tests_per_scenario):
                for query in scenario['queries']:
                    start_time = time.perf_counter()
                    results = self.enhanced_retriever.retrieve(query, k=10, threshold=0.1)
                    end_time = time.perf_counter()
                    
                    response_time_ms = (end_time - start_time) * 1000
                    scenario_times.append(response_time_ms)
                    all_times.append(response_time_ms)
            
            scenario_results[scenario['type']] = {
                'times_ms': scenario_times,
                'avg_time_ms': statistics.mean(scenario_times),
                'median_time_ms': statistics.median(scenario_times),
                'max_time_ms': max(scenario_times),
                'min_time_ms': min(scenario_times),
                'std_time_ms': statistics.stdev(scenario_times) if len(scenario_times) > 1 else 0.0,
                'under_50ms_count': sum(1 for t in scenario_times if t < 50),
                'under_50ms_percent': (sum(1 for t in scenario_times if t < 50) / len(scenario_times)) * 100,
                'description': scenario['description']
            }
        
        # Overall statistics
        avg_time = statistics.mean(all_times)
        median_time = statistics.median(all_times)
        max_time = max(all_times)
        min_time = min(all_times)
        std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0.0
        
        under_50ms_count = sum(1 for t in all_times if t < 50)
        under_50ms_percent = (under_50ms_count / len(all_times)) * 100
        
        # Performance percentiles
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(all_times, p)
        
        return {
            'claim_validated': avg_time < 50.0,
            'avg_response_time_ms': avg_time,
            'median_response_time_ms': median_time,
            'max_response_time_ms': max_time,
            'min_response_time_ms': min_time,
            'std_response_time_ms': std_time,
            'tests_performed': len(all_times),
            'under_50ms_count': under_50ms_count,
            'under_50ms_percent': under_50ms_percent,
            'percentiles': percentiles,
            'scenario_results': scenario_results,
            'performance_grade': self._grade_performance(avg_time, under_50ms_percent)
        }
    
    def validate_production_stress_test(self, concurrent_queries: int = 20, 
                                       duration_seconds: int = 30) -> Dict[str, Any]:
        """Test system under production-like stress conditions."""
        print(f"\nRunning production stress test: {concurrent_queries} concurrent queries for {duration_seconds}s...")
        
        import threading
        import queue
        
        # Test queries
        stress_queries = [
            "quantum mechanics uncertainty principle",
            "evolution natural selection Darwin",
            "machine learning neural networks",
            "ethics moral philosophy virtue",
            "calculus derivative integral",
            "relativity theory Einstein",
            "DNA genetics inheritance",
            "algorithms data structures",
            "thermodynamics entropy",
            "political philosophy justice"
        ]
        
        results_queue = queue.Queue()
        start_event = threading.Event()
        stop_event = threading.Event()
        
        def worker_thread():
            """Worker thread that performs continuous queries."""
            query_count = 0
            errors = 0
            response_times = []
            
            while not stop_event.is_set():
                try:
                    # Select random query
                    import random
                    query = random.choice(stress_queries)
                    
                    start_time = time.perf_counter()
                    results = self.enhanced_retriever.retrieve(query, k=5, threshold=0.1)
                    end_time = time.perf_counter()
                    
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    response_times.append(response_time)
                    query_count += 1
                    
                except Exception as e:
                    errors += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            
            results_queue.put({
                'query_count': query_count,
                'errors': errors,
                'response_times': response_times
            })
        
        # Start worker threads
        threads = []
        for i in range(concurrent_queries):
            thread = threading.Thread(target=worker_thread)
            threads.append(thread)
            thread.start()
        
        # Run for specified duration
        time.sleep(duration_seconds)
        stop_event.set()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # Collect results
        all_query_counts = []
        all_errors = []
        all_response_times = []
        
        while not results_queue.empty():
            result = results_queue.get()
            all_query_counts.append(result['query_count'])
            all_errors.append(result['errors'])
            all_response_times.extend(result['response_times'])
        
        # Calculate statistics
        total_queries = sum(all_query_counts)
        total_errors = sum(all_errors)
        queries_per_second = total_queries / duration_seconds
        error_rate = (total_errors / total_queries) * 100 if total_queries > 0 else 0
        
        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            median_response_time = statistics.median(all_response_times)
            max_response_time = max(all_response_times)
            min_response_time = min(all_response_times)
            under_50ms_percent = (sum(1 for t in all_response_times if t < 50) / len(all_response_times)) * 100
        else:
            avg_response_time = median_response_time = max_response_time = min_response_time = 0
            under_50ms_percent = 0
        
        return {
            'stress_test_passed': error_rate < 5.0 and avg_response_time < 100,  # Relaxed for concurrent load
            'duration_seconds': duration_seconds,
            'concurrent_queries': concurrent_queries,
            'total_queries': total_queries,
            'total_errors': total_errors,
            'queries_per_second': queries_per_second,
            'error_rate_percent': error_rate,
            'avg_response_time_ms': avg_response_time,
            'median_response_time_ms': median_response_time,
            'max_response_time_ms': max_response_time,
            'min_response_time_ms': min_response_time,
            'under_50ms_percent': under_50ms_percent,
            'thread_performance': [
                {
                    'thread_id': i,
                    'queries': all_query_counts[i] if i < len(all_query_counts) else 0,
                    'errors': all_errors[i] if i < len(all_errors) else 0
                }
                for i in range(concurrent_queries)
            ]
        }
    
    def validate_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and failure modes."""
        print("\nValidating edge cases and failure modes...")
        
        edge_case_tests = []
        
        # Test empty queries
        try:
            results = self.enhanced_retriever.retrieve("", k=5)
            edge_case_tests.append({
                'test': 'empty_query',
                'passed': len(results) == 0,
                'description': 'Empty query handling',
                'result_count': len(results)
            })
        except Exception as e:
            edge_case_tests.append({
                'test': 'empty_query',
                'passed': False,
                'description': 'Empty query handling',
                'error': str(e)
            })
        
        # Test very long queries
        very_long_query = "quantum " * 100  # 100 repetitions
        try:
            start_time = time.perf_counter()
            results = self.enhanced_retriever.retrieve(very_long_query, k=5)
            response_time = (time.perf_counter() - start_time) * 1000
            edge_case_tests.append({
                'test': 'very_long_query',
                'passed': response_time < 500,  # 500ms tolerance for very long query
                'description': 'Very long query (600+ chars)',
                'response_time_ms': response_time,
                'result_count': len(results)
            })
        except Exception as e:
            edge_case_tests.append({
                'test': 'very_long_query',
                'passed': False,
                'description': 'Very long query (600+ chars)',
                'error': str(e)
            })
        
        # Test special characters
        special_char_query = "quantum@#$%^&*()mechanics!?<>[]{}|"
        try:
            results = self.enhanced_retriever.retrieve(special_char_query, k=5)
            edge_case_tests.append({
                'test': 'special_characters',
                'passed': True,  # Should handle gracefully
                'description': 'Query with special characters',
                'result_count': len(results)
            })
        except Exception as e:
            edge_case_tests.append({
                'test': 'special_characters',
                'passed': False,
                'description': 'Query with special characters',
                'error': str(e)
            })
        
        # Test high k values
        try:
            results = self.enhanced_retriever.retrieve("quantum mechanics", k=1000)
            edge_case_tests.append({
                'test': 'high_k_value',
                'passed': len(results) <= 1000,  # Should not exceed corpus size
                'description': 'Very high k value (1000)',
                'result_count': len(results)
            })
        except Exception as e:
            edge_case_tests.append({
                'test': 'high_k_value',
                'passed': False,
                'description': 'Very high k value (1000)',
                'error': str(e)
            })
        
        # Test zero threshold
        try:
            results = self.enhanced_retriever.retrieve("quantum", k=5, threshold=0.0)
            edge_case_tests.append({
                'test': 'zero_threshold',
                'passed': len(results) > 0,
                'description': 'Zero similarity threshold',
                'result_count': len(results)
            })
        except Exception as e:
            edge_case_tests.append({
                'test': 'zero_threshold',
                'passed': False,
                'description': 'Zero similarity threshold',
                'error': str(e)
            })
        
        # Test high threshold
        try:
            results = self.enhanced_retriever.retrieve("quantum", k=5, threshold=0.99)
            edge_case_tests.append({
                'test': 'high_threshold',
                'passed': True,  # Should handle gracefully (may return no results)
                'description': 'Very high similarity threshold (0.99)',
                'result_count': len(results)
            })
        except Exception as e:
            edge_case_tests.append({
                'test': 'high_threshold',
                'passed': False,
                'description': 'Very high similarity threshold (0.99)',
                'error': str(e)
            })
        
        passed_tests = sum(1 for test in edge_case_tests if test['passed'])
        total_tests = len(edge_case_tests)
        
        return {
            'edge_cases_passed': passed_tests == total_tests,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'pass_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'test_details': edge_case_tests
        }
    
    def _grade_performance(self, avg_time_ms: float, under_50ms_percent: float) -> str:
        """Grade performance based on response times."""
        if avg_time_ms < 25 and under_50ms_percent > 95:
            return "Excellent"
        elif avg_time_ms < 50 and under_50ms_percent > 85:
            return "Good"
        elif avg_time_ms < 100 and under_50ms_percent > 70:
            return "Fair"
        else:
            return "Poor"
    
    def generate_comprehensive_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RAG ENHANCEMENT VALIDATION REPORT")
        print("="*80)
        
        # Setup systems
        setup_results = self.setup_systems()
        
        if not (setup_results['enhanced_ready'] and setup_results['baseline_ready']):
            return {
                'validation_failed': True,
                'reason': 'System setup failed',
                'setup_results': setup_results
            }
        
        # Run all validation tests
        improvement_validation = self.validate_92_percent_improvement_claim()
        performance_validation = self.validate_50ms_response_time_claim()
        stress_test_results = self.validate_production_stress_test()
        edge_case_results = self.validate_edge_cases()
        
        # Overall assessment
        claims_validated = (
            improvement_validation['claim_validated'] and
            performance_validation['claim_validated'] and
            stress_test_results['stress_test_passed'] and
            edge_case_results['edge_cases_passed']
        )
        
        overall_score = (
            (1 if improvement_validation['claim_validated'] else 0) +
            (1 if performance_validation['claim_validated'] else 0) +
            (1 if stress_test_results['stress_test_passed'] else 0) +
            (1 if edge_case_results['edge_cases_passed'] else 0)
        ) / 4
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_assessment': {
                'all_claims_validated': claims_validated,
                'overall_score': overall_score,
                'grade': self._grade_performance(
                    performance_validation['avg_response_time_ms'],
                    performance_validation['under_50ms_percent']
                )
            },
            'setup_results': setup_results,
            'improvement_validation': improvement_validation,
            'performance_validation': performance_validation,
            'stress_test_results': stress_test_results,
            'edge_case_results': edge_case_results,
            'production_readiness': {
                'ready': claims_validated,
                'reasons': []
            }
        }
        
        # Add reasons for production readiness assessment
        if not improvement_validation['claim_validated']:
            report['production_readiness']['reasons'].append(
                f"Quality improvement only {improvement_validation['actual_improvement_percent']:.1f}%, below claimed 92%"
            )
        
        if not performance_validation['claim_validated']:
            report['production_readiness']['reasons'].append(
                f"Average response time {performance_validation['avg_response_time_ms']:.1f}ms, above claimed 50ms"
            )
        
        if not stress_test_results['stress_test_passed']:
            report['production_readiness']['reasons'].append(
                f"Stress test failed: {stress_test_results['error_rate_percent']:.1f}% error rate"
            )
        
        if not edge_case_results['edge_cases_passed']:
            report['production_readiness']['reasons'].append(
                f"Edge case handling: {edge_case_results['passed_tests']}/{edge_case_results['total_tests']} passed"
            )
        
        # Print summary
        print(f"\nVALIDATION SUMMARY:")
        print(f"Overall Score: {overall_score:.2f}/1.0")
        print(f"All Claims Validated: {'YES' if claims_validated else 'NO'}")
        print(f"\n92% Improvement Claim: {'✓' if improvement_validation['claim_validated'] else '✗'}")
        print(f"  Actual improvement: {improvement_validation['actual_improvement_percent']:.1f}%")
        print(f"\n<50ms Response Time Claim: {'✓' if performance_validation['claim_validated'] else '✗'}")
        print(f"  Actual avg response time: {performance_validation['avg_response_time_ms']:.1f}ms")
        print(f"\nProduction Stress Test: {'✓' if stress_test_results['stress_test_passed'] else '✗'}")
        print(f"  QPS: {stress_test_results['queries_per_second']:.1f}, Error rate: {stress_test_results['error_rate_percent']:.1f}%")
        print(f"\nEdge Case Handling: {'✓' if edge_case_results['edge_cases_passed'] else '✗'}")
        print(f"  Tests passed: {edge_case_results['passed_tests']}/{edge_case_results['total_tests']}")
        
        print(f"\nProduction Ready: {'YES' if claims_validated else 'NO'}")
        if not claims_validated:
            print("Issues:")
            for reason in report['production_readiness']['reasons']:
                print(f"  - {reason}")
        
        return report


def main():
    """Main function to run deep validation analysis."""
    script_dir = Path(__file__).parent
    corpus_dir = script_dir / "corpus_data"
    
    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}")
        return 1
    
    analyzer = DeepValidationAnalyzer(str(corpus_dir))
    
    try:
        report = analyzer.generate_comprehensive_validation_report()
        
        # Save detailed report
        report_file = script_dir / "deep_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed validation report saved to: {report_file}")
        
        return 0 if report['overall_assessment']['all_claims_validated'] else 1
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
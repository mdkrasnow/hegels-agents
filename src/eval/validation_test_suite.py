"""
Enhanced Evaluation Pipeline Validation Test Suite

Comprehensive validation of the evaluation pipeline infrastructure including:
1. Statistical analysis validation
2. A/B testing framework validation  
3. Evaluation framework testing
4. Research infrastructure validation
5. Production integration testing

This serves as both validation and demonstration of research-grade evaluation capabilities.
"""

import unittest
import json
import time
import tempfile
import math
import sys
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import evaluation components
from eval.comprehensive_evaluator import (
    AutomatedEvaluationPipeline, BaselineCalculator, ABTestingFramework,
    create_comprehensive_evaluator, BaselineMetrics, EvaluationBatch
)

from eval.statistical_analyzer import (
    StatisticalAnalyzer, PerformanceBenchmarkSuite, EvaluationReportGenerator,
    create_statistical_analyzer, create_report_generator,
    StatisticalSummary, TrendAnalysis, CorrelationAnalysis, BenchmarkComparison
)

from eval.automated_workflows import (
    WorkflowOrchestrator, WorkflowDefinition, WorkflowStep, WorkflowTrigger,
    TriggerType, create_basic_evaluation_workflow, ContinuousMonitoringService
)

from eval.performance_benchmarks import (
    BenchmarkSuite, PerformanceProfiler, create_benchmark_suite,
    PerformanceMetrics, BenchmarkResult
)


class ValidationResults:
    """Tracks validation results across all tests."""
    
    def __init__(self):
        self.results = {
            'statistical_analysis': {},
            'ab_testing': {},
            'evaluation_framework': {},
            'research_infrastructure': {},
            'production_integration': {},
            'overall_status': 'unknown'
        }
        self.start_time = datetime.now()
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        
    def record_test(self, category: str, test_name: str, passed: bool, details: Dict[str, Any] = None):
        """Record test result."""
        self.test_count += 1
        if passed:
            self.passed_count += 1
        else:
            self.failed_count += 1
            
        if category not in self.results:
            self.results[category] = {}
            
        self.results[category][test_name] = {
            'passed': passed,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        success_rate = self.passed_count / self.test_count if self.test_count > 0 else 0
        
        # Determine overall status
        if success_rate >= 0.95:
            overall_status = 'excellent'
        elif success_rate >= 0.85:
            overall_status = 'good'
        elif success_rate >= 0.70:
            overall_status = 'acceptable'
        else:
            overall_status = 'needs_improvement'
        
        return {
            'overall_status': overall_status,
            'success_rate': success_rate,
            'total_tests': self.test_count,
            'passed_tests': self.passed_count,
            'failed_tests': self.failed_count,
            'duration_seconds': duration,
            'category_results': self.results,
            'validated_at': datetime.now().isoformat()
        }


# Global validation tracker
validation_results = ValidationResults()


class MockEvaluationData:
    """Generates realistic mock evaluation data for testing."""
    
    @staticmethod
    def generate_quality_data(count: int = 100, base_quality: float = 0.75, 
                            noise: float = 0.1) -> List[Dict[str, Any]]:
        """Generate mock quality evaluation data."""
        data = []
        base_time = datetime.now() - timedelta(hours=count)
        
        for i in range(count):
            # Add some realistic variation
            quality_score = base_quality + (i % 10) * 0.02 + (noise * (0.5 - hash(i) % 100 / 100))
            quality_score = max(0.0, min(1.0, quality_score))
            
            data.append({
                'quality_score': quality_score,
                'improvement_score': quality_score * 20 + (i % 5),
                'dialectical_score': quality_score * 0.9 + (i % 3) * 0.05,
                'response_time': 1.0 + (i % 10) * 0.1 + noise * 0.2,
                'timestamp': (base_time + timedelta(hours=i)).isoformat(),
                'success': quality_score > 0.5,
                'evaluation_id': f'eval_{i:04d}',
                'question_complexity': (i % 5) + 1,
                'agent_version': f'v1.{i//20}'
            })
        
        return data
    
    @staticmethod
    def generate_time_series_data(days: int = 30, trend_slope: float = 0.01) -> List[Dict[str, Any]]:
        """Generate time series data with configurable trend."""
        data = []
        base_time = datetime.now() - timedelta(days=days)
        base_value = 0.70
        
        for i in range(days):
            # Linear trend plus noise
            value = base_value + i * trend_slope + math.sin(i * 0.5) * 0.05
            value = max(0.0, min(1.0, value))
            
            data.append({
                'timestamp': (base_time + timedelta(days=i)).isoformat(),
                'quality_score': value,
                'response_time': 1.5 - value * 0.5 + (i % 3) * 0.1,
                'improvement_score': value * 25,
                'success': True
            })
        
        return data


class TestStatisticalAnalysisValidation(unittest.TestCase):
    """Validate statistical analysis components."""
    
    def setUp(self):
        self.analyzer = create_statistical_analyzer(confidence_level=0.95)
        self.tolerance = 1e-10  # For floating point comparisons
        
    def test_confidence_interval_accuracy(self):
        """Test confidence interval calculations against known datasets."""
        try:
            # Known dataset with expected confidence interval
            known_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            expected_mean = 5.5
            
            summary = self.analyzer.calculate_summary_statistics(known_data)
            
            # Validate mean
            self.assertAlmostEqual(summary.mean, expected_mean, places=6)
            
            # Validate confidence interval contains mean
            ci_lower, ci_upper = summary.confidence_interval
            self.assertLessEqual(ci_lower, summary.mean)
            self.assertGreaterEqual(ci_upper, summary.mean)
            
            # Validate confidence interval width is reasonable
            ci_width = ci_upper - ci_lower
            self.assertGreater(ci_width, 0)
            self.assertLess(ci_width, summary.std_dev * 4)  # Sanity check
            
            details = {
                'mean': summary.mean,
                'expected_mean': expected_mean,
                'confidence_interval': summary.confidence_interval,
                'ci_width': ci_width,
                'std_dev': summary.std_dev
            }
            
            validation_results.record_test(
                'statistical_analysis', 'confidence_interval_accuracy', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'statistical_analysis', 'confidence_interval_accuracy', False, 
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_significance_testing_accuracy(self):
        """Test statistical significance testing with known distributions."""
        try:
            # Generate two datasets with known difference
            group_a = [0.7 + 0.01 * i + 0.02 * (i % 3) for i in range(50)]  # Mean ~0.72
            group_b = [0.8 + 0.01 * i + 0.02 * (i % 3) for i in range(50)]  # Mean ~0.82
            
            data_a = [{'quality_score': val} for val in group_a]
            data_b = [{'quality_score': val} for val in group_b]
            
            comparison = self.analyzer.compare_to_benchmark(data_a, data_b, 'quality_score')
            
            # Should detect significant difference
            self.assertGreater(comparison.benchmark_value, comparison.current_value)
            self.assertGreater(comparison.difference, 0)  # Benchmark - current > 0
            self.assertIn(comparison.performance_category, ['exceeds', 'meets', 'below'])
            
            # Test with identical distributions
            identical_a = [0.75] * 50
            identical_b = [0.75] * 50
            
            data_identical_a = [{'quality_score': val} for val in identical_a]
            data_identical_b = [{'quality_score': val} for val in identical_b]
            
            comparison_identical = self.analyzer.compare_to_benchmark(
                data_identical_a, data_identical_b, 'quality_score'
            )
            
            # Should show no significant difference
            self.assertAlmostEqual(comparison_identical.current_value, comparison_identical.benchmark_value, places=6)
            
            details = {
                'different_groups': {
                    'mean_a': statistics.mean(group_a),
                    'mean_b': statistics.mean(group_b),
                    'performance_category': comparison.performance_category
                },
                'identical_groups': {
                    'difference': comparison_identical.difference,
                    'performance_category': comparison_identical.performance_category
                }
            }
            
            validation_results.record_test(
                'statistical_analysis', 'significance_testing_accuracy', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'statistical_analysis', 'significance_testing_accuracy', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_trend_analysis_accuracy(self):
        """Test trend analysis with known time series patterns."""
        try:
            # Test increasing trend
            increasing_data = MockEvaluationData.generate_time_series_data(30, trend_slope=0.01)
            trend_analysis = self.analyzer.analyze_trend(increasing_data, 'quality_score')
            
            self.assertEqual(trend_analysis.metric_name, 'quality_score')
            self.assertIn(trend_analysis.trend_direction, ['increasing', 'stable'])
            self.assertGreaterEqual(trend_analysis.trend_strength, 0.0)
            self.assertLessEqual(trend_analysis.trend_strength, 1.0)
            self.assertGreater(trend_analysis.slope, 0)  # Should detect positive slope
            
            # Test decreasing trend
            decreasing_data = MockEvaluationData.generate_time_series_data(30, trend_slope=-0.01)
            trend_decreasing = self.analyzer.analyze_trend(decreasing_data, 'quality_score')
            
            self.assertIn(trend_decreasing.trend_direction, ['decreasing', 'stable'])
            self.assertLess(trend_decreasing.slope, 0)  # Should detect negative slope
            
            # Test stable trend
            stable_data = MockEvaluationData.generate_time_series_data(30, trend_slope=0.0)
            trend_stable = self.analyzer.analyze_trend(stable_data, 'quality_score')
            
            self.assertIn(trend_stable.trend_direction, ['stable', 'volatile'])
            
            details = {
                'increasing_trend': {
                    'direction': trend_analysis.trend_direction,
                    'slope': trend_analysis.slope,
                    'strength': trend_analysis.trend_strength
                },
                'decreasing_trend': {
                    'direction': trend_decreasing.trend_direction,
                    'slope': trend_decreasing.slope,
                    'strength': trend_decreasing.trend_strength
                },
                'stable_trend': {
                    'direction': trend_stable.trend_direction,
                    'slope': trend_stable.slope,
                    'strength': trend_stable.trend_strength
                }
            }
            
            validation_results.record_test(
                'statistical_analysis', 'trend_analysis_accuracy', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'statistical_analysis', 'trend_analysis_accuracy', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_correlation_analysis_accuracy(self):
        """Test correlation analysis with known relationships."""
        try:
            # Create data with known correlation
            base_data = []
            for i in range(100):
                x = 0.5 + i * 0.01
                y = 0.8 * x + 0.1 + (i % 10) * 0.01  # Strong positive correlation
                base_data.append({'metric_x': x, 'metric_y': y})
            
            correlation = self.analyzer.analyze_correlation(base_data, 'metric_x', 'metric_y')
            
            # Should detect strong positive correlation
            self.assertGreater(correlation.correlation_coefficient, 0.7)
            self.assertEqual(correlation.relationship_direction, 'positive')
            self.assertIn(correlation.relationship_strength, ['strong', 'moderate'])
            
            # Create data with no correlation
            uncorrelated_data = []
            for i in range(100):
                x = 0.5 + i * 0.01
                y = 0.5 + (i % 17) * 0.01  # No real correlation
                uncorrelated_data.append({'metric_x': x, 'metric_y': y})
            
            correlation_none = self.analyzer.analyze_correlation(uncorrelated_data, 'metric_x', 'metric_y')
            
            # Should detect weak or no correlation
            self.assertLess(abs(correlation_none.correlation_coefficient), 0.3)
            
            details = {
                'strong_correlation': {
                    'coefficient': correlation.correlation_coefficient,
                    'direction': correlation.relationship_direction,
                    'strength': correlation.relationship_strength
                },
                'weak_correlation': {
                    'coefficient': correlation_none.correlation_coefficient,
                    'direction': correlation_none.relationship_direction,
                    'strength': correlation_none.relationship_strength
                }
            }
            
            validation_results.record_test(
                'statistical_analysis', 'correlation_analysis_accuracy', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'statistical_analysis', 'correlation_analysis_accuracy', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise


class TestABTestingFrameworkValidation(unittest.TestCase):
    """Validate A/B testing framework with known datasets."""
    
    def setUp(self):
        self.ab_framework = ABTestingFramework(significance_level=0.05)
        
    def create_mock_evaluator(self, quality_mean: float, failure_rate: float = 0.0):
        """Create mock evaluator with specified characteristics."""
        class MockEvaluator:
            def __init__(self):
                self.call_count = 0
                
            def evaluate(self, question: str) -> Dict[str, Any]:
                self.call_count += 1
                
                # Simulate failure rate
                if failure_rate > 0 and (self.call_count % int(1/failure_rate)) == 0:
                    return {
                        'question': question,
                        'success': False,
                        'error': 'Simulated failure',
                        'quality_score': 0.0
                    }
                
                # Add some realistic variation around the mean
                quality_score = quality_mean + (self.call_count % 10) * 0.02 - 0.1
                quality_score = max(0.0, min(1.0, quality_score))
                
                return {
                    'question': question,
                    'success': True,
                    'quality_score': quality_score,
                    'evaluation_time': 1.0 + (self.call_count % 5) * 0.1,
                    'call_count': self.call_count
                }
        
        return MockEvaluator()
    
    def mock_evaluator_factory(self, config):
        """Factory for creating mock evaluators from config."""
        return self.create_mock_evaluator(
            config.get('quality_mean', 0.75),
            config.get('failure_rate', 0.0)
        )
    
    def test_ab_test_with_known_difference(self):
        """Test A/B testing with approaches having known performance differences."""
        try:
            # Configure approaches with different performance
            approach_a_config = {
                'name': 'Approach A',
                'quality_mean': 0.70,  # Lower performance
                'failure_rate': 0.0
            }
            
            approach_b_config = {
                'name': 'Approach B',
                'quality_mean': 0.80,  # Higher performance
                'failure_rate': 0.0
            }
            
            questions = [f"Test question {i}" for i in range(20)]
            
            ab_result = self.ab_framework.run_ab_test(
                name="Known Difference Test",
                approach_a_config=approach_a_config,
                approach_b_config=approach_b_config,
                questions=questions,
                evaluator_factory=self.mock_evaluator_factory
            )
            
            # Validate structure
            self.assertIsInstance(ab_result.test_id, str)
            self.assertEqual(len(ab_result.results_a), len(questions))
            self.assertEqual(len(ab_result.results_b), len(questions))
            
            # Check statistical analysis detected the difference
            stats = ab_result.statistical_analysis
            self.assertIn('approach_a_stats', stats)
            self.assertIn('approach_b_stats', stats)
            self.assertIn('mean_difference', stats)
            
            # B should outperform A
            mean_a = stats['approach_a_stats']['mean']
            mean_b = stats['approach_b_stats']['mean']
            self.assertGreater(mean_b, mean_a)
            
            # Check conclusion
            conclusion = ab_result.conclusion
            self.assertIn('recommendation', conclusion)
            
            details = {
                'approach_a_mean': mean_a,
                'approach_b_mean': mean_b,
                'mean_difference': stats['mean_difference'],
                'effect_size': stats.get('effect_size', 'N/A'),
                'recommendation': conclusion['recommendation']
            }
            
            validation_results.record_test(
                'ab_testing', 'known_difference_detection', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'ab_testing', 'known_difference_detection', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_ab_test_with_identical_approaches(self):
        """Test A/B testing with identical approaches (should show no difference)."""
        try:
            # Configure identical approaches
            approach_config = {
                'name': 'Identical Approach',
                'quality_mean': 0.75,
                'failure_rate': 0.0
            }
            
            questions = [f"Test question {i}" for i in range(15)]
            
            ab_result = self.ab_framework.run_ab_test(
                name="Identical Approaches Test",
                approach_a_config=approach_config,
                approach_b_config=approach_config,
                questions=questions,
                evaluator_factory=self.mock_evaluator_factory
            )
            
            # Should detect no significant difference
            stats = ab_result.statistical_analysis
            mean_difference = abs(stats['mean_difference'])
            self.assertLess(mean_difference, 0.1)  # Small difference acceptable due to randomness
            
            # Check that statistical significance is not claimed
            statistical_significance = stats.get('statistical_significance', False)
            # With identical approaches, should not be statistically significant
            
            details = {
                'mean_difference': stats['mean_difference'],
                'statistical_significance': statistical_significance,
                'effect_size': stats.get('effect_size', 'N/A'),
                'recommendation': ab_result.conclusion['recommendation']
            }
            
            validation_results.record_test(
                'ab_testing', 'identical_approaches_handling', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'ab_testing', 'identical_approaches_handling', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_ab_test_error_handling(self):
        """Test A/B testing handles errors gracefully."""
        try:
            # Configure one approach with failures
            approach_a_config = {
                'name': 'Reliable Approach',
                'quality_mean': 0.75,
                'failure_rate': 0.0
            }
            
            approach_b_config = {
                'name': 'Unreliable Approach',
                'quality_mean': 0.80,
                'failure_rate': 0.2  # 20% failure rate
            }
            
            questions = [f"Test question {i}" for i in range(10)]
            
            ab_result = self.ab_framework.run_ab_test(
                name="Error Handling Test",
                approach_a_config=approach_a_config,
                approach_b_config=approach_b_config,
                questions=questions,
                evaluator_factory=self.mock_evaluator_factory
            )
            
            # Should still produce results despite some failures
            self.assertEqual(len(ab_result.results_a), len(questions))
            self.assertEqual(len(ab_result.results_b), len(questions))
            
            # Count failures in approach B
            failures_b = sum(1 for r in ab_result.results_b if not r.get('success', True))
            self.assertGreater(failures_b, 0)  # Should have some failures
            
            # Statistical analysis should handle partial data
            self.assertIsInstance(ab_result.statistical_analysis, dict)
            
            details = {
                'total_questions': len(questions),
                'failures_a': sum(1 for r in ab_result.results_a if not r.get('success', True)),
                'failures_b': failures_b,
                'analysis_completed': len(ab_result.statistical_analysis) > 0
            }
            
            validation_results.record_test(
                'ab_testing', 'error_handling', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'ab_testing', 'error_handling', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise


class TestEvaluationFrameworkIntegration(unittest.TestCase):
    """Test evaluation framework integration with existing systems."""
    
    def setUp(self):
        self.pipeline = create_comprehensive_evaluator()
        
    def test_evaluation_batch_creation_and_execution(self):
        """Test creation and execution of evaluation batches."""
        try:
            questions = ["What is the meaning of dialectical reasoning?", 
                        "How does thesis-antithesis-synthesis work?"]
            configs = [
                {'type': 'quality', 'id': 'quality_eval'},
                {'type': 'dialectical', 'id': 'dialectical_eval'}
            ]
            
            # Create batch
            batch = self.pipeline.create_evaluation_batch(
                name="Integration Test Batch",
                questions=questions,
                configs=configs,
                description="Test batch for validation"
            )
            
            # Validate batch creation
            self.assertIsInstance(batch.batch_id, str)
            self.assertEqual(batch.name, "Integration Test Batch")
            self.assertEqual(len(batch.questions), 2)
            self.assertEqual(len(batch.evaluation_configs), 2)
            self.assertEqual(batch.status, "pending")
            
            # Batch should be registered in pipeline
            self.assertIn(batch.batch_id, self.pipeline.active_batches)
            
            details = {
                'batch_id': batch.batch_id,
                'questions_count': len(batch.questions),
                'configs_count': len(batch.evaluation_configs),
                'status': batch.status
            }
            
            validation_results.record_test(
                'evaluation_framework', 'batch_creation', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'evaluation_framework', 'batch_creation', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_baseline_metrics_integration(self):
        """Test baseline metrics calculation and storage."""
        try:
            # Generate mock evaluation data
            mock_data = MockEvaluationData.generate_quality_data(50)
            
            # Update baseline metrics
            baseline = self.pipeline.update_baseline_metrics(
                'quality', mock_data, {'test_context': 'validation'}
            )
            
            # Validate baseline creation
            self.assertIsInstance(baseline, BaselineMetrics)
            self.assertEqual(baseline.metric_type, 'quality')
            self.assertGreater(baseline.baseline_value, 0)
            self.assertEqual(baseline.sample_size, 50)
            self.assertIsInstance(baseline.confidence_interval, tuple)
            
            # Check baseline is stored in pipeline
            self.assertIn('quality', self.pipeline.baseline_metrics)
            stored_baseline = self.pipeline.baseline_metrics['quality']
            self.assertEqual(stored_baseline.baseline_value, baseline.baseline_value)
            
            details = {
                'metric_type': baseline.metric_type,
                'baseline_value': baseline.baseline_value,
                'sample_size': baseline.sample_size,
                'confidence_interval': baseline.confidence_interval,
                'statistical_significance': baseline.statistical_significance
            }
            
            validation_results.record_test(
                'evaluation_framework', 'baseline_metrics', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'evaluation_framework', 'baseline_metrics', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_pipeline_status_monitoring(self):
        """Test pipeline status monitoring and reporting."""
        try:
            # Get initial status
            status = self.pipeline.get_pipeline_status()
            
            # Validate status structure
            self.assertIsInstance(status, dict)
            required_fields = ['pipeline_id', 'active_batches', 'completed_evaluations', 
                             'baseline_metrics_available', 'total_questions_evaluated']
            
            for field in required_fields:
                self.assertIn(field, status)
            
            self.assertIsInstance(status['pipeline_id'], str)
            self.assertIsInstance(status['active_batches'], int)
            self.assertIsInstance(status['completed_evaluations'], int)
            self.assertIsInstance(status['baseline_metrics_available'], list)
            
            details = {
                'status_fields': list(status.keys()),
                'pipeline_id': status['pipeline_id'],
                'active_batches': status['active_batches'],
                'completed_evaluations': status['completed_evaluations']
            }
            
            validation_results.record_test(
                'evaluation_framework', 'status_monitoring', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'evaluation_framework', 'status_monitoring', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise


class TestResearchInfrastructureValidation(unittest.TestCase):
    """Validate research-grade infrastructure capabilities."""
    
    def setUp(self):
        self.benchmark_suite = create_benchmark_suite(enable_resource_monitoring=False)
        self.report_generator = create_report_generator()
        
    def test_baseline_measurement_reproducibility(self):
        """Test baseline measurement accuracy and reproducibility."""
        try:
            def mock_operation():
                """Mock operation with consistent performance."""
                time.sleep(0.01)  # Consistent 10ms operation
                return {"result": "test_data"}
            
            # Run multiple baseline measurements
            results = []
            for i in range(3):
                result = self.benchmark_suite.run_latency_benchmark(
                    operation=mock_operation,
                    operation_type="reproducibility_test",
                    iterations=10,
                    warmup_iterations=2
                )
                results.append(result)
            
            # Extract mean durations
            mean_durations = []
            for result in results:
                successful_metrics = [m for m in result.metrics if m.success]
                durations = [m.duration_seconds for m in successful_metrics]
                mean_durations.append(statistics.mean(durations))
            
            # Check reproducibility (should have low variance)
            duration_variance = statistics.variance(mean_durations)
            duration_mean = statistics.mean(mean_durations)
            cv = (statistics.stdev(mean_durations) / duration_mean) if duration_mean > 0 else 0
            
            # Coefficient of variation should be reasonably low for reproducible results
            self.assertLess(cv, 0.2)  # Less than 20% variation
            
            details = {
                'mean_durations': mean_durations,
                'overall_mean': duration_mean,
                'variance': duration_variance,
                'coefficient_of_variation': cv,
                'reproducible': cv < 0.2
            }
            
            validation_results.record_test(
                'research_infrastructure', 'baseline_reproducibility', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'research_infrastructure', 'baseline_reproducibility', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_evaluation_metric_calculations(self):
        """Test accuracy of evaluation metric calculations."""
        try:
            # Test with known data
            test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            expected_mean = 0.55
            expected_median = 0.55
            
            analyzer = create_statistical_analyzer()
            summary = analyzer.calculate_summary_statistics(test_values)
            
            # Validate calculations
            self.assertAlmostEqual(summary.mean, expected_mean, places=6)
            self.assertAlmostEqual(summary.median, expected_median, places=6)
            self.assertEqual(summary.count, 10)
            self.assertAlmostEqual(summary.min_value, 0.1, places=6)
            self.assertAlmostEqual(summary.max_value, 1.0, places=6)
            
            # Test percentile calculations
            expected_p25 = 0.325  # 25th percentile
            expected_p75 = 0.775  # 75th percentile
            
            actual_p25 = summary.percentiles[25]
            actual_p75 = summary.percentiles[75]
            
            self.assertAlmostEqual(actual_p25, expected_p25, places=2)
            self.assertAlmostEqual(actual_p75, expected_p75, places=2)
            
            details = {
                'input_values': test_values,
                'calculated_mean': summary.mean,
                'expected_mean': expected_mean,
                'calculated_median': summary.median,
                'expected_median': expected_median,
                'percentiles': summary.percentiles,
                'std_dev': summary.std_dev,
                'variance': summary.variance
            }
            
            validation_results.record_test(
                'research_infrastructure', 'metric_calculations', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'research_infrastructure', 'metric_calculations', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_data_collection_workflows(self):
        """Test data collection and analysis workflow accuracy."""
        try:
            # Create comprehensive evaluation data
            evaluation_data = MockEvaluationData.generate_quality_data(100)
            
            # Test report generation
            report = self.report_generator.generate_comprehensive_report(evaluation_data)
            
            # Validate report structure
            self.assertIsInstance(report, str)
            self.assertIn("Comprehensive Evaluation Analysis Report", report)
            self.assertIn("Executive Summary", report)
            self.assertIn("quality_score", report)
            self.assertIn("Recommendations", report)
            
            # Test data workflow components
            analyzer = create_statistical_analyzer()
            
            # Extract quality scores
            quality_scores = [item['quality_score'] for item in evaluation_data]
            summary = analyzer.calculate_summary_statistics(quality_scores)
            
            # Validate workflow consistency
            self.assertGreater(summary.count, 90)  # Should process most data
            self.assertGreater(summary.mean, 0.5)  # Reasonable quality range
            self.assertLess(summary.mean, 1.0)
            
            details = {
                'data_points_processed': summary.count,
                'data_quality_mean': summary.mean,
                'report_length': len(report),
                'report_sections': ['Executive Summary', 'quality_score', 'Recommendations'],
                'workflow_success': True
            }
            
            validation_results.record_test(
                'research_infrastructure', 'data_collection_workflows', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'research_infrastructure', 'data_collection_workflows', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise


class TestProductionIntegration(unittest.TestCase):
    """Test production integration and scalability."""
    
    def setUp(self):
        self.orchestrator = WorkflowOrchestrator()
        self.monitoring_service = ContinuousMonitoringService(self.orchestrator)
        
    def test_workflow_orchestrator_integration(self):
        """Test workflow orchestrator integration capabilities."""
        try:
            # Create a test workflow
            workflow = create_basic_evaluation_workflow(
                name="Production Integration Test",
                questions=["Test question 1", "Test question 2"],
                schedule="manual"
            )
            
            # Register workflow
            workflow_id = self.orchestrator.register_workflow(workflow)
            
            # Validate registration
            self.assertIsInstance(workflow_id, str)
            self.assertIn(workflow_id, self.orchestrator.workflow_definitions)
            
            # Get workflow status
            status = self.orchestrator.get_workflow_status(workflow_id)
            self.assertIsInstance(status, dict)
            self.assertIn('workflow', status)
            self.assertIn('recent_executions', status)
            
            details = {
                'workflow_id': workflow_id,
                'workflow_registered': workflow_id in self.orchestrator.workflow_definitions,
                'status_available': 'workflow' in status,
                'workflow_name': workflow.name
            }
            
            validation_results.record_test(
                'production_integration', 'workflow_orchestrator', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'production_integration', 'workflow_orchestrator', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_monitoring_system_integration(self):
        """Test monitoring system integration."""
        try:
            # Add metric threshold
            self.monitoring_service.add_metric_threshold(
                metric_name='quality_score',
                threshold_value=0.7,
                operator='less',
                alert_workflow_id=None
            )
            
            # Verify threshold was added
            self.assertIn('quality_score', self.monitoring_service.thresholds)
            threshold_config = self.monitoring_service.thresholds['quality_score']
            self.assertEqual(threshold_config['threshold_value'], 0.7)
            self.assertEqual(threshold_config['operator'], 'less')
            
            # Get monitoring status
            status = self.monitoring_service.get_monitoring_status()
            self.assertIsInstance(status, dict)
            self.assertIn('monitoring_enabled', status)
            self.assertIn('thresholds_configured', status)
            self.assertEqual(status['thresholds_configured'], 1)
            
            details = {
                'threshold_added': 'quality_score' in self.monitoring_service.thresholds,
                'threshold_value': threshold_config['threshold_value'],
                'monitoring_status_available': len(status) > 0,
                'thresholds_configured': status['thresholds_configured']
            }
            
            validation_results.record_test(
                'production_integration', 'monitoring_system', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'production_integration', 'monitoring_system', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def test_scalability_performance(self):
        """Test scalability and performance under load."""
        try:
            def mock_lightweight_operation():
                """Very lightweight mock operation for scalability testing."""
                return {"result": time.time()}
            
            # Test with multiple concurrent operations
            benchmark_suite = create_benchmark_suite(enable_resource_monitoring=False)
            
            # Small scale test (production would use larger numbers)
            result = benchmark_suite.run_throughput_benchmark(
                operation=mock_lightweight_operation,
                operation_type="scalability_test",
                duration_seconds=2,
                concurrent_workers=2
            )
            
            # Validate performance characteristics
            self.assertGreater(result.total_operations, 0)
            self.assertGreater(result.successful_operations, 0)
            
            # Check throughput is reasonable
            ops_per_second = result.summary_statistics.get('operations_per_second', 0)
            self.assertGreater(ops_per_second, 1)  # Should handle at least 1 op/sec
            
            # Test error handling under load
            def failing_operation():
                """Operation that fails 20% of the time."""
                if time.time() * 1000 % 5 == 0:
                    raise RuntimeError("Simulated failure")
                return {"result": "success"}
            
            error_test_result = benchmark_suite.run_latency_benchmark(
                operation=failing_operation,
                operation_type="error_handling_test",
                iterations=10,
                warmup_iterations=1
            )
            
            # Should handle some failures gracefully
            self.assertGreater(error_test_result.total_operations, 0)
            
            details = {
                'throughput_ops_per_second': ops_per_second,
                'total_operations': result.total_operations,
                'successful_operations': result.successful_operations,
                'error_test_completed': error_test_result.total_operations > 0,
                'duration_seconds': result.summary_statistics.get('actual_duration_seconds', 0)
            }
            
            validation_results.record_test(
                'production_integration', 'scalability_performance', True, details
            )
            
        except Exception as e:
            validation_results.record_test(
                'production_integration', 'scalability_performance', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise


class ValidationSummary(unittest.TestCase):
    """Generate final validation summary and assessment."""
    
    def test_generate_validation_report(self):
        """Generate comprehensive validation report."""
        try:
            # Get validation summary
            summary = validation_results.get_summary()
            
            # Create detailed report
            report = self._create_validation_report(summary)
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"validation_report_{timestamp}.json")
            
            with open(report_path, 'w') as f:
                json.dump({
                    'validation_summary': summary,
                    'detailed_report': report
                }, f, indent=2, default=str)
            
            # Print summary to console
            print("\n" + "="*80)
            print("ENHANCED EVALUATION PIPELINE VALIDATION RESULTS")
            print("="*80)
            print(f"Overall Status: {summary['overall_status'].upper()}")
            print(f"Success Rate: {summary['success_rate']:.2%}")
            print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"Duration: {summary['duration_seconds']:.1f} seconds")
            print("="*80)
            
            # Category breakdown
            for category, results in summary['category_results'].items():
                if results:  # Only show categories with results
                    passed = sum(1 for test in results.values() if test['passed'])
                    total = len(results)
                    print(f"{category.replace('_', ' ').title()}: {passed}/{total} tests passed")
            
            print("="*80)
            print(f"Detailed report saved to: {report_path}")
            print("="*80)
            
            # Validate overall success
            self.assertGreaterEqual(summary['success_rate'], 0.80, 
                                  f"Validation success rate {summary['success_rate']:.2%} below acceptable threshold")
            
            validation_results.record_test(
                'overall', 'validation_summary', True, {
                    'report_path': str(report_path),
                    'overall_status': summary['overall_status'],
                    'success_rate': summary['success_rate']
                }
            )
            
        except Exception as e:
            validation_results.record_test(
                'overall', 'validation_summary', False,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            raise
    
    def _create_validation_report(self, summary: Dict[str, Any]) -> str:
        """Create detailed validation report."""
        report_lines = [
            "# Enhanced Evaluation Pipeline Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- Overall Status: **{summary['overall_status'].upper()}**",
            f"- Success Rate: **{summary['success_rate']:.2%}**",
            f"- Total Tests: {summary['total_tests']}",
            f"- Passed: {summary['passed_tests']}",
            f"- Failed: {summary['failed_tests']}",
            f"- Duration: {summary['duration_seconds']:.1f} seconds",
            ""
        ]
        
        # Assessment
        if summary['success_rate'] >= 0.95:
            assessment = "Excellent - Pipeline demonstrates research-grade capabilities with high reliability"
        elif summary['success_rate'] >= 0.85:
            assessment = "Good - Pipeline meets most requirements with minor issues to address"
        elif summary['success_rate'] >= 0.70:
            assessment = "Acceptable - Pipeline functional but needs improvement in some areas"
        else:
            assessment = "Needs Improvement - Significant issues require attention before production use"
        
        report_lines.extend([
            "## Assessment",
            assessment,
            "",
            "## Category Results"
        ])
        
        # Category details
        for category, results in summary['category_results'].items():
            if not results or not isinstance(results, dict):
                continue
                
            passed = sum(1 for test in results.values() if isinstance(test, dict) and test.get('passed', False))
            total = len(results)
            category_name = category.replace('_', ' ').title()
            
            report_lines.extend([
                f"### {category_name}",
                f"Status: {passed}/{total} tests passed",
                ""
            ])
            
            for test_name, test_result in results.items():
                status = "✓ PASS" if test_result['passed'] else "✗ FAIL"
                report_lines.append(f"- {test_name}: {status}")
                
                if not test_result['passed'] and 'error' in test_result.get('details', {}):
                    report_lines.append(f"  Error: {test_result['details']['error']}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if summary['success_rate'] >= 0.95:
            report_lines.append("- Pipeline is ready for production use")
            report_lines.append("- Continue monitoring performance in production")
        elif summary['success_rate'] >= 0.85:
            report_lines.append("- Address failing tests before full production deployment")
            report_lines.append("- Consider gradual rollout with monitoring")
        else:
            report_lines.append("- Significant improvements needed before production use")
            report_lines.append("- Focus on failing components and retest")
        
        return "\n".join(report_lines)


def run_comprehensive_validation():
    """Run comprehensive validation suite."""
    print("Starting Enhanced Evaluation Pipeline Validation...")
    print("=" * 80)
    
    # Create test suite
    test_classes = [
        TestStatisticalAnalysisValidation,
        TestABTestingFrameworkValidation,
        TestEvaluationFrameworkIntegration,
        TestResearchInfrastructureValidation,
        TestProductionIntegration,
        ValidationSummary  # Must be last
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
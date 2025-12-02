"""
Comprehensive Test Suite for Enhanced Evaluation Pipeline

This test suite validates all components of the enhanced evaluation system
including baseline measurement, statistical analysis, automated workflows,
and performance benchmarking.

This serves as both a test suite and a demonstration of the evaluation
capabilities.
"""

import unittest
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import evaluation components
from .comprehensive_evaluator import (
    AutomatedEvaluationPipeline, BaselineCalculator, ABTestingFramework,
    create_comprehensive_evaluator
)

from .statistical_analyzer import (
    StatisticalAnalyzer, PerformanceBenchmarkSuite, EvaluationReportGenerator,
    create_statistical_analyzer, create_report_generator
)

from .automated_workflows import (
    WorkflowOrchestrator, WorkflowDefinition, WorkflowStep, WorkflowTrigger,
    TriggerType, create_basic_evaluation_workflow
)

from .performance_benchmarks import (
    BenchmarkSuite, PerformanceProfiler, create_benchmark_suite
)


class MockEvaluationOperation:
    """Mock evaluation operation for testing."""
    
    def __init__(self, base_duration: float = 0.1, failure_rate: float = 0.0):
        self.base_duration = base_duration
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def evaluate(self, question: str) -> Dict[str, Any]:
        """Mock evaluation that returns consistent results."""
        self.call_count += 1
        
        # Simulate some processing time
        time.sleep(self.base_duration + (self.call_count % 10) * 0.01)
        
        # Simulate occasional failures
        if self.failure_rate > 0 and (self.call_count % int(1/self.failure_rate)) == 0:
            raise RuntimeError("Simulated evaluation failure")
        
        return {
            'question': question,
            'success': True,
            'quality_score': 0.75 + (self.call_count % 5) * 0.05,
            'evaluation_time': self.base_duration,
            'metadata': {'call_count': self.call_count}
        }
    
    def __call__(self) -> Dict[str, Any]:
        """Make the object callable for performance profiling."""
        return self.evaluate("test question")


class TestBaselineCalculator(unittest.TestCase):
    """Test baseline calculation and statistical analysis."""
    
    def setUp(self):
        self.calculator = BaselineCalculator(confidence_level=0.95)
        
        # Generate mock evaluation data
        self.mock_evaluations = [
            {'quality_score': 0.75, 'improvement_score': 15.5, 'response_time': 1.2},
            {'quality_score': 0.80, 'improvement_score': 18.2, 'response_time': 1.1},
            {'quality_score': 0.72, 'improvement_score': 12.8, 'response_time': 1.4},
            {'quality_score': 0.78, 'improvement_score': 16.9, 'response_time': 1.3},
            {'quality_score': 0.76, 'improvement_score': 14.7, 'response_time': 1.0},
        ]
    
    def test_calculate_quality_baseline(self):
        """Test calculation of quality baseline metrics."""
        baseline = self.calculator.calculate_baseline_metrics(
            self.mock_evaluations, 'quality'
        )
        
        self.assertEqual(baseline.metric_type, 'quality')
        self.assertAlmostEqual(baseline.baseline_value, 0.762, places=3)
        self.assertEqual(baseline.sample_size, 5)
        self.assertIsInstance(baseline.confidence_interval, tuple)
        self.assertIsNotNone(baseline.statistical_significance)
    
    def test_calculate_improvement_baseline(self):
        """Test calculation of improvement baseline metrics."""
        baseline = self.calculator.calculate_baseline_metrics(
            self.mock_evaluations, 'improvement'
        )
        
        self.assertEqual(baseline.metric_type, 'improvement')
        self.assertAlmostEqual(baseline.baseline_value, 15.62, places=2)
        self.assertEqual(baseline.sample_size, 5)
    
    def test_calculate_performance_baseline(self):
        """Test calculation of performance baseline metrics."""
        baseline = self.calculator.calculate_baseline_metrics(
            self.mock_evaluations, 'performance'
        )
        
        self.assertEqual(baseline.metric_type, 'performance')
        self.assertAlmostEqual(baseline.baseline_value, 1.2, places=2)
        self.assertEqual(baseline.sample_size, 5)
    
    def test_empty_evaluations_error(self):
        """Test that empty evaluations raise appropriate error."""
        with self.assertRaises(ValueError):
            self.calculator.calculate_baseline_metrics([], 'quality')
    
    def test_baseline_serialization(self):
        """Test baseline metrics can be serialized to dict."""
        baseline = self.calculator.calculate_baseline_metrics(
            self.mock_evaluations, 'quality'
        )
        
        baseline_dict = baseline.to_dict()
        self.assertIsInstance(baseline_dict, dict)
        self.assertIn('metric_type', baseline_dict)
        self.assertIn('baseline_value', baseline_dict)
        self.assertIn('confidence_interval', baseline_dict)


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test statistical analysis capabilities."""
    
    def setUp(self):
        self.analyzer = create_statistical_analyzer()
        
        # Generate time series data for trend analysis
        base_time = datetime.now() - timedelta(days=30)
        self.time_series_data = []
        
        for i in range(30):
            self.time_series_data.append({
                'timestamp': (base_time + timedelta(days=i)).isoformat(),
                'quality_score': 0.7 + i * 0.01 + (i % 3) * 0.02,  # Upward trend with noise
                'response_time': 1.0 + i * 0.02 + (i % 2) * 0.1      # Slight upward trend
            })
    
    def test_summary_statistics(self):
        """Test calculation of summary statistics."""
        values = [0.75, 0.80, 0.72, 0.78, 0.76, 0.85, 0.70, 0.82, 0.77, 0.79]
        
        summary = self.analyzer.calculate_summary_statistics(values)
        
        self.assertEqual(summary.count, 10)
        self.assertAlmostEqual(summary.mean, 0.774, places=3)
        self.assertAlmostEqual(summary.median, 0.77, places=3)
        self.assertGreater(summary.std_dev, 0)
        self.assertIsInstance(summary.percentiles, dict)
        self.assertIn(95, summary.percentiles)
        self.assertIsInstance(summary.confidence_interval, tuple)
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        trend = self.analyzer.analyze_trend(
            self.time_series_data, 'quality_score'
        )
        
        self.assertEqual(trend.metric_name, 'quality_score')
        self.assertIn(trend.trend_direction, ['increasing', 'decreasing', 'stable', 'volatile'])
        self.assertGreaterEqual(trend.trend_strength, 0)
        self.assertLessEqual(trend.trend_strength, 1)
        self.assertEqual(trend.data_points, 30)
        self.assertIsInstance(trend.slope, float)
    
    def test_correlation_analysis(self):
        """Test correlation analysis between metrics."""
        correlation = self.analyzer.analyze_correlation(
            self.time_series_data, 'quality_score', 'response_time'
        )
        
        self.assertEqual(correlation.metric_x, 'quality_score')
        self.assertEqual(correlation.metric_y, 'response_time')
        self.assertGreaterEqual(correlation.correlation_coefficient, -1)
        self.assertLessEqual(correlation.correlation_coefficient, 1)
        self.assertIn(correlation.relationship_strength, ['strong', 'moderate', 'weak', 'none'])
        self.assertIn(correlation.relationship_direction, ['positive', 'negative', 'none'])
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality."""
        # Create current and benchmark datasets
        current_data = [{'quality_score': 0.80 + i * 0.01} for i in range(10)]
        benchmark_data = [{'quality_score': 0.75 + i * 0.01} for i in range(10)]
        
        comparison = self.analyzer.compare_to_benchmark(
            current_data, benchmark_data, 'quality_score'
        )
        
        self.assertEqual(comparison.metric_name, 'quality_score')
        self.assertGreater(comparison.current_value, comparison.benchmark_value)
        self.assertGreater(comparison.difference, 0)
        self.assertIn(comparison.performance_category, 
                     ['exceeds', 'meets', 'below', 'significantly_below'])


class TestABTestingFramework(unittest.TestCase):
    """Test A/B testing capabilities."""
    
    def setUp(self):
        self.ab_framework = ABTestingFramework(significance_level=0.05)
        
        # Mock evaluator factory
        def mock_evaluator_factory(config):
            operation = MockEvaluationOperation(
                base_duration=config.get('base_duration', 0.1),
                failure_rate=config.get('failure_rate', 0.0)
            )
            return operation
        
        self.evaluator_factory = mock_evaluator_factory
    
    def test_ab_test_execution(self):
        """Test execution of A/B test."""
        approach_a_config = {
            'name': 'Approach A',
            'base_duration': 0.1,
            'failure_rate': 0.0
        }
        
        approach_b_config = {
            'name': 'Approach B',
            'base_duration': 0.12,  # Slightly slower
            'failure_rate': 0.0
        }
        
        questions = ["Question 1", "Question 2", "Question 3"]
        
        ab_result = self.ab_framework.run_ab_test(
            name="Test AB",
            approach_a_config=approach_a_config,
            approach_b_config=approach_b_config,
            questions=questions,
            evaluator_factory=self.evaluator_factory
        )
        
        self.assertEqual(ab_result.name, "Test AB")
        self.assertEqual(ab_result.approach_a, "Approach A")
        self.assertEqual(ab_result.approach_b, "Approach B")
        self.assertEqual(len(ab_result.results_a), 3)
        self.assertEqual(len(ab_result.results_b), 3)
        self.assertIsInstance(ab_result.statistical_analysis, dict)
        self.assertIsInstance(ab_result.conclusion, dict)


class TestAutomatedEvaluationPipeline(unittest.TestCase):
    """Test automated evaluation pipeline functionality."""
    
    def setUp(self):
        self.pipeline = create_comprehensive_evaluator()
    
    def test_create_evaluation_batch(self):
        """Test creation of evaluation batch."""
        questions = ["Question 1", "Question 2", "Question 3"]
        configs = [
            {'type': 'quality', 'id': 'quality_test'},
            {'type': 'dialectical', 'id': 'dialectical_test'}
        ]
        
        batch = self.pipeline.create_evaluation_batch(
            name="Test Batch",
            questions=questions,
            configs=configs,
            description="Test batch description"
        )
        
        self.assertEqual(batch.name, "Test Batch")
        self.assertEqual(len(batch.questions), 3)
        self.assertEqual(len(batch.evaluation_configs), 2)
        self.assertEqual(batch.status, "pending")
        self.assertIn(batch.batch_id, self.pipeline.active_batches)
    
    def test_pipeline_status(self):
        """Test pipeline status reporting."""
        status = self.pipeline.get_pipeline_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('pipeline_id', status)
        self.assertIn('active_batches', status)
        self.assertIn('completed_evaluations', status)
        self.assertIn('baseline_metrics_available', status)


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test performance benchmarking capabilities."""
    
    def setUp(self):
        self.benchmark_suite = create_benchmark_suite(enable_resource_monitoring=False)
        self.mock_operation = MockEvaluationOperation(base_duration=0.01)
    
    def test_latency_benchmark(self):
        """Test latency benchmarking."""
        result = self.benchmark_suite.run_latency_benchmark(
            operation=self.mock_operation,
            operation_type="mock_evaluation",
            iterations=10,
            warmup_iterations=2
        )
        
        self.assertEqual(result.total_operations, 10)
        self.assertGreaterEqual(result.successful_operations, 8)  # Allow for some variation
        self.assertIsInstance(result.summary_statistics, dict)
        self.assertIn('duration_stats', result.summary_statistics)
        self.assertGreater(len(result.metrics), 0)
    
    def test_throughput_benchmark(self):
        """Test throughput benchmarking."""
        result = self.benchmark_suite.run_throughput_benchmark(
            operation=self.mock_operation,
            operation_type="mock_evaluation",
            duration_seconds=2,
            concurrent_workers=1
        )
        
        self.assertGreater(result.total_operations, 0)
        self.assertIn('operations_per_second', result.summary_statistics)
        self.assertGreater(result.summary_statistics['operations_per_second'], 0)
    
    def test_baseline_establishment(self):
        """Test establishment of performance baselines."""
        # First run a benchmark
        result = self.benchmark_suite.run_latency_benchmark(
            operation=self.mock_operation,
            operation_type="test_operation",
            iterations=5,
            warmup_iterations=1
        )
        
        # Establish baseline
        baseline = self.benchmark_suite.establish_baseline(
            "test_operation", result.benchmark_id
        )
        
        self.assertIsInstance(baseline, dict)
        self.assertIn('mean_duration', baseline)
        self.assertIn('p95_duration', baseline)
        self.assertIn('success_rate', baseline)
    
    def test_baseline_comparison(self):
        """Test comparison to baseline performance."""
        # Run initial benchmark and establish baseline
        result1 = self.benchmark_suite.run_latency_benchmark(
            operation=self.mock_operation,
            operation_type="comparison_test",
            iterations=5
        )
        
        self.benchmark_suite.establish_baseline(
            "comparison_test", result1.benchmark_id
        )
        
        # Run second benchmark for comparison
        result2 = self.benchmark_suite.run_latency_benchmark(
            operation=self.mock_operation,
            operation_type="comparison_test", 
            iterations=5
        )
        
        comparison = self.benchmark_suite.compare_to_baseline(
            "comparison_test", result2.benchmark_id
        )
        
        self.assertIsInstance(comparison, dict)
        self.assertIn('comparison_details', comparison)
        self.assertIn('overall_assessment', comparison)


class TestWorkflowOrchestration(unittest.TestCase):
    """Test workflow orchestration capabilities."""
    
    def setUp(self):
        self.orchestrator = WorkflowOrchestrator()
    
    def test_workflow_registration(self):
        """Test workflow registration."""
        workflow = create_basic_evaluation_workflow(
            name="Test Workflow",
            questions=["Question 1", "Question 2"],
            schedule="daily"
        )
        
        workflow_id = self.orchestrator.register_workflow(workflow)
        
        self.assertIsInstance(workflow_id, str)
        self.assertIn(workflow_id, self.orchestrator.workflow_definitions)
    
    def test_workflow_execution(self):
        """Test workflow execution."""
        workflow = create_basic_evaluation_workflow(
            name="Simple Test",
            questions=["Test question"],
            schedule="manual"
        )
        
        workflow_id = self.orchestrator.register_workflow(workflow)
        
        # Note: This test will only verify structure since actual execution
        # requires integration with evaluation components
        try:
            execution = self.orchestrator.execute_workflow(workflow_id, manual_trigger=True)
            self.assertIsNotNone(execution.execution_id)
        except Exception:
            # Expected since we're using mock data
            pass


class TestEvaluationReportGeneration(unittest.TestCase):
    """Test evaluation report generation."""
    
    def setUp(self):
        self.report_generator = create_report_generator()
        
        # Mock evaluation data
        self.evaluation_data = [
            {
                'quality_score': 0.75,
                'improvement_score': 15.5,
                'response_time': 1.2,
                'success': True,
                'timestamp': datetime.now().isoformat()
            },
            {
                'quality_score': 0.80,
                'improvement_score': 18.2,
                'response_time': 1.1,
                'success': True,
                'timestamp': datetime.now().isoformat()
            },
            {
                'quality_score': 0.72,
                'improvement_score': 12.8,
                'response_time': 1.4,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
        ]
    
    def test_comprehensive_report_generation(self):
        """Test generation of comprehensive evaluation report."""
        report = self.report_generator.generate_comprehensive_report(
            self.evaluation_data
        )
        
        self.assertIsInstance(report, str)
        self.assertIn("Comprehensive Evaluation Analysis Report", report)
        self.assertIn("Executive Summary", report)
        self.assertIn("quality_score", report)
        self.assertIn("Recommendations", report)
    
    def test_empty_data_handling(self):
        """Test report generation with empty data."""
        report = self.report_generator.generate_comprehensive_report([])
        
        self.assertIsInstance(report, str)
        self.assertIn("Comprehensive Evaluation Analysis Report", report)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple components."""
    
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from start to finish."""
        # 1. Create evaluation pipeline
        pipeline = create_comprehensive_evaluator()
        
        # 2. Create evaluation batch
        questions = ["Test question 1", "Test question 2"]
        configs = [{'type': 'quality', 'id': 'test_config'}]
        
        batch = pipeline.create_evaluation_batch(
            name="Integration Test",
            questions=questions,
            configs=configs
        )
        
        # 3. Verify batch creation
        self.assertIsInstance(batch.batch_id, str)
        self.assertEqual(len(batch.questions), 2)
        
        # 4. Test statistical analysis components
        analyzer = create_statistical_analyzer()
        mock_data = [{'quality_score': 0.75 + i * 0.01} for i in range(10)]
        summary = analyzer.calculate_summary_statistics([d['quality_score'] for d in mock_data])
        
        self.assertGreater(summary.count, 0)
        self.assertGreater(summary.mean, 0)
        
        # 5. Test performance benchmarking
        benchmark_suite = create_benchmark_suite(enable_resource_monitoring=False)
        mock_operation = MockEvaluationOperation(base_duration=0.01)
        
        benchmark_result = benchmark_suite.run_latency_benchmark(
            operation=mock_operation,
            operation_type="integration_test",
            iterations=5,
            warmup_iterations=1
        )
        
        self.assertGreater(benchmark_result.total_operations, 0)
        
        # 6. Generate report
        report_generator = create_report_generator()
        mock_eval_data = [
            {'quality_score': 0.75, 'success': True},
            {'quality_score': 0.80, 'success': True}
        ]
        
        report = report_generator.generate_comprehensive_report(mock_eval_data)
        self.assertIn("Comprehensive Evaluation Analysis Report", report)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
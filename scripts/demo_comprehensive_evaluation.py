#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline Demonstration

This script demonstrates the enhanced evaluation capabilities including:
- Baseline performance measurement and statistical analysis
- A/B testing infrastructure and automated workflows
- Performance benchmarking and trend analysis
- Comprehensive reporting and monitoring

This serves as both a demonstration and a validation of the evaluation system.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# Import comprehensive evaluation components
from eval import (
    # Core evaluation pipeline
    create_comprehensive_evaluator, BaselineCalculator, ABTestingFramework,
    
    # Statistical analysis
    create_statistical_analyzer, create_report_generator, create_benchmark_suite,
    
    # Automated workflows
    WorkflowOrchestrator, create_basic_evaluation_workflow, create_monitoring_workflow,
    
    # Performance benchmarking
    create_benchmark_suite, PerformanceProfiler
)

from agents.utils import AgentLogger


class MockEvaluationSystem:
    """
    Mock evaluation system for demonstration purposes.
    
    In production, this would integrate with actual agents and evaluation logic.
    """
    
    def __init__(self, base_quality: float = 0.75, variability: float = 0.1):
        self.base_quality = base_quality
        self.variability = variability
        self.call_count = 0
        self.logger = AgentLogger("mock_evaluator")
    
    def evaluate_single_agent(self, question: str) -> Dict[str, Any]:
        """Mock single agent evaluation."""
        self.call_count += 1
        
        # Simulate evaluation time
        time.sleep(0.1 + (self.call_count % 5) * 0.02)
        
        # Generate mock metrics with some variability
        import random
        quality_score = max(0, min(1, random.normalvariate(self.base_quality, self.variability)))
        
        return {
            'evaluation_type': 'single_agent',
            'question': question,
            'quality_score': quality_score,
            'response_time': 0.1 + (self.call_count % 5) * 0.02,
            'success': True,
            'confidence': quality_score * 0.9,
            'metadata': {
                'call_count': self.call_count,
                'evaluator': 'mock_single'
            }
        }
    
    def evaluate_dialectical(self, question: str) -> Dict[str, Any]:
        """Mock dialectical evaluation."""
        self.call_count += 1
        
        # Simulate longer evaluation time for dialectical process
        time.sleep(0.2 + (self.call_count % 3) * 0.03)
        
        # Generate mock metrics - dialectical typically performs better
        import random
        quality_score = max(0, min(1, random.normalvariate(self.base_quality + 0.1, self.variability)))
        improvement = max(0, (quality_score - self.base_quality) / self.base_quality * 100)
        
        return {
            'evaluation_type': 'dialectical',
            'question': question,
            'quality_score': quality_score,
            'improvement_percentage': improvement,
            'response_time': 0.2 + (self.call_count % 3) * 0.03,
            'success': True,
            'dialectical_score': quality_score + 0.05,
            'metadata': {
                'call_count': self.call_count,
                'evaluator': 'mock_dialectical'
            }
        }
    
    def evaluate_question(self, question: str, eval_type: str = 'single') -> Dict[str, Any]:
        """Evaluate a question with specified type."""
        if eval_type == 'dialectical':
            return self.evaluate_dialectical(question)
        else:
            return self.evaluate_single_agent(question)


def demonstrate_baseline_measurement():
    """Demonstrate baseline performance measurement capabilities."""
    print("\n" + "="*60)
    print("🔬 BASELINE MEASUREMENT DEMONSTRATION")
    print("="*60)
    
    # Create baseline calculator
    calculator = BaselineCalculator(confidence_level=0.95)
    mock_system = MockEvaluationSystem(base_quality=0.75, variability=0.05)
    
    print("Collecting baseline evaluation data...")
    
    # Collect baseline evaluation data
    baseline_questions = [
        "What is artificial intelligence?",
        "Explain quantum computing principles",
        "Describe machine learning algorithms",
        "What are neural networks?",
        "How does natural language processing work?",
        "Explain deep learning concepts",
        "What is computer vision?", 
        "Describe reinforcement learning",
        "What are large language models?",
        "Explain data science methodology"
    ]
    
    baseline_evaluations = []
    for i, question in enumerate(baseline_questions):
        print(f"  Evaluating baseline question {i+1}/{len(baseline_questions)}")
        result = mock_system.evaluate_single_agent(question)
        baseline_evaluations.append(result)
    
    # Calculate baseline metrics
    quality_baseline = calculator.calculate_baseline_metrics(
        baseline_evaluations, 'quality', {'dataset': 'demonstration_set'}
    )
    
    performance_baseline = calculator.calculate_baseline_metrics(
        baseline_evaluations, 'performance', {'dataset': 'demonstration_set'}
    )
    
    print(f"\n📊 Baseline Results:")
    print(f"Quality Baseline: {quality_baseline.baseline_value:.3f} ± {quality_baseline.confidence_interval[1] - quality_baseline.baseline_value:.3f}")
    print(f"Performance Baseline: {performance_baseline.baseline_value:.3f}s ± {performance_baseline.confidence_interval[1] - performance_baseline.baseline_value:.3f}s")
    print(f"Sample Size: {quality_baseline.sample_size} evaluations")
    print(f"Statistical Significance: p={quality_baseline.statistical_significance:.4f}" if quality_baseline.statistical_significance else "Statistical Significance: Not calculated")
    
    return baseline_evaluations, quality_baseline, performance_baseline


def demonstrate_ab_testing():
    """Demonstrate A/B testing infrastructure."""
    print("\n" + "="*60)
    print("🧪 A/B TESTING DEMONSTRATION")
    print("="*60)
    
    ab_framework = ABTestingFramework(significance_level=0.05)
    
    # Mock evaluator factory for A/B testing
    def create_evaluator(config):
        return MockEvaluationSystem(
            base_quality=config.get('base_quality', 0.75),
            variability=config.get('variability', 0.05)
        )
    
    # Define test approaches
    approach_a_config = {
        'name': 'Standard Evaluation',
        'base_quality': 0.75,
        'variability': 0.05
    }
    
    approach_b_config = {
        'name': 'Enhanced Evaluation', 
        'base_quality': 0.80,  # Slightly better performance
        'variability': 0.05
    }
    
    # Test questions
    test_questions = [
        "Compare different machine learning approaches",
        "Analyze the ethics of AI development", 
        "Evaluate quantum computing applications",
        "Discuss neural network architectures",
        "Examine data privacy concerns"
    ]
    
    print(f"Running A/B test with {len(test_questions)} questions...")
    
    # Run A/B test
    ab_result = ab_framework.run_ab_test(
        name="Evaluation Method Comparison",
        approach_a_config=approach_a_config,
        approach_b_config=approach_b_config,
        questions=test_questions,
        evaluator_factory=create_evaluator
    )
    
    print(f"\n📈 A/B Test Results:")
    print(f"Test: {ab_result.name}")
    print(f"Approach A ({ab_result.approach_a}): {len(ab_result.results_a)} evaluations")
    print(f"Approach B ({ab_result.approach_b}): {len(ab_result.results_b)} evaluations")
    
    stats = ab_result.statistical_analysis
    print(f"\nStatistical Analysis:")
    print(f"  Approach A Mean: {stats['approach_a_stats']['mean']:.3f}")
    print(f"  Approach B Mean: {stats['approach_b_stats']['mean']:.3f}")
    print(f"  Difference: {stats['mean_difference']:.3f}")
    print(f"  Effect Size: {stats['effect_size']:.3f}")
    print(f"  Statistical Significance: {stats['statistical_significance']}")
    
    conclusion = ab_result.conclusion
    print(f"\nConclusion:")
    print(f"  Recommendation: {conclusion['recommendation']}")
    print(f"  Confidence: {conclusion['confidence']}")
    print(f"  Reason: {conclusion['reason']}")
    
    return ab_result


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARKING DEMONSTRATION")
    print("="*60)
    
    benchmark_suite = create_benchmark_suite(enable_resource_monitoring=True)
    mock_system = MockEvaluationSystem(base_quality=0.75)
    
    # Define benchmark operation
    def evaluation_operation():
        return mock_system.evaluate_single_agent("Benchmark test question")
    
    print("Running latency benchmark...")
    
    # Run latency benchmark
    latency_result = benchmark_suite.run_latency_benchmark(
        operation=evaluation_operation,
        operation_type="mock_evaluation",
        iterations=20,
        warmup_iterations=5
    )
    
    print(f"\n⏱️ Latency Benchmark Results:")
    print(f"Total Operations: {latency_result.total_operations}")
    print(f"Successful Operations: {latency_result.successful_operations}")
    print(f"Success Rate: {(latency_result.successful_operations/latency_result.total_operations)*100:.1f}%")
    
    duration_stats = latency_result.summary_statistics.get('duration_stats', {})
    print(f"Duration Statistics:")
    print(f"  Mean: {duration_stats.get('mean', 0):.3f}s")
    print(f"  Median: {duration_stats.get('median', 0):.3f}s")
    print(f"  95th percentile: {duration_stats.get('percentile_95', 0):.3f}s")
    print(f"  Standard deviation: {duration_stats.get('std_dev', 0):.3f}s")
    
    print("\nRunning throughput benchmark...")
    
    # Run throughput benchmark
    throughput_result = benchmark_suite.run_throughput_benchmark(
        operation=evaluation_operation,
        operation_type="mock_evaluation",
        duration_seconds=10,
        concurrent_workers=2
    )
    
    print(f"\n🚀 Throughput Benchmark Results:")
    print(f"Total Operations: {throughput_result.total_operations}")
    print(f"Operations per Second: {throughput_result.summary_statistics.get('operations_per_second', 0):.2f}")
    print(f"Concurrent Workers: {throughput_result.summary_statistics.get('concurrent_workers', 1)}")
    print(f"Duration: {throughput_result.summary_statistics.get('actual_duration_seconds', 0):.1f}s")
    
    # Establish and compare baselines
    print("\nEstablishing performance baseline...")
    baseline = benchmark_suite.establish_baseline(
        "mock_evaluation", latency_result.benchmark_id
    )
    
    print(f"Baseline established: {baseline['mean_duration']:.3f}s mean duration")
    
    return latency_result, throughput_result, baseline


def demonstrate_statistical_analysis():
    """Demonstrate statistical analysis and reporting capabilities."""
    print("\n" + "="*60)
    print("📊 STATISTICAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = create_statistical_analyzer()
    mock_system = MockEvaluationSystem(base_quality=0.75)
    
    # Generate time series data for trend analysis
    print("Generating time series evaluation data...")
    
    time_series_data = []
    base_time = datetime.now() - timedelta(days=14)
    
    for day in range(14):
        for hour in range(0, 24, 6):  # Every 6 hours
            timestamp = base_time + timedelta(days=day, hours=hour)
            
            # Simulate improving performance over time
            quality_trend = 0.70 + (day / 14) * 0.10  # Gradual improvement
            
            result = mock_system.evaluate_single_agent("Daily evaluation question")
            result['timestamp'] = timestamp.isoformat()
            result['quality_score'] = max(0.6, min(0.9, quality_trend + (result['quality_score'] - 0.75)))
            
            time_series_data.append(result)
    
    print(f"Generated {len(time_series_data)} data points over 14 days")
    
    # Perform trend analysis
    quality_trend = analyzer.analyze_trend(time_series_data, 'quality_score')
    
    print(f"\n📈 Trend Analysis Results:")
    print(f"Metric: {quality_trend.metric_name}")
    print(f"Time Period: {quality_trend.time_period}")
    print(f"Trend Direction: {quality_trend.trend_direction}")
    print(f"Trend Strength (R²): {quality_trend.trend_strength:.3f}")
    print(f"Slope: {quality_trend.slope:.6f}")
    print(f"Data Points: {quality_trend.data_points}")
    
    # Perform correlation analysis
    correlation = analyzer.analyze_correlation(
        time_series_data, 'quality_score', 'response_time'
    )
    
    print(f"\n🔗 Correlation Analysis:")
    print(f"Metrics: {correlation.metric_x} vs {correlation.metric_y}")
    print(f"Correlation Coefficient: {correlation.correlation_coefficient:.3f}")
    print(f"Relationship Strength: {correlation.relationship_strength}")
    print(f"Relationship Direction: {correlation.relationship_direction}")
    print(f"Sample Size: {correlation.sample_size}")
    
    # Generate comprehensive report
    report_generator = create_report_generator()
    comprehensive_report = report_generator.generate_comprehensive_report(
        time_series_data,
        {'title': 'Evaluation System Performance Analysis'}
    )
    
    print(f"\n📄 Generated comprehensive report ({len(comprehensive_report)} characters)")
    print("Report preview (first 500 characters):")
    print("-" * 50)
    print(comprehensive_report[:500] + "..." if len(comprehensive_report) > 500 else comprehensive_report)
    
    return time_series_data, quality_trend, correlation, comprehensive_report


def demonstrate_automated_workflows():
    """Demonstrate automated workflow capabilities."""
    print("\n" + "="*60)
    print("🔄 AUTOMATED WORKFLOWS DEMONSTRATION") 
    print("="*60)
    
    # Create workflow orchestrator
    orchestrator = WorkflowOrchestrator()
    
    # Create a basic evaluation workflow
    test_questions = [
        "Test workflow question 1",
        "Test workflow question 2"
    ]
    
    evaluation_workflow = create_basic_evaluation_workflow(
        name="Daily Evaluation Workflow",
        questions=test_questions,
        schedule="daily"
    )
    
    print("Registering evaluation workflow...")
    workflow_id = orchestrator.register_workflow(evaluation_workflow)
    print(f"Workflow registered with ID: {workflow_id}")
    
    # Create a monitoring workflow
    monitoring_workflow = create_monitoring_workflow(
        metric_name="quality_score",
        threshold=0.70,
        operator="less"
    )
    
    print("Registering monitoring workflow...")
    monitoring_id = orchestrator.register_workflow(monitoring_workflow)
    print(f"Monitoring workflow registered with ID: {monitoring_id}")
    
    # Get workflow status
    eval_status = orchestrator.get_workflow_status(workflow_id)
    
    print(f"\n📋 Workflow Status:")
    print(f"Name: {eval_status['workflow']['name']}")
    print(f"Description: {eval_status['workflow']['description']}")
    print(f"Trigger Type: {eval_status['workflow']['trigger']['trigger_type']}")
    print(f"Steps: {len(eval_status['workflow']['steps'])}")
    print(f"Recent Executions: {len(eval_status['recent_executions'])}")
    
    # Start orchestrator (in a real scenario)
    print("\nWorkflow orchestrator ready for scheduling...")
    print("Note: In production, workflows would execute automatically based on triggers")
    
    return orchestrator, workflow_id, monitoring_id


def demonstrate_comprehensive_integration():
    """Demonstrate integration of all components working together."""
    print("\n" + "="*60)
    print("🔗 COMPREHENSIVE INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create comprehensive evaluation pipeline
    pipeline = create_comprehensive_evaluator()
    
    # Create evaluation batch
    integration_questions = [
        "Integration test question 1",
        "Integration test question 2", 
        "Integration test question 3"
    ]
    
    evaluation_configs = [
        {'type': 'quality', 'id': 'quality_eval'},
        {'type': 'dialectical', 'id': 'dialectical_eval'}
    ]
    
    print("Creating comprehensive evaluation batch...")
    batch = pipeline.create_evaluation_batch(
        name="Integration Demonstration",
        questions=integration_questions,
        configs=evaluation_configs,
        description="Demonstration of integrated evaluation pipeline"
    )
    
    print(f"Created batch: {batch.name} ({batch.batch_id})")
    print(f"Questions: {len(batch.questions)}")
    print(f"Configurations: {len(batch.evaluation_configs)}")
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    
    print(f"\n🎯 Pipeline Integration Status:")
    print(f"Pipeline ID: {status['pipeline_id']}")
    print(f"Active Batches: {status['active_batches']}")
    print(f"Completed Evaluations: {status['completed_evaluations']}")
    print(f"Available Baseline Metrics: {', '.join(status['baseline_metrics_available']) if status['baseline_metrics_available'] else 'None'}")
    
    # Export results to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"\nExporting results to temporary directory...")
        pipeline.export_results(temp_path, include_raw_data=True)
        
        # List exported files
        exported_files = list(temp_path.glob("*.json"))
        print(f"Exported {len(exported_files)} result files")
        
        for file in exported_files:
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    return pipeline, batch


def main():
    """Run comprehensive evaluation demonstration."""
    print("🚀 COMPREHENSIVE EVALUATION PIPELINE DEMONSTRATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track demonstration duration
    start_time = datetime.now()
    
    try:
        # 1. Baseline Measurement
        baseline_data, quality_baseline, performance_baseline = demonstrate_baseline_measurement()
        
        # 2. A/B Testing
        ab_result = demonstrate_ab_testing()
        
        # 3. Performance Benchmarking
        latency_result, throughput_result, benchmark_baseline = demonstrate_performance_benchmarking()
        
        # 4. Statistical Analysis
        time_series_data, trend_analysis, correlation_analysis, report = demonstrate_statistical_analysis()
        
        # 5. Automated Workflows
        orchestrator, workflow_id, monitoring_id = demonstrate_automated_workflows()
        
        # 6. Comprehensive Integration
        pipeline, batch = demonstrate_comprehensive_integration()
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print(f"Total Duration: {duration:.1f} seconds")
        print(f"Components Demonstrated:")
        print(f"  ✓ Baseline Performance Measurement")
        print(f"  ✓ A/B Testing Infrastructure")
        print(f"  ✓ Performance Benchmarking Suite")
        print(f"  ✓ Statistical Analysis & Reporting")
        print(f"  ✓ Automated Workflow Orchestration")
        print(f"  ✓ Comprehensive Pipeline Integration")
        
        print(f"\nKey Results:")
        print(f"  Quality Baseline: {quality_baseline.baseline_value:.3f}")
        print(f"  A/B Test Winner: {ab_result.conclusion['winner']}")
        print(f"  Latency P95: {latency_result.summary_statistics.get('duration_stats', {}).get('percentile_95', 0):.3f}s")
        print(f"  Throughput: {throughput_result.summary_statistics.get('operations_per_second', 0):.1f} ops/sec")
        print(f"  Quality Trend: {trend_analysis.trend_direction}")
        print(f"  Workflows Registered: 2")
        print(f"  Evaluation Batches: 1")
        
        print(f"\n🎉 All evaluation pipeline components are functioning correctly!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
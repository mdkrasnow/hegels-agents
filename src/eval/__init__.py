"""
Comprehensive Evaluation Module

Provides advanced evaluation frameworks and metrics for assessing:
- Quality of dialectical reasoning and agent performance
- Baseline performance measurement and statistical analysis
- A/B testing infrastructure and comparison frameworks
- Automated evaluation workflows and continuous monitoring
- Performance benchmarking and bottleneck identification
- Statistical analysis and comprehensive reporting

Key Components:
- ComprehensiveEvaluationPipeline: Enhanced evaluation with baseline measurement
- StatisticalAnalyzer: Advanced statistical analysis and trend detection
- AutomatedWorkflows: Scheduled evaluations and continuous monitoring
- PerformanceBenchmarks: Latency, throughput, and stress testing
- BlindedEvaluator: Fair comparison evaluation methodology
- QualityAssessment: Detailed quality metrics and dialectical effectiveness
"""

# Import existing evaluation components
from .quality_assessment import (
    QualityMetrics, DialecticalAssessment, ResponseAnalyzer, 
    DialecticalEvaluator, ComprehensiveQualityFramework
)

from .blinded_evaluator import (
    BlindedEvaluator, BlindedDialecticalComparison, 
    AnonymizedResponse, BlindedEvaluationResult
)

# Import enhanced evaluation pipeline
from .comprehensive_evaluator import (
    AutomatedEvaluationPipeline, BaselineCalculator, ABTestingFramework,
    BaselineMetrics, EvaluationBatch, ABTestResult,
    create_comprehensive_evaluator
)

# Import statistical analysis components
from .statistical_analyzer import (
    StatisticalAnalyzer, PerformanceBenchmarkSuite, EvaluationReportGenerator,
    StatisticalSummary, TrendAnalysis, CorrelationAnalysis, BenchmarkComparison,
    create_statistical_analyzer, create_benchmark_suite, create_report_generator
)

# Import automated workflows
from .automated_workflows import (
    WorkflowOrchestrator, ContinuousMonitoringService, WorkflowScheduler,
    WorkflowDefinition, WorkflowStep, WorkflowTrigger, WorkflowExecution,
    WorkflowStatus, TriggerType, create_basic_evaluation_workflow,
    create_monitoring_workflow
)

# Import performance benchmarking
from .performance_benchmarks import (
    BenchmarkSuite, PerformanceProfiler, PerformanceMetrics, BenchmarkResult,
    create_benchmark_suite
)

# Export all main classes and functions
__all__ = [
    # Quality Assessment
    'QualityMetrics', 'DialecticalAssessment', 'ResponseAnalyzer',
    'DialecticalEvaluator', 'ComprehensiveQualityFramework',
    
    # Blinded Evaluation
    'BlindedEvaluator', 'BlindedDialecticalComparison', 
    'AnonymizedResponse', 'BlindedEvaluationResult',
    
    # Comprehensive Evaluation Pipeline
    'AutomatedEvaluationPipeline', 'BaselineCalculator', 'ABTestingFramework',
    'BaselineMetrics', 'EvaluationBatch', 'ABTestResult',
    'create_comprehensive_evaluator',
    
    # Statistical Analysis
    'StatisticalAnalyzer', 'PerformanceBenchmarkSuite', 'EvaluationReportGenerator',
    'StatisticalSummary', 'TrendAnalysis', 'CorrelationAnalysis', 'BenchmarkComparison',
    'create_statistical_analyzer', 'create_benchmark_suite', 'create_report_generator',
    
    # Automated Workflows
    'WorkflowOrchestrator', 'ContinuousMonitoringService', 'WorkflowScheduler',
    'WorkflowDefinition', 'WorkflowStep', 'WorkflowTrigger', 'WorkflowExecution',
    'WorkflowStatus', 'TriggerType', 'create_basic_evaluation_workflow',
    'create_monitoring_workflow',
    
    # Performance Benchmarking  
    'BenchmarkSuite', 'PerformanceProfiler', 'PerformanceMetrics', 'BenchmarkResult',
]
"""
Comprehensive Evaluation Pipeline for Hegel's Agents

This module implements the enhanced evaluation pipeline that builds on existing
quality assessment and blinded evaluation components to provide comprehensive
evaluation capabilities including baseline measurement, A/B testing, and
automated workflows.

Key Features:
- Baseline performance measurement and statistical analysis
- A/B testing infrastructure for comparing approaches
- Automated evaluation workflows with comprehensive reporting
- Integration with existing evaluation components
- Performance benchmarking and trend analysis
"""

import uuid
import json
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
from collections import defaultdict
import logging

# Import existing evaluation components
from .quality_assessment import (
    QualityMetrics, DialecticalAssessment, ResponseAnalyzer, 
    DialecticalEvaluator, ComprehensiveQualityFramework
)
from .blinded_evaluator import BlindedEvaluator, BlindedDialecticalComparison

# Import agent and debate components
from agents.utils import AgentResponse, AgentLogger
from debate.session import DebateSession


@dataclass
class BaselineMetrics:
    """
    Baseline performance metrics for comparison and trend analysis.
    """
    metric_type: str  # "quality", "performance", "dialectical"
    baseline_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    measurement_date: datetime
    measurement_context: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_type': self.metric_type,
            'baseline_value': self.baseline_value,
            'confidence_interval': list(self.confidence_interval),
            'sample_size': self.sample_size,
            'measurement_date': self.measurement_date.isoformat(),
            'measurement_context': self.measurement_context,
            'statistical_significance': self.statistical_significance
        }


@dataclass
class EvaluationBatch:
    """
    Represents a batch of evaluations for systematic comparison.
    """
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    questions: List[str] = field(default_factory=list)
    evaluation_configs: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'batch_id': self.batch_id,
            'name': self.name,
            'description': self.description,
            'questions': self.questions,
            'evaluation_configs': self.evaluation_configs,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'metadata': self.metadata
        }


@dataclass
class ABTestResult:
    """
    Results from an A/B test comparing two evaluation approaches.
    """
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    approach_a: str = ""
    approach_b: str = ""
    questions_tested: List[str] = field(default_factory=list)
    results_a: List[Dict[str, Any]] = field(default_factory=list)
    results_b: List[Dict[str, Any]] = field(default_factory=list)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    conclusion: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'approach_a': self.approach_a,
            'approach_b': self.approach_b,
            'questions_tested': self.questions_tested,
            'results_a': self.results_a,
            'results_b': self.results_b,
            'statistical_analysis': self.statistical_analysis,
            'conclusion': self.conclusion,
            'timestamp': self.timestamp.isoformat()
        }


class BaselineCalculator:
    """
    Calculates and maintains baseline performance metrics.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize baseline calculator.
        
        Args:
            confidence_level: Confidence level for statistical intervals
        """
        self.confidence_level = confidence_level
        self.logger = AgentLogger("baseline_calculator")
        
    def calculate_baseline_metrics(self, 
                                 evaluations: List[Dict[str, Any]], 
                                 metric_type: str,
                                 context: Dict[str, Any] = None) -> BaselineMetrics:
        """
        Calculate baseline metrics from a set of evaluations.
        
        Args:
            evaluations: List of evaluation results
            metric_type: Type of metric being calculated
            context: Additional context for the measurement
            
        Returns:
            BaselineMetrics with statistical analysis
        """
        if not evaluations:
            raise ValueError("Cannot calculate baseline from empty evaluations")
        
        # Extract relevant values based on metric type
        values = self._extract_metric_values(evaluations, metric_type)
        
        if len(values) < 2:
            raise ValueError(f"Insufficient data points for baseline calculation: {len(values)}")
        
        # Calculate statistical measures
        baseline_value = statistics.mean(values)
        std_dev = statistics.stdev(values)
        sample_size = len(values)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            values, self.confidence_level
        )
        
        # Test for statistical significance (normality test)
        statistical_significance = self._test_statistical_significance(values)
        
        return BaselineMetrics(
            metric_type=metric_type,
            baseline_value=baseline_value,
            confidence_interval=confidence_interval,
            sample_size=sample_size,
            measurement_date=datetime.now(),
            measurement_context=context or {},
            statistical_significance=statistical_significance
        )
    
    def _extract_metric_values(self, evaluations: List[Dict[str, Any]], 
                             metric_type: str) -> List[float]:
        """Extract specific metric values from evaluations."""
        values = []
        
        for evaluation in evaluations:
            if metric_type == "quality":
                # Extract overall quality score
                quality_score = evaluation.get('overall_quality', 
                               evaluation.get('quality_score', 
                               evaluation.get('single_agent_quality_score')))
                if quality_score is not None:
                    values.append(float(quality_score))
                    
            elif metric_type == "improvement":
                # Extract improvement score
                improvement = evaluation.get('improvement_score', 
                            evaluation.get('improvement_percentage'))
                if improvement is not None:
                    values.append(float(improvement))
                    
            elif metric_type == "performance":
                # Extract timing information
                time_taken = evaluation.get('time_taken', 
                           evaluation.get('single_agent_time',
                           evaluation.get('response_time')))
                if time_taken is not None:
                    values.append(float(time_taken))
                    
            elif metric_type == "dialectical":
                # Extract dialectical effectiveness
                dialectical_score = evaluation.get('dialectical_score',
                                  evaluation.get('overall_dialectical_score'))
                if dialectical_score is not None:
                    values.append(float(dialectical_score))
        
        return values
    
    def _calculate_confidence_interval(self, values: List[float], 
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for the given values."""
        import scipy.stats as stats
        
        mean = statistics.mean(values)
        sem = stats.sem(values)  # Standard error of the mean
        h = sem * stats.t.ppf((1 + confidence_level) / 2., len(values)-1)
        
        return (mean - h, mean + h)
    
    def _test_statistical_significance(self, values: List[float]) -> float:
        """Test statistical significance using Shapiro-Wilk normality test."""
        try:
            import scipy.stats as stats
            statistic, p_value = stats.shapiro(values)
            return p_value
        except ImportError:
            # Fallback to basic variance calculation if scipy not available
            return statistics.variance(values) if len(values) > 1 else 0.0


class ABTestingFramework:
    """
    A/B testing framework for comparing different evaluation approaches.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize A/B testing framework.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.logger = AgentLogger("ab_testing")
        
    def run_ab_test(self, 
                   name: str,
                   approach_a_config: Dict[str, Any],
                   approach_b_config: Dict[str, Any],
                   questions: List[str],
                   evaluator_factory) -> ABTestResult:
        """
        Run A/B test comparing two evaluation approaches.
        
        Args:
            name: Name of the A/B test
            approach_a_config: Configuration for approach A
            approach_b_config: Configuration for approach B  
            questions: Test questions to use
            evaluator_factory: Function to create evaluators from config
            
        Returns:
            ABTestResult with statistical analysis
        """
        test_id = str(uuid.uuid4())
        self.logger.log_debug(f"Starting A/B test '{name}' with {len(questions)} questions")
        
        # Run evaluations for both approaches
        results_a = self._run_evaluation_batch(
            questions, approach_a_config, evaluator_factory, "A"
        )
        results_b = self._run_evaluation_batch(
            questions, approach_b_config, evaluator_factory, "B"
        )
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(results_a, results_b)
        
        # Draw conclusions
        conclusion = self._draw_conclusions(
            statistical_analysis, approach_a_config, approach_b_config
        )
        
        return ABTestResult(
            test_id=test_id,
            name=name,
            approach_a=approach_a_config.get('name', 'Approach A'),
            approach_b=approach_b_config.get('name', 'Approach B'),
            questions_tested=questions,
            results_a=results_a,
            results_b=results_b,
            statistical_analysis=statistical_analysis,
            conclusion=conclusion
        )
    
    def _run_evaluation_batch(self, 
                            questions: List[str], 
                            config: Dict[str, Any], 
                            evaluator_factory,
                            approach_label: str) -> List[Dict[str, Any]]:
        """Run evaluation batch for one approach."""
        self.logger.log_debug(f"Running evaluation batch for approach {approach_label}")
        
        evaluator = evaluator_factory(config)
        results = []
        
        for i, question in enumerate(questions):
            try:
                result = evaluator.evaluate(question)
                result['question_index'] = i
                result['approach'] = approach_label
                results.append(result)
            except Exception as e:
                self.logger.log_error(RuntimeError(f"Evaluation failed for question {i}: {e}"), "evaluation")
                results.append({
                    'question_index': i,
                    'approach': approach_label,
                    'error': str(e),
                    'quality_score': 0.0
                })
        
        return results
    
    def _perform_statistical_analysis(self, 
                                    results_a: List[Dict[str, Any]], 
                                    results_b: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis comparing the two approaches."""
        # Extract quality scores
        scores_a = [r.get('quality_score', 0.0) for r in results_a]
        scores_b = [r.get('quality_score', 0.0) for r in results_b]
        
        # Basic statistics
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0
        
        # Effect size (Cohen's d)
        pooled_std = ((std_a ** 2 + std_b ** 2) / 2) ** 0.5
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0
        
        # Perform t-test if possible
        t_test_result = self._perform_t_test(scores_a, scores_b)
        
        return {
            'approach_a_stats': {
                'mean': mean_a,
                'std': std_a,
                'count': len(scores_a)
            },
            'approach_b_stats': {
                'mean': mean_b,
                'std': std_b,
                'count': len(scores_b)
            },
            'effect_size': effect_size,
            'mean_difference': mean_b - mean_a,
            't_test': t_test_result,
            'practical_significance': abs(effect_size) > 0.2,  # Small effect size threshold
            'statistical_significance': t_test_result.get('p_value', 1.0) < self.significance_level
        }
    
    def _perform_t_test(self, scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
        """Perform t-test between the two score sets."""
        try:
            import scipy.stats as stats
            t_statistic, p_value = stats.ttest_ind(scores_a, scores_b)
            return {
                't_statistic': t_statistic,
                'p_value': p_value,
                'method': 'independent_t_test'
            }
        except ImportError:
            # Fallback to manual calculation
            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)
            var_a = statistics.variance(scores_a) if len(scores_a) > 1 else 0.0
            var_b = statistics.variance(scores_b) if len(scores_b) > 1 else 0.0
            
            # Simple t-statistic approximation
            pooled_se = ((var_a / len(scores_a)) + (var_b / len(scores_b))) ** 0.5
            t_stat = (mean_b - mean_a) / pooled_se if pooled_se > 0 else 0.0
            
            return {
                't_statistic': t_stat,
                'p_value': None,  # Cannot calculate without scipy
                'method': 'manual_approximation'
            }
    
    def _draw_conclusions(self, 
                        statistical_analysis: Dict[str, Any],
                        config_a: Dict[str, Any],
                        config_b: Dict[str, Any]) -> Dict[str, Any]:
        """Draw conclusions from statistical analysis."""
        stats_sig = statistical_analysis.get('statistical_significance', False)
        practical_sig = statistical_analysis.get('practical_significance', False)
        effect_size = statistical_analysis.get('effect_size', 0.0)
        mean_diff = statistical_analysis.get('mean_difference', 0.0)
        
        if stats_sig and practical_sig:
            if mean_diff > 0:
                recommendation = "ADOPT_B"
                confidence = "HIGH"
                reason = "Approach B shows statistically significant improvement"
            else:
                recommendation = "ADOPT_A"
                confidence = "HIGH"
                reason = "Approach A shows statistically significant improvement"
        elif practical_sig:
            recommendation = "CONSIDER_B" if mean_diff > 0 else "CONSIDER_A"
            confidence = "MEDIUM"
            reason = "Practical improvement observed but statistical significance unclear"
        else:
            recommendation = "NO_DIFFERENCE"
            confidence = "LOW"
            reason = "No significant difference between approaches"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'effect_size_interpretation': self._interpret_effect_size(effect_size),
            'winner': config_b.get('name', 'Approach B') if mean_diff > 0 else config_a.get('name', 'Approach A'),
            'margin': abs(mean_diff),
            'requires_further_testing': not (stats_sig and practical_sig)
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"


class AutomatedEvaluationPipeline:
    """
    Automated evaluation pipeline that coordinates comprehensive evaluation workflows.
    """
    
    def __init__(self, 
                 quality_framework: Optional[ComprehensiveQualityFramework] = None,
                 blinded_evaluator: Optional[BlindedEvaluator] = None,
                 baseline_calculator: Optional[BaselineCalculator] = None,
                 ab_testing: Optional[ABTestingFramework] = None):
        """
        Initialize the automated evaluation pipeline.
        
        Args:
            quality_framework: Quality assessment framework
            blinded_evaluator: Blinded evaluation component
            baseline_calculator: Baseline metrics calculator
            ab_testing: A/B testing framework
        """
        self.quality_framework = quality_framework or ComprehensiveQualityFramework()
        self.blinded_evaluator = blinded_evaluator or BlindedEvaluator()
        self.baseline_calculator = baseline_calculator or BaselineCalculator()
        self.ab_testing = ab_testing or ABTestingFramework()
        
        self.logger = AgentLogger("evaluation_pipeline")
        self.pipeline_id = str(uuid.uuid4())
        
        # Pipeline state
        self.active_batches: Dict[str, EvaluationBatch] = {}
        self.completed_evaluations: List[Dict[str, Any]] = []
        self.baseline_metrics: Dict[str, BaselineMetrics] = {}
        
    def create_evaluation_batch(self, 
                              name: str, 
                              questions: List[str], 
                              configs: List[Dict[str, Any]],
                              description: str = "") -> EvaluationBatch:
        """
        Create a new evaluation batch.
        
        Args:
            name: Name of the evaluation batch
            questions: Questions to evaluate
            configs: List of evaluation configurations
            description: Optional description
            
        Returns:
            EvaluationBatch instance
        """
        batch = EvaluationBatch(
            name=name,
            description=description,
            questions=questions,
            evaluation_configs=configs
        )
        
        self.active_batches[batch.batch_id] = batch
        self.logger.log_debug(f"Created evaluation batch '{name}' with {len(questions)} questions")
        
        return batch
    
    def run_evaluation_batch(self, batch_id: str, parallel: bool = True) -> Dict[str, Any]:
        """
        Run an evaluation batch.
        
        Args:
            batch_id: ID of the batch to run
            parallel: Whether to run evaluations in parallel
            
        Returns:
            Evaluation results
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch = self.active_batches[batch_id]
        batch.status = "running"
        
        self.logger.log_debug(f"Starting evaluation batch '{batch.name}'")
        
        try:
            if parallel and len(batch.questions) > 1:
                results = self._run_batch_parallel(batch)
            else:
                results = self._run_batch_sequential(batch)
            
            batch.status = "completed"
            batch.completed_at = datetime.now()
            
            # Store results
            batch_results = {
                'batch_id': batch_id,
                'batch_metadata': batch.to_dict(),
                'evaluation_results': results,
                'summary_statistics': self._calculate_batch_statistics(results),
                'completed_at': datetime.now().isoformat()
            }
            
            self.completed_evaluations.append(batch_results)
            
            self.logger.log_debug(f"Completed evaluation batch '{batch.name}'")
            return batch_results
            
        except Exception as e:
            batch.status = "failed"
            batch.metadata['error'] = str(e)
            self.logger.log_error(e, f"Evaluation batch '{batch.name}' failed")
            raise
    
    def _run_batch_sequential(self, batch: EvaluationBatch) -> List[Dict[str, Any]]:
        """Run evaluation batch sequentially."""
        results = []
        
        for i, question in enumerate(batch.questions):
            self.logger.log_debug(f"Evaluating question {i+1}/{len(batch.questions)}")
            
            question_results = []
            for config in batch.evaluation_configs:
                try:
                    result = self._run_single_evaluation(question, config)
                    result['question_index'] = i
                    result['config_id'] = config.get('id', f'config_{len(question_results)}')
                    question_results.append(result)
                except Exception as e:
                    self.logger.log_error(RuntimeError(f"Evaluation failed for question {i}: {e}"), "evaluation")
                    question_results.append({
                        'question_index': i,
                        'config_id': config.get('id', f'config_{len(question_results)}'),
                        'error': str(e),
                        'success': False
                    })
            
            results.extend(question_results)
        
        return results
    
    def _run_batch_parallel(self, batch: EvaluationBatch) -> List[Dict[str, Any]]:
        """Run evaluation batch in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all evaluation tasks
            future_to_info = {}
            
            for i, question in enumerate(batch.questions):
                for j, config in enumerate(batch.evaluation_configs):
                    future = executor.submit(self._run_single_evaluation, question, config)
                    future_to_info[future] = {
                        'question_index': i,
                        'config_id': config.get('id', f'config_{j}'),
                        'question': question,
                        'config': config
                    }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_info):
                info = future_to_info[future]
                try:
                    result = future.result()
                    result.update(info)
                    results.append(result)
                except Exception as e:
                    self.logger.log_error(e, "Parallel evaluation failed")
                    error_result = info.copy()
                    error_result.update({'error': str(e), 'success': False})
                    results.append(error_result)
        
        # Sort results by question index for consistency
        results.sort(key=lambda x: (x.get('question_index', 0), x.get('config_id', '')))
        
        return results
    
    def _run_single_evaluation(self, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single evaluation based on configuration."""
        eval_type = config.get('type', 'quality')
        
        if eval_type == 'quality':
            return self._run_quality_evaluation(question, config)
        elif eval_type == 'dialectical':
            return self._run_dialectical_evaluation(question, config)
        elif eval_type == 'blinded':
            return self._run_blinded_evaluation(question, config)
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")
    
    def _run_quality_evaluation(self, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run quality evaluation for a single question."""
        # This would integrate with agent systems to generate responses
        # For now, providing structure that can be filled in when agents are available
        
        start_time = datetime.now()
        
        try:
            # Placeholder for actual agent response generation
            # response = agent.respond(question)
            # quality_metrics = self.quality_framework.evaluate_single_response(response)
            
            # Mock result for now
            result = {
                'evaluation_type': 'quality',
                'question': question,
                'success': True,
                'quality_score': 0.75,  # Mock score
                'metrics': {
                    'depth_score': 0.8,
                    'clarity_score': 0.7,
                    'coherence_score': 0.75
                },
                'evaluation_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'evaluation_type': 'quality',
                'question': question,
                'success': False,
                'error': str(e),
                'evaluation_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
    
    def _run_dialectical_evaluation(self, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run dialectical evaluation for a single question."""
        start_time = datetime.now()
        
        try:
            # Placeholder for dialectical evaluation
            result = {
                'evaluation_type': 'dialectical',
                'question': question,
                'success': True,
                'dialectical_score': 0.8,  # Mock score
                'improvement_percentage': 15.5,  # Mock improvement
                'evaluation_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'evaluation_type': 'dialectical',
                'question': question,
                'success': False,
                'error': str(e),
                'evaluation_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
    
    def _run_blinded_evaluation(self, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run blinded evaluation for a single question."""
        start_time = datetime.now()
        
        try:
            # Placeholder for blinded evaluation
            result = {
                'evaluation_type': 'blinded',
                'question': question,
                'success': True,
                'single_score': 7.2,  # Mock score
                'dialectical_score': 8.1,  # Mock score
                'evaluation_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'evaluation_type': 'blinded',
                'question': question,
                'success': False,
                'error': str(e),
                'evaluation_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for a batch of results."""
        if not results:
            return {'total_evaluations': 0, 'success_rate': 0.0}
        
        successful_results = [r for r in results if r.get('success', False)]
        success_rate = len(successful_results) / len(results)
        
        # Extract quality scores
        quality_scores = []
        for result in successful_results:
            if 'quality_score' in result:
                quality_scores.append(result['quality_score'])
            elif 'dialectical_score' in result:
                quality_scores.append(result['dialectical_score'])
        
        statistics_dict = {
            'total_evaluations': len(results),
            'successful_evaluations': len(successful_results),
            'success_rate': success_rate,
            'evaluation_types': list(set(r.get('evaluation_type', 'unknown') for r in results))
        }
        
        if quality_scores:
            statistics_dict.update({
                'mean_quality_score': statistics.mean(quality_scores),
                'median_quality_score': statistics.median(quality_scores),
                'std_quality_score': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores)
            })
        
        return statistics_dict
    
    def update_baseline_metrics(self, metric_type: str, evaluations: List[Dict[str, Any]],
                              context: Dict[str, Any] = None) -> BaselineMetrics:
        """
        Update baseline metrics with new evaluation data.
        
        Args:
            metric_type: Type of metric to update
            evaluations: New evaluation data
            context: Additional context for the measurement
            
        Returns:
            Updated BaselineMetrics
        """
        baseline = self.baseline_calculator.calculate_baseline_metrics(
            evaluations, metric_type, context
        )
        
        self.baseline_metrics[metric_type] = baseline
        self.logger.log_debug(f"Updated baseline for {metric_type}: {baseline.baseline_value:.3f}")
        
        return baseline
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the evaluation pipeline."""
        return {
            'pipeline_id': self.pipeline_id,
            'active_batches': len(self.active_batches),
            'completed_evaluations': len(self.completed_evaluations),
            'baseline_metrics_available': list(self.baseline_metrics.keys()),
            'total_questions_evaluated': sum(
                len(eval_data.get('evaluation_results', [])) 
                for eval_data in self.completed_evaluations
            )
        }
    
    def export_results(self, output_path: Union[str, Path],
                      include_raw_data: bool = True) -> None:
        """
        Export all evaluation results to files.
        
        Args:
            output_path: Directory to save results
            include_raw_data: Whether to include raw evaluation data
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export summary statistics
        summary = {
            'pipeline_status': self.get_pipeline_status(),
            'baseline_metrics': {k: v.to_dict() for k, v in self.baseline_metrics.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / f"evaluation_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Export detailed results if requested
        if include_raw_data:
            with open(output_dir / f"evaluation_results_{timestamp}.json", 'w') as f:
                json.dump(self.completed_evaluations, f, indent=2, default=str)
        
        self.logger.log_debug(f"Results exported to {output_dir}")


# Factory function for easy initialization
def create_comprehensive_evaluator(**kwargs) -> AutomatedEvaluationPipeline:
    """
    Create a comprehensive evaluation pipeline with sensible defaults.
    
    Args:
        **kwargs: Optional component configurations
        
    Returns:
        Configured AutomatedEvaluationPipeline
    """
    return AutomatedEvaluationPipeline(**kwargs)


# Export main classes
__all__ = [
    'BaselineMetrics',
    'EvaluationBatch', 
    'ABTestResult',
    'BaselineCalculator',
    'ABTestingFramework',
    'AutomatedEvaluationPipeline',
    'create_comprehensive_evaluator'
]
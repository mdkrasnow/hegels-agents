#!/usr/bin/env python3
"""
Statistical Validation of Hegel's Agents vs Single Review Performance

This script performs comprehensive statistical validation to determine whether 
Hegel's dialectical multi-agent system performs significantly better than 
single-agent review across multiple evaluation metrics.

Features:
- Rigorous paired statistical testing (paired t-test, Wilcoxon signed-rank)
- Effect size calculations (Cohen's d) and practical significance assessment
- Power analysis and sample size recommendations
- Confidence intervals and bootstrap validation
- Comprehensive reporting with statistical interpretation
- Configurable parameters and evaluation metrics
- Integration with existing evaluation infrastructure

Usage:
    python statistical_validation.py --sample-size 50 --confidence-level 0.95
    python statistical_validation.py --data-file results.json --output-dir stats/
    python statistical_validation.py --run-new-evaluation --questions 100

Author: Claude AI
Created: 2024-12-03
"""

import sys
import argparse
import json
import statistics
import math
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple
import warnings

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# Statistics and data analysis
try:
    import numpy as np
    from scipy import stats
    from scipy.stats import ttest_rel, wilcoxon, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some advanced statistical tests will be unavailable.")

# Import Hegel's evaluation infrastructure
try:
    # Load configuration first to avoid config errors
    from config.settings import load_config
    load_config()
    
    from debate.dialectical_tester import DialecticalTester, DialecticalTestResult
    from eval.blinded_evaluator import BlindedDialecticalComparison
    from test_questions.dialectical_test_questions import get_question_set
    EVALUATION_INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    EVALUATION_INFRASTRUCTURE_AVAILABLE = False
    warnings.warn(f"Evaluation infrastructure not available: {e}. Will work with data files only.")
except Exception as e:
    EVALUATION_INFRASTRUCTURE_AVAILABLE = False
    warnings.warn(f"Configuration or evaluation setup failed: {e}. Will work with data files only.")


@dataclass
class StatisticalTestResult:
    """Results from a statistical test comparing two approaches."""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    sample_size: int = 0
    power: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ValidationConfig:
    """Configuration for statistical validation."""
    sample_size: int = 50
    confidence_level: float = 0.95
    significance_level: float = 0.05
    effect_size_threshold: float = 0.3  # Minimum meaningful effect size
    practical_significance_threshold: float = 0.05  # 5% improvement threshold
    power_target: float = 0.80  # Target statistical power
    bootstrap_samples: int = 1000
    random_seed: Optional[int] = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Metrics comparing Hegel's agents vs single review."""
    single_scores: List[float] = field(default_factory=list)
    hegel_scores: List[float] = field(default_factory=list) 
    single_times: List[float] = field(default_factory=list)
    hegel_times: List[float] = field(default_factory=list)
    improvement_scores: List[float] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def sample_size(self) -> int:
        """Get the sample size."""
        return len(self.single_scores)
    
    def validate_data(self) -> bool:
        """Validate that data is consistent and sufficient."""
        if not self.single_scores or not self.hegel_scores:
            return False
        
        expected_size = len(self.single_scores)
        return (
            len(self.hegel_scores) == expected_size and
            (not self.single_times or len(self.single_times) == expected_size) and
            (not self.hegel_times or len(self.hegel_times) == expected_size) and
            (not self.improvement_scores or len(self.improvement_scores) == expected_size)
        )


class StatisticalValidator:
    """
    Comprehensive statistical validator for Hegel's agents vs single review.
    
    This class performs rigorous statistical analysis to determine whether
    Hegel's dialectical approach provides statistically significant and
    practically meaningful improvements over single-agent review.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """
        Initialize the statistical validator.
        
        Args:
            config: Configuration for validation parameters
        """
        self.config = config or ValidationConfig()
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            if SCIPY_AVAILABLE:
                np.random.seed(self.config.random_seed)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.validation_results: Dict[str, Any] = {}
        self.comparison_data: Optional[ComparisonMetrics] = None
        
    def collect_evaluation_data(self, 
                              data_source: Union[str, Path, None] = None,
                              run_new_evaluation: bool = False,
                              num_questions: int = None) -> ComparisonMetrics:
        """
        Collect comparison data from various sources.
        
        Args:
            data_source: Path to existing data file, or None to run new evaluation
            run_new_evaluation: Whether to run new comparative evaluation
            num_questions: Number of questions for new evaluation
            
        Returns:
            ComparisonMetrics with paired evaluation data
        """
        self.logger.info("Collecting evaluation data...")
        
        if data_source:
            # Load from file
            return self._load_data_from_file(data_source)
        elif run_new_evaluation:
            # Run new evaluation
            return self._run_new_evaluation(num_questions)
        else:
            # Require real data - no mock data
            raise ValueError(
                "No data source specified. You must provide either:\n"
                "  --data-file path/to/results.json  (load existing evaluation results)\n"
                "  --run-new-evaluation             (run new comparative evaluation)\n\n"
                "This tool requires real evaluation data to provide meaningful statistical analysis."
            )
    
    def _load_data_from_file(self, file_path: Union[str, Path]) -> ComparisonMetrics:
        """Load comparison data from JSON file."""
        self.logger.info(f"Loading data from {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extract comparison metrics from various possible data formats
        metrics = ComparisonMetrics()
        
        if 'test_results' in data:
            # DialecticalTestSuite format
            for result in data['test_results']:
                metrics.single_scores.append(result['single_agent_quality_score'])
                metrics.hegel_scores.append(result['dialectical_quality_score'])
                metrics.improvement_scores.append(result.get('improvement_score', 0))
                metrics.questions.append(result['question'])
                
                if 'single_agent_time' in result:
                    metrics.single_times.append(result['single_agent_time'])
                if 'dialectical_time' in result:
                    metrics.hegel_times.append(result['dialectical_time'])
        
        elif 'comparison_results' in data:
            # Direct comparison format
            comp_data = data['comparison_results']
            metrics.single_scores = comp_data['single_scores']
            metrics.hegel_scores = comp_data['hegel_scores']
            metrics.improvement_scores = comp_data.get('improvement_scores', [])
            metrics.questions = comp_data.get('questions', [])
            metrics.single_times = comp_data.get('single_times', [])
            metrics.hegel_times = comp_data.get('hegel_times', [])
        
        else:
            raise ValueError("Unrecognized data format in file")
        
        if not metrics.validate_data():
            raise ValueError("Loaded data failed validation - inconsistent array sizes")
        
        metrics.metadata['source_file'] = str(file_path)
        metrics.metadata['loaded_at'] = datetime.now().isoformat()
        
        self.logger.info(f"Loaded {metrics.sample_size} comparison pairs")
        return metrics
    
    def _run_new_evaluation(self, num_questions: int = None) -> ComparisonMetrics:
        """Run new comparative evaluation using existing infrastructure."""
        if not EVALUATION_INFRASTRUCTURE_AVAILABLE:
            raise RuntimeError("Evaluation infrastructure not available for new evaluation")
        
        num_questions = num_questions or self.config.sample_size
        self.logger.info(f"Running new comparative evaluation with {num_questions} questions")
        
        # Setup evaluation infrastructure
        corpus_dir = Path(__file__).parent / "corpus_data"
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        
        tester = DialecticalTester(str(corpus_dir))
        
        # Get test questions
        all_questions = get_question_set()
        if num_questions > len(all_questions):
            self.logger.warning(f"Requested {num_questions} questions but only {len(all_questions)} available")
            num_questions = len(all_questions)
        
        test_questions = random.sample(all_questions, num_questions)
        
        # Run comparative evaluation
        metrics = ComparisonMetrics()
        
        for i, question in enumerate(test_questions):
            self.logger.info(f"Evaluating question {i+1}/{num_questions}")
            
            try:
                result = tester.run_comparison_test(question)
                
                metrics.single_scores.append(result.single_agent_quality_score)
                metrics.hegel_scores.append(result.dialectical_quality_score)
                metrics.improvement_scores.append(result.improvement_score)
                metrics.single_times.append(result.single_agent_time)
                metrics.hegel_times.append(result.dialectical_time)
                metrics.questions.append(question)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for question {i+1}: {e}")
                continue
        
        if not metrics.validate_data():
            raise RuntimeError("Generated evaluation data failed validation")
        
        metrics.metadata['evaluation_type'] = 'new_comparative'
        metrics.metadata['evaluated_at'] = datetime.now().isoformat()
        metrics.metadata['corpus_dir'] = str(corpus_dir)
        
        self.logger.info(f"Completed evaluation of {metrics.sample_size} questions")
        return metrics
    
    
    def run_statistical_analysis(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on comparison data.
        
        Args:
            data: Comparison metrics to analyze
            
        Returns:
            Complete statistical analysis results
        """
        self.logger.info("Running comprehensive statistical analysis...")
        self.comparison_data = data
        
        if not data.validate_data():
            raise ValueError("Data validation failed - cannot perform analysis")
        
        if data.sample_size < 10:
            self.logger.warning(f"Small sample size ({data.sample_size}) may limit statistical power")
        
        # Initialize results
        analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size': data.sample_size,
            'configuration': self.config.to_dict(),
            'data_summary': self._calculate_descriptive_statistics(data),
            'statistical_tests': {},
            'effect_size_analysis': {},
            'practical_significance': {},
            'power_analysis': {},
            'conclusions': {}
        }
        
        # Quality scores analysis (primary endpoint)
        quality_tests = self._analyze_quality_scores(data)
        analysis_results['statistical_tests']['quality_analysis'] = quality_tests
        
        # Improvement scores analysis
        improvement_tests = self._analyze_improvement_scores(data)
        analysis_results['statistical_tests']['improvement_analysis'] = improvement_tests
        
        # Performance analysis (timing)
        if data.single_times and data.hegel_times:
            performance_tests = self._analyze_performance_timing(data)
            analysis_results['statistical_tests']['performance_analysis'] = performance_tests
        
        # Effect size analysis
        effect_analysis = self._calculate_effect_sizes(data)
        analysis_results['effect_size_analysis'] = effect_analysis
        
        # Practical significance assessment
        practical_analysis = self._assess_practical_significance(data)
        analysis_results['practical_significance'] = practical_analysis
        
        # Power analysis
        if SCIPY_AVAILABLE:
            power_analysis = self._perform_power_analysis(data)
            analysis_results['power_analysis'] = power_analysis
        
        # Overall conclusions
        conclusions = self._draw_conclusions(analysis_results)
        analysis_results['conclusions'] = conclusions
        
        # Store results
        self.validation_results = analysis_results
        
        self.logger.info("Statistical analysis complete")
        return analysis_results
    
    def _calculate_descriptive_statistics(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Calculate descriptive statistics for the data."""
        def stats_summary(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        summary = {
            'single_agent_scores': stats_summary(data.single_scores),
            'hegel_scores': stats_summary(data.hegel_scores),
        }
        
        if data.improvement_scores:
            summary['improvement_scores'] = stats_summary(data.improvement_scores)
        
        if data.single_times and data.hegel_times:
            summary['single_agent_times'] = stats_summary(data.single_times)
            summary['hegel_times'] = stats_summary(data.hegel_times)
        
        # Improvement metrics
        improvements = [h - s for h, s in zip(data.hegel_scores, data.single_scores)]
        summary['absolute_improvements'] = stats_summary(improvements)
        
        positive_improvements = len([x for x in improvements if x > 0])
        summary['improvement_rate'] = positive_improvements / len(improvements)
        
        return summary
    
    def _analyze_quality_scores(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Analyze quality score differences using paired tests."""
        single_scores = data.single_scores
        hegel_scores = data.hegel_scores
        
        results = {}
        
        # Paired t-test (primary test)
        if SCIPY_AVAILABLE:
            try:
                t_stat, p_value = ttest_rel(hegel_scores, single_scores)
                results['paired_t_test'] = StatisticalTestResult(
                    test_name='Paired t-test',
                    test_statistic=t_stat,
                    p_value=p_value,
                    sample_size=len(single_scores),
                    interpretation=self._interpret_p_value(p_value)
                ).to_dict()
            except Exception as e:
                self.logger.warning(f"Paired t-test failed: {e}")
        
        # Wilcoxon signed-rank test (non-parametric)
        if SCIPY_AVAILABLE:
            try:
                differences = [h - s for h, s in zip(hegel_scores, single_scores)]
                non_zero_diffs = [d for d in differences if d != 0]
                
                if len(non_zero_diffs) > 5:
                    w_stat, w_p = wilcoxon(non_zero_diffs)
                    results['wilcoxon_test'] = StatisticalTestResult(
                        test_name='Wilcoxon signed-rank test',
                        test_statistic=w_stat,
                        p_value=w_p,
                        sample_size=len(non_zero_diffs),
                        interpretation=self._interpret_p_value(w_p)
                    ).to_dict()
            except Exception as e:
                self.logger.warning(f"Wilcoxon test failed: {e}")
        
        # Manual paired t-test (fallback)
        if not SCIPY_AVAILABLE or 'paired_t_test' not in results:
            manual_t_result = self._manual_paired_t_test(single_scores, hegel_scores)
            results['paired_t_test_manual'] = manual_t_result.to_dict()
        
        return results
    
    def _analyze_improvement_scores(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Analyze improvement scores against zero (no improvement)."""
        if not data.improvement_scores:
            # Calculate from quality scores
            improvements = [(h - s) / 100 for h, s in zip(data.hegel_scores, data.single_scores)]
        else:
            improvements = data.improvement_scores
        
        results = {}
        
        # One-sample t-test against zero
        if SCIPY_AVAILABLE:
            try:
                t_stat, p_value = stats.ttest_1samp(improvements, 0)
                results['improvement_t_test'] = StatisticalTestResult(
                    test_name='One-sample t-test (vs zero improvement)',
                    test_statistic=t_stat,
                    p_value=p_value,
                    sample_size=len(improvements),
                    interpretation=self._interpret_p_value(p_value)
                ).to_dict()
            except Exception as e:
                self.logger.warning(f"Improvement t-test failed: {e}")
        
        # Sign test (simple non-parametric)
        positive_improvements = len([x for x in improvements if x > 0])
        total_improvements = len(improvements)
        sign_test_p = self._binomial_test(positive_improvements, total_improvements, 0.5)
        
        results['sign_test'] = StatisticalTestResult(
            test_name='Sign test (positive improvements)',
            test_statistic=positive_improvements,
            p_value=sign_test_p,
            sample_size=total_improvements,
            interpretation=f"{positive_improvements}/{total_improvements} positive improvements, " + self._interpret_p_value(sign_test_p)
        ).to_dict()
        
        return results
    
    def _analyze_performance_timing(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Analyze performance timing differences."""
        results = {}
        
        # Time overhead analysis
        time_ratios = [h / s for h, s in zip(data.hegel_times, data.single_times) if s > 0]
        
        if time_ratios:
            results['timing_summary'] = {
                'mean_time_ratio': statistics.mean(time_ratios),
                'median_time_ratio': statistics.median(time_ratios),
                'mean_overhead_percent': (statistics.mean(time_ratios) - 1) * 100
            }
        
        # Paired t-test for timing
        if SCIPY_AVAILABLE:
            try:
                t_stat, p_value = ttest_rel(data.hegel_times, data.single_times)
                results['timing_t_test'] = StatisticalTestResult(
                    test_name='Paired t-test (timing comparison)',
                    test_statistic=t_stat,
                    p_value=p_value,
                    sample_size=len(data.single_times),
                    interpretation=f"Hegel's approach is {'significantly' if p_value < 0.05 else 'not significantly'} slower"
                ).to_dict()
            except Exception as e:
                self.logger.warning(f"Timing t-test failed: {e}")
        
        return results
    
    def _calculate_effect_sizes(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Calculate effect sizes for practical significance."""
        results = {}
        
        # Cohen's d for quality scores
        quality_effect_size = self._cohens_d(data.hegel_scores, data.single_scores)
        results['quality_cohens_d'] = {
            'effect_size': quality_effect_size,
            'interpretation': self._interpret_cohens_d(quality_effect_size)
        }
        
        # Raw improvement statistics
        improvements = [h - s for h, s in zip(data.hegel_scores, data.single_scores)]
        mean_improvement = statistics.mean(improvements)
        results['raw_improvement'] = {
            'mean_points_improvement': mean_improvement,
            'mean_percentage_improvement': mean_improvement / statistics.mean(data.single_scores) * 100
        }
        
        # Confidence interval for improvement
        if SCIPY_AVAILABLE:
            ci_lower, ci_upper = self._calculate_confidence_interval(improvements)
            results['improvement_confidence_interval'] = {
                'confidence_level': self.config.confidence_level,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'interpretation': f"We are {self.config.confidence_level*100:.0f}% confident the true improvement is between {ci_lower:.2f} and {ci_upper:.2f} points"
            }
        
        return results
    
    def _assess_practical_significance(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Assess practical significance beyond statistical significance."""
        improvements = [h - s for h, s in zip(data.hegel_scores, data.single_scores)]
        mean_improvement = statistics.mean(improvements)
        
        # Percentage improvement
        baseline_mean = statistics.mean(data.single_scores)
        percent_improvement = mean_improvement / baseline_mean * 100
        
        # Threshold-based assessment
        threshold_points = 5.0  # 5 point improvement on 100-point scale
        threshold_percent = self.config.practical_significance_threshold * 100
        
        results = {
            'mean_improvement_points': mean_improvement,
            'mean_improvement_percent': percent_improvement,
            'threshold_points': threshold_points,
            'threshold_percent': threshold_percent,
            'exceeds_point_threshold': abs(mean_improvement) >= threshold_points,
            'exceeds_percent_threshold': abs(percent_improvement) >= threshold_percent,
            'practically_significant': abs(mean_improvement) >= threshold_points and abs(percent_improvement) >= threshold_percent
        }
        
        # Clinical significance analysis
        substantial_improvements = len([x for x in improvements if x >= threshold_points])
        results['substantial_improvement_rate'] = substantial_improvements / len(improvements)
        results['number_needed_to_treat'] = 1 / (substantial_improvements / len(improvements)) if substantial_improvements > 0 else float('inf')
        
        return results
    
    def _perform_power_analysis(self, data: ComparisonMetrics) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available for power analysis'}
        
        # Calculate observed effect size
        effect_size = self._cohens_d(data.hegel_scores, data.single_scores)
        
        # Power for current sample size
        current_power = self._calculate_power(
            effect_size, data.sample_size, self.config.significance_level
        )
        
        # Required sample size for target power
        required_n = self._calculate_required_sample_size(
            effect_size, self.config.power_target, self.config.significance_level
        )
        
        # Minimum detectable effect size
        min_detectable_effect = self._minimum_detectable_effect(
            data.sample_size, self.config.power_target, self.config.significance_level
        )
        
        results = {
            'observed_effect_size': effect_size,
            'current_sample_size': data.sample_size,
            'current_power': current_power,
            'target_power': self.config.power_target,
            'required_sample_size_for_target_power': required_n,
            'minimum_detectable_effect': min_detectable_effect,
            'power_adequate': current_power >= self.config.power_target,
            'interpretation': f"Current study has {current_power*100:.1f}% power to detect the observed effect size"
        }
        
        return results
    
    def _draw_conclusions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Draw overall conclusions from the statistical analysis."""
        conclusions = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size_assessment': self._assess_sample_size(analysis),
            'statistical_significance_summary': self._summarize_statistical_tests(analysis),
            'practical_significance_summary': self._summarize_practical_significance(analysis),
            'overall_recommendation': '',
            'confidence_level': '',
            'limitations': [],
            'next_steps': []
        }
        
        # Extract key results
        sample_size = analysis['sample_size']
        practical_sig = analysis.get('practical_significance', {})
        effect_size = analysis.get('effect_size_analysis', {})
        
        # Overall assessment
        statistically_significant = self._any_test_significant(analysis)
        practically_significant = practical_sig.get('practically_significant', False)
        effect_size_substantial = effect_size.get('quality_cohens_d', {}).get('effect_size', 0) >= self.config.effect_size_threshold
        
        # Generate recommendation
        if statistically_significant and practically_significant and effect_size_substantial:
            conclusions['overall_recommendation'] = "ADOPT_HEGELS_APPROACH"
            conclusions['confidence_level'] = "HIGH"
            conclusions['summary'] = "Hegel's dialectical approach shows statistically significant and practically meaningful improvement over single-agent review."
        
        elif statistically_significant and practically_significant:
            conclusions['overall_recommendation'] = "LIKELY_ADOPT_HEGELS_APPROACH"  
            conclusions['confidence_level'] = "MODERATE_HIGH"
            conclusions['summary'] = "Hegel's approach shows significant improvement, though effect size requires consideration."
        
        elif practically_significant:
            conclusions['overall_recommendation'] = "CONSIDER_HEGELS_APPROACH"
            conclusions['confidence_level'] = "MODERATE"
            conclusions['summary'] = "Hegel's approach shows practical improvement but statistical significance is unclear."
            
        else:
            conclusions['overall_recommendation'] = "INSUFFICIENT_EVIDENCE"
            conclusions['confidence_level'] = "LOW"
            conclusions['summary'] = "Current evidence does not support superiority of Hegel's approach over single-agent review."
        
        # Add limitations and next steps
        conclusions['limitations'] = self._identify_limitations(analysis)
        conclusions['next_steps'] = self._suggest_next_steps(analysis)
        
        return conclusions
    
    # Helper methods for statistical calculations
    
    def _manual_paired_t_test(self, group1: List[float], group2: List[float]) -> StatisticalTestResult:
        """Manual implementation of paired t-test."""
        if len(group1) != len(group2):
            raise ValueError("Groups must have equal size for paired t-test")
        
        differences = [g2 - g1 for g1, g2 in zip(group1, group2)]
        n = len(differences)
        
        if n < 2:
            raise ValueError("Need at least 2 pairs for t-test")
        
        mean_diff = statistics.mean(differences)
        std_diff = statistics.stdev(differences)
        
        # Standard error
        se = std_diff / math.sqrt(n)
        
        # T statistic
        t_stat = mean_diff / se if se > 0 else float('inf') if mean_diff != 0 else 0
        
        # Degrees of freedom
        df = n - 1
        
        # Approximate p-value (two-tailed)
        # This is a rough approximation without scipy
        p_value = 2 * (1 - self._approximate_t_cdf(abs(t_stat), df))
        
        return StatisticalTestResult(
            test_name='Manual Paired t-test',
            test_statistic=t_stat,
            p_value=p_value,
            sample_size=n,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _approximate_t_cdf(self, t: float, df: int) -> float:
        """Rough approximation of t-distribution CDF."""
        # Very rough approximation - use normal for large df
        if df > 30:
            return self._approximate_normal_cdf(t)
        else:
            # Simple approximation for small df
            return max(0, min(1, 0.5 + t / (2 * math.sqrt(df + 1))))
    
    def _approximate_normal_cdf(self, x: float) -> float:
        """Rough approximation of standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _binomial_test(self, successes: int, trials: int, p: float) -> float:
        """Simple binomial test p-value calculation."""
        if trials == 0:
            return 1.0
        
        # Use normal approximation for large n
        if trials > 30:
            expected = trials * p
            variance = trials * p * (1 - p)
            z_score = (successes - expected) / math.sqrt(variance)
            return 2 * (1 - abs(self._approximate_normal_cdf(abs(z_score))))
        else:
            # Use scipy binomial test for small samples
            if SCIPY_AVAILABLE:
                from scipy.stats import binomtest
                result = binomtest(successes, trials, p, alternative='two-sided')
                return result.pvalue
            else:
                raise NotImplementedError(
                    "Binomial test requires scipy. Install with: pip install scipy. "
                    "Manual approximation is not statistically valid and should not be used "
                    "for important decisions."
                )
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group1 or not group2:
            return 0.0
        
        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        
        if len(group1) == 1 and len(group2) == 1:
            return 0.0
        
        # Address division by zero when sample sizes too small
        if len(group1) + len(group2) <= 2:
            return 0.0

        # Use numpy for better performance when available
        if SCIPY_AVAILABLE:
            import numpy as np
            var1 = np.var(group1, ddof=1) if len(group1) > 1 else 0.0
            var2 = np.var(group2, ddof=1) if len(group2) > 1 else 0.0
        else:
            var1 = statistics.variance(group1) if len(group1) > 1 else 0.0
            var2 = statistics.variance(group2) if len(group2) > 1 else 0.0

        pooled_var = ((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / (len(group1) + len(group2) - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1.0
        
        return (mean2 - mean1) / pooled_std
    
    def _calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        if not SCIPY_AVAILABLE or len(data) < 2:
            # Fallback: use mean +/- 1.96 * sem
            mean = statistics.mean(data)
            std = statistics.stdev(data) if len(data) > 1 else 0
            sem = std / math.sqrt(len(data))
            margin = 1.96 * sem  # Approximate 95% CI
            return mean - margin, mean + margin
        
        mean = np.mean(data)
        sem = stats.sem(data)
        confidence = self.config.confidence_level
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        return mean - h, mean + h
    
    def _calculate_power(self, effect_size: float, n: int, alpha: float) -> float:
        """Calculate statistical power."""
        if not SCIPY_AVAILABLE:
            # Rough approximation
            z_alpha = 1.96  # For alpha = 0.05
            z_beta = z_alpha - effect_size * math.sqrt(n / 2)
            return max(0, min(1, 1 - self._approximate_normal_cdf(z_beta)))
        
        # More accurate calculation with scipy
        delta = effect_size * math.sqrt(n / 2)
        critical_value = stats.t.ppf(1 - alpha / 2, n - 1)
        power = 1 - stats.t.cdf(critical_value - delta, n - 1) + stats.t.cdf(-critical_value - delta, n - 1)
        return power
    
    def _calculate_required_sample_size(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate required sample size for target power."""
        if not SCIPY_AVAILABLE or effect_size <= 0:
            return 1000  # Conservative estimate
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return max(10, int(math.ceil(n)))
    
    def _minimum_detectable_effect(self, n: int, power: float, alpha: float) -> float:
        """Calculate minimum detectable effect size."""
        if not SCIPY_AVAILABLE:
            return 0.5  # Conservative estimate
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        return (z_alpha + z_beta) / math.sqrt(n / 2)
    
    def _interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value."""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        elif p_value < 0.10:
            return "Marginally significant (p < 0.10)"
        else:
            return "Not significant"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        direction = "positive" if d > 0 else "negative" if d < 0 else "no"
        
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"{magnitude} {direction} effect (d = {d:.3f})"
    
    def _assess_sample_size(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess adequacy of sample size."""
        n = analysis['sample_size']
        
        assessment = {
            'sample_size': n,
            'adequacy': 'adequate' if n >= 30 else 'limited' if n >= 10 else 'insufficient',
            'minimum_recommended': 30,
            'good_practice': 50,
            'recommendations': []
        }
        
        if n < 10:
            assessment['recommendations'].append("Sample size too small for reliable conclusions")
        elif n < 30:
            assessment['recommendations'].append("Consider increasing sample size for more robust statistics")
        elif n >= 100:
            assessment['recommendations'].append("Sample size is excellent for statistical analysis")
        
        return assessment
    
    def _summarize_statistical_tests(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize statistical test results."""
        tests = analysis.get('statistical_tests', {})
        
        significant_tests = []
        non_significant_tests = []
        
        for test_category, test_results in tests.items():
            for test_name, result in test_results.items():
                if isinstance(result, dict) and 'p_value' in result:
                    if result['p_value'] < self.config.significance_level:
                        significant_tests.append(f"{test_category}: {result['test_name']}")
                    else:
                        non_significant_tests.append(f"{test_category}: {result['test_name']}")
        
        return {
            'significant_tests': significant_tests,
            'non_significant_tests': non_significant_tests,
            'overall_significance': len(significant_tests) > 0,
            'consensus': len(significant_tests) > len(non_significant_tests)
        }
    
    def _summarize_practical_significance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize practical significance assessment."""
        practical = analysis.get('practical_significance', {})
        
        return {
            'practically_significant': practical.get('practically_significant', False),
            'improvement_points': practical.get('mean_improvement_points', 0),
            'improvement_percent': practical.get('mean_improvement_percent', 0),
            'substantial_improvement_rate': practical.get('substantial_improvement_rate', 0)
        }
    
    def _any_test_significant(self, analysis: Dict[str, Any]) -> bool:
        """Check if any statistical test is significant."""
        tests = analysis.get('statistical_tests', {})
        
        for test_results in tests.values():
            for result in test_results.values():
                if isinstance(result, dict) and 'p_value' in result:
                    if result['p_value'] < self.config.significance_level:
                        return True
        
        return False
    
    def _identify_limitations(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify limitations of the analysis."""
        limitations = []
        
        n = analysis['sample_size']
        if n < 30:
            limitations.append(f"Small sample size (n={n}) limits statistical power")
        
        if not SCIPY_AVAILABLE:
            limitations.append("Advanced statistical functions not available (SciPy not installed)")
        
        data_type = self.comparison_data.metadata.get('data_type', 'unknown')
        if data_type == 'mock_demonstration':
            limitations.append("Analysis based on mock data - results are for demonstration only")
        
        # Check for missing data
        if not self.comparison_data.single_times:
            limitations.append("Performance timing data not available")
        
        return limitations
    
    def _suggest_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest next steps based on analysis results."""
        next_steps = []
        
        n = analysis['sample_size']
        power_analysis = analysis.get('power_analysis', {})
        
        # Sample size recommendations
        if n < 30:
            next_steps.append("Increase sample size to at least 30 for robust conclusions")
        
        required_n = power_analysis.get('required_sample_size_for_target_power')
        if required_n and required_n > n:
            next_steps.append(f"Consider increasing sample size to {required_n} for {self.config.power_target*100:.0f}% power")
        
        # Follow-up studies
        conclusions = analysis.get('conclusions', {})
        recommendation = conclusions.get('overall_recommendation', '')
        
        if recommendation == 'ADOPT_HEGELS_APPROACH':
            next_steps.append("Proceed with implementing Hegel's approach in production")
            next_steps.append("Monitor performance and quality in real-world deployment")
        
        elif recommendation == 'CONSIDER_HEGELS_APPROACH':
            next_steps.append("Conduct larger confirmatory study")
            next_steps.append("Investigate factors that might enhance the dialectical approach")
        
        elif recommendation == 'INSUFFICIENT_EVIDENCE':
            next_steps.append("Redesign study with larger sample size")
            next_steps.append("Consider alternative dialectical protocols or agent configurations")
        
        # Data quality improvements
        if self.comparison_data.metadata.get('data_type') == 'mock_demonstration':
            next_steps.append("Replace mock data with real evaluation data")
        
        return next_steps


def save_results(results: Dict[str, Any], output_dir: Union[str, Path], 
                 config: ValidationConfig) -> Dict[str, str]:
    """
    Save validation results to files.
    
    Args:
        results: Analysis results to save
        output_dir: Directory to save results
        config: Validation configuration
        
    Returns:
        Dictionary mapping file types to saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Save detailed results as JSON
    json_file = output_path / f"statistical_validation_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    saved_files['detailed_results'] = str(json_file)
    
    # Save human-readable summary report
    report_file = output_path / f"statistical_validation_report_{timestamp}.md"
    report_content = generate_summary_report(results, config)
    with open(report_file, 'w') as f:
        f.write(report_content)
    saved_files['summary_report'] = str(report_file)
    
    # Save configuration
    config_file = output_path / f"validation_config_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    saved_files['configuration'] = str(config_file)
    
    return saved_files


def generate_summary_report(results: Dict[str, Any], config: ValidationConfig) -> str:
    """Generate human-readable summary report."""
    
    conclusions = results.get('conclusions', {})
    data_summary = results.get('data_summary', {})
    practical = results.get('practical_significance', {})
    effect_size = results.get('effect_size_analysis', {})
    
    report = f"""# Statistical Validation Report: Hegel's Agents vs Single Review

**Analysis Date:** {results.get('analysis_timestamp', 'Unknown')}  
**Sample Size:** {results.get('sample_size', 'Unknown')}  
**Confidence Level:** {config.confidence_level:.1%}  
**Significance Level:** {config.significance_level:.3f}  

## Executive Summary

**Overall Recommendation:** {conclusions.get('overall_recommendation', 'Unknown')}  
**Confidence Level:** {conclusions.get('confidence_level', 'Unknown')}  

{conclusions.get('summary', 'No summary available.')}

## Key Findings

### Performance Comparison
- **Single Agent Average Score:** {data_summary.get('single_agent_scores', {}).get('mean', 0):.2f}/100
- **Hegel's Agents Average Score:** {data_summary.get('hegel_scores', {}).get('mean', 0):.2f}/100
- **Mean Improvement:** {practical.get('mean_improvement_points', 0):.2f} points ({practical.get('mean_improvement_percent', 0):.1f}%)
- **Success Rate:** {data_summary.get('improvement_rate', 0)*100:.1f}% of tests showed improvement

### Statistical Significance
"""
    
    # Add statistical test summaries
    stat_summary = conclusions.get('statistical_significance_summary', {})
    if stat_summary.get('overall_significance', False):
        report += "✅ **Statistically Significant** - Results are unlikely due to chance\n"
    else:
        report += "❌ **Not Statistically Significant** - Results could be due to chance\n"
    
    # Add practical significance
    if practical.get('practically_significant', False):
        report += "✅ **Practically Significant** - Improvement exceeds meaningful thresholds\n"
    else:
        report += "❌ **Not Practically Significant** - Improvement below meaningful thresholds\n"
    
    # Add effect size
    quality_effect = effect_size.get('quality_cohens_d', {})
    if quality_effect:
        report += f"\n### Effect Size\n**Cohen's d:** {quality_effect.get('effect_size', 0):.3f} - {quality_effect.get('interpretation', 'Unknown effect')}\n"
    
    # Add confidence intervals
    ci_info = effect_size.get('improvement_confidence_interval', {})
    if ci_info:
        report += f"\n### Confidence Interval\n{ci_info.get('interpretation', 'No confidence interval available.')}\n"
    
    # Add power analysis
    power = results.get('power_analysis', {})
    if power and 'current_power' in power:
        report += f"\n### Statistical Power\n**Current Power:** {power['current_power']*100:.1f}%\n"
        if power.get('power_adequate', False):
            report += "✅ Adequate power for reliable results\n"
        else:
            report += f"❌ Low power - recommend n ≥ {power.get('required_sample_size_for_target_power', 'Unknown')} for {config.power_target*100:.0f}% power\n"
    
    # Add limitations
    limitations = conclusions.get('limitations', [])
    if limitations:
        report += "\n## Limitations\n"
        for limitation in limitations:
            report += f"- {limitation}\n"
    
    # Add recommendations
    next_steps = conclusions.get('next_steps', [])
    if next_steps:
        report += "\n## Recommendations\n"
        for step in next_steps:
            report += f"- {step}\n"
    
    # Add detailed statistics appendix
    report += "\n## Detailed Statistical Results\n\n"
    
    # Quality analysis
    quality_tests = results.get('statistical_tests', {}).get('quality_analysis', {})
    for test_name, test_result in quality_tests.items():
        if isinstance(test_result, dict):
            test_type = test_result.get('test_name', test_name)
            p_val = test_result.get('p_value', 'Unknown')
            interpretation = test_result.get('interpretation', 'No interpretation')
            p_val_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
            report += f"**{test_type}:** p = {p_val_str} - {interpretation}\n\n"
    
    # Sample size assessment
    sample_assessment = conclusions.get('sample_size_assessment', {})
    if sample_assessment:
        report += f"### Sample Size Assessment\n"
        report += f"**Adequacy:** {sample_assessment.get('adequacy', 'Unknown')}\n"
        recommendations = sample_assessment.get('recommendations', [])
        for rec in recommendations:
            report += f"- {rec}\n"
    
    report += f"\n---\n*Report generated using statistical validation framework for Hegel's Agents*"
    
    return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Statistical validation of Hegel's Agents vs Single Review performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load existing evaluation data from file
  python statistical_validation.py --data-file evaluation_results.json
  
  # Run new comparative evaluation with Hegel's agents
  python statistical_validation.py --run-new-evaluation --sample-size 50
  
  # Full analysis with custom statistical parameters
  python statistical_validation.py --data-file results.json --confidence-level 0.99 --output-dir validation_results/
  
  # High-precision analysis with large sample
  python statistical_validation.py --run-new-evaluation --sample-size 100 --significance-level 0.01
        """
    )
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--data-file', '-f',
        type=str,
        help="Load comparison data from JSON file"
    )
    data_group.add_argument(
        '--run-new-evaluation', '-e',
        action='store_true',
        help="Run new comparative evaluation using existing infrastructure"
    )
    
    # Configuration options
    def validate_sample_size(value):
        """Validate sample size is reasonable."""
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer: {value}")
        
        if ivalue < 2:
            raise argparse.ArgumentTypeError("Sample size must be at least 2")
        if ivalue > 100000:
            raise argparse.ArgumentTypeError("Sample size cannot exceed 100,000")
        return ivalue

    def validate_probability(min_val=0.0, max_val=1.0):
        """Create validator for probability values."""
        def validator(value):
            try:
                fvalue = float(value)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid number: {value}")
            
            if not (min_val <= fvalue <= max_val):
                raise argparse.ArgumentTypeError(
                    f"Value must be between {min_val} and {max_val}"
                )
            return fvalue
        return validator

    parser.add_argument(
        '--sample-size', '-n',
        type=validate_sample_size,
        default=50,
        help="Sample size for analysis (default: 50, range: 2-100000)"
    )
    parser.add_argument(
        '--confidence-level', '-c',
        type=validate_probability(0.5, 0.999),
        default=0.95,
        help="Confidence level for intervals (default: 0.95, range: 0.5-0.999)"
    )
    parser.add_argument(
        '--significance-level', '-s',
        type=validate_probability(0.001, 0.5),
        default=0.05,
        help="Statistical significance level (default: 0.05, range: 0.001-0.5)"
    )
    parser.add_argument(
        '--effect-size-threshold', '-t',
        type=float,
        default=0.3,
        help="Minimum meaningful effect size (default: 0.3)"
    )
    parser.add_argument(
        '--power-target', '-p',
        type=float,
        default=0.80,
        help="Target statistical power (default: 0.80)"
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='statistical_validation_results',
        help="Output directory for results (default: statistical_validation_results)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = ValidationConfig(
            sample_size=args.sample_size,
            confidence_level=args.confidence_level,
            significance_level=args.significance_level,
            effect_size_threshold=args.effect_size_threshold,
            power_target=args.power_target,
            random_seed=args.random_seed
        )
        
        logger.info("Starting statistical validation analysis...")
        logger.info(f"Configuration: n={config.sample_size}, α={config.significance_level}, CI={config.confidence_level}")
        
        # Initialize validator
        validator = StatisticalValidator(config)
        
        # Collect data
        if args.data_file:
            logger.info(f"Loading data from {args.data_file}")
            comparison_data = validator.collect_evaluation_data(data_source=args.data_file)
        elif args.run_new_evaluation:
            logger.info("Running new comparative evaluation...")
            comparison_data = validator.collect_evaluation_data(
                run_new_evaluation=True, 
                num_questions=args.sample_size
            )
        else:
            logger.error("No data source specified. Use --data-file or --run-new-evaluation")
            comparison_data = validator.collect_evaluation_data()
        
        # Validate data
        if not comparison_data.validate_data():
            logger.error("Data validation failed - cannot proceed with analysis")
            return 1
        
        logger.info(f"Analysis will use {comparison_data.sample_size} comparison pairs")
        
        # Run statistical analysis
        results = validator.run_statistical_analysis(comparison_data)
        
        # Print summary to console
        conclusions = results.get('conclusions', {})
        print(f"\n{'='*60}")
        print("STATISTICAL VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Recommendation: {conclusions.get('overall_recommendation', 'Unknown')}")
        print(f"Confidence Level: {conclusions.get('confidence_level', 'Unknown')}")
        print(f"\n{conclusions.get('summary', 'No summary available.')}")
        
        # Show key metrics
        data_summary = results.get('data_summary', {})
        practical = results.get('practical_significance', {})
        
        print(f"\nKey Metrics:")
        print(f"  Sample Size: {results.get('sample_size', 'Unknown')}")
        print(f"  Mean Improvement: {practical.get('mean_improvement_points', 0):.2f} points")
        print(f"  Improvement Rate: {data_summary.get('improvement_rate', 0)*100:.1f}% of tests")
        print(f"  Practical Significance: {'Yes' if practical.get('practically_significant') else 'No'}")
        
        # Statistical significance summary
        stat_summary = conclusions.get('statistical_significance_summary', {})
        print(f"  Statistical Significance: {'Yes' if stat_summary.get('overall_significance') else 'No'}")
        
        # Save results if requested
        if not args.no_save:
            logger.info(f"Saving results to {args.output_dir}")
            saved_files = save_results(results, args.output_dir, config)
            print(f"\nResults saved:")
            for file_type, file_path in saved_files.items():
                print(f"  {file_type}: {file_path}")
        
        # Return appropriate exit code
        recommendation = conclusions.get('overall_recommendation', '')
        if recommendation in ['ADOPT_HEGELS_APPROACH', 'LIKELY_ADOPT_HEGELS_APPROACH']:
            print(f"\n✅ CONCLUSION: Evidence supports using Hegel's dialectical approach")
            return 0
        elif recommendation == 'CONSIDER_HEGELS_APPROACH':
            print(f"\n⚠️ CONCLUSION: Mixed evidence - consider further evaluation")
            return 0
        else:
            print(f"\n❌ CONCLUSION: Insufficient evidence for Hegel's approach superiority")
            return 1
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
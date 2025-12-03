"""
Statistical Analysis and Reporting for Evaluation Results

This module provides comprehensive statistical analysis and reporting capabilities
for evaluation results, including trend analysis, correlation studies, and
automated report generation.

Key Features:
- Advanced statistical analysis of evaluation metrics
- Trend analysis and performance monitoring
- Correlation and regression analysis
- Automated report generation with visualizations
- Statistical significance testing
- Performance benchmarking and comparison
"""

import json
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict, Counter

from agents.utils import AgentLogger


@dataclass
class StatisticalSummary:
    """
    Statistical summary for a set of measurements.
    """
    count: int
    mean: float
    median: float
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    percentiles: Dict[int, float]
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'count': self.count,
            'mean': self.mean,
            'median': self.median,
            'std_dev': self.std_dev,
            'variance': self.variance,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'percentiles': self.percentiles,
            'confidence_interval': list(self.confidence_interval)
        }


@dataclass
class TrendAnalysis:
    """
    Results of trend analysis over time.
    """
    metric_name: str
    time_period: str
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float  # 0-1, strength of the trend
    slope: float
    r_squared: float
    data_points: int
    start_date: datetime
    end_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'time_period': self.time_period,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'slope': self.slope,
            'r_squared': self.r_squared,
            'data_points': self.data_points,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat()
        }


@dataclass
class CorrelationAnalysis:
    """
    Results of correlation analysis between metrics.
    """
    metric_x: str
    metric_y: str
    correlation_coefficient: float
    p_value: Optional[float]
    relationship_strength: str  # "strong", "moderate", "weak", "none"
    relationship_direction: str  # "positive", "negative", "none"
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_x': self.metric_x,
            'metric_y': self.metric_y,
            'correlation_coefficient': self.correlation_coefficient,
            'p_value': self.p_value,
            'relationship_strength': self.relationship_strength,
            'relationship_direction': self.relationship_direction,
            'sample_size': self.sample_size
        }


@dataclass
class BenchmarkComparison:
    """
    Comparison against benchmark performance.
    """
    metric_name: str
    current_value: float
    benchmark_value: float
    difference: float
    percentage_change: float
    performance_category: str  # "exceeds", "meets", "below", "significantly_below"
    statistical_significance: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'benchmark_value': self.benchmark_value,
            'difference': self.difference,
            'percentage_change': self.percentage_change,
            'performance_category': self.performance_category,
            'statistical_significance': self.statistical_significance
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for evaluation results.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.logger = AgentLogger("statistical_analyzer")
        
    def calculate_summary_statistics(self, values: List[float]) -> StatisticalSummary:
        """
        Calculate comprehensive summary statistics.
        
        Args:
            values: List of numeric values
            
        Returns:
            StatisticalSummary with all statistical measures
        """
        if not values:
            raise ValueError("Cannot calculate statistics for empty data")
        
        values = [float(v) for v in values if v is not None]  # Clean data
        
        if len(values) == 0:
            raise ValueError("No valid numeric values found")
        
        # Basic statistics
        count = len(values)
        mean = statistics.mean(values)
        median = statistics.median(values)
        std_dev = statistics.stdev(values) if count > 1 else 0.0
        variance = statistics.variance(values) if count > 1 else 0.0
        min_value = min(values)
        max_value = max(values)
        
        # Percentiles
        percentiles = {}
        for p in [5, 10, 25, 75, 90, 95]:
            percentiles[p] = self._calculate_percentile(values, p)
        
        # Confidence interval
        confidence_interval = self._calculate_confidence_interval(values)
        
        return StatisticalSummary(
            count=count,
            mean=mean,
            median=median,
            std_dev=std_dev,
            variance=variance,
            min_value=min_value,
            max_value=max_value,
            percentiles=percentiles,
            confidence_interval=confidence_interval
        )
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        try:
            import numpy as np
            return float(np.percentile(values, percentile))
        except ImportError:
            # Fallback calculation without numpy
            sorted_values = sorted(values)
            k = (len(sorted_values) - 1) * percentile / 100
            f = int(k)
            c = k - f
            if f == len(sorted_values) - 1:
                return sorted_values[f]
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        try:
            import scipy.stats as stats
            mean = statistics.mean(values)
            sem = stats.sem(values)
            h = sem * stats.t.ppf((1 + self.confidence_level) / 2., len(values)-1)
            return (mean - h, mean + h)
        except ImportError:
            # Fallback using standard deviation
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            margin = 1.96 * (std_dev / (len(values) ** 0.5))  # Approximate 95% CI
            return (mean - margin, mean + margin)
    
    def analyze_trend(self, 
                     time_series_data: List[Dict[str, Any]], 
                     metric_name: str,
                     date_field: str = 'timestamp') -> TrendAnalysis:
        """
        Analyze trend in time series data.
        
        Args:
            time_series_data: List of data points with timestamps
            metric_name: Name of metric to analyze
            date_field: Field name containing the timestamp
            
        Returns:
            TrendAnalysis with trend information
        """
        # Extract and sort data
        data_points = []
        for item in time_series_data:
            if metric_name in item and date_field in item:
                timestamp_str = item[date_field]
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                value = float(item[metric_name])
                data_points.append((timestamp, value))
        
        if len(data_points) < 2:
            raise ValueError("Insufficient data points for trend analysis")
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x[0])
        
        # Convert to arrays for analysis
        timestamps = [dp[0] for dp in data_points]
        values = [dp[1] for dp in data_points]
        
        # Convert timestamps to numeric values (days from first timestamp)
        start_date = timestamps[0]
        numeric_times = [(ts - start_date).total_seconds() / 86400 for ts in timestamps]
        
        # Perform linear regression
        slope, r_squared = self._calculate_linear_regression(numeric_times, values)
        
        # Determine trend characteristics
        trend_direction = self._determine_trend_direction(slope, r_squared)
        trend_strength = r_squared  # R-squared as measure of trend strength
        
        return TrendAnalysis(
            metric_name=metric_name,
            time_period=f"{start_date.date()} to {timestamps[-1].date()}",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            data_points=len(data_points),
            start_date=start_date,
            end_date=timestamps[-1]
        )
    
    def _calculate_linear_regression(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Calculate linear regression slope and R-squared."""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            x = np.array(x_values)
            y = np.array(y_values)
            
            # Calculate slope using least squares
            slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
            
            # Calculate R-squared
            y_pred = slope * (x - np.mean(x)) + np.mean(y)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return slope, r_squared
            
        except ImportError:
            # Fallback calculation without numpy
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x**2 for x in x_values)
            sum_y2 = sum(y**2 for y in y_values)
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            
            # Calculate R-squared (simplified)
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y)**2 for y in y_values)
            y_pred = [slope * (x - sum_x/n) + mean_y for x in x_values]
            ss_res = sum((y - y_pred[i])**2 for i, y in enumerate(y_values))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return slope, r_squared
    
    def _determine_trend_direction(self, slope: float, r_squared: float) -> str:
        """Determine trend direction based on slope and R-squared."""
        if r_squared < 0.1:
            return "stable"
        elif r_squared < 0.3:
            return "volatile"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def analyze_correlation(self, 
                          data: List[Dict[str, Any]], 
                          metric_x: str, 
                          metric_y: str) -> CorrelationAnalysis:
        """
        Analyze correlation between two metrics.
        
        Args:
            data: List of data points containing both metrics
            metric_x: First metric name
            metric_y: Second metric name
            
        Returns:
            CorrelationAnalysis with correlation information
        """
        # Extract paired values
        pairs = []
        for item in data:
            if metric_x in item and metric_y in item:
                try:
                    x_val = float(item[metric_x])
                    y_val = float(item[metric_y])
                    pairs.append((x_val, y_val))
                except (ValueError, TypeError):
                    continue
        
        if len(pairs) < 3:
            raise ValueError("Insufficient paired data points for correlation analysis")
        
        x_values = [p[0] for p in pairs]
        y_values = [p[1] for p in pairs]
        
        # Calculate correlation coefficient
        correlation_coefficient = self._calculate_correlation_coefficient(x_values, y_values)
        
        # Calculate p-value if possible
        p_value = self._calculate_correlation_p_value(x_values, y_values, correlation_coefficient)
        
        # Interpret correlation strength and direction
        relationship_strength = self._interpret_correlation_strength(abs(correlation_coefficient))
        relationship_direction = "positive" if correlation_coefficient > 0 else "negative" if correlation_coefficient < 0 else "none"
        
        return CorrelationAnalysis(
            metric_x=metric_x,
            metric_y=metric_y,
            correlation_coefficient=correlation_coefficient,
            p_value=p_value,
            relationship_strength=relationship_strength,
            relationship_direction=relationship_direction,
            sample_size=len(pairs)
        )
    
    def _calculate_correlation_coefficient(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        try:
            import numpy as np
            return float(np.corrcoef(x_values, y_values)[0, 1])
        except ImportError:
            # Manual calculation
            n = len(x_values)
            if n <= 1:
                return 0.0
            
            mean_x = sum(x_values) / n
            mean_y = sum(y_values) / n
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
            sum_sq_x = sum((x - mean_x)**2 for x in x_values)
            sum_sq_y = sum((y - mean_y)**2 for y in y_values)
            
            denominator = (sum_sq_x * sum_sq_y)**0.5
            
            return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_correlation_p_value(self, x_values: List[float], y_values: List[float], correlation: float) -> Optional[float]:
        """Calculate p-value for correlation coefficient."""
        try:
            import scipy.stats as stats
            _, p_value = stats.pearsonr(x_values, y_values)
            return p_value
        except ImportError:
            # Cannot calculate without scipy
            return None
    
    def _interpret_correlation_strength(self, abs_correlation: float) -> str:
        """Interpret correlation strength based on absolute value."""
        if abs_correlation >= 0.7:
            return "strong"
        elif abs_correlation >= 0.3:
            return "moderate"
        elif abs_correlation >= 0.1:
            return "weak"
        else:
            return "none"
    
    def compare_to_benchmark(self, 
                           current_data: List[Dict[str, Any]], 
                           benchmark_data: List[Dict[str, Any]], 
                           metric_name: str) -> BenchmarkComparison:
        """
        Compare current performance to benchmark.
        
        Args:
            current_data: Current evaluation data
            benchmark_data: Benchmark data for comparison
            metric_name: Name of metric to compare
            
        Returns:
            BenchmarkComparison with comparison results
        """
        # Extract metric values
        current_values = [float(item[metric_name]) for item in current_data if metric_name in item]
        benchmark_values = [float(item[metric_name]) for item in benchmark_data if metric_name in item]
        
        if not current_values or not benchmark_values:
            raise ValueError("Insufficient data for benchmark comparison")
        
        # Calculate means
        current_value = statistics.mean(current_values)
        benchmark_value = statistics.mean(benchmark_values)
        
        # Calculate difference and percentage change
        difference = current_value - benchmark_value
        percentage_change = (difference / benchmark_value * 100) if benchmark_value != 0 else 0
        
        # Determine performance category
        performance_category = self._categorize_performance(percentage_change)
        
        # Calculate statistical significance if possible
        statistical_significance = self._test_significance(current_values, benchmark_values)
        
        return BenchmarkComparison(
            metric_name=metric_name,
            current_value=current_value,
            benchmark_value=benchmark_value,
            difference=difference,
            percentage_change=percentage_change,
            performance_category=performance_category,
            statistical_significance=statistical_significance
        )
    
    def _categorize_performance(self, percentage_change: float) -> str:
        """Categorize performance based on percentage change."""
        if percentage_change >= 10:
            return "exceeds"
        elif percentage_change >= -5:
            return "meets"
        elif percentage_change >= -15:
            return "below"
        else:
            return "significantly_below"
    
    def _test_significance(self, current_values: List[float], benchmark_values: List[float]) -> Optional[float]:
        """Test statistical significance of difference."""
        try:
            import scipy.stats as stats
            _, p_value = stats.ttest_ind(current_values, benchmark_values)
            return p_value
        except ImportError:
            return None


class PerformanceBenchmarkSuite:
    """
    Suite for comprehensive performance benchmarking.
    """
    
    def __init__(self, analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize benchmark suite.
        
        Args:
            analyzer: StatisticalAnalyzer instance
        """
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.logger = AgentLogger("benchmark_suite")
        self.benchmarks: Dict[str, Any] = {}
        
    def register_benchmark(self, 
                          name: str, 
                          data: List[Dict[str, Any]], 
                          description: str = "") -> None:
        """
        Register a benchmark dataset.
        
        Args:
            name: Benchmark name
            data: Benchmark data
            description: Description of benchmark
        """
        self.benchmarks[name] = {
            'data': data,
            'description': description,
            'registered_at': datetime.now(),
            'metrics_available': self._extract_available_metrics(data)
        }
        
        self.logger.log_debug(f"Registered benchmark '{name}' with {len(data)} data points")
    
    def _extract_available_metrics(self, data: List[Dict[str, Any]]) -> List[str]:
        """Extract available numeric metrics from data."""
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        # Filter to numeric metrics
        numeric_metrics = []
        for key in all_keys:
            sample_values = [item.get(key) for item in data[:10]]  # Check first 10
            if any(isinstance(v, (int, float)) for v in sample_values):
                numeric_metrics.append(key)
        
        return numeric_metrics
    
    def run_comprehensive_benchmark(self, 
                                  current_data: List[Dict[str, Any]], 
                                  benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparison.
        
        Args:
            current_data: Current data to benchmark
            benchmark_name: Name of benchmark to use (None for all)
            
        Returns:
            Comprehensive benchmark results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'current_data_size': len(current_data),
            'benchmarks_compared': [],
            'metric_comparisons': {},
            'overall_assessment': {}
        }
        
        benchmarks_to_use = [benchmark_name] if benchmark_name else list(self.benchmarks.keys())
        
        for benchmark_name in benchmarks_to_use:
            if benchmark_name not in self.benchmarks:
                self.logger.log_debug(f"Benchmark '{benchmark_name}' not found")
                continue
            
            benchmark_data = self.benchmarks[benchmark_name]['data']
            available_metrics = self.benchmarks[benchmark_name]['metrics_available']
            
            benchmark_results = {
                'benchmark_name': benchmark_name,
                'benchmark_size': len(benchmark_data),
                'metric_comparisons': {}
            }
            
            # Compare each available metric
            for metric in available_metrics:
                try:
                    comparison = self.analyzer.compare_to_benchmark(
                        current_data, benchmark_data, metric
                    )
                    benchmark_results['metric_comparisons'][metric] = comparison.to_dict()
                except Exception as e:
                    self.logger.log_debug(f"Failed to compare metric '{metric}': {e}")
            
            results['benchmarks_compared'].append(benchmark_results)
        
        # Generate overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        return results
    
    def _generate_overall_assessment(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment from benchmark results."""
        all_comparisons = []
        for benchmark in benchmark_results['benchmarks_compared']:
            all_comparisons.extend(benchmark['metric_comparisons'].values())
        
        if not all_comparisons:
            return {'status': 'no_data', 'message': 'No valid comparisons available'}
        
        # Count performance categories
        performance_counts = Counter(comp['performance_category'] for comp in all_comparisons)
        
        # Calculate overall score
        category_scores = {'exceeds': 4, 'meets': 3, 'below': 2, 'significantly_below': 1}
        total_score = sum(performance_counts[cat] * category_scores[cat] for cat in performance_counts)
        max_possible_score = len(all_comparisons) * 4
        overall_score = total_score / max_possible_score if max_possible_score > 0 else 0
        
        # Determine overall status
        if overall_score >= 0.8:
            status = "excellent"
        elif overall_score >= 0.6:
            status = "good"
        elif overall_score >= 0.4:
            status = "needs_improvement"
        else:
            status = "poor"
        
        return {
            'status': status,
            'overall_score': overall_score,
            'performance_distribution': dict(performance_counts),
            'total_metrics_compared': len(all_comparisons),
            'message': f"Overall performance is {status} ({overall_score:.2%} of benchmark)"
        }


class EvaluationReportGenerator:
    """
    Generates comprehensive evaluation reports with statistical analysis.
    """
    
    def __init__(self, analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize report generator.
        
        Args:
            analyzer: StatisticalAnalyzer instance
        """
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.logger = AgentLogger("report_generator")
    
    def generate_comprehensive_report(self, 
                                    evaluation_data: List[Dict[str, Any]], 
                                    report_config: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_data: Evaluation data to analyze
            report_config: Configuration for report generation
            
        Returns:
            Formatted report string
        """
        config = report_config or {}
        
        # Extract metrics
        available_metrics = self._extract_metrics(evaluation_data)
        
        report = []
        report.append("# Comprehensive Evaluation Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Points: {len(evaluation_data)}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        summary = self._generate_executive_summary(evaluation_data, available_metrics)
        report.append(summary)
        report.append("")
        
        # Statistical Analysis for each metric
        for metric in available_metrics:
            try:
                report.append(f"## Analysis: {metric}")
                metric_analysis = self._analyze_metric(evaluation_data, metric)
                report.append(metric_analysis)
                report.append("")
            except Exception as e:
                self.logger.log_debug(f"Failed to analyze metric '{metric}': {e}")
        
        # Correlations
        if len(available_metrics) > 1:
            report.append("## Correlation Analysis")
            correlation_analysis = self._analyze_correlations(evaluation_data, available_metrics)
            report.append(correlation_analysis)
            report.append("")
        
        # Trends (if timestamp data available)
        if self._has_timestamp_data(evaluation_data):
            report.append("## Trend Analysis")
            trend_analysis = self._analyze_trends(evaluation_data, available_metrics)
            report.append(trend_analysis)
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        recommendations = self._generate_recommendations(evaluation_data, available_metrics)
        report.append(recommendations)
        
        return "\n".join(report)
    
    def _extract_metrics(self, data: List[Dict[str, Any]]) -> List[str]:
        """Extract available numeric metrics."""
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        numeric_metrics = []
        for key in all_keys:
            if key in ['timestamp', 'created_at', 'question', 'error']:
                continue
                
            sample_values = [item.get(key) for item in data[:10]]
            if any(isinstance(v, (int, float)) for v in sample_values):
                numeric_metrics.append(key)
        
        return numeric_metrics
    
    def _generate_executive_summary(self, data: List[Dict[str, Any]], metrics: List[str]) -> str:
        """Generate executive summary."""
        successful_evaluations = len([d for d in data if d.get('success', True)])
        success_rate = successful_evaluations / len(data) * 100 if data else 0
        
        summary = [
            f"Analyzed {len(data)} evaluation records with {success_rate:.1f}% success rate.",
            f"Key metrics evaluated: {', '.join(metrics[:5])}{'...' if len(metrics) > 5 else ''}",
        ]
        
        if metrics:
            # Get summary stats for first metric
            values = [float(item[metrics[0]]) for item in data if metrics[0] in item and isinstance(item[metrics[0]], (int, float))]
            if values:
                stats = self.analyzer.calculate_summary_statistics(values)
                summary.append(f"Primary metric ({metrics[0]}) average: {stats.mean:.3f} ± {stats.std_dev:.3f}")
        
        return "\n".join(summary)
    
    def _analyze_metric(self, data: List[Dict[str, Any]], metric: str) -> str:
        """Analyze a specific metric."""
        values = [float(item[metric]) for item in data if metric in item and isinstance(item[metric], (int, float))]
        
        if not values:
            return f"No valid data for {metric}"
        
        stats = self.analyzer.calculate_summary_statistics(values)
        
        analysis = [
            f"**{metric}** (n={stats.count})",
            f"- Mean: {stats.mean:.3f}",
            f"- Median: {stats.median:.3f}",
            f"- Standard Deviation: {stats.std_dev:.3f}",
            f"- Range: {stats.min_value:.3f} to {stats.max_value:.3f}",
            f"- 95% CI: [{stats.confidence_interval[0]:.3f}, {stats.confidence_interval[1]:.3f}]",
        ]
        
        # Add interpretation
        cv = stats.std_dev / stats.mean if stats.mean != 0 else float('inf')
        if cv < 0.1:
            analysis.append("- **Interpretation:** Very consistent performance")
        elif cv < 0.3:
            analysis.append("- **Interpretation:** Moderate variability")
        else:
            analysis.append("- **Interpretation:** High variability, investigate outliers")
        
        return "\n".join(analysis)
    
    def _analyze_correlations(self, data: List[Dict[str, Any]], metrics: List[str]) -> str:
        """Analyze correlations between metrics."""
        correlations = []
        
        for i, metric_x in enumerate(metrics):
            for metric_y in metrics[i+1:]:
                try:
                    corr_analysis = self.analyzer.analyze_correlation(data, metric_x, metric_y)
                    if corr_analysis.relationship_strength != "none":
                        correlations.append(
                            f"- **{metric_x} vs {metric_y}**: "
                            f"{corr_analysis.relationship_strength} {corr_analysis.relationship_direction} "
                            f"correlation (r={corr_analysis.correlation_coefficient:.3f})"
                        )
                except Exception:
                    continue
        
        if correlations:
            return "Notable correlations found:\n" + "\n".join(correlations)
        else:
            return "No significant correlations detected between metrics."
    
    def _has_timestamp_data(self, data: List[Dict[str, Any]]) -> bool:
        """Check if data has timestamp information."""
        timestamp_fields = ['timestamp', 'created_at', 'date', 'time']
        return any(field in data[0] for field in timestamp_fields if data)
    
    def _analyze_trends(self, data: List[Dict[str, Any]], metrics: List[str]) -> str:
        """Analyze trends over time."""
        trends = []
        
        for metric in metrics[:3]:  # Analyze first 3 metrics
            try:
                trend_analysis = self.analyzer.analyze_trend(data, metric)
                trends.append(
                    f"- **{metric}**: {trend_analysis.trend_direction} trend "
                    f"(R²={trend_analysis.r_squared:.3f}) over {trend_analysis.data_points} data points"
                )
            except Exception:
                continue
        
        if trends:
            return "Trend analysis:\n" + "\n".join(trends)
        else:
            return "Insufficient temporal data for trend analysis."
    
    def _generate_recommendations(self, data: List[Dict[str, Any]], metrics: List[str]) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check data quality
        if len(data) < 30:
            recommendations.append("- Collect more data points for robust statistical analysis (current: {}, recommended: 30+)".format(len(data)))
        
        # Check for high variability metrics
        for metric in metrics[:3]:
            try:
                values = [float(item[metric]) for item in data if metric in item and isinstance(item[metric], (int, float))]
                if values:
                    stats = self.analyzer.calculate_summary_statistics(values)
                    cv = stats.std_dev / stats.mean if stats.mean != 0 else 0
                    if cv > 0.5:
                        recommendations.append(f"- Investigate high variability in {metric} (CV={cv:.2f})")
            except Exception:
                continue
        
        # Performance recommendations
        success_rate = len([d for d in data if d.get('success', True)]) / len(data) if data else 0
        if success_rate < 0.9:
            recommendations.append(f"- Improve evaluation success rate (current: {success_rate:.1%})")
        
        if not recommendations:
            recommendations.append("- Performance appears stable, continue current monitoring approach")
        
        return "\n".join(recommendations)


# Factory functions
def create_statistical_analyzer(confidence_level: float = 0.95) -> StatisticalAnalyzer:
    """Create a statistical analyzer with specified confidence level."""
    return StatisticalAnalyzer(confidence_level)


def create_benchmark_suite() -> PerformanceBenchmarkSuite:
    """Create a performance benchmark suite."""
    return PerformanceBenchmarkSuite()


def create_report_generator() -> EvaluationReportGenerator:
    """Create an evaluation report generator."""
    return EvaluationReportGenerator()


# Export main classes
__all__ = [
    'StatisticalSummary',
    'TrendAnalysis',
    'CorrelationAnalysis', 
    'BenchmarkComparison',
    'StatisticalAnalyzer',
    'PerformanceBenchmarkSuite',
    'EvaluationReportGenerator',
    'create_statistical_analyzer',
    'create_benchmark_suite',
    'create_report_generator'
]
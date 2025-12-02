"""
Performance Benchmarking Suite

This module provides comprehensive performance benchmarking capabilities for
evaluation systems, including latency measurement, throughput analysis,
resource utilization monitoring, and performance comparison.

Key Features:
- Comprehensive performance metrics collection
- Baseline performance establishment and comparison
- Resource utilization monitoring (CPU, memory, etc.)
- Performance regression detection
- Load testing and stress testing capabilities
- Performance profiling and bottleneck identification
"""

import time
import psutil
import threading
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import statistics
from collections import defaultdict, deque
import concurrent.futures

from agents.utils import AgentLogger


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a single operation.
    """
    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    thread_count: Optional[int] = None
    io_read_bytes: Optional[int] = None
    io_write_bytes: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'thread_count': self.thread_count,
            'io_read_bytes': self.io_read_bytes,
            'io_write_bytes': self.io_write_bytes,
            'success': self.success,
            'error_message': self.error_message,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class BenchmarkResult:
    """
    Results from a complete benchmark run.
    """
    benchmark_id: str
    benchmark_name: str
    test_description: str
    start_time: datetime
    end_time: datetime
    total_operations: int
    successful_operations: int
    failed_operations: int
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_id': self.benchmark_id,
            'benchmark_name': self.benchmark_name,
            'test_description': self.test_description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'metrics': [m.to_dict() for m in self.metrics],
            'summary_statistics': self.summary_statistics,
            'resource_usage': self.resource_usage
        }


class PerformanceProfiler:
    """
    Profiles performance of operations with comprehensive monitoring.
    """
    
    def __init__(self, enable_resource_monitoring: bool = True):
        """
        Initialize performance profiler.
        
        Args:
            enable_resource_monitoring: Whether to monitor system resources
        """
        self.enable_resource_monitoring = enable_resource_monitoring
        self.logger = AgentLogger("performance_profiler")
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._resource_samples: deque = deque(maxlen=1000)
        
    def profile_operation(self, 
                         operation: Callable, 
                         operation_type: str,
                         operation_id: str = None,
                         *args, **kwargs) -> PerformanceMetrics:
        """
        Profile a single operation with comprehensive metrics.
        
        Args:
            operation: Function to profile
            operation_type: Type/category of operation
            operation_id: Optional operation identifier
            *args, **kwargs: Arguments for the operation
            
        Returns:
            PerformanceMetrics with detailed performance data
        """
        operation_id = operation_id or f"{operation_type}_{int(time.time())}"
        
        # Start resource monitoring if enabled
        if self.enable_resource_monitoring and not self._monitoring_active:
            self._start_resource_monitoring()
        
        # Capture initial state
        start_time = datetime.now()
        initial_memory = self._get_memory_usage()
        initial_io = self._get_io_stats()
        
        # Force garbage collection for clean measurement
        gc.collect()
        
        success = True
        error_message = None
        result = None
        
        try:
            # Execute operation
            result = operation(*args, **kwargs)
            
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.log_warning(f"Operation {operation_id} failed: {e}")
        
        # Capture final state
        end_time = datetime.now()
        final_memory = self._get_memory_usage()
        final_io = self._get_io_stats()
        
        # Calculate metrics
        duration = (end_time - start_time).total_seconds()
        peak_memory = self._get_peak_memory_during_operation(start_time, end_time)
        
        # Get CPU usage (average during operation)
        cpu_usage = self._get_average_cpu_usage(start_time, end_time)
        
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=final_memory,
            peak_memory_mb=peak_memory,
            thread_count=threading.active_count(),
            io_read_bytes=final_io.get('read_bytes', 0) - initial_io.get('read_bytes', 0),
            io_write_bytes=final_io.get('write_bytes', 0) - initial_io.get('write_bytes', 0),
            success=success,
            error_message=error_message,
            custom_metrics={'result_size': len(str(result)) if result else 0}
        )
        
        return metrics
    
    def _start_resource_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._resource_monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _resource_monitoring_loop(self) -> None:
        """Continuous resource monitoring loop."""
        while self._monitoring_active:
            try:
                sample = {
                    'timestamp': datetime.now(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict()
                }
                
                self._resource_samples.append(sample)
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                self.logger.log_warning(f"Resource monitoring error: {e}")
                time.sleep(1)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_io_stats(self) -> Dict[str, int]:
        """Get current I/O statistics."""
        try:
            process = psutil.Process()
            io_counters = process.io_counters()
            return {
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes
            }
        except Exception:
            return {'read_bytes': 0, 'write_bytes': 0}
    
    def _get_peak_memory_during_operation(self, start_time: datetime, end_time: datetime) -> Optional[float]:
        """Get peak memory usage during operation."""
        relevant_samples = [
            sample for sample in self._resource_samples
            if start_time <= sample['timestamp'] <= end_time
        ]
        
        if relevant_samples:
            return max(sample['memory_mb'] for sample in relevant_samples)
        return None
    
    def _get_average_cpu_usage(self, start_time: datetime, end_time: datetime) -> Optional[float]:
        """Get average CPU usage during operation."""
        relevant_samples = [
            sample for sample in self._resource_samples
            if start_time <= sample['timestamp'] <= end_time
        ]
        
        if relevant_samples:
            return statistics.mean(sample['cpu_percent'] for sample in relevant_samples)
        return None
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for evaluation operations.
    """
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        """
        Initialize benchmark suite.
        
        Args:
            profiler: PerformanceProfiler instance
        """
        self.profiler = profiler or PerformanceProfiler()
        self.logger = AgentLogger("benchmark_suite")
        
        # Benchmark results storage
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
    def run_latency_benchmark(self, 
                            operation: Callable,
                            operation_type: str,
                            iterations: int = 100,
                            warmup_iterations: int = 10) -> BenchmarkResult:
        """
        Run latency benchmark for an operation.
        
        Args:
            operation: Function to benchmark
            operation_type: Type of operation
            iterations: Number of test iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            BenchmarkResult with latency analysis
        """
        benchmark_id = f"latency_{operation_type}_{int(time.time())}"
        
        self.logger.log_info(f"Running latency benchmark: {benchmark_id}")
        
        start_time = datetime.now()
        metrics = []
        
        # Warmup iterations (not recorded)
        self.logger.log_info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            try:
                operation()
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark iterations
        self.logger.log_info(f"Running {iterations} benchmark iterations...")
        successful_operations = 0
        failed_operations = 0
        
        for i in range(iterations):
            operation_id = f"{benchmark_id}_iter_{i}"
            
            try:
                metric = self.profiler.profile_operation(
                    operation, operation_type, operation_id
                )
                metrics.append(metric)
                
                if metric.success:
                    successful_operations += 1
                else:
                    failed_operations += 1
                    
            except Exception as e:
                self.logger.log_warning(f"Benchmark iteration {i} failed: {e}")
                failed_operations += 1
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_statistics = self._calculate_summary_statistics(successful_metrics)
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(metrics)
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=f"Latency Benchmark - {operation_type}",
            test_description=f"Latency test with {iterations} iterations",
            start_time=start_time,
            end_time=end_time,
            total_operations=iterations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            metrics=metrics,
            summary_statistics=summary_statistics,
            resource_usage=resource_usage
        )
        
        self.benchmark_results[benchmark_id] = result
        return result
    
    def run_throughput_benchmark(self, 
                                operation: Callable,
                                operation_type: str,
                                duration_seconds: int = 60,
                                concurrent_workers: int = 1) -> BenchmarkResult:
        """
        Run throughput benchmark to measure operations per second.
        
        Args:
            operation: Function to benchmark
            operation_type: Type of operation
            duration_seconds: How long to run the test
            concurrent_workers: Number of concurrent workers
            
        Returns:
            BenchmarkResult with throughput analysis
        """
        benchmark_id = f"throughput_{operation_type}_{int(time.time())}"
        
        self.logger.log_info(f"Running throughput benchmark: {benchmark_id}")
        
        start_time = datetime.now()
        end_time_target = start_time + timedelta(seconds=duration_seconds)
        
        metrics = []
        operation_counter = 0
        successful_operations = 0
        failed_operations = 0
        
        def worker_function():
            nonlocal operation_counter, successful_operations, failed_operations
            
            while datetime.now() < end_time_target:
                operation_id = f"{benchmark_id}_op_{operation_counter}"
                operation_counter += 1
                
                try:
                    metric = self.profiler.profile_operation(
                        operation, operation_type, operation_id
                    )
                    metrics.append(metric)
                    
                    if metric.success:
                        successful_operations += 1
                    else:
                        failed_operations += 1
                        
                except Exception as e:
                    self.logger.log_warning(f"Throughput operation failed: {e}")
                    failed_operations += 1
        
        # Run with concurrent workers
        if concurrent_workers == 1:
            worker_function()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
                futures = [executor.submit(worker_function) for _ in range(concurrent_workers)]
                concurrent.futures.wait(futures)
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_statistics = self._calculate_summary_statistics(successful_metrics)
        
        # Add throughput-specific metrics
        summary_statistics.update({
            'operations_per_second': successful_operations / actual_duration,
            'concurrent_workers': concurrent_workers,
            'actual_duration_seconds': actual_duration
        })
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(metrics)
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=f"Throughput Benchmark - {operation_type}",
            test_description=f"Throughput test with {concurrent_workers} workers for {duration_seconds}s",
            start_time=start_time,
            end_time=end_time,
            total_operations=len(metrics),
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            metrics=metrics,
            summary_statistics=summary_statistics,
            resource_usage=resource_usage
        )
        
        self.benchmark_results[benchmark_id] = result
        return result
    
    def run_stress_test(self, 
                       operation: Callable,
                       operation_type: str,
                       max_concurrent_workers: int = 10,
                       ramp_up_duration: int = 30,
                       steady_duration: int = 60) -> BenchmarkResult:
        """
        Run stress test with gradually increasing load.
        
        Args:
            operation: Function to benchmark
            operation_type: Type of operation
            max_concurrent_workers: Maximum number of concurrent workers
            ramp_up_duration: Time to ramp up to max workers (seconds)
            steady_duration: Time to maintain max load (seconds)
            
        Returns:
            BenchmarkResult with stress test analysis
        """
        benchmark_id = f"stress_{operation_type}_{int(time.time())}"
        
        self.logger.log_info(f"Running stress test: {benchmark_id}")
        
        start_time = datetime.now()
        metrics = []
        operation_counter = 0
        successful_operations = 0
        failed_operations = 0
        
        # Track performance degradation
        performance_timeline = []
        
        def worker_function(worker_id: int, end_time: datetime):
            nonlocal operation_counter, successful_operations, failed_operations
            
            while datetime.now() < end_time:
                operation_id = f"{benchmark_id}_worker_{worker_id}_op_{operation_counter}"
                operation_counter += 1
                
                try:
                    metric = self.profiler.profile_operation(
                        operation, operation_type, operation_id
                    )
                    metrics.append(metric)
                    
                    if metric.success:
                        successful_operations += 1
                    else:
                        failed_operations += 1
                        
                except Exception as e:
                    failed_operations += 1
                
                time.sleep(0.01)  # Brief pause to prevent overwhelming
        
        # Ramp-up phase
        self.logger.log_info(f"Ramping up to {max_concurrent_workers} workers over {ramp_up_duration}s")
        
        ramp_end_time = start_time + timedelta(seconds=ramp_up_duration)
        for current_workers in range(1, max_concurrent_workers + 1):
            phase_duration = ramp_up_duration / max_concurrent_workers
            phase_end = datetime.now() + timedelta(seconds=phase_duration)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
                futures = [
                    executor.submit(worker_function, i, min(phase_end, ramp_end_time))
                    for i in range(current_workers)
                ]
                concurrent.futures.wait(futures)
            
            # Record performance at this load level
            recent_metrics = [m for m in metrics if m.start_time >= datetime.now() - timedelta(seconds=10)]
            if recent_metrics:
                avg_duration = statistics.mean(m.duration_seconds for m in recent_metrics if m.success)
                performance_timeline.append({
                    'workers': current_workers,
                    'avg_duration': avg_duration,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Steady-state phase
        self.logger.log_info(f"Maintaining {max_concurrent_workers} workers for {steady_duration}s")
        steady_end_time = datetime.now() + timedelta(seconds=steady_duration)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers) as executor:
            futures = [
                executor.submit(worker_function, i, steady_end_time)
                for i in range(max_concurrent_workers)
            ]
            concurrent.futures.wait(futures)
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_statistics = self._calculate_summary_statistics(successful_metrics)
        
        # Add stress test specific metrics
        summary_statistics.update({
            'max_concurrent_workers': max_concurrent_workers,
            'performance_timeline': performance_timeline,
            'performance_degradation': self._calculate_performance_degradation(performance_timeline)
        })
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(metrics)
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=f"Stress Test - {operation_type}",
            test_description=f"Stress test ramping to {max_concurrent_workers} workers",
            start_time=start_time,
            end_time=end_time,
            total_operations=len(metrics),
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            metrics=metrics,
            summary_statistics=summary_statistics,
            resource_usage=resource_usage
        )
        
        self.benchmark_results[benchmark_id] = result
        return result
    
    def _calculate_summary_statistics(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics from performance metrics."""
        if not metrics:
            return {}
        
        durations = [m.duration_seconds for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics if m.memory_usage_mb is not None]
        cpu_usage = [m.cpu_usage_percent for m in metrics if m.cpu_usage_percent is not None]
        
        stats = {
            'count': len(metrics),
            'duration_stats': {
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations),
                'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0.0,
                'percentile_95': self._calculate_percentile(durations, 95),
                'percentile_99': self._calculate_percentile(durations, 99)
            }
        }
        
        if memory_usage:
            stats['memory_stats'] = {
                'mean_mb': statistics.mean(memory_usage),
                'max_mb': max(memory_usage),
                'min_mb': min(memory_usage)
            }
        
        if cpu_usage:
            stats['cpu_stats'] = {
                'mean_percent': statistics.mean(cpu_usage),
                'max_percent': max(cpu_usage),
                'min_percent': min(cpu_usage)
            }
        
        return stats
    
    def _calculate_resource_usage(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate overall resource usage statistics."""
        if not metrics:
            return {}
        
        total_io_read = sum(m.io_read_bytes or 0 for m in metrics)
        total_io_write = sum(m.io_write_bytes or 0 for m in metrics)
        
        return {
            'total_io_read_mb': total_io_read / 1024 / 1024,
            'total_io_write_mb': total_io_write / 1024 / 1024,
            'peak_thread_count': max(m.thread_count or 0 for m in metrics),
            'avg_thread_count': statistics.mean(m.thread_count or 0 for m in metrics)
        }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def _calculate_performance_degradation(self, performance_timeline: List[Dict]) -> Dict[str, Any]:
        """Calculate performance degradation during stress test."""
        if len(performance_timeline) < 2:
            return {}
        
        initial_duration = performance_timeline[0]['avg_duration']
        final_duration = performance_timeline[-1]['avg_duration']
        
        degradation_percent = ((final_duration - initial_duration) / initial_duration) * 100
        
        return {
            'initial_avg_duration': initial_duration,
            'final_avg_duration': final_duration,
            'degradation_percent': degradation_percent,
            'acceptable_degradation': degradation_percent < 50  # Threshold for acceptable degradation
        }
    
    def establish_baseline(self, operation_type: str, benchmark_id: str) -> Dict[str, float]:
        """
        Establish baseline metrics from a benchmark result.
        
        Args:
            operation_type: Type of operation
            benchmark_id: ID of benchmark to use as baseline
            
        Returns:
            Baseline metrics dictionary
        """
        if benchmark_id not in self.benchmark_results:
            raise ValueError(f"Benchmark {benchmark_id} not found")
        
        result = self.benchmark_results[benchmark_id]
        successful_metrics = [m for m in result.metrics if m.success]
        
        if not successful_metrics:
            raise ValueError("No successful operations in benchmark")
        
        durations = [m.duration_seconds for m in successful_metrics]
        
        baseline = {
            'mean_duration': statistics.mean(durations),
            'median_duration': statistics.median(durations),
            'p95_duration': self._calculate_percentile(durations, 95),
            'p99_duration': self._calculate_percentile(durations, 99),
            'success_rate': len(successful_metrics) / len(result.metrics),
            'established_at': datetime.now().isoformat(),
            'baseline_benchmark_id': benchmark_id
        }
        
        self.baseline_metrics[operation_type] = baseline
        self.logger.log_info(f"Established baseline for {operation_type}")
        
        return baseline
    
    def compare_to_baseline(self, operation_type: str, benchmark_id: str) -> Dict[str, Any]:
        """
        Compare benchmark results to established baseline.
        
        Args:
            operation_type: Type of operation
            benchmark_id: ID of benchmark to compare
            
        Returns:
            Comparison results
        """
        if operation_type not in self.baseline_metrics:
            raise ValueError(f"No baseline established for {operation_type}")
        
        if benchmark_id not in self.benchmark_results:
            raise ValueError(f"Benchmark {benchmark_id} not found")
        
        baseline = self.baseline_metrics[operation_type]
        result = self.benchmark_results[benchmark_id]
        
        successful_metrics = [m for m in result.metrics if m.success]
        durations = [m.duration_seconds for m in successful_metrics]
        
        if not durations:
            return {'error': 'No successful operations to compare'}
        
        current_stats = {
            'mean_duration': statistics.mean(durations),
            'median_duration': statistics.median(durations),
            'p95_duration': self._calculate_percentile(durations, 95),
            'p99_duration': self._calculate_percentile(durations, 99),
            'success_rate': len(successful_metrics) / len(result.metrics)
        }
        
        comparison = {}
        for metric, current_value in current_stats.items():
            baseline_value = baseline[metric]
            difference = current_value - baseline_value
            percentage_change = (difference / baseline_value) * 100 if baseline_value != 0 else 0
            
            comparison[metric] = {
                'current': current_value,
                'baseline': baseline_value,
                'difference': difference,
                'percentage_change': percentage_change,
                'performance_category': self._categorize_performance_change(percentage_change, metric)
            }
        
        return {
            'comparison_details': comparison,
            'overall_assessment': self._assess_overall_performance(comparison),
            'compared_at': datetime.now().isoformat()
        }
    
    def _categorize_performance_change(self, percentage_change: float, metric: str) -> str:
        """Categorize performance change."""
        # For duration metrics, lower is better
        if 'duration' in metric:
            if percentage_change <= -10:
                return "significant_improvement"
            elif percentage_change <= -5:
                return "improvement"
            elif percentage_change <= 5:
                return "stable"
            elif percentage_change <= 20:
                return "degradation"
            else:
                return "significant_degradation"
        
        # For success rate, higher is better
        elif metric == 'success_rate':
            if percentage_change >= 2:
                return "improvement"
            elif percentage_change >= -2:
                return "stable"
            else:
                return "degradation"
        
        return "unknown"
    
    def _assess_overall_performance(self, comparison: Dict[str, Any]) -> str:
        """Assess overall performance compared to baseline."""
        categories = [details['performance_category'] for details in comparison.values()]
        
        if any('significant_degradation' in cat for cat in categories):
            return "significant_regression"
        elif any('degradation' in cat for cat in categories):
            return "performance_regression"
        elif any('significant_improvement' in cat for cat in categories):
            return "significant_improvement"
        elif any('improvement' in cat for cat in categories):
            return "performance_improvement"
        else:
            return "stable_performance"
    
    def export_benchmark_results(self, output_path: Path) -> None:
        """Export all benchmark results to files."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export individual results
        for benchmark_id, result in self.benchmark_results.items():
            result_file = output_path / f"benchmark_{benchmark_id}_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Export summary
        summary = {
            'total_benchmarks': len(self.benchmark_results),
            'benchmark_ids': list(self.benchmark_results.keys()),
            'baseline_metrics': self.baseline_metrics,
            'exported_at': datetime.now().isoformat()
        }
        
        summary_file = output_path / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.log_info(f"Exported {len(self.benchmark_results)} benchmark results to {output_path}")


# Factory function
def create_benchmark_suite(enable_resource_monitoring: bool = True) -> BenchmarkSuite:
    """
    Create a benchmark suite with performance profiler.
    
    Args:
        enable_resource_monitoring: Whether to enable resource monitoring
        
    Returns:
        BenchmarkSuite instance
    """
    profiler = PerformanceProfiler(enable_resource_monitoring)
    return BenchmarkSuite(profiler)


# Export main classes
__all__ = [
    'PerformanceMetrics',
    'BenchmarkResult', 
    'PerformanceProfiler',
    'BenchmarkSuite',
    'create_benchmark_suite'
]
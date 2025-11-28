#!/usr/bin/env python3
"""
Test runner for Hegels Agents project.

This script provides a comprehensive test runner that can execute different
types of tests (unit, integration, performance) and generate reports.
It's designed to support the research and development workflow.
"""

import sys
import os
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
import threading


@dataclass
class TestResult:
    """Results from running a test suite or individual test."""
    name: str
    passed: bool
    duration: float
    output: str
    error: Optional[str] = None
    skipped: int = 0
    failed: int = 0
    total: int = 0
    coverage: Optional[float] = None


class TestRunner:
    """Comprehensive test runner for the Hegels Agents project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the test runner.
        
        Args:
            project_root: Path to project root directory. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (directory containing this script's parent)
            script_dir = Path(__file__).parent
            self.project_root = script_dir.parent
        else:
            self.project_root = Path(project_root)
        
        # Test directories and patterns
        self.test_dirs = [
            self.project_root / "tests",
            self.project_root / "src" / "tests",
        ]
        
        # Results storage
        self.results: List[TestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Configuration
        self.verbose = False
        self.parallel = False
        self.max_workers = 4
        self.coverage_enabled = False
        
        # Output settings
        self.output_formats = ["console"]  # can include "json", "xml", "html"
        self.output_dir = self.project_root / "test_reports"
    
    def find_test_files(self, pattern: str = "test_*.py") -> List[Path]:
        """
        Find test files matching the given pattern.
        
        Args:
            pattern: Glob pattern for test files
            
        Returns:
            List of test file paths
        """
        test_files = []
        
        for test_dir in self.test_dirs:
            if test_dir.exists():
                test_files.extend(test_dir.rglob(pattern))
        
        # Also look for pytest-style test files
        if pattern == "test_*.py":
            for test_dir in self.test_dirs:
                if test_dir.exists():
                    test_files.extend(test_dir.rglob("*_test.py"))
        
        return sorted(test_files)
    
    def check_test_environment(self) -> Dict[str, Any]:
        """
        Check if the test environment is properly set up.
        
        Returns:
            Dictionary with environment check results
        """
        results = {
            "python_executable": sys.executable,
            "project_root": str(self.project_root),
            "test_directories": [],
            "pytest_available": False,
            "coverage_available": False,
            "test_files_found": 0,
            "issues": []
        }
        
        # Check test directories
        for test_dir in self.test_dirs:
            dir_info = {
                "path": str(test_dir),
                "exists": test_dir.exists(),
                "test_files": len(list(test_dir.rglob("test_*.py"))) if test_dir.exists() else 0
            }
            results["test_directories"].append(dir_info)
            
            if dir_info["exists"]:
                results["test_files_found"] += dir_info["test_files"]
        
        # Check pytest availability
        try:
            import pytest
            results["pytest_available"] = True
            results["pytest_version"] = pytest.__version__
        except ImportError:
            results["issues"].append("pytest not available - install with: pip install pytest")
        
        # Check coverage availability
        try:
            import coverage
            results["coverage_available"] = True
            results["coverage_version"] = coverage.__version__
        except ImportError:
            results["coverage_available"] = False
        
        # Check if PYTHONPATH includes src directory
        src_dir = self.project_root / "src"
        if str(src_dir) not in sys.path:
            results["issues"].append(f"src directory ({src_dir}) not in Python path")
        
        return results
    
    def setup_test_environment(self):
        """Set up the test environment (PYTHONPATH, etc.)."""
        # Add src directory to Python path
        src_dir = self.project_root / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        # Set environment variables for tests
        os.environ["PYTHONPATH"] = str(src_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.environ["HEGELS_AGENTS_ROOT"] = str(self.project_root)
        os.environ["HEGELS_AGENTS_ENV"] = "test"
    
    def run_pytest(self, 
                  test_path: Optional[str] = None,
                  markers: Optional[List[str]] = None,
                  args: Optional[List[str]] = None) -> TestResult:
        """
        Run tests using pytest.
        
        Args:
            test_path: Specific test file or directory to run
            markers: Pytest markers to filter tests (e.g., ["unit", "not slow"])
            args: Additional pytest arguments
            
        Returns:
            TestResult object with results
        """
        start_time = time.time()
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add test path or default to test directories
        if test_path:
            cmd.append(test_path)
        else:
            # Add existing test directories
            for test_dir in self.test_dirs:
                if test_dir.exists():
                    cmd.append(str(test_dir))
        
        # Add markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add verbose output if requested
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add coverage if enabled
        if self.coverage_enabled:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
        
        # Add JSON output for parsing
        json_report = self.output_dir / "pytest_report.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])
        
        # Add additional arguments
        if args:
            cmd.extend(args)
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(exist_ok=True)
            
            # Run pytest
            if self.verbose:
                print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            
            # Parse results from JSON report if available
            test_stats = self._parse_pytest_json_report(json_report)
            
            # Create result
            test_result = TestResult(
                name="pytest",
                passed=result.returncode == 0,
                duration=duration,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                **test_stats
            )
            
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="pytest",
                passed=False,
                duration=duration,
                output="",
                error=f"Failed to run pytest: {str(e)}"
            )
    
    def _parse_pytest_json_report(self, json_file: Path) -> Dict[str, Any]:
        """Parse pytest JSON report for detailed statistics."""
        stats = {"total": 0, "failed": 0, "skipped": 0}
        
        try:
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    stats = {
                        "total": summary.get("total", 0),
                        "failed": summary.get("failed", 0),
                        "skipped": summary.get("skipped", 0)
                    }
        except Exception:
            pass  # Fallback to default stats
        
        return stats
    
    def run_individual_test(self, test_file: Path) -> TestResult:
        """
        Run an individual test file.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            TestResult object
        """
        start_time = time.time()
        
        try:
            cmd = [sys.executable, str(test_file)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            
            return TestResult(
                name=test_file.name,
                passed=result.returncode == 0,
                duration=duration,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                total=1,
                failed=1 if result.returncode != 0 else 0
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_file.name,
                passed=False,
                duration=duration,
                output="",
                error=f"Failed to run test: {str(e)}",
                total=1,
                failed=1
            )
    
    def run_parallel_tests(self, test_files: List[Path]) -> List[TestResult]:
        """
        Run test files in parallel.
        
        Args:
            test_files: List of test files to run
            
        Returns:
            List of TestResult objects
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.run_individual_test, test_file): test_file
                for test_file in test_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                test_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if self.verbose:
                        status = "✅" if result.passed else "❌"
                        print(f"{status} {test_file.name} ({result.duration:.2f}s)")
                        
                except Exception as e:
                    results.append(TestResult(
                        name=test_file.name,
                        passed=False,
                        duration=0.0,
                        output="",
                        error=f"Exception during test execution: {str(e)}"
                    ))
        
        return results
    
    def run_all_tests(self, 
                     test_type: str = "all",
                     pattern: Optional[str] = None,
                     markers: Optional[List[str]] = None) -> List[TestResult]:
        """
        Run all tests of a specified type.
        
        Args:
            test_type: Type of tests to run ("all", "unit", "integration", "performance")
            pattern: Custom file pattern for test discovery
            markers: Pytest markers to filter tests
            
        Returns:
            List of TestResult objects
        """
        self.start_time = datetime.now()
        
        # Set up test environment
        self.setup_test_environment()
        
        results = []
        
        if test_type in ["all", "pytest"]:
            # Run using pytest (preferred)
            pytest_markers = markers or []
            
            # Add type-specific markers
            if test_type == "unit":
                pytest_markers.append("unit")
            elif test_type == "integration":
                pytest_markers.append("integration")
            elif test_type == "performance":
                pytest_markers.append("performance")
            
            pytest_result = self.run_pytest(markers=pytest_markers if pytest_markers else None)
            results.append(pytest_result)
        
        # If pytest failed or we're running individual files
        if test_type == "individual" or (results and not results[0].passed):
            test_files = self.find_test_files(pattern or "test_*.py")
            
            if test_files:
                if self.parallel and len(test_files) > 1:
                    individual_results = self.run_parallel_tests(test_files)
                else:
                    individual_results = [
                        self.run_individual_test(test_file)
                        for test_file in test_files
                    ]
                
                results.extend(individual_results)
        
        self.end_time = datetime.now()
        self.results.extend(results)
        
        return results
    
    def generate_report(self, results: List[TestResult], format_type: str = "console"):
        """
        Generate test report in specified format.
        
        Args:
            results: List of TestResult objects
            format_type: Report format ("console", "json", "html")
        """
        if format_type == "console":
            self._generate_console_report(results)
        elif format_type == "json":
            self._generate_json_report(results)
        else:
            print(f"⚠️  Unsupported report format: {format_type}")
    
    def _generate_console_report(self, results: List[TestResult]):
        """Generate console report."""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_duration = sum(r.duration for r in results)
        total_tests = sum(r.total for r in results)
        total_failed = sum(r.failed for r in results)
        total_skipped = sum(r.skipped for r in results)
        total_passed = total_tests - total_failed - total_skipped
        
        # Overall summary
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed} (✅)")
        print(f"Failed: {total_failed} ({'❌' if total_failed > 0 else '✅'})")
        print(f"Skipped: {total_skipped}")
        
        # Success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Individual test results
        print(f"\nDETAILED RESULTS:")
        print("-" * 60)
        
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} | {result.name:30} | {result.duration:6.2f}s")
            
            if not result.passed and result.error:
                print(f"   Error: {result.error}")
        
        # Coverage information
        coverage_results = [r for r in results if r.coverage is not None]
        if coverage_results:
            avg_coverage = sum(r.coverage for r in coverage_results) / len(coverage_results)
            print(f"\nCode Coverage: {avg_coverage:.1f}%")
        
        print("\n" + "=" * 60)
    
    def _generate_json_report(self, results: List[TestResult]):
        """Generate JSON report."""
        self.output_dir.mkdir(exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
            "summary": {
                "total_tests": sum(r.total for r in results),
                "passed": sum(r.total - r.failed - r.skipped for r in results),
                "failed": sum(r.failed for r in results),
                "skipped": sum(r.skipped for r in results),
                "success_rate": 0  # Will be calculated below
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "total": r.total,
                    "failed": r.failed,
                    "skipped": r.skipped,
                    "error": r.error,
                    "coverage": r.coverage
                }
                for r in results
            ]
        }
        
        # Calculate success rate
        total = report_data["summary"]["total_tests"]
        passed = report_data["summary"]["passed"]
        report_data["summary"]["success_rate"] = (passed / total * 100) if total > 0 else 0
        
        # Write report
        report_file = self.output_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📊 JSON report saved to: {report_file}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for Hegels Agents project"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "performance", "pytest", "individual"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        help="File pattern for test discovery (e.g., 'test_agent*.py')"
    )
    
    parser.add_argument(
        "--markers", "-m",
        action="append",
        help="Pytest markers to filter tests (can be used multiple times)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable code coverage reporting"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "both"],
        default="console",
        help="Output format for test results"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check test environment and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize test runner
        runner = TestRunner()
        runner.verbose = args.verbose
        runner.parallel = args.parallel
        runner.coverage_enabled = args.coverage
        
        # Check environment if requested
        if args.check_env:
            env_check = runner.check_test_environment()
            print("🔍 Test Environment Check")
            print("=" * 40)
            print(f"Python: {env_check['python_executable']}")
            print(f"Project Root: {env_check['project_root']}")
            print(f"Pytest Available: {'✅' if env_check['pytest_available'] else '❌'}")
            print(f"Coverage Available: {'✅' if env_check['coverage_available'] else '❌'}")
            print(f"Test Files Found: {env_check['test_files_found']}")
            
            if env_check['issues']:
                print("\n⚠️  Issues:")
                for issue in env_check['issues']:
                    print(f"  - {issue}")
                sys.exit(1)
            else:
                print("\n✅ Environment looks good!")
                sys.exit(0)
        
        # Run tests
        print("🧪 Starting test execution...")
        results = runner.run_all_tests(
            test_type=args.type,
            pattern=args.pattern,
            markers=args.markers
        )
        
        # Generate reports
        if args.output_format in ["console", "both"]:
            runner.generate_report(results, "console")
        
        if args.output_format in ["json", "both"]:
            runner.generate_report(results, "json")
        
        # Exit with appropriate code
        all_passed = all(r.passed for r in results)
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during test execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Enhanced Evaluation Pipeline Validation Runner

This script runs comprehensive validation tests for the evaluation pipeline
and generates detailed reports on the research-grade capabilities.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set working directory to project root
os.chdir(project_root)

def check_dependencies():
    """Check required dependencies are available."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    
    if missing:
        print(f"Warning: Missing optional dependencies: {', '.join(missing)}")
        print("Some advanced statistical features may not be available.")
        print("Install with: pip install " + " ".join(missing))
        print()
    
    return len(missing) == 0

def run_validation():
    """Run comprehensive validation suite."""
    print("Enhanced Evaluation Pipeline Validation")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check dependencies
    has_all_deps = check_dependencies()
    
    try:
        # Import and run validation
        from eval.validation_test_suite import run_comprehensive_validation
        
        print("Running comprehensive validation tests...")
        print("-" * 40)
        
        success = run_comprehensive_validation()
        
        print()
        print("=" * 80)
        if success:
            print("✓ VALIDATION SUCCESSFUL")
            print("Enhanced Evaluation Pipeline meets research-grade standards")
        else:
            print("✗ VALIDATION ISSUES DETECTED")
            print("See detailed report for specific failures")
        
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return success
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nTrying to fix import issues...")
        
        # Try to import individual components to diagnose
        try:
            from eval.comprehensive_evaluator import create_comprehensive_evaluator
            print("✓ Comprehensive evaluator import successful")
        except ImportError as e:
            print(f"✗ Comprehensive evaluator import failed: {e}")
        
        try:
            from eval.statistical_analyzer import create_statistical_analyzer
            print("✓ Statistical analyzer import successful")  
        except ImportError as e:
            print(f"✗ Statistical analyzer import failed: {e}")
        
        try:
            from eval.automated_workflows import WorkflowOrchestrator
            print("✓ Workflow orchestrator import successful")
        except ImportError as e:
            print(f"✗ Workflow orchestrator import failed: {e}")
        
        try:
            from eval.performance_benchmarks import create_benchmark_suite
            print("✓ Performance benchmarks import successful")
        except ImportError as e:
            print(f"✗ Performance benchmarks import failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"Validation Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def run_quick_validation():
    """Run a quick validation of core components."""
    print("Quick Validation of Core Components")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Statistical Analyzer
    try:
        from eval.statistical_analyzer import create_statistical_analyzer
        analyzer = create_statistical_analyzer()
        
        # Test basic functionality
        test_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        summary = analyzer.calculate_summary_statistics(test_data)
        
        assert abs(summary.mean - 0.55) < 0.001
        assert summary.count == 10
        
        results['statistical_analyzer'] = {
            'status': 'PASS',
            'details': f'Mean: {summary.mean:.3f}, Count: {summary.count}'
        }
        print("✓ Statistical Analyzer: PASS")
        
    except Exception as e:
        results['statistical_analyzer'] = {
            'status': 'FAIL', 
            'error': str(e)
        }
        print(f"✗ Statistical Analyzer: FAIL - {e}")
    
    # Test 2: Comprehensive Evaluator
    try:
        from eval.comprehensive_evaluator import create_comprehensive_evaluator
        pipeline = create_comprehensive_evaluator()
        
        # Test batch creation
        batch = pipeline.create_evaluation_batch(
            name="Quick Test",
            questions=["Test question"],
            configs=[{'type': 'quality', 'id': 'test'}]
        )
        
        assert batch.name == "Quick Test"
        assert len(batch.questions) == 1
        
        results['comprehensive_evaluator'] = {
            'status': 'PASS',
            'details': f'Batch ID: {batch.batch_id}'
        }
        print("✓ Comprehensive Evaluator: PASS")
        
    except Exception as e:
        results['comprehensive_evaluator'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Comprehensive Evaluator: FAIL - {e}")
    
    # Test 3: Performance Benchmarks
    try:
        from eval.performance_benchmarks import create_benchmark_suite
        benchmark_suite = create_benchmark_suite(enable_resource_monitoring=False)
        
        # Test with simple operation
        def test_op():
            return {"result": "test"}
        
        result = benchmark_suite.run_latency_benchmark(
            operation=test_op,
            operation_type="quick_test", 
            iterations=3,
            warmup_iterations=1
        )
        
        assert result.total_operations == 3
        assert result.successful_operations > 0
        
        results['performance_benchmarks'] = {
            'status': 'PASS',
            'details': f'Operations: {result.total_operations}, Success: {result.successful_operations}'
        }
        print("✓ Performance Benchmarks: PASS")
        
    except Exception as e:
        results['performance_benchmarks'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Performance Benchmarks: FAIL - {e}")
    
    # Test 4: Workflow Orchestrator
    try:
        from eval.automated_workflows import WorkflowOrchestrator, create_basic_evaluation_workflow
        orchestrator = WorkflowOrchestrator()
        
        # Test workflow creation
        workflow = create_basic_evaluation_workflow(
            name="Quick Test Workflow",
            questions=["Test"],
            schedule="manual"
        )
        
        workflow_id = orchestrator.register_workflow(workflow)
        status = orchestrator.get_workflow_status(workflow_id)
        
        assert isinstance(workflow_id, str)
        assert 'workflow' in status
        
        results['workflow_orchestrator'] = {
            'status': 'PASS',
            'details': f'Workflow ID: {workflow_id[:8]}...'
        }
        print("✓ Workflow Orchestrator: PASS")
        
    except Exception as e:
        results['workflow_orchestrator'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"✗ Workflow Orchestrator: FAIL - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    total = len(results)
    
    print(f"Quick Validation Results: {passed}/{total} components passed")
    
    if passed == total:
        print("✓ All core components functional")
        return True
    else:
        print("✗ Some components have issues")
        return False

def main():
    """Main validation runner."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        return run_quick_validation()
    else:
        return run_validation()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
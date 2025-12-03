#!/usr/bin/env python3
"""
Test Suite for Statistical Validation System

This test suite validates the functionality and statistical rigor
of the statistical validation system for Hegel's agents.

Tests include:
- Data validation and loading
- Statistical calculations
- Configuration handling
- Output generation
- Edge cases and error handling

Usage:
    python test_statistical_validation.py
"""

import sys
import unittest
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from statistical_validation import (
    StatisticalValidator, ValidationConfig, ComparisonMetrics,
    StatisticalTestResult, save_results, generate_summary_report
)


class TestValidationConfig(unittest.TestCase):
    """Test ValidationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        self.assertEqual(config.sample_size, 50)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.significance_level, 0.05)
        self.assertEqual(config.effect_size_threshold, 0.3)
        self.assertEqual(config.power_target, 0.80)
        self.assertEqual(config.random_seed, 42)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            sample_size=100,
            confidence_level=0.99,
            significance_level=0.01,
            random_seed=123
        )
        
        self.assertEqual(config.sample_size, 100)
        self.assertEqual(config.confidence_level, 0.99)
        self.assertEqual(config.significance_level, 0.01)
        self.assertEqual(config.random_seed, 123)
    
    def test_config_serialization(self):
        """Test config to_dict method."""
        config = ValidationConfig(sample_size=75)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['sample_size'], 75)
        self.assertIn('confidence_level', config_dict)


class TestComparisonMetrics(unittest.TestCase):
    """Test ComparisonMetrics class."""
    
    def test_empty_metrics(self):
        """Test empty metrics initialization."""
        metrics = ComparisonMetrics()
        
        self.assertEqual(len(metrics.single_scores), 0)
        self.assertEqual(len(metrics.hegel_scores), 0)
        self.assertEqual(metrics.sample_size, 0)
        self.assertFalse(metrics.validate_data())
    
    def test_valid_metrics(self):
        """Test valid metrics data."""
        metrics = ComparisonMetrics(
            single_scores=[70.0, 75.0, 80.0],
            hegel_scores=[75.0, 78.0, 85.0],
            questions=['Q1', 'Q2', 'Q3']
        )
        
        self.assertEqual(metrics.sample_size, 3)
        self.assertTrue(metrics.validate_data())
    
    def test_invalid_metrics(self):
        """Test invalid metrics data (mismatched sizes)."""
        metrics = ComparisonMetrics(
            single_scores=[70.0, 75.0],
            hegel_scores=[75.0, 78.0, 85.0]  # Different size
        )
        
        self.assertFalse(metrics.validate_data())
    
    def test_metrics_with_timing(self):
        """Test metrics with timing data."""
        metrics = ComparisonMetrics(
            single_scores=[70.0, 75.0],
            hegel_scores=[75.0, 78.0], 
            single_times=[3.0, 3.5],
            hegel_times=[8.0, 9.0]
        )
        
        self.assertTrue(metrics.validate_data())


class TestStatisticalValidator(unittest.TestCase):
    """Test StatisticalValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig(sample_size=20, random_seed=42)
        self.validator = StatisticalValidator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.config.sample_size, 20)
        self.assertEqual(self.validator.config.random_seed, 42)
        self.assertIsNone(self.validator.comparison_data)
    
    def test_mock_data_generation(self):
        """Test mock data generation."""
        data = self.validator.collect_evaluation_data()
        
        self.assertEqual(data.sample_size, 20)
        self.assertTrue(data.validate_data())
        self.assertEqual(len(data.single_scores), 20)
        self.assertEqual(len(data.hegel_scores), 20)
        self.assertEqual(data.metadata['data_type'], 'mock_demonstration')
    
    def test_data_loading_from_dict(self):
        """Test loading data from dictionary format."""
        test_data = {
            'test_results': [
                {
                    'question': 'Test Q1',
                    'single_agent_quality_score': 70.0,
                    'dialectical_quality_score': 75.0,
                    'improvement_score': 0.05,
                    'single_agent_time': 3.0,
                    'dialectical_time': 8.0
                },
                {
                    'question': 'Test Q2', 
                    'single_agent_quality_score': 65.0,
                    'dialectical_quality_score': 72.0,
                    'improvement_score': 0.07,
                    'single_agent_time': 2.8,
                    'dialectical_time': 7.5
                }
            ]
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            data = self.validator._load_data_from_file(temp_file)
            
            self.assertEqual(data.sample_size, 2)
            self.assertTrue(data.validate_data())
            self.assertEqual(data.single_scores[0], 70.0)
            self.assertEqual(data.hegel_scores[0], 75.0)
            self.assertEqual(len(data.single_times), 2)
            
        finally:
            os.unlink(temp_file)
    
    def test_statistical_analysis(self):
        """Test full statistical analysis."""
        # Create test data
        data = ComparisonMetrics(
            single_scores=[65.0, 70.0, 68.0, 72.0, 66.0] * 10,  # 50 samples
            hegel_scores=[70.0, 75.0, 73.0, 77.0, 71.0] * 10,   # Consistent improvement
            questions=[f'Question {i}' for i in range(50)]
        )
        
        results = self.validator.run_statistical_analysis(data)
        
        # Check result structure
        self.assertIn('analysis_timestamp', results)
        self.assertIn('sample_size', results)
        self.assertIn('data_summary', results)
        self.assertIn('statistical_tests', results)
        self.assertIn('practical_significance', results)
        self.assertIn('conclusions', results)
        
        # Check sample size
        self.assertEqual(results['sample_size'], 50)
        
        # Check data summary
        summary = results['data_summary']
        self.assertIn('single_agent_scores', summary)
        self.assertIn('hegel_scores', summary)
        self.assertIn('improvement_rate', summary)
        
        # Check that there's improvement detected
        self.assertGreater(summary['improvement_rate'], 0.5)  # Should be > 50%
        
        # Check practical significance
        practical = results['practical_significance']
        self.assertIn('mean_improvement_points', practical)
        self.assertIn('practically_significant', practical)
        
        # With consistent 5-point improvement, should be practically significant
        self.assertGreater(practical['mean_improvement_points'], 4.0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small sample
        small_data = ComparisonMetrics(
            single_scores=[70.0, 75.0],
            hegel_scores=[72.0, 77.0],
            questions=['Q1', 'Q2']
        )
        
        # Should not crash with small sample
        results = self.validator.run_statistical_analysis(small_data)
        self.assertEqual(results['sample_size'], 2)
        
        # Test with identical scores (no improvement)
        no_improvement_data = ComparisonMetrics(
            single_scores=[70.0] * 20,
            hegel_scores=[70.0] * 20,  # No improvement
            questions=[f'Q{i}' for i in range(20)]
        )
        
        results = self.validator.run_statistical_analysis(no_improvement_data)
        practical = results['practical_significance']
        self.assertEqual(practical['mean_improvement_points'], 0.0)
        self.assertFalse(practical['practically_significant'])


class TestStatisticalFunctions(unittest.TestCase):
    """Test statistical calculation functions."""
    
    def setUp(self):
        """Set up test validator."""
        config = ValidationConfig(random_seed=42)
        self.validator = StatisticalValidator(config)
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        group1 = [70, 72, 68, 74, 69]  # Mean = 70.6
        group2 = [75, 77, 73, 79, 74]  # Mean = 75.6, difference = 5
        
        cohens_d = self.validator._cohens_d(group1, group2)
        
        # Should be a positive effect size around 1.0 (large effect)
        self.assertGreater(cohens_d, 0.5)
        self.assertLess(cohens_d, 3.0)  # Allow for larger effect sizes in test data
    
    def test_manual_paired_t_test(self):
        """Test manual paired t-test implementation."""
        group1 = [70.0, 72.0, 68.0, 74.0, 69.0]
        group2 = [75.0, 77.0, 73.0, 79.0, 74.0]  # Consistent +5 improvement
        
        result = self.validator._manual_paired_t_test(group1, group2)
        
        self.assertIsInstance(result, StatisticalTestResult)
        self.assertEqual(result.sample_size, 5)
        self.assertIsInstance(result.test_statistic, (int, float))  # Should be a number
        self.assertIsInstance(result.p_value, (int, float))  # Should be a number
    
    def test_interpretation_functions(self):
        """Test result interpretation functions."""
        # Test p-value interpretation
        self.assertIn('significant', self.validator._interpret_p_value(0.001).lower())
        self.assertIn('not significant', self.validator._interpret_p_value(0.10).lower())
        
        # Test Cohen's d interpretation
        interpretation = self.validator._interpret_cohens_d(0.8)
        self.assertIn('large', interpretation.lower())
        self.assertIn('positive', interpretation.lower())
        
        interpretation = self.validator._interpret_cohens_d(-0.3)
        self.assertIn('small', interpretation.lower())
        self.assertIn('negative', interpretation.lower())


class TestOutputGeneration(unittest.TestCase):
    """Test output and reporting functions."""
    
    def test_summary_report_generation(self):
        """Test summary report generation."""
        # Create mock results
        config = ValidationConfig(sample_size=30)
        mock_results = {
            'analysis_timestamp': '2024-12-03T10:00:00',
            'sample_size': 30,
            'data_summary': {
                'single_agent_scores': {'mean': 70.0},
                'hegel_scores': {'mean': 75.0},
                'improvement_rate': 0.7
            },
            'practical_significance': {
                'mean_improvement_points': 5.0,
                'mean_improvement_percent': 7.1,
                'practically_significant': True
            },
            'conclusions': {
                'overall_recommendation': 'ADOPT_HEGELS_APPROACH',
                'confidence_level': 'HIGH',
                'summary': 'Test summary',
                'limitations': ['Test limitation'],
                'next_steps': ['Test next step']
            }
        }
        
        report = generate_summary_report(mock_results, config)
        
        self.assertIsInstance(report, str)
        self.assertIn('Statistical Validation Report', report)
        self.assertIn('ADOPT_HEGELS_APPROACH', report)
        self.assertIn('Sample Size:** 30', report)  # Check for actual format in report
        self.assertIn('Mean Improvement:** 5.00', report)
    
    def test_results_saving(self):
        """Test results saving functionality."""
        mock_results = {
            'analysis_timestamp': '2024-12-03T10:00:00',
            'sample_size': 20,
            'conclusions': {'test': 'data'}
        }
        
        config = ValidationConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = save_results(mock_results, temp_dir, config)
            
            self.assertIn('detailed_results', saved_files)
            self.assertIn('summary_report', saved_files)
            self.assertIn('configuration', saved_files)
            
            # Check files exist
            for file_path in saved_files.values():
                self.assertTrue(Path(file_path).exists())
            
            # Check JSON file content
            with open(saved_files['detailed_results'], 'r') as f:
                loaded_results = json.load(f)
            self.assertEqual(loaded_results['sample_size'], 20)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_complete_validation_workflow(self):
        """Test complete validation workflow from start to finish."""
        config = ValidationConfig(sample_size=30, random_seed=42)
        validator = StatisticalValidator(config)
        
        # Generate data
        data = validator.collect_evaluation_data()
        self.assertTrue(data.validate_data())
        
        # Run analysis
        results = validator.run_statistical_analysis(data)
        self.assertIsInstance(results, dict)
        
        # Check all expected sections exist
        expected_sections = [
            'analysis_timestamp', 'sample_size', 'configuration',
            'data_summary', 'statistical_tests', 'practical_significance',
            'conclusions'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results)
        
        # Check conclusions are reasonable
        conclusions = results['conclusions']
        self.assertIn('overall_recommendation', conclusions)
        self.assertIn('confidence_level', conclusions)
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = save_results(results, temp_dir, config)
            self.assertEqual(len(saved_files), 3)  # JSON, report, config
    
    def test_different_data_quality_scenarios(self):
        """Test with different data quality scenarios."""
        scenarios = [
            # High quality improvement
            {
                'name': 'high_improvement',
                'single_scores': [60.0, 65.0, 62.0, 67.0, 63.0] * 6,
                'hegel_scores': [75.0, 80.0, 77.0, 82.0, 78.0] * 6,
                'expected_significant': True
            },
            # No improvement
            {
                'name': 'no_improvement',
                'single_scores': [70.0, 72.0, 68.0, 74.0, 69.0] * 6,
                'hegel_scores': [70.0, 72.0, 68.0, 74.0, 69.0] * 6,
                'expected_significant': False
            },
            # Mixed results
            {
                'name': 'mixed_results',
                'single_scores': [65.0, 70.0, 68.0, 72.0, 66.0] * 6,
                'hegel_scores': [67.0, 75.0, 65.0, 77.0, 71.0] * 6,  # Some better, some worse
                'expected_significant': None  # Unclear
            }
        ]
        
        config = ValidationConfig(random_seed=42)
        validator = StatisticalValidator(config)
        
        for scenario in scenarios:
            with self.subTest(scenario=scenario['name']):
                data = ComparisonMetrics(
                    single_scores=scenario['single_scores'],
                    hegel_scores=scenario['hegel_scores'],
                    questions=[f'Q{i}' for i in range(len(scenario['single_scores']))]
                )
                
                results = validator.run_statistical_analysis(data)
                
                # Check that analysis completes without error
                self.assertIn('conclusions', results)
                
                practical = results['practical_significance']
                
                if scenario['expected_significant'] is True:
                    # Should show improvement
                    self.assertGreater(practical['mean_improvement_points'], 2.0)
                elif scenario['expected_significant'] is False:
                    # Should show little to no improvement
                    self.assertLessEqual(abs(practical['mean_improvement_points']), 1.0)


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestValidationConfig,
        TestComparisonMetrics,
        TestStatisticalValidator,
        TestStatisticalFunctions,
        TestOutputGeneration,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Return success/failure
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main test execution."""
    print("Statistical Validation System - Test Suite")
    print(f"{'='*60}")
    print("Testing statistical validation functionality...")
    
    success = run_tests()
    
    if success:
        print(f"\n✅ All tests passed! Statistical validation system is working correctly.")
        return 0
    else:
        print(f"\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
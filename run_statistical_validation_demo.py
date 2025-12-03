#!/usr/bin/env python3
"""
Statistical Validation Demo Script

This script demonstrates how to use the statistical validation system to
compare Hegel's dialectical multi-agent approach vs single-agent review.

This demo shows:
1. Basic usage with mock data
2. Advanced configuration options  
3. Different analysis scenarios
4. How to interpret results

Usage:
    python run_statistical_validation_demo.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the validation system
from statistical_validation import (
    StatisticalValidator, ValidationConfig, ComparisonMetrics,
    save_results, generate_summary_report
)


def demo_basic_validation():
    """Demonstrate basic statistical validation with default settings."""
    print("="*70)
    print("DEMO 1: Basic Statistical Validation")
    print("="*70)
    
    # Create validator with default configuration
    config = ValidationConfig(
        sample_size=50,
        confidence_level=0.95,
        significance_level=0.05,
        random_seed=42  # For reproducible results
    )
    
    print(f"Configuration:")
    print(f"  Sample size: {config.sample_size}")
    print(f"  Confidence level: {config.confidence_level}")
    print(f"  Significance level: {config.significance_level}")
    
    validator = StatisticalValidator(config)
    
    # Collect mock data (simulates real evaluation results)
    print(f"\nGenerating mock comparative evaluation data...")
    comparison_data = validator.collect_evaluation_data()
    
    print(f"Generated {comparison_data.sample_size} comparison pairs")
    print(f"Sample data preview:")
    print(f"  Single agent scores: {comparison_data.single_scores[:5]} ...")
    print(f"  Hegel scores: {comparison_data.hegel_scores[:5]} ...")
    
    # Run statistical analysis
    print(f"\nRunning comprehensive statistical analysis...")
    results = validator.run_statistical_analysis(comparison_data)
    
    # Show key results
    conclusions = results['conclusions']
    practical = results['practical_significance']
    data_summary = results['data_summary']
    
    print(f"\nResults Summary:")
    print(f"  Overall recommendation: {conclusions['overall_recommendation']}")
    print(f"  Confidence level: {conclusions['confidence_level']}")
    print(f"  Mean improvement: {practical['mean_improvement_points']:.2f} points")
    print(f"  Improvement percentage: {practical['mean_improvement_percent']:.1f}%")
    print(f"  Success rate: {data_summary['improvement_rate']*100:.1f}% of tests improved")
    print(f"  Practically significant: {practical['practically_significant']}")
    
    # Show statistical tests
    stat_tests = results.get('statistical_tests', {})
    quality_tests = stat_tests.get('quality_analysis', {})
    
    print(f"\nStatistical Tests:")
    for test_name, test_result in quality_tests.items():
        if isinstance(test_result, dict):
            p_val = test_result.get('p_value', 'N/A')
            interpretation = test_result.get('interpretation', 'No interpretation')
            print(f"  {test_result.get('test_name', test_name)}: p={p_val:.4f if isinstance(p_val, float) else p_val}")
            print(f"    → {interpretation}")
    
    return results


def demo_power_analysis():
    """Demonstrate power analysis and sample size recommendations."""
    print("\n" + "="*70)
    print("DEMO 2: Power Analysis and Sample Size Planning")
    print("="*70)
    
    # Test different sample sizes to show power analysis
    sample_sizes = [10, 25, 50, 100, 200]
    
    print("Sample Size Analysis:")
    print("N\tPower\tRecommendation")
    print("-" * 40)
    
    for n in sample_sizes:
        config = ValidationConfig(sample_size=n, random_seed=42)
        validator = StatisticalValidator(config)
        
        # Generate data for this sample size
        data = validator.collect_evaluation_data()
        results = validator.run_statistical_analysis(data)
        
        power_analysis = results.get('power_analysis', {})
        current_power = power_analysis.get('current_power', 0)
        
        recommendation = "✓ Good" if current_power >= 0.8 else "⚠ Low power" if current_power >= 0.5 else "✗ Inadequate"
        
        print(f"{n}\t{current_power*100:.1f}%\t{recommendation}")
    
    # Show detailed power analysis for optimal sample size
    print(f"\nDetailed Power Analysis (n=100):")
    config = ValidationConfig(sample_size=100, random_seed=42)
    validator = StatisticalValidator(config)
    data = validator.collect_evaluation_data()
    results = validator.run_statistical_analysis(data)
    
    power_info = results['power_analysis']
    print(f"  Observed effect size: {power_info['observed_effect_size']:.3f}")
    print(f"  Current power: {power_info['current_power']*100:.1f}%")
    print(f"  Required n for 80% power: {power_info['required_sample_size_for_target_power']}")
    print(f"  Minimum detectable effect: {power_info['minimum_detectable_effect']:.3f}")


def demo_different_scenarios():
    """Demonstrate validation under different scenarios."""
    print("\n" + "="*70)
    print("DEMO 3: Different Performance Scenarios")
    print("="*70)
    
    scenarios = [
        {
            'name': 'Strong Improvement',
            'single_base': 60.0,
            'improvement': 15.0,  # Large improvement
            'description': 'Hegel\'s approach shows large improvements'
        },
        {
            'name': 'Marginal Improvement', 
            'single_base': 75.0,
            'improvement': 3.0,  # Small improvement
            'description': 'Hegel\'s approach shows small improvements'
        },
        {
            'name': 'No Improvement',
            'single_base': 70.0, 
            'improvement': 0.0,  # No improvement
            'description': 'Hegel\'s approach shows no improvement'
        },
        {
            'name': 'Mixed Results',
            'single_base': 65.0,
            'improvement': 5.0,  # Moderate improvement
            'description': 'Hegel\'s approach shows moderate improvements'
        }
    ]
    
    print("Scenario\t\tRecommendation\t\tPractical Sig\tStatistical Sig")
    print("-" * 80)
    
    for scenario in scenarios:
        # Create custom mock data for this scenario
        config = ValidationConfig(sample_size=50, random_seed=42)
        validator = StatisticalValidator(config)
        
        # Generate scenario-specific data
        import random
        random.seed(42)
        
        data = ComparisonMetrics()
        n = config.sample_size
        
        for i in range(n):
            single_score = random.gauss(scenario['single_base'], 10.0)
            single_score = max(10.0, min(100.0, single_score))
            
            # Add improvement with some variance
            improvement = random.gauss(scenario['improvement'], 4.0)
            hegel_score = single_score + improvement
            hegel_score = max(10.0, min(100.0, hegel_score))
            
            data.single_scores.append(single_score)
            data.hegel_scores.append(hegel_score)
            data.improvement_scores.append(improvement / 100.0)
            data.questions.append(f"Question {i+1}")
        
        # Analyze this scenario
        results = validator.run_statistical_analysis(data)
        
        # Extract key results
        conclusions = results['conclusions']
        practical = results['practical_significance']
        stat_summary = conclusions['statistical_significance_summary']
        
        recommendation = conclusions['overall_recommendation'].replace('_', ' ')[:15]
        practical_sig = "✓" if practical['practically_significant'] else "✗"
        stat_sig = "✓" if stat_summary['overall_significance'] else "✗"
        
        print(f"{scenario['name']:<15}\t{recommendation:<15}\t{practical_sig}\t\t{stat_sig}")
    
    print(f"\nScenario Interpretations:")
    for scenario in scenarios:
        print(f"  {scenario['name']}: {scenario['description']}")


def demo_real_world_usage():
    """Demonstrate how to use the system with real evaluation data."""
    print("\n" + "="*70)
    print("DEMO 4: Real-World Usage Examples")
    print("="*70)
    
    print("Example 1: Loading data from evaluation results")
    print("-" * 50)
    
    # Show how to create data in the expected format
    example_data = {
        "test_results": [
            {
                "question": "What are the main causes of climate change?",
                "single_agent_quality_score": 72.5,
                "dialectical_quality_score": 78.2,
                "improvement_score": 0.057,
                "single_agent_time": 3.2,
                "dialectical_time": 8.7
            },
            {
                "question": "How do neural networks learn?",
                "single_agent_quality_score": 68.3,
                "dialectical_quality_score": 75.1,
                "improvement_score": 0.068,
                "single_agent_time": 2.9,
                "dialectical_time": 7.4
            },
            # ... more results would go here
        ]
    }
    
    print("Expected data format (JSON):")
    print(json.dumps(example_data, indent=2))
    
    print(f"\nExample 2: Command-line usage")
    print("-" * 50)
    
    examples = [
        "# Basic analysis with default settings",
        "python statistical_validation.py",
        "",
        "# Load data from file", 
        "python statistical_validation.py --data-file evaluation_results.json",
        "",
        "# Run new evaluation with specific sample size",
        "python statistical_validation.py --run-new-evaluation --sample-size 75",
        "",
        "# Custom statistical parameters",
        "python statistical_validation.py --confidence-level 0.99 --significance-level 0.01",
        "",
        "# Full analysis with output",
        "python statistical_validation.py --sample-size 100 --output-dir validation_results/ --verbose"
    ]
    
    for example in examples:
        print(example)
    
    print(f"\nExample 3: Programmatic usage")
    print("-" * 50)
    
    programmatic_example = '''
# Import the validation system
from statistical_validation import StatisticalValidator, ValidationConfig

# Create configuration
config = ValidationConfig(
    sample_size=100,
    confidence_level=0.95,
    significance_level=0.05,
    power_target=0.80
)

# Initialize validator
validator = StatisticalValidator(config)

# Load your evaluation data
data = validator.collect_evaluation_data(data_source="your_results.json")

# Run analysis
results = validator.run_statistical_analysis(data)

# Check recommendation
recommendation = results['conclusions']['overall_recommendation']
if recommendation == 'ADOPT_HEGELS_APPROACH':
    print("✓ Evidence supports using Hegel's approach")
elif recommendation == 'INSUFFICIENT_EVIDENCE':
    print("✗ Need more evidence or larger sample size")
'''
    
    print(programmatic_example)


def demo_output_interpretation():
    """Demonstrate how to interpret the statistical validation output."""
    print("\n" + "="*70)
    print("DEMO 5: Output Interpretation Guide")
    print("="*70)
    
    # Generate example results
    config = ValidationConfig(sample_size=75, random_seed=42)
    validator = StatisticalValidator(config)
    data = validator.collect_evaluation_data()
    results = validator.run_statistical_analysis(data)
    
    print("Understanding the Output:")
    print("=" * 30)
    
    print(f"\n1. OVERALL RECOMMENDATION")
    recommendation = results['conclusions']['overall_recommendation']
    confidence = results['conclusions']['confidence_level']
    
    interpretation_map = {
        'ADOPT_HEGELS_APPROACH': '✓ Strong evidence favors Hegel\'s approach',
        'LIKELY_ADOPT_HEGELS_APPROACH': '✓ Good evidence favors Hegel\'s approach', 
        'CONSIDER_HEGELS_APPROACH': '⚠ Mixed evidence - proceed with caution',
        'INSUFFICIENT_EVIDENCE': '✗ No clear evidence of superiority'
    }
    
    print(f"   Recommendation: {recommendation}")
    print(f"   Confidence: {confidence}")
    print(f"   Interpretation: {interpretation_map.get(recommendation, 'Unknown')}")
    
    print(f"\n2. STATISTICAL SIGNIFICANCE")
    stat_tests = results.get('statistical_tests', {})
    quality_tests = stat_tests.get('quality_analysis', {})
    
    print("   What it means: Are results likely due to real differences (not chance)?")
    for test_name, test_result in quality_tests.items():
        if isinstance(test_result, dict) and 'p_value' in test_result:
            p_val = test_result.get('p_value')
            significant = p_val < 0.05 if isinstance(p_val, float) else False
            print(f"   {test_result.get('test_name', test_name)}: {'✓ Significant' if significant else '✗ Not significant'} (p={p_val:.4f if isinstance(p_val, float) else p_val})")
    
    print(f"\n3. PRACTICAL SIGNIFICANCE")
    practical = results.get('practical_significance', {})
    improvement_points = practical.get('mean_improvement_points', 0)
    improvement_percent = practical.get('mean_improvement_percent', 0)
    practically_significant = practical.get('practically_significant', False)
    
    print("   What it means: Is the improvement large enough to matter in practice?")
    print(f"   Mean improvement: {improvement_points:.2f} points ({improvement_percent:.1f}%)")
    print(f"   Practically significant: {'✓ Yes' if practically_significant else '✗ No'}")
    
    print(f"\n4. EFFECT SIZE") 
    effect_size = results.get('effect_size_analysis', {})
    cohens_d = effect_size.get('quality_cohens_d', {})
    
    print("   What it means: How large is the difference between approaches?")
    if cohens_d:
        d_value = cohens_d.get('effect_size', 0)
        interpretation = cohens_d.get('interpretation', 'Unknown')
        print(f"   Cohen's d: {d_value:.3f} - {interpretation}")
    
    print(f"\n5. CONFIDENCE INTERVALS")
    ci_info = effect_size.get('improvement_confidence_interval', {})
    if ci_info:
        print("   What it means: Range of likely true improvement values")
        print(f"   {ci_info.get('interpretation', 'No confidence interval available')}")
    
    print(f"\n6. POWER ANALYSIS")
    power = results.get('power_analysis', {})
    if power:
        current_power = power.get('current_power', 0)
        required_n = power.get('required_sample_size_for_target_power', 0)
        
        print("   What it means: How likely are we to detect a real difference?")
        print(f"   Current power: {current_power*100:.1f}%")
        if current_power < 0.8:
            print(f"   ⚠ Low power - consider increasing sample size to {required_n}")
        else:
            print(f"   ✓ Adequate power for reliable results")


def main():
    """Run all demonstration examples."""
    print("Statistical Validation System - Comprehensive Demo")
    print("=" * 70)
    print("This demo shows how to use the statistical validation system")
    print("to compare Hegel's dialectical agents vs single-agent review.")
    print(f"Demo started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all demos
        demo_basic_validation()
        demo_power_analysis() 
        demo_different_scenarios()
        demo_real_world_usage()
        demo_output_interpretation()
        
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("✓ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Run the full analysis: python statistical_validation.py")
        print("2. Try with your own data: python statistical_validation.py --data-file your_results.json")
        print("3. Customize parameters as needed for your evaluation requirements")
        print("\nFor help: python statistical_validation.py --help")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
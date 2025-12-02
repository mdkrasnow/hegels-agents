#!/usr/bin/env python3
"""
Dialectical Test Validation Script - Phase 0.5.3

This script runs the critical validation test that determines whether dialectical
debate improves AI reasoning quality. It performs comprehensive comparison between
single-agent responses and dialectical synthesis across multiple test questions.

This is the make-or-break test for the entire hegels-agents project.

Usage:
    python scripts/run_dialectical_test.py [--questions N] [--output DIR] [--verbose]
"""

import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add src and test_questions to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

from debate.dialectical_tester import DialecticalTester, DialecticalTestSuite
from test_questions.dialectical_test_questions import (
    get_question_set, get_question_metadata, validate_question_coverage
)


def setup_test_environment():
    """
    Set up the test environment and validate prerequisites.
    
    Returns:
        Dict with environment status
    """
    print("Setting up dialectical test environment...")
    
    # Load configuration first
    from config.settings import load_config
    try:
        config = load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"✅ API key configured: {bool(config.get_gemini_api_key())}")
        print(f"✅ Database configured: {bool(config.get_database_url())}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        raise
    
    # Check corpus availability
    corpus_dir = Path(__file__).parent.parent / "corpus_data"
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    corpus_files = list(corpus_dir.glob("*.txt"))
    if len(corpus_files) < 10:
        print(f"Warning: Only {len(corpus_files)} corpus files found, may affect test quality")
    
    # Validate question coverage
    coverage = validate_question_coverage()
    print(f"Test Questions: {coverage['total_questions']}")
    print(f"Domains Covered: {coverage['domain_count']}")
    print(f"Question Type Distribution: {coverage['question_type_distribution']}")
    
    return {
        'corpus_dir': str(corpus_dir),
        'corpus_files': len(corpus_files),
        'question_coverage': coverage,
        'setup_time': datetime.now().isoformat()
    }


def run_validation_test(num_questions: int = None, 
                       output_dir: str = None,
                       verbose: bool = False) -> DialecticalTestSuite:
    """
    Run the complete dialectical validation test.
    
    Args:
        num_questions: Number of questions to test (None for all)
        output_dir: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Complete test suite results
    """
    print("\n" + "="*60)
    print("DIALECTICAL VALIDATION TEST - PHASE 0.5.3")
    print("="*60)
    
    # Setup environment
    env_status = setup_test_environment()
    
    # Initialize tester
    tester = DialecticalTester(env_status['corpus_dir'])
    
    # Get test questions
    all_questions = get_question_set()
    questions_to_test = all_questions[:num_questions] if num_questions else all_questions
    
    print(f"\nTesting {len(questions_to_test)} questions...")
    print(f"This will validate the core hypothesis that dialectical debate improves AI reasoning.\n")
    
    # Run the test suite
    start_time = time.time()
    test_suite = tester.run_test_suite(questions_to_test)
    end_time = time.time()
    
    print(f"\nTest suite completed in {end_time - start_time:.2f} seconds")
    
    # Save results if output directory specified
    if output_dir:
        try:
            save_test_results(test_suite, output_dir, verbose)
        except Exception as save_error:
            print(f"Warning: Failed to save results: {save_error}")
            print("Test results are still available in memory for summary display")
    
    return test_suite


def make_json_serializable(obj):
    """
    Convert objects to JSON serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object
    """
    # Handle numpy boolean types
    if hasattr(obj, '__class__') and 'bool_' in str(obj.__class__):
        return bool(obj)
    # Handle numpy numeric types
    elif hasattr(obj, '__class__') and any(x in str(obj.__class__) for x in ['int_', 'float_', 'number']):
        return float(obj) if 'float' in str(obj.__class__) else int(obj)
    # Handle regular Python types
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    # Handle basic JSON serializable types
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # For anything else, try to convert to string as a fallback
    else:
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)

def save_test_results(test_suite: DialecticalTestSuite, 
                     output_dir: str, 
                     verbose: bool = False):
    """
    Save test results to files.
    
    Args:
        test_suite: Test suite results to save
        output_dir: Directory to save results
        verbose: Enable verbose output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = test_suite.timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    results_file = output_path / f"dialectical_test_results_{timestamp}.json"
    
    # Convert test suite to serializable format
    raw_results = {
        'timestamp': test_suite.timestamp.isoformat(),
        'summary_statistics': test_suite.summary_statistics,
        'hypothesis_validation': test_suite.hypothesis_validation,
        'test_results': [
            {
                'question': result.question,
                'single_agent_quality_score': result.single_agent_quality_score,
                'dialectical_quality_score': result.dialectical_quality_score,
                'improvement_score': result.improvement_score,
                'single_agent_time': result.single_agent_time,
                'dialectical_time': result.dialectical_time,
                'conflict_identified': result.conflict_identified,
                'synthesis_effectiveness': result.synthesis_effectiveness,
                'timestamp': result.timestamp.isoformat(),
                'test_metadata': result.test_metadata,
                'single_agent_response': {
                    'content': result.single_agent_response.content,
                    'reasoning': result.single_agent_response.reasoning,
                    'confidence': result.single_agent_response.confidence,
                    'sources': result.single_agent_response.sources,
                    'metadata': result.single_agent_response.metadata
                },
                'dialectical_synthesis': {
                    'content': result.dialectical_synthesis.content,
                    'reasoning': result.dialectical_synthesis.reasoning,
                    'confidence': result.dialectical_synthesis.confidence,
                    'sources': result.dialectical_synthesis.sources,
                    'metadata': result.dialectical_synthesis.metadata
                },
                'debate_session_summary': result.debate_session.get_summary()
            }
            for result in test_suite.test_results
        ]
    }
    
    # Ensure all values are JSON serializable
    serializable_results = make_json_serializable(raw_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    
    # Save human-readable report
    from debate.dialectical_tester import DialecticalTester
    tester = DialecticalTester("dummy")  # Just for report generation
    report = tester.generate_detailed_report(test_suite)
    
    report_file = output_path / f"dialectical_test_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {report_file}")
    
    # Save individual debate transcripts if verbose
    if verbose:
        transcripts_dir = output_path / f"transcripts_{timestamp}"
        transcripts_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(test_suite.test_results, 1):
            transcript_file = transcripts_dir / f"debate_{i}_transcript.txt"
            with open(transcript_file, 'w') as f:
                f.write(result.debate_session.export_transcript())
        
        print(f"Debate transcripts saved to: {transcripts_dir}")


def print_summary_results(test_suite: DialecticalTestSuite):
    """
    Print a summary of test results to console.
    
    Args:
        test_suite: Test suite results to summarize
    """
    stats = test_suite.summary_statistics
    hypothesis = test_suite.hypothesis_validation
    
    print("\n" + "="*60)
    print("DIALECTICAL TEST RESULTS SUMMARY")
    print("="*60)
    
    # Core hypothesis result
    if hypothesis['hypothesis_supported']:
        print("🎉 HYPOTHESIS VALIDATED: Dialectical debate improves AI reasoning quality!")
    else:
        print("❌ HYPOTHESIS NOT VALIDATED: Dialectical approach needs refinement")
    
    print(f"\nKey Metrics:")
    print(f"  Average Improvement: {stats['improvement_scores']['mean']:.1%}")
    print(f"  Tests Showing Improvement: {stats['improvement_analysis']['positive_percentage']:.1f}%")
    print(f"  Single Agent Quality: {stats['quality_scores']['single_agent']['mean']:.2f}/10")
    print(f"  Dialectical Quality: {stats['quality_scores']['dialectical']['mean']:.2f}/10")
    
    print(f"\nPerformance:")
    print(f"  Single Agent Time: {stats['performance']['single_agent_time']['mean']:.2f}s avg")
    print(f"  Dialectical Time: {stats['performance']['dialectical_time']['mean']:.2f}s avg")
    print(f"  Time Overhead: {stats['performance']['time_overhead_ratio']:.1f}x")
    
    print(f"\nDebate Analysis:")
    print(f"  Conflicts Identified: {stats['conflicts_and_synthesis']['conflicts_identified_count']}/{stats['num_tests']}")
    print(f"  Synthesis Effectiveness: {stats['conflicts_and_synthesis']['mean_synthesis_effectiveness']:.2f}/10")
    
    # Individual results summary
    print(f"\nIndividual Test Results:")
    for i, result in enumerate(test_suite.test_results, 1):
        status = "✓" if result.improvement_score > 0 else "✗"
        print(f"  {i:2d}. {status} {result.improvement_score:+.1%} | "
              f"Quality: {result.single_agent_quality_score:.1f}→{result.dialectical_quality_score:.1f}")
    
    print(f"\nRecommendation: {hypothesis.get('interpretation', {}).get('overall_recommendation', 'Unknown')}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run dialectical test validation for Phase 0.5.3"
    )
    parser.add_argument(
        "--questions", "-q", 
        type=int, 
        help="Number of questions to test (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="test_results",
        help="Output directory for results (default: test_results)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output and save debate transcripts"
    )
    parser.add_argument(
        "--no-save",
        action="store_true", 
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    try:
        # Run validation test
        test_suite = run_validation_test(
            num_questions=args.questions,
            output_dir=args.output if not args.no_save else None,
            verbose=args.verbose
        )
        
        # Print summary
        print_summary_results(test_suite)
        
        # Return appropriate exit code
        if test_suite.hypothesis_validation['hypothesis_supported']:
            print("\n✅ Phase 0.5.3 validation SUCCESSFUL - proceed to Phase 1")
            return 0
        else:
            print("\n❌ Phase 0.5.3 validation FAILED - reassess approach before Phase 1")
            return 1
    
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
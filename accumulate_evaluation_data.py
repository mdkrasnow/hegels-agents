#!/usr/bin/env python3
"""
Evaluation Data Accumulator for Hegel's Agents

This script runs multiple evaluation rounds and accumulates results for
statistical analysis. It can:

1. Load existing evaluation data from previous runs
2. Run new dialectical evaluations 
3. Merge and append new results to existing data
4. Save accumulated dataset for statistical validation
5. Optionally run statistical analysis on accumulated data

Usage:
    # Run 5 new evaluations and accumulate to existing data
    python accumulate_evaluation_data.py --run-count 5 --data-file accumulated_results.json

    # Start fresh with 10 evaluations
    python accumulate_evaluation_data.py --run-count 10 --output-file fresh_results.json

    # Run statistical analysis on accumulated data
    python accumulate_evaluation_data.py --analyze-only --data-file accumulated_results.json
"""

import sys
import argparse
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

try:
    # Load configuration first
    from config.settings import load_config
    config = load_config()
    
    from debate.dialectical_tester import DialecticalTester, DialecticalTestResult
    from test_questions.dialectical_test_questions import get_question_set
    EVALUATION_INFRASTRUCTURE_AVAILABLE = True
except Exception as e:
    EVALUATION_INFRASTRUCTURE_AVAILABLE = False
    print(f"Warning: Evaluation infrastructure not available: {e}")
    print("Will work in data analysis mode only.")

# Import statistical validation
from statistical_validation import (
    StatisticalValidator, ValidationConfig, ComparisonMetrics, 
    save_results, generate_summary_report
)


class EvaluationDataAccumulator:
    """
    Accumulates evaluation data across multiple runs for robust statistical analysis.
    """
    
    def __init__(self, 
                 data_file: Optional[str] = None,
                 corpus_dir: Optional[str] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the accumulator.
        
        Args:
            data_file: Path to existing data file to load from
            corpus_dir: Directory containing corpus data  
            random_seed: Random seed for reproducible results
        """
        self.data_file = Path(data_file) if data_file else None
        self.corpus_dir = Path(corpus_dir) if corpus_dir else (project_root / "corpus_data")
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.accumulated_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_evaluations': 0,
                'unique_questions_evaluated': set(),
                'evaluation_sessions': []
            },
            'test_results': []
        }
        
        # Load existing data if provided
        if self.data_file and self.data_file.exists():
            self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing evaluation data from file."""
        self.logger.info(f"Loading existing data from {self.data_file}")
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            # Handle different data formats
            if 'test_results' in data:
                # Direct format
                self.accumulated_data = data
                # Convert set back from list if needed
                if 'unique_questions_evaluated' in self.accumulated_data['metadata']:
                    unique_questions = self.accumulated_data['metadata']['unique_questions_evaluated']
                    if isinstance(unique_questions, list):
                        self.accumulated_data['metadata']['unique_questions_evaluated'] = set(unique_questions)
            
            elif 'comparison_results' in data:
                # Convert from comparison format
                self._convert_comparison_format(data)
            
            else:
                self.logger.warning("Unrecognized data format, starting fresh")
                return
            
            self.logger.info(f"Loaded {len(self.accumulated_data['test_results'])} existing evaluation results")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing data: {e}")
            self.logger.info("Starting with empty dataset")
    
    def _convert_comparison_format(self, data: Dict[str, Any]):
        """Convert comparison results format to test results format."""
        comp_data = data['comparison_results']
        
        for i in range(len(comp_data['single_scores'])):
            result = {
                'question': comp_data.get('questions', [f'Question {i+1}'])[i] if i < len(comp_data.get('questions', [])) else f'Question {i+1}',
                'single_agent_quality_score': comp_data['single_scores'][i],
                'dialectical_quality_score': comp_data['hegel_scores'][i],
                'improvement_score': comp_data.get('improvement_scores', [0])[i] if i < len(comp_data.get('improvement_scores', [])) else 0,
                'single_agent_time': comp_data.get('single_times', [3.0])[i] if i < len(comp_data.get('single_times', [])) else 3.0,
                'dialectical_time': comp_data.get('hegel_times', [8.0])[i] if i < len(comp_data.get('hegel_times', [])) else 8.0,
                'imported_from': 'comparison_format',
                'evaluation_timestamp': data.get('metadata', {}).get('loaded_at', datetime.now().isoformat())
            }
            self.accumulated_data['test_results'].append(result)
        
        # Update metadata
        self.accumulated_data['metadata']['total_evaluations'] = len(self.accumulated_data['test_results'])
        self.accumulated_data['metadata']['unique_questions_evaluated'] = set(
            result['question'] for result in self.accumulated_data['test_results']
        )
    
    def run_evaluation_round(self, num_questions: int = 10, question_selection: str = 'random') -> List[Dict[str, Any]]:
        """
        Run a single round of evaluation with multiple questions.
        
        Args:
            num_questions: Number of questions to evaluate
            question_selection: How to select questions ('random', 'sequential', 'all')
            
        Returns:
            List of evaluation results
        """
        if not EVALUATION_INFRASTRUCTURE_AVAILABLE:
            raise RuntimeError("Cannot run evaluations - infrastructure not available")
        
        self.logger.info(f"Running evaluation round with {num_questions} questions")
        
        # Setup tester
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
        
        tester = DialecticalTester(str(self.corpus_dir))
        
        # Get questions
        all_questions = get_question_set()
        
        if question_selection == 'all':
            test_questions = all_questions[:num_questions]
        elif question_selection == 'sequential':
            # Start from where we left off, cycling through
            start_idx = len(self.accumulated_data['test_results']) % len(all_questions)
            test_questions = []
            for i in range(num_questions):
                test_questions.append(all_questions[(start_idx + i) % len(all_questions)])
        else:  # random
            test_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
        
        # Run evaluations
        round_results = []
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, question in enumerate(test_questions):
            self.logger.info(f"Evaluating question {i+1}/{len(test_questions)}: {question[:100]}...")
            
            try:
                result = tester.run_comparison_test(question)
                
                # Convert to our format
                evaluation_result = {
                    'question': question,
                    'single_agent_quality_score': result.single_agent_quality_score,
                    'dialectical_quality_score': result.dialectical_quality_score,
                    'improvement_score': result.improvement_score,
                    'single_agent_time': result.single_agent_time,
                    'dialectical_time': result.dialectical_time,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'question_index_in_session': i,
                    'metadata': {
                        'corpus_dir': str(self.corpus_dir),
                        'question_selection': question_selection
                    }
                }
                
                round_results.append(evaluation_result)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for question {i+1}: {e}")
                continue
        
        # Update accumulated data
        self.accumulated_data['test_results'].extend(round_results)
        self.accumulated_data['metadata']['total_evaluations'] += len(round_results)
        self.accumulated_data['metadata']['evaluation_sessions'].append({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'questions_evaluated': len(round_results),
            'question_selection': question_selection
        })
        
        # Update unique questions
        for result in round_results:
            self.accumulated_data['metadata']['unique_questions_evaluated'].add(result['question'])
        
        self.logger.info(f"Completed evaluation round: {len(round_results)} successful evaluations")
        return round_results
    
    def run_multiple_rounds(self, 
                          num_rounds: int,
                          questions_per_round: int = 5,
                          question_selection: str = 'random') -> List[List[Dict[str, Any]]]:
        """
        Run multiple evaluation rounds.
        
        Args:
            num_rounds: Number of evaluation rounds
            questions_per_round: Questions per round
            question_selection: Question selection strategy
            
        Returns:
            List of round results
        """
        all_rounds = []
        
        for round_num in range(num_rounds):
            self.logger.info(f"Starting round {round_num + 1}/{num_rounds}")
            
            try:
                round_results = self.run_evaluation_round(
                    num_questions=questions_per_round,
                    question_selection=question_selection
                )
                all_rounds.append(round_results)
                
                # Save progress after each round
                if self.data_file:
                    self.save_data(self.data_file)
                
            except Exception as e:
                self.logger.error(f"Round {round_num + 1} failed: {e}")
                continue
        
        return all_rounds
    
    def get_comparison_metrics(self) -> ComparisonMetrics:
        """
        Convert accumulated data to ComparisonMetrics format for statistical analysis.
        
        Returns:
            ComparisonMetrics object ready for analysis
        """
        if not self.accumulated_data['test_results']:
            raise ValueError("No evaluation data available")
        
        metrics = ComparisonMetrics()
        
        for result in self.accumulated_data['test_results']:
            metrics.single_scores.append(result['single_agent_quality_score'])
            metrics.hegel_scores.append(result['dialectical_quality_score'])
            metrics.improvement_scores.append(result['improvement_score'])
            metrics.single_times.append(result['single_agent_time'])
            metrics.hegel_times.append(result['dialectical_time'])
            metrics.questions.append(result['question'])
        
        # Add metadata
        metrics.metadata = {
            'data_source': 'accumulated_evaluations',
            'total_evaluations': self.accumulated_data['metadata']['total_evaluations'],
            'unique_questions': len(self.accumulated_data['metadata']['unique_questions_evaluated']),
            'evaluation_sessions': len(self.accumulated_data['metadata']['evaluation_sessions']),
            'accumulated_at': datetime.now().isoformat(),
            'corpus_dir': str(self.corpus_dir)
        }
        
        if not metrics.validate_data():
            raise ValueError("Generated comparison metrics failed validation")
        
        return metrics
    
    def save_data(self, output_file: Path):
        """
        Save accumulated data to file.
        
        Args:
            output_file: Path to save data
        """
        # Convert set to list for JSON serialization
        data_to_save = self.accumulated_data.copy()
        data_to_save['metadata'] = self.accumulated_data['metadata'].copy()
        data_to_save['metadata']['unique_questions_evaluated'] = list(
            self.accumulated_data['metadata']['unique_questions_evaluated']
        )
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(self.accumulated_data['test_results'])} evaluation results to {output_file}")
    
    def run_statistical_analysis(self, 
                                config: Optional[ValidationConfig] = None,
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run statistical analysis on accumulated data.
        
        Args:
            config: Validation configuration
            output_dir: Directory to save analysis results
            
        Returns:
            Analysis results
        """
        self.logger.info("Running statistical analysis on accumulated data")
        
        # Get comparison metrics
        metrics = self.get_comparison_metrics()
        
        # Setup validator
        if config is None:
            config = ValidationConfig(sample_size=metrics.sample_size)
        
        validator = StatisticalValidator(config)
        
        # Run analysis
        results = validator.run_statistical_analysis(metrics)
        
        # Save results if output directory specified
        if output_dir:
            saved_files = save_results(results, output_dir, config)
            self.logger.info("Analysis results saved:")
            for file_type, file_path in saved_files.items():
                self.logger.info(f"  {file_type}: {file_path}")
        
        return results
    
    def print_summary(self):
        """Print summary of accumulated data."""
        total = self.accumulated_data['metadata']['total_evaluations']
        unique_questions = len(self.accumulated_data['metadata']['unique_questions_evaluated'])
        sessions = len(self.accumulated_data['metadata']['evaluation_sessions'])
        
        print(f"\n{'='*60}")
        print("ACCUMULATED EVALUATION DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total Evaluations: {total}")
        print(f"Unique Questions Evaluated: {unique_questions}")
        print(f"Evaluation Sessions: {sessions}")
        
        if total > 0:
            # Calculate basic stats
            single_scores = [r['single_agent_quality_score'] for r in self.accumulated_data['test_results']]
            hegel_scores = [r['dialectical_quality_score'] for r in self.accumulated_data['test_results']]
            improvements = [h - s for h, s in zip(hegel_scores, single_scores)]
            
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            
            print(f"\nQuick Stats:")
            print(f"  Average Single Agent Score: {sum(single_scores)/len(single_scores):.1f}")
            print(f"  Average Hegel Score: {sum(hegel_scores)/len(hegel_scores):.1f}")
            print(f"  Average Improvement: {sum(improvements)/len(improvements):.1f} points")
            print(f"  Positive Improvements: {positive_improvements}/{total} ({positive_improvements/total*100:.1f}%)")
        
        print(f"\nReady for statistical analysis: {'Yes' if total >= 10 else f'No (need {10-total} more evaluations)'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Accumulate evaluation data across multiple runs for robust statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 5 evaluation rounds and accumulate results
    python accumulate_evaluation_data.py --run-count 5 --data-file accumulated_results.json
    
    # Start fresh with 20 evaluations (4 rounds of 5)
    python accumulate_evaluation_data.py --run-count 4 --questions-per-round 5 --output-file fresh_data.json
    
    # Add 10 more evaluations to existing data
    python accumulate_evaluation_data.py --run-count 2 --data-file existing_data.json --questions-per-round 5
    
    # Analyze existing accumulated data
    python accumulate_evaluation_data.py --analyze-only --data-file accumulated_results.json
    
    # Run evaluations and analysis in one command
    python accumulate_evaluation_data.py --run-count 3 --analyze --output-dir analysis_results/
        """
    )
    
    # Data options
    parser.add_argument(
        '--data-file', '-f',
        type=str,
        help="Path to existing data file to load and append to"
    )
    parser.add_argument(
        '--output-file', '-o',
        type=str,
        help="Path to save accumulated data (defaults to data-file if provided)"
    )
    
    # Evaluation options
    parser.add_argument(
        '--run-count', '-r',
        type=int,
        default=0,
        help="Number of evaluation rounds to run (default: 0)"
    )
    parser.add_argument(
        '--questions-per-round', '-q',
        type=int,
        default=5,
        help="Number of questions per evaluation round (default: 5)"
    )
    parser.add_argument(
        '--question-selection',
        choices=['random', 'sequential', 'all'],
        default='random',
        help="How to select questions (default: random)"
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        help="Path to corpus directory (default: corpus_data/)"
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help="Only run analysis on existing data, don't run new evaluations"
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help="Run statistical analysis after collecting data"
    )
    parser.add_argument(
        '--analysis-output-dir',
        type=str,
        default='accumulated_analysis_results',
        help="Directory to save analysis results (default: accumulated_analysis_results)"
    )
    
    # Configuration options  
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help="Confidence level for statistical analysis (default: 0.95)"
    )
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)"
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.analyze_only and args.run_count > 0:
        parser.error("Cannot use --analyze-only with --run-count > 0")
    
    if not args.analyze_only and args.run_count == 0:
        parser.error("Must specify --run-count > 0 or use --analyze-only")
    
    if args.analyze_only and not args.data_file:
        parser.error("--analyze-only requires --data-file")
    
    try:
        # Initialize accumulator
        accumulator = EvaluationDataAccumulator(
            data_file=args.data_file,
            corpus_dir=args.corpus_dir,
            random_seed=args.random_seed
        )
        
        # Run evaluations if requested
        if not args.analyze_only and args.run_count > 0:
            print(f"Running {args.run_count} evaluation rounds with {args.questions_per_round} questions each...")
            
            accumulator.run_multiple_rounds(
                num_rounds=args.run_count,
                questions_per_round=args.questions_per_round,
                question_selection=args.question_selection
            )
            
            print(f"Completed {args.run_count} evaluation rounds")
        
        # Save data
        output_file = Path(args.output_file) if args.output_file else (
            Path(args.data_file) if args.data_file else Path('accumulated_evaluation_data.json')
        )
        accumulator.save_data(output_file)
        
        # Print summary
        accumulator.print_summary()
        
        # Run analysis if requested
        if args.analyze or args.analyze_only:
            print(f"\nRunning statistical analysis...")
            
            config = ValidationConfig(
                sample_size=accumulator.accumulated_data['metadata']['total_evaluations'],
                confidence_level=args.confidence_level,
                significance_level=args.significance_level,
                random_seed=args.random_seed
            )
            
            results = accumulator.run_statistical_analysis(
                config=config,
                output_dir=args.analysis_output_dir
            )
            
            # Print detailed results
            conclusions = results.get('conclusions', {})
            data_summary = results.get('data_summary', {})
            statistics = results.get('statistical_tests', {})
            effect_size = results.get('effect_size_analysis', {})
            power_analysis = results.get('power_analysis', {})
            practical_sig = conclusions.get('practical_significance_summary', {})
            
            print(f"\n{'='*60}")
            print("STATISTICAL ANALYSIS RESULTS")
            print(f"{'='*60}")
            print(f"Recommendation: {conclusions.get('overall_recommendation', 'Unknown')}")
            print(f"Confidence: {conclusions.get('confidence_level', 'Unknown')}")
            print(f"Summary: {conclusions.get('summary', 'No summary available')}")
            
            # Key performance metrics
            single_agent_data = data_summary.get('single_agent_scores', {})
            hegel_data = data_summary.get('hegel_scores', {})
            print(f"\n{'Key Findings:'}")
            print(f"• Single Agent Avg: {single_agent_data.get('mean', 0):.1f}/100")
            print(f"• Hegel's Agents Avg: {hegel_data.get('mean', 0):.1f}/100") 
            print(f"• Mean Improvement: {practical_sig.get('improvement_points', 0):.1f} points ({practical_sig.get('improvement_percent', 0):.1f}%)")
            
            # Statistical significance
            quality_tests = statistics.get('quality_analysis', {})
            t_test = quality_tests.get('paired_t_test', {}) if quality_tests else {}
            wilcoxon = quality_tests.get('wilcoxon_signed_rank', {}) if quality_tests else {}
            
            print(f"\n{'Statistical Significance:'}")
            if t_test:
                print(f"• Paired t-test: p = {t_test.get('p_value', 1):.4f} ({'✅ Significant' if t_test.get('p_value', 1) < 0.05 else '❌ Not significant'})")
            if wilcoxon:
                print(f"• Wilcoxon test: p = {wilcoxon.get('p_value', 1):.4f} ({'✅ Significant' if wilcoxon.get('p_value', 1) < 0.05 else '❌ Not significant'})")
            
            # Overall significance
            sig_summary = conclusions.get('statistical_significance_summary', {})
            if sig_summary.get('overall_significance'):
                print(f"• Overall: ✅ Statistically significant improvement")
            else:
                print(f"• Overall: ❌ No significant improvement found")
            
            # Effect size and confidence
            print(f"\n{'Effect Size & Confidence:'}")
            quality_effect = effect_size.get('quality_analysis', {})
            if quality_effect:
                cohens_d = quality_effect.get('cohens_d', 0)
                effect_interp = quality_effect.get('interpretation', 'Unknown')
                print(f"• Cohen's d: {cohens_d:.3f} ({effect_interp})")
                
                ci = quality_effect.get('confidence_interval', {})
                if ci:
                    ci_lower = ci.get('lower', 0)
                    ci_upper = ci.get('upper', 0) 
                    print(f"• 95% Confidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}] points")
            
            # Power analysis
            quality_power = power_analysis.get('quality_analysis', {})
            if quality_power:
                current_power = quality_power.get('current_power', 0) * 100
                print(f"• Statistical Power: {current_power:.1f}% ({'✅ Good' if current_power >= 80 else '⚠️  Low power'})")
                
                if current_power < 80:
                    recommended_n = quality_power.get('recommended_sample_size_80_power', 'Unknown')
                    print(f"• Recommended sample size for 80% power: {recommended_n}")
            
            # Sample size assessment
            sample_assessment = conclusions.get('sample_size_assessment', {})
            if sample_assessment:
                adequacy = sample_assessment.get('adequacy', 'Unknown')
                min_recommended = sample_assessment.get('minimum_recommended', 'Unknown')
                print(f"\n{'Sample Size Assessment:'}")
                print(f"• Adequacy: {adequacy} (n={results.get('sample_size', 0)})")
                print(f"• Minimum recommended: {min_recommended}")
            
            # File locations
            if args.analysis_output_dir:
                print(f"\n{'Detailed reports saved to:'}")
                print(f"• Analysis results: {args.analysis_output_dir}/")
                print("• Check the .md report file for full statistical details")
        
        print(f"\nData saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
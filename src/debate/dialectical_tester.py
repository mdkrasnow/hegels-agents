"""
DialecticalTester - Core Dialectical Test Implementation for Phase 0.5.3

This module implements the critical validation test that determines whether
dialectical debate improves AI reasoning quality. It provides structured
comparison between single-agent responses and dialectical synthesis.
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from agents.worker import BasicWorkerAgent
from agents.reviewer import BasicReviewerAgent
from agents.utils import AgentResponse
from corpus.file_retriever import FileCorpusRetriever
from .session import DebateSession
from eval.blinded_evaluator import BlindedDialecticalComparison


@dataclass
class DialecticalTestResult:
    """
    Results from a single dialectical test comparison.
    """
    question: str
    single_agent_response: AgentResponse
    dialectical_synthesis: AgentResponse
    debate_session: DebateSession
    
    # Quality metrics
    single_agent_quality_score: float
    dialectical_quality_score: float
    improvement_score: float
    
    # Performance metrics
    single_agent_time: float
    dialectical_time: float
    
    # Analysis metrics
    conflict_identified: bool
    synthesis_effectiveness: float
    
    # Metadata
    timestamp: datetime
    test_metadata: Dict[str, Any]
    evaluation_metadata: Dict[str, Any]  # Blinded evaluation metadata


@dataclass
class DialecticalTestSuite:
    """
    Complete results from running dialectical tests on multiple questions.
    """
    test_results: List[DialecticalTestResult]
    summary_statistics: Dict[str, Any]
    hypothesis_validation: Dict[str, Any]
    timestamp: datetime


class DialecticalTester:
    """
    Core dialectical test implementation that validates whether dialectical
    debate improves reasoning quality over single-agent responses.
    
    This is the critical validation component for the entire project.
    """
    
    def __init__(self, 
                 corpus_dir: str,
                 worker_1_id: str = "worker_1",
                 worker_2_id: str = "worker_2", 
                 reviewer_id: str = "reviewer"):
        """
        Initialize the dialectical tester.
        
        Args:
            corpus_dir: Path to corpus directory for knowledge retrieval
            worker_1_id: ID for first worker agent
            worker_2_id: ID for second worker agent  
            reviewer_id: ID for reviewer agent
        """
        self.corpus_dir = corpus_dir
        
        # Initialize agents
        self.worker_1 = BasicWorkerAgent(worker_1_id)
        self.worker_2 = BasicWorkerAgent(worker_2_id)
        self.reviewer = BasicReviewerAgent(reviewer_id)
        
        # Initialize corpus retriever
        self.corpus_retriever = FileCorpusRetriever(corpus_dir)
        self.corpus_retriever.load_corpus()
        self.corpus_retriever.build_search_index()
        
        # Initialize blinded evaluator for fair comparison
        self.blinded_evaluator = BlindedDialecticalComparison()
        
        # Test configuration (kept for backwards compatibility, but no longer used)
        self.quality_evaluation_prompt = """
        Please evaluate the quality of this response on a scale from 1-100 based on:
        1. Accuracy and factual correctness
        2. Comprehensiveness and depth of analysis  
        3. Clarity and organization of reasoning
        4. Use of evidence and supporting information
        5. Acknowledgment of limitations or uncertainties
        
        Provide only a numeric score from 1-100, no explanation needed.
        """
        
    def run_single_agent_test(self, question: str) -> Tuple[AgentResponse, float]:
        """
        Run single-agent test for baseline comparison.
        
        Args:
            question: Question to answer
            
        Returns:
            Tuple of (response, time_taken)
        """
        start_time = time.time()
        
        # Get context from corpus
        context = self.corpus_retriever.retrieve_for_question(question, max_results=3)
        
        # Get single agent response
        response = self.worker_1.respond(question, external_context=context)
        
        end_time = time.time()
        return response, end_time - start_time
    
    def run_dialectical_test(self, question: str) -> Tuple[AgentResponse, DebateSession, float]:
        """
        Run dialectical test with worker debate and synthesis.
        
        Args:
            question: Question to answer
            
        Returns:
            Tuple of (synthesis_response, debate_session, time_taken)
        """
        start_time = time.time()
        
        # Initialize debate session
        debate_session = DebateSession(question)
        
        # Get context from corpus
        context = self.corpus_retriever.retrieve_for_question(question, max_results=3)
        
        # Worker 1 initial response
        worker1_response = self.worker_1.respond(question, external_context=context)
        debate_session.add_turn("worker_1", worker1_response)
        
        # Worker 2 alternative response (with knowledge of Worker 1's response)
        worker2_context = f"Previous response to consider: {worker1_response.content}\n\n{context if context else ''}"
        worker2_response = self.worker_2.respond(question, external_context=worker2_context)
        debate_session.add_turn("worker_2", worker2_response)
        
        # Reviewer synthesis
        synthesis_response = self.reviewer.synthesize_responses(
            question, [worker1_response, worker2_response]
        )
        debate_session.add_turn("reviewer_synthesis", synthesis_response)
        
        # Analyze the debate
        debate_session.analyze_debate([worker1_response, worker2_response], synthesis_response)
        
        end_time = time.time()
        return synthesis_response, debate_session, end_time - start_time
    
    def evaluate_response_quality(self, response: AgentResponse) -> float:
        """
        DEPRECATED: Use blinded evaluation instead for fair comparison.
        
        This method is kept for backwards compatibility but should not be used
        for new dialectical tests as it introduces evaluator bias.
        
        Args:
            response: Response to evaluate
            
        Returns:
            Quality score (1-100)
        """
        import warnings
        warnings.warn(
            "evaluate_response_quality is deprecated and biased. Use blinded evaluation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Simple fallback for backwards compatibility
        return 5.0  # Neutral score
    
    def calculate_improvement_score(self, single_score: float, dialectical_score: float) -> float:
        """
        Calculate improvement score from single to dialectical response.
        
        Args:
            single_score: Single agent quality score
            dialectical_score: Dialectical synthesis quality score
            
        Returns:
            Improvement score (-1 to 1, positive indicates improvement)
        """
        if single_score == 0:
            return 0.0
        
        improvement = (dialectical_score - single_score) / 100.0
        return max(-1.0, min(1.0, improvement))
    
    def run_comparison_test(self, question: str) -> DialecticalTestResult:
        """
        Run complete comparison test for a single question.
        
        Args:
            question: Question to test
            
        Returns:
            DialecticalTestResult with all metrics
        """
        print(f"Testing question: {question[:100]}...")
        
        # Run single agent test
        single_response, single_time = self.run_single_agent_test(question)
        print(f"  Single agent response time: {single_time:.2f}s")
        
        # Run dialectical test
        dialectical_response, debate_session, dialectical_time = self.run_dialectical_test(question)
        print(f"  Dialectical response time: {dialectical_time:.2f}s")
        
        # Evaluate quality using blinded evaluation for fairness
        print("  Performing blinded evaluation...")
        blinded_comparison = self.blinded_evaluator.compare_approaches(
            question, single_response, dialectical_response
        )
        
        single_quality = blinded_comparison["scores"]["single_agent"]
        dialectical_quality = blinded_comparison["scores"]["dialectical"]
        improvement_score = blinded_comparison["improvement"]["improvement_score"]
        
        print(f"  Quality scores (blinded) - Single: {single_quality}, Dialectical: {dialectical_quality}")
        print(f"  Evaluation fairness: Independent evaluators, anonymized responses")
        
        # Create result
        result = DialecticalTestResult(
            question=question,
            single_agent_response=single_response,
            dialectical_synthesis=dialectical_response,
            debate_session=debate_session,
            single_agent_quality_score=single_quality,
            dialectical_quality_score=dialectical_quality,
            improvement_score=improvement_score,
            single_agent_time=single_time,
            dialectical_time=dialectical_time,
            conflict_identified=debate_session.conflicts_identified,
            synthesis_effectiveness=debate_session.synthesis_effectiveness,
            timestamp=datetime.now(),
            test_metadata={
                'corpus_dir': self.corpus_dir,
                'worker_1_id': self.worker_1.agent_id,
                'worker_2_id': self.worker_2.agent_id,
                'reviewer_id': self.reviewer.agent_id
            },
            evaluation_metadata=blinded_comparison["evaluation_metadata"]
        )
        
        return result
    
    def run_test_suite(self, questions: List[str]) -> DialecticalTestSuite:
        """
        Run complete dialectical test suite on multiple questions.
        
        Args:
            questions: List of questions to test
            
        Returns:
            DialecticalTestSuite with complete results
        """
        print(f"Running dialectical test suite on {len(questions)} questions...")
        
        test_results = []
        for i, question in enumerate(questions, 1):
            print(f"\n=== Test {i}/{len(questions)} ===")
            result = self.run_comparison_test(question)
            test_results.append(result)
            
            print(f"  Improvement score: {result.improvement_score:.3f}")
            if result.improvement_score > 0:
                print("  ✓ Dialectical approach improved quality")
            else:
                print("  ✗ Dialectical approach did not improve quality")
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(test_results)
        
        # Validate hypothesis
        hypothesis_validation = self._validate_hypothesis(test_results, summary_stats)
        
        # Create test suite result
        test_suite = DialecticalTestSuite(
            test_results=test_results,
            summary_statistics=summary_stats,
            hypothesis_validation=hypothesis_validation,
            timestamp=datetime.now()
        )
        
        return test_suite
    
    def _calculate_summary_statistics(self, results: List[DialecticalTestResult]) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        if not results:
            print("Warning: No test results provided for statistics calculation")
            return {}
        
        improvement_scores = [r.improvement_score for r in results]
        single_quality_scores = [r.single_agent_quality_score for r in results]
        dialectical_quality_scores = [r.dialectical_quality_score for r in results]
        single_times = [r.single_agent_time for r in results]
        dialectical_times = [r.dialectical_time for r in results]
        
        positive_improvements = [s for s in improvement_scores if s > 0]
        negative_improvements = [s for s in improvement_scores if s < 0]
        
        return {
            'num_tests': len(results),
            'improvement_scores': {
                'mean': statistics.mean(improvement_scores),
                'median': statistics.median(improvement_scores),
                'stdev': statistics.stdev(improvement_scores) if len(improvement_scores) > 1 else 0.0,
                'min': min(improvement_scores),
                'max': max(improvement_scores)
            },
            'quality_scores': {
                'single_agent': {
                    'mean': statistics.mean(single_quality_scores),
                    'median': statistics.median(single_quality_scores),
                    'stdev': statistics.stdev(single_quality_scores) if len(single_quality_scores) > 1 else 0.0
                },
                'dialectical': {
                    'mean': statistics.mean(dialectical_quality_scores), 
                    'median': statistics.median(dialectical_quality_scores),
                    'stdev': statistics.stdev(dialectical_quality_scores) if len(dialectical_quality_scores) > 1 else 0.0
                }
            },
            'performance': {
                'single_agent_time': {
                    'mean': statistics.mean(single_times),
                    'median': statistics.median(single_times)
                },
                'dialectical_time': {
                    'mean': statistics.mean(dialectical_times),
                    'median': statistics.median(dialectical_times)
                },
                'time_overhead_ratio': statistics.mean(dialectical_times) / statistics.mean(single_times) if single_times and statistics.mean(single_times) > 0 else 1.0
            },
            'improvement_analysis': {
                'positive_improvements': len(positive_improvements),
                'negative_improvements': len(negative_improvements),
                'neutral_improvements': len(improvement_scores) - len(positive_improvements) - len(negative_improvements),
                'positive_percentage': len(positive_improvements) / len(results) * 100 if results else 0.0,
                'mean_positive_improvement': statistics.mean(positive_improvements) if positive_improvements else 0.0,
                'mean_negative_improvement': statistics.mean(negative_improvements) if negative_improvements else 0.0
            },
            'conflicts_and_synthesis': {
                'conflicts_identified_count': sum(1 for r in results if r.conflict_identified),
                'mean_synthesis_effectiveness': statistics.mean([r.synthesis_effectiveness for r in results])
            }
        }
    
    def _validate_hypothesis(self, results: List[DialecticalTestResult], 
                           summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the core hypothesis that dialectical debate improves reasoning.
        """
        improvement_scores = [r.improvement_score for r in results]
        
        # Basic statistical tests
        mean_improvement = summary_stats['improvement_scores']['mean']
        positive_rate = summary_stats['improvement_analysis']['positive_percentage']
        
        # Significance thresholds
        SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement
        MAJORITY_THRESHOLD = 60  # 60% of tests should show improvement
        
        # Hypothesis validation
        hypothesis_supported = (
            mean_improvement > SIGNIFICANT_IMPROVEMENT_THRESHOLD and 
            positive_rate > MAJORITY_THRESHOLD
        )
        
        # Effect size analysis
        if len(improvement_scores) > 1:
            effect_size = mean_improvement / summary_stats['improvement_scores']['stdev']
        else:
            effect_size = None
        
        # Statistical significance (basic t-test against zero improvement)
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(improvement_scores, 0)
            statistically_significant = p_value < 0.05
        except ImportError:
            # Fallback if scipy not available
            t_stat, p_value = None, None
            statistically_significant = None
        
        return {
            'hypothesis_supported': hypothesis_supported,
            'mean_improvement': mean_improvement,
            'positive_improvement_rate': positive_rate,
            'effect_size': effect_size,
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_05': statistically_significant
            },
            'interpretation': {
                'practical_significance': mean_improvement > SIGNIFICANT_IMPROVEMENT_THRESHOLD,
                'consistency': positive_rate > MAJORITY_THRESHOLD,
                'overall_recommendation': (
                    "Proceed with dialectical approach" if hypothesis_supported 
                    else "Reconsider dialectical approach"
                )
            }
        }
    
    @staticmethod
    def generate_detailed_report(test_suite: DialecticalTestSuite) -> str:
        """
        Generate a detailed analysis report of the test results.
        
        Args:
            test_suite: Completed test suite results
            
        Returns:
            Detailed report string
        """
        stats = test_suite.summary_statistics
        hypothesis = test_suite.hypothesis_validation
        
        report = f"""
# Dialectical Test Report - Phase 0.5.3 Validation

**Test Timestamp:** {test_suite.timestamp}
**Number of Tests:** {stats['num_tests']}

## Executive Summary

The core hypothesis that dialectical debate improves AI reasoning quality has been 
{"**VALIDATED**" if hypothesis['hypothesis_supported'] else "**NOT VALIDATED**"}.

**Key Findings:**
- Average improvement: {stats['improvement_scores']['mean']:.1%}
- Tests showing improvement: {stats['improvement_analysis']['positive_percentage']:.1f}%
- Statistical significance: {"Yes" if hypothesis.get('statistical_significance', {}).get('significant_at_05') else "No/Unknown"}

## Detailed Results

### Quality Score Analysis
- **Single Agent Average:** {stats['quality_scores']['single_agent']['mean']:.2f}/100
- **Dialectical Average:** {stats['quality_scores']['dialectical']['mean']:.2f}/100
- **Quality Improvement:** {(stats['quality_scores']['dialectical']['mean'] - stats['quality_scores']['single_agent']['mean']):.2f} points

### Performance Analysis  
- **Single Agent Time:** {stats['performance']['single_agent_time']['mean']:.2f}s average
- **Dialectical Time:** {stats['performance']['dialectical_time']['mean']:.2f}s average
- **Time Overhead:** {stats['performance']['time_overhead_ratio']:.1f}x

### Improvement Distribution
- **Positive Improvements:** {stats['improvement_analysis']['positive_improvements']} tests
- **No Change/Negative:** {stats['improvement_analysis']['negative_improvements'] + stats['improvement_analysis'].get('neutral_improvements', 0)} tests
- **Mean Positive Improvement:** {stats['improvement_analysis']['mean_positive_improvement']:.1%}

### Debate Analysis
- **Conflicts Identified:** {stats['conflicts_and_synthesis']['conflicts_identified_count']}/{stats['num_tests']} debates
- **Average Synthesis Effectiveness:** {stats['conflicts_and_synthesis']['mean_synthesis_effectiveness']:.2f}/100

## Individual Test Results

"""
        
        for i, result in enumerate(test_suite.test_results, 1):
            improvement_indicator = "✓" if result.improvement_score > 0 else "✗"
            report += f"""
### Test {i}: {improvement_indicator}
**Question:** {result.question[:100]}{"..." if len(result.question) > 100 else ""}
- Single Agent Quality: {result.single_agent_quality_score:.1f}/100
- Dialectical Quality: {result.dialectical_quality_score:.1f}/100  
- Improvement: {result.improvement_score:.1%}
- Conflict Identified: {result.conflict_identified}
"""

        report += f"""

## Statistical Analysis

### Central Tendency
- **Mean Improvement:** {stats['improvement_scores']['mean']:.1%}
- **Median Improvement:** {stats['improvement_scores']['median']:.1%}
- **Standard Deviation:** {stats['improvement_scores']['stdev']:.1%}

### Effect Size
- **Cohen's d:** {hypothesis.get('effect_size', 'N/A')}

### Statistical Significance
- **T-statistic:** {hypothesis.get('statistical_significance', {}).get('t_statistic', 'N/A')}
- **P-value:** {hypothesis.get('statistical_significance', {}).get('p_value', 'N/A')}

## Hypothesis Validation

**Core Hypothesis:** Dialectical debate between multiple AI agents improves reasoning quality.

**Validation Criteria:**
1. Mean improvement > 5%: {"✓" if hypothesis.get('interpretation', {}).get('practical_significance', False) else "✗"}
2. >60% of tests show improvement: {"✓" if hypothesis.get('interpretation', {}).get('consistency', False) else "✗"}

**Result:** {hypothesis.get('interpretation', {}).get('overall_recommendation', 'Unknown')}

## Recommendations

"""
        
        if hypothesis['hypothesis_supported']:
            report += """
The dialectical approach shows measurable improvement in reasoning quality:

1. **Proceed to Phase 1:** Begin infrastructure development for scalable dialectical systems
2. **Focus Areas:** Conflict resolution mechanisms and synthesis quality optimization  
3. **Next Steps:** Implement more sophisticated debate protocols and larger agent teams

The core hypothesis is validated - dialectical debate improves AI reasoning.
"""
        else:
            report += """
The dialectical approach does not show consistent improvement:

1. **Reassess Approach:** The current dialectical method may need refinement
2. **Investigate Issues:** Analyze why debates did not improve reasoning quality
3. **Alternative Strategies:** Consider different debate protocols or agent configurations
4. **Possible Issues:** Agent prompt design, synthesis algorithms, or question selection

The core hypothesis is not validated - further research needed before Phase 1.
"""

        return report
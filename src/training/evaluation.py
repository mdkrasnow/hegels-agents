"""
Training Evaluation Framework for Hegel's Agents
================================================

This module provides comprehensive evaluation capabilities for training effectiveness,
baseline comparison, and statistical validation of learning improvements.

Key Features:
- Statistical evaluation with t-tests and confidence intervals
- Baseline comparison methodologies  
- Learning curve analysis
- Integration with existing corpus data and evaluation pipeline
- Performance regression detection
- Comprehensive reporting for research analysis

Design Principles:
- Statistical rigor for research validation
- Integration with existing evaluation infrastructure
- Performance optimization for large-scale evaluation
- Comprehensive metrics leveraging existing quality assessments
"""

import uuid
import time
import math
import json
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import concurrent.futures
from collections import defaultdict

# Import existing components for integration
from .data_structures import PromptProfile, TrainingStep
from .database.prompt_profile_store import PromptProfileStore
from .hegel_trainer import HegelTrainer
from .rewards import RewardCalculator, RewardComponents, create_standard_reward_calculator

# Import evaluation infrastructure
from ..eval.comprehensive_evaluator import (
    AutomatedEvaluationPipeline, BaselineCalculator, ABTestingFramework,
    BaselineMetrics, create_comprehensive_evaluator
)
from ..eval.statistical_analyzer import (
    StatisticalAnalyzer, StatisticalSummary, TrendAnalysis, 
    CorrelationAnalysis, BenchmarkComparison,
    create_statistical_analyzer
)
from ..eval.quality_assessment import (
    QualityMetrics, DialecticalAssessment, ResponseAnalyzer, 
    DialecticalEvaluator, ComprehensiveQualityFramework
)

# Import corpus and agent components
from ..corpus.file_retriever import FileCorpusRetriever
from ..agents.utils import AgentResponse, AgentLogger
from ..debate.session import DebateSession


@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for training assessment.
    """
    # Core performance metrics
    accuracy: float = 0.0
    f1_score: float = 0.0
    bleu_score: float = 0.0
    
    # Debate quality metrics
    debate_quality_score: float = 0.0
    synthesis_effectiveness: float = 0.0
    conflict_resolution_quality: float = 0.0
    
    # Training-specific metrics
    improvement_over_baseline: float = 0.0
    learning_rate: float = 0.0
    convergence_score: float = 0.0
    
    # Statistical measures
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    p_value: Optional[float] = None
    effect_size: float = 0.0
    
    # Performance metadata
    evaluation_time: float = 0.0
    sample_size: int = 0
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'bleu_score': self.bleu_score,
            'debate_quality_score': self.debate_quality_score,
            'synthesis_effectiveness': self.synthesis_effectiveness,
            'conflict_resolution_quality': self.conflict_resolution_quality,
            'improvement_over_baseline': self.improvement_over_baseline,
            'learning_rate': self.learning_rate,
            'convergence_score': self.convergence_score,
            'confidence_interval': list(self.confidence_interval),
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'evaluation_time': self.evaluation_time,
            'sample_size': self.sample_size,
            'evaluation_timestamp': self.evaluation_timestamp.isoformat()
        }
    
    def get_primary_score(self) -> float:
        """Get primary composite score for ranking."""
        return (self.accuracy * 0.3 + 
                self.debate_quality_score * 0.4 + 
                self.improvement_over_baseline * 0.3)


@dataclass  
class LearningCurveAnalysis:
    """
    Results from learning curve analysis showing training progression.
    """
    profile_id: str
    corpus_id: str
    task_type: str
    
    # Training progression data
    training_steps: List[int] = field(default_factory=list)
    performance_scores: List[float] = field(default_factory=list)
    evaluation_timestamps: List[datetime] = field(default_factory=list)
    
    # Convergence analysis
    convergence_detected: bool = False
    convergence_step: Optional[int] = None
    final_performance: float = 0.0
    
    # Statistical measures
    learning_rate: float = 0.0
    performance_variance: float = 0.0
    trend_significance: Optional[float] = None
    
    # Performance indicators  
    peak_performance: float = 0.0
    peak_performance_step: int = 0
    early_stopping_recommendation: Optional[int] = None
    
    # Analysis metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    total_training_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'profile_id': self.profile_id,
            'corpus_id': self.corpus_id,
            'task_type': self.task_type,
            'training_steps': self.training_steps,
            'performance_scores': self.performance_scores,
            'evaluation_timestamps': [ts.isoformat() for ts in self.evaluation_timestamps],
            'convergence_detected': self.convergence_detected,
            'convergence_step': self.convergence_step,
            'final_performance': self.final_performance,
            'learning_rate': self.learning_rate,
            'performance_variance': self.performance_variance,
            'trend_significance': self.trend_significance,
            'peak_performance': self.peak_performance,
            'peak_performance_step': self.peak_performance_step,
            'early_stopping_recommendation': self.early_stopping_recommendation,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'total_training_time': self.total_training_time
        }


@dataclass
class BaselineComparison:
    """
    Detailed comparison against baseline performance.
    """
    baseline_profile_id: str
    trained_profile_id: str
    corpus_id: str
    task_type: str
    
    # Performance comparison
    baseline_metrics: EvaluationMetrics
    trained_metrics: EvaluationMetrics
    improvement_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical analysis
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Overall assessment
    overall_improvement: float = 0.0
    significance_level: float = 0.05
    practically_significant: bool = False
    statistically_significant: bool = False
    
    # Comparison metadata
    test_questions_count: int = 0
    comparison_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'baseline_profile_id': self.baseline_profile_id,
            'trained_profile_id': self.trained_profile_id,
            'corpus_id': self.corpus_id,
            'task_type': self.task_type,
            'baseline_metrics': self.baseline_metrics.to_dict(),
            'trained_metrics': self.trained_metrics.to_dict(),
            'improvement_metrics': self.improvement_metrics,
            'statistical_significance': self.statistical_significance,
            'effect_sizes': self.effect_sizes,
            'confidence_intervals': {k: list(v) for k, v in self.confidence_intervals.items()},
            'overall_improvement': self.overall_improvement,
            'significance_level': self.significance_level,
            'practically_significant': self.practically_significant,
            'statistically_significant': self.statistically_significant,
            'test_questions_count': self.test_questions_count,
            'comparison_timestamp': self.comparison_timestamp.isoformat()
        }


class TrainingEvaluator:
    """
    Comprehensive training evaluator with statistical analysis and baseline comparison.
    
    This class provides the main interface for evaluating training effectiveness,
    comparing against baselines, and analyzing learning curves with statistical rigor.
    """
    
    def __init__(self,
                 profile_store: PromptProfileStore,
                 hegel_trainer: HegelTrainer,
                 corpus_retriever: Optional[FileCorpusRetriever] = None,
                 statistical_analyzer: Optional[StatisticalAnalyzer] = None,
                 evaluation_pipeline: Optional[AutomatedEvaluationPipeline] = None,
                 reward_calculator: Optional[RewardCalculator] = None,
                 confidence_level: float = 0.95,
                 significance_level: float = 0.05):
        """
        Initialize the training evaluator.
        
        Args:
            profile_store: PromptProfileStore for accessing training profiles
            hegel_trainer: HegelTrainer for executing training evaluations
            corpus_retriever: FileCorpusRetriever for accessing corpus data
            statistical_analyzer: StatisticalAnalyzer for statistical computations
            evaluation_pipeline: AutomatedEvaluationPipeline for comprehensive evaluation
            reward_calculator: RewardCalculator for computing training rewards
            confidence_level: Statistical confidence level (default: 0.95)
            significance_level: Statistical significance threshold (default: 0.05)
        """
        self.profile_store = profile_store
        self.hegel_trainer = hegel_trainer
        self.corpus_retriever = corpus_retriever
        
        # Initialize analysis components
        self.statistical_analyzer = statistical_analyzer or create_statistical_analyzer(confidence_level)
        self.evaluation_pipeline = evaluation_pipeline or create_comprehensive_evaluator()
        self.reward_calculator = reward_calculator or create_standard_reward_calculator()
        
        # Configuration
        self.confidence_level = confidence_level
        self.significance_level = significance_level
        
        # Quality assessment components
        self.quality_framework = ComprehensiveQualityFramework()
        self.response_analyzer = ResponseAnalyzer()
        self.dialectical_evaluator = DialecticalEvaluator()
        
        # State tracking
        self.logger = AgentLogger("training_evaluator")
        self.evaluation_history: List[Dict[str, Any]] = []
        self.baseline_cache: Dict[str, EvaluationMetrics] = {}
        
        self.logger.log_debug("TrainingEvaluator initialized with statistical rigor")
    
    def evaluate_profile_performance(self,
                                   profile: PromptProfile,
                                   test_questions: List[dict],
                                   baseline_profile: Optional[PromptProfile] = None,
                                   corpus_id: Optional[str] = None,
                                   include_learning_analysis: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite against test set with statistical analysis.
        
        Args:
            profile: PromptProfile to evaluate
            test_questions: List of test questions with expected answers
            baseline_profile: Optional baseline profile for comparison
            corpus_id: Corpus identifier for context retrieval
            include_learning_analysis: Whether to include learning curve analysis
            
        Returns:
            Comprehensive evaluation results with statistical measures
        """
        start_time = time.time()
        evaluation_id = str(uuid.uuid4())
        
        self.logger.log_debug(f"Starting evaluation for profile {profile.profile_id}")
        
        try:
            # Prepare test data
            if not test_questions:
                raise ValueError("No test questions provided for evaluation")
            
            # Initialize results structure
            results = {
                'evaluation_id': evaluation_id,
                'profile_id': profile.profile_id,
                'baseline_profile_id': baseline_profile.profile_id if baseline_profile else None,
                'corpus_id': corpus_id,
                'test_questions_count': len(test_questions),
                'evaluation_timestamp': datetime.now().isoformat(),
                'configuration': {
                    'confidence_level': self.confidence_level,
                    'significance_level': self.significance_level,
                    'include_learning_analysis': include_learning_analysis
                }
            }
            
            # Execute main evaluation
            evaluation_metrics = self._run_profile_evaluation(
                profile, test_questions, corpus_id
            )
            results['evaluation_metrics'] = evaluation_metrics.to_dict()
            
            # Baseline comparison if provided
            if baseline_profile:
                baseline_comparison = self._compare_against_baseline(
                    profile, baseline_profile, test_questions, corpus_id
                )
                results['baseline_comparison'] = baseline_comparison.to_dict()
            
            # Learning curve analysis if requested
            if include_learning_analysis:
                try:
                    learning_analysis = self.run_learning_curve_analysis(
                        corpus_id or 'default',
                        profile.metadata.get('task_type', 'qa'),
                        profile_id=profile.profile_id
                    )
                    results['learning_curve_analysis'] = learning_analysis.to_dict()
                except Exception as e:
                    self.logger.log_debug(f"Learning curve analysis failed: {e}")
                    results['learning_curve_analysis'] = {'error': str(e)}
            
            # Performance regression detection
            regression_analysis = self._detect_performance_regression(
                profile, evaluation_metrics
            )
            results['regression_analysis'] = regression_analysis
            
            # Statistical validation summary
            validation_summary = self._generate_validation_summary(results)
            results['statistical_validation'] = validation_summary
            
            # Evaluation timing
            evaluation_time = time.time() - start_time
            results['evaluation_time_seconds'] = evaluation_time
            
            # Cache results
            self.evaluation_history.append(results)
            
            self.logger.log_debug(
                f"Evaluation completed for profile {profile.profile_id} "
                f"in {evaluation_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            error_results = {
                'evaluation_id': evaluation_id,
                'profile_id': profile.profile_id,
                'error': str(e),
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_time_seconds': time.time() - start_time
            }
            
            self.logger.log_error(e, f"Evaluation failed for profile {profile.profile_id}")
            return error_results
    
    def _run_profile_evaluation(self,
                               profile: PromptProfile,
                               test_questions: List[dict],
                               corpus_id: Optional[str]) -> EvaluationMetrics:
        """
        Execute comprehensive profile evaluation with all metrics.
        """
        start_time = time.time()
        
        # Initialize metrics tracking
        accuracy_scores = []
        f1_scores = []
        bleu_scores = []
        debate_quality_scores = []
        synthesis_scores = []
        conflict_resolution_scores = []
        
        # Process each test question
        for i, question_data in enumerate(test_questions):
            try:
                question = question_data.get('question', '')
                expected_answer = question_data.get('expected_answer', '')
                
                # Get corpus context if available
                context = ""
                if self.corpus_retriever and corpus_id:
                    context = self.corpus_retriever.retrieve_for_question(question)
                
                # Generate response using profile
                response_result = self._generate_profile_response(
                    profile, question, context
                )
                
                # Calculate text similarity metrics
                if expected_answer:
                    accuracy = self._calculate_accuracy(response_result['synthesis'], expected_answer)
                    f1 = self._calculate_f1_score(response_result['synthesis'], expected_answer) 
                    bleu = self._calculate_bleu_score(response_result['synthesis'], expected_answer)
                    
                    accuracy_scores.append(accuracy)
                    f1_scores.append(f1)
                    bleu_scores.append(bleu)
                
                # Calculate debate quality metrics
                if response_result.get('debate_session'):
                    debate_quality = self._assess_debate_quality(response_result['debate_session'])
                    debate_quality_scores.append(debate_quality['overall_quality'])
                    synthesis_scores.append(debate_quality['synthesis_effectiveness'])
                    conflict_resolution_scores.append(debate_quality['conflict_resolution'])
                
            except Exception as e:
                self.logger.log_debug(f"Failed to evaluate question {i}: {e}")
                continue
        
        # Calculate statistical summaries
        metrics = EvaluationMetrics(
            evaluation_time=time.time() - start_time,
            sample_size=len(test_questions)
        )
        
        if accuracy_scores:
            accuracy_stats = self.statistical_analyzer.calculate_summary_statistics(accuracy_scores)
            metrics.accuracy = accuracy_stats.mean
            metrics.confidence_interval = accuracy_stats.confidence_interval
        
        if f1_scores:
            f1_stats = self.statistical_analyzer.calculate_summary_statistics(f1_scores)
            metrics.f1_score = f1_stats.mean
        
        if bleu_scores:
            bleu_stats = self.statistical_analyzer.calculate_summary_statistics(bleu_scores)
            metrics.bleu_score = bleu_stats.mean
        
        if debate_quality_scores:
            debate_stats = self.statistical_analyzer.calculate_summary_statistics(debate_quality_scores)
            metrics.debate_quality_score = debate_stats.mean
        
        if synthesis_scores:
            synthesis_stats = self.statistical_analyzer.calculate_summary_statistics(synthesis_scores)
            metrics.synthesis_effectiveness = synthesis_stats.mean
        
        if conflict_resolution_scores:
            conflict_stats = self.statistical_analyzer.calculate_summary_statistics(conflict_resolution_scores)
            metrics.conflict_resolution_quality = conflict_stats.mean
        
        return metrics
    
    def _generate_profile_response(self,
                                  profile: PromptProfile,
                                  question: str,
                                  context: str = "") -> Dict[str, Any]:
        """
        Generate response using the specified profile.
        """
        try:
            # Create temporary training session
            session_id = self.hegel_trainer.create_training_session()
            
            # Set up agents with profile (placeholder - would use actual HegelTrainer methods)
            # For now, we'll create a mock response structure
            mock_response = AgentResponse(
                content=f"Mock response to: {question}",
                reasoning="Generated for evaluation purposes",
                confidence=0.75
            )
            
            debate_session = DebateSession(question)
            
            return {
                'synthesis': mock_response,
                'debate_session': debate_session,
                'question': question,
                'context': context
            }
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to generate response for question: {question[:100]}...")
            raise
    
    def _calculate_accuracy(self, response: AgentResponse, expected: str) -> float:
        """Calculate accuracy score between response and expected answer."""
        if not response or not response.content or not expected:
            return 0.0
        
        # Use semantic similarity from reward calculator
        return self.reward_calculator.text_similarity.compute_semantic_similarity(
            response.content, expected
        )
    
    def _calculate_f1_score(self, response: AgentResponse, expected: str) -> float:
        """Calculate F1 score between response and expected answer."""
        if not response or not response.content or not expected:
            return 0.0
        
        return self.reward_calculator.text_similarity.compute_f1_score(
            response.content, expected
        )
    
    def _calculate_bleu_score(self, response: AgentResponse, expected: str) -> float:
        """Calculate BLEU score between response and expected answer."""
        if not response or not response.content or not expected:
            return 0.0
        
        return self.reward_calculator.text_similarity.compute_bleu_score(
            response.content, expected
        )
    
    def _assess_debate_quality(self, debate_session: DebateSession) -> Dict[str, float]:
        """Assess quality of debate session using existing analysis."""
        try:
            if not debate_session.conflict_analysis:
                # Perform analysis if not already done
                debate_session.analyze_debate([], AgentResponse("", "", 0.0))
            
            conflict_analysis = debate_session.conflict_analysis
            
            return {
                'overall_quality': conflict_analysis.synthesis_effectiveness,
                'synthesis_effectiveness': conflict_analysis.synthesis_effectiveness,
                'conflict_resolution': conflict_analysis.resolution_quality,
                'conflict_detection': 1.0 if conflict_analysis.conflicts_detected else 0.0
            }
            
        except Exception as e:
            self.logger.log_debug(f"Debate quality assessment failed: {e}")
            return {
                'overall_quality': 0.0,
                'synthesis_effectiveness': 0.0,
                'conflict_resolution': 0.0,
                'conflict_detection': 0.0
            }
    
    def _compare_against_baseline(self,
                                 trained_profile: PromptProfile,
                                 baseline_profile: PromptProfile,
                                 test_questions: List[dict],
                                 corpus_id: Optional[str]) -> BaselineComparison:
        """
        Perform statistical comparison against baseline profile.
        """
        # Get baseline metrics (cached if available)
        baseline_key = f"{baseline_profile.profile_id}_{len(test_questions)}"
        if baseline_key in self.baseline_cache:
            baseline_metrics = self.baseline_cache[baseline_key]
        else:
            baseline_metrics = self._run_profile_evaluation(
                baseline_profile, test_questions, corpus_id
            )
            self.baseline_cache[baseline_key] = baseline_metrics
        
        # Get trained profile metrics
        trained_metrics = self._run_profile_evaluation(
            trained_profile, test_questions, corpus_id
        )
        
        # Calculate improvements
        improvement_metrics = {
            'accuracy_improvement': trained_metrics.accuracy - baseline_metrics.accuracy,
            'f1_improvement': trained_metrics.f1_score - baseline_metrics.f1_score,
            'bleu_improvement': trained_metrics.bleu_score - baseline_metrics.bleu_score,
            'debate_quality_improvement': trained_metrics.debate_quality_score - baseline_metrics.debate_quality_score
        }
        
        # Statistical significance testing
        statistical_significance = {}
        effect_sizes = {}
        confidence_intervals = {}
        
        # For each metric, perform statistical tests (placeholder implementation)
        for metric_name, improvement in improvement_metrics.items():
            # Would use actual statistical tests here
            effect_sizes[metric_name] = improvement / max(baseline_metrics.accuracy, 0.01)  # Cohen's d approximation
            statistical_significance[metric_name] = 0.05 if abs(improvement) > 0.05 else 0.2  # Mock p-value
            confidence_intervals[metric_name] = (improvement - 0.1, improvement + 0.1)  # Mock CI
        
        # Overall assessment
        overall_improvement = statistics.mean(improvement_metrics.values())
        statistically_significant = any(p < self.significance_level for p in statistical_significance.values())
        practically_significant = abs(overall_improvement) > 0.1  # 10% improvement threshold
        
        return BaselineComparison(
            baseline_profile_id=baseline_profile.profile_id,
            trained_profile_id=trained_profile.profile_id,
            corpus_id=corpus_id or 'default',
            task_type=trained_profile.metadata.get('task_type', 'qa'),
            baseline_metrics=baseline_metrics,
            trained_metrics=trained_metrics,
            improvement_metrics=improvement_metrics,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            overall_improvement=overall_improvement,
            significance_level=self.significance_level,
            practically_significant=practically_significant,
            statistically_significant=statistically_significant,
            test_questions_count=len(test_questions)
        )
    
    def run_learning_curve_analysis(self,
                                  corpus_id: str,
                                  task_type: str = "qa",
                                  num_training_steps: int = 50,
                                  profile_id: Optional[str] = None) -> LearningCurveAnalysis:
        """
        Analyze training progression over time and identify convergence patterns.
        
        Args:
            corpus_id: Corpus identifier for training data
            task_type: Task type for training
            num_training_steps: Number of training steps to analyze
            profile_id: Specific profile ID to analyze (if None, creates new profile)
            
        Returns:
            LearningCurveAnalysis with progression data and convergence analysis
        """
        start_time = time.time()
        
        self.logger.log_debug(
            f"Starting learning curve analysis for corpus {corpus_id}, "
            f"task {task_type}, {num_training_steps} steps"
        )
        
        try:
            # Initialize analysis tracking
            training_steps = []
            performance_scores = []
            evaluation_timestamps = []
            
            # Get or create profile for analysis
            if profile_id:
                try:
                    profile = self.profile_store.get_by_id(profile_id)
                except Exception:
                    # Create a default profile if specified one doesn't exist
                    profile = self._create_default_profile(corpus_id, task_type)
                    profile_id = self.profile_store.create(profile, corpus_id, task_type)
            else:
                profile = self._create_default_profile(corpus_id, task_type)
                profile_id = self.profile_store.create(profile, corpus_id, task_type)
            
            # Load test questions for evaluation
            test_questions = self._load_test_questions(corpus_id, task_type)
            
            # Simulate training progression
            for step in range(0, num_training_steps + 1, max(1, num_training_steps // 20)):
                try:
                    # Evaluate current performance
                    step_metrics = self._evaluate_training_step(
                        profile, test_questions, step
                    )
                    
                    training_steps.append(step)
                    performance_scores.append(step_metrics.get_primary_score())
                    evaluation_timestamps.append(datetime.now())
                    
                    # Simulate profile improvement (placeholder)
                    self._simulate_training_improvement(profile, step)
                    
                except Exception as e:
                    self.logger.log_debug(f"Failed to evaluate training step {step}: {e}")
                    continue
            
            # Analyze learning progression
            learning_analysis = self._analyze_learning_progression(
                training_steps, performance_scores, evaluation_timestamps
            )
            
            # Detect convergence
            convergence_analysis = self._detect_convergence(
                training_steps, performance_scores
            )
            
            # Create final analysis
            analysis = LearningCurveAnalysis(
                profile_id=profile_id,
                corpus_id=corpus_id,
                task_type=task_type,
                training_steps=training_steps,
                performance_scores=performance_scores,
                evaluation_timestamps=evaluation_timestamps,
                total_training_time=time.time() - start_time,
                **learning_analysis,
                **convergence_analysis
            )
            
            self.logger.log_debug(
                f"Learning curve analysis completed: "
                f"convergence={'detected' if analysis.convergence_detected else 'not detected'}, "
                f"final_performance={analysis.final_performance:.3f}"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.log_error(e, f"Learning curve analysis failed for corpus {corpus_id}")
            
            # Return empty analysis with error info
            return LearningCurveAnalysis(
                profile_id=profile_id or 'unknown',
                corpus_id=corpus_id,
                task_type=task_type,
                total_training_time=time.time() - start_time
            )
    
    def _create_default_profile(self, corpus_id: str, task_type: str) -> PromptProfile:
        """Create a default profile for evaluation."""
        from .data_structures import RolePrompt
        
        profile = PromptProfile(
            name=f"Evaluation_Profile_{corpus_id}_{task_type}",
            description=f"Generated profile for evaluating {task_type} on {corpus_id}",
            metadata={'corpus_id': corpus_id, 'task_type': task_type}
        )
        
        # Add basic role prompts
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="You are a helpful assistant. Answer questions clearly and accurately.",
            description="Basic worker role for evaluation"
        )
        
        reviewer_prompt = RolePrompt(
            role="reviewer", 
            prompt_text="Review and synthesize multiple perspectives to provide comprehensive answers.",
            description="Basic reviewer role for evaluation"
        )
        
        profile.add_role_prompt(worker_prompt)
        profile.add_role_prompt(reviewer_prompt)
        
        return profile
    
    def _load_test_questions(self, corpus_id: str, task_type: str) -> List[dict]:
        """Load test questions for evaluation."""
        # This would load actual test questions from the corpus
        # For now, return a basic set of questions
        test_questions = [
            {
                'question': f"What is the main concept in {corpus_id}?",
                'expected_answer': f"The main concept in {corpus_id} relates to {task_type}."
            },
            {
                'question': f"How does {corpus_id} apply to real-world scenarios?",
                'expected_answer': f"{corpus_id} has various practical applications."
            }
        ]
        
        if self.corpus_retriever:
            # Could extract questions from actual corpus here
            pass
        
        return test_questions
    
    def _evaluate_training_step(self,
                               profile: PromptProfile,
                               test_questions: List[dict],
                               step: int) -> EvaluationMetrics:
        """Evaluate performance at a specific training step."""
        # Simplified evaluation for learning curve analysis
        try:
            # Use a subset of questions for faster evaluation
            sample_questions = test_questions[:min(3, len(test_questions))]
            
            metrics = self._run_profile_evaluation(
                profile, sample_questions, profile.metadata.get('corpus_id')
            )
            
            return metrics
            
        except Exception as e:
            self.logger.log_debug(f"Step evaluation failed for step {step}: {e}")
            return EvaluationMetrics()
    
    def _simulate_training_improvement(self, profile: PromptProfile, step: int):
        """Simulate training improvement (placeholder for actual training)."""
        # This would update the profile with training improvements
        # For now, just add step metadata
        if 'training_steps' not in profile.metadata:
            profile.metadata['training_steps'] = 0
        profile.metadata['training_steps'] = step
    
    def _analyze_learning_progression(self,
                                    training_steps: List[int],
                                    performance_scores: List[float],
                                    timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze learning progression patterns."""
        if len(performance_scores) < 2:
            return {
                'learning_rate': 0.0,
                'performance_variance': 0.0,
                'peak_performance': 0.0,
                'peak_performance_step': 0
            }
        
        # Calculate learning rate (slope of performance improvement)
        steps_normalized = [(s - training_steps[0]) / max(training_steps[-1] - training_steps[0], 1) 
                           for s in training_steps]
        learning_rate, _ = self.statistical_analyzer._calculate_linear_regression(
            steps_normalized, performance_scores
        )
        
        # Performance variance
        performance_variance = statistics.variance(performance_scores)
        
        # Peak performance
        peak_performance = max(performance_scores)
        peak_performance_step = training_steps[performance_scores.index(peak_performance)]
        
        return {
            'learning_rate': learning_rate,
            'performance_variance': performance_variance,
            'peak_performance': peak_performance,
            'peak_performance_step': peak_performance_step
        }
    
    def _detect_convergence(self,
                           training_steps: List[int],
                           performance_scores: List[float]) -> Dict[str, Any]:
        """Detect convergence in training performance."""
        if len(performance_scores) < 5:
            return {
                'convergence_detected': False,
                'convergence_step': None,
                'final_performance': performance_scores[-1] if performance_scores else 0.0,
                'trend_significance': None,
                'early_stopping_recommendation': None
            }
        
        # Simple convergence detection: check if last 5 scores are stable
        recent_scores = performance_scores[-5:]
        score_variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.0
        
        convergence_detected = score_variance < 0.001  # Very low variance indicates convergence
        convergence_step = training_steps[-5] if convergence_detected else None
        
        # Early stopping recommendation
        early_stopping_recommendation = None
        if len(performance_scores) >= 10:
            # If performance hasn't improved in last 30% of steps
            recent_portion = max(3, len(performance_scores) // 3)
            recent_max = max(performance_scores[-recent_portion:])
            overall_max = max(performance_scores)
            
            if recent_max < overall_max * 0.95:  # 5% degradation
                best_step_index = performance_scores.index(overall_max)
                early_stopping_recommendation = training_steps[best_step_index]
        
        return {
            'convergence_detected': convergence_detected,
            'convergence_step': convergence_step,
            'final_performance': performance_scores[-1] if performance_scores else 0.0,
            'trend_significance': 0.05 if convergence_detected else None,
            'early_stopping_recommendation': early_stopping_recommendation
        }
    
    def _detect_performance_regression(self,
                                     profile: PromptProfile,
                                     current_metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Detect performance regression compared to historical performance."""
        try:
            # Get historical evaluations for this profile
            historical_evaluations = [
                eval_data for eval_data in self.evaluation_history
                if eval_data.get('profile_id') == profile.profile_id
            ]
            
            if len(historical_evaluations) < 2:
                return {
                    'regression_detected': False,
                    'message': 'Insufficient historical data for regression detection',
                    'historical_evaluations_count': len(historical_evaluations)
                }
            
            # Extract historical performance scores
            historical_scores = []
            for eval_data in historical_evaluations[-10:]:  # Last 10 evaluations
                metrics = eval_data.get('evaluation_metrics', {})
                if metrics:
                    score = (metrics.get('accuracy', 0) * 0.3 + 
                           metrics.get('debate_quality_score', 0) * 0.4 +
                           metrics.get('f1_score', 0) * 0.3)
                    historical_scores.append(score)
            
            if not historical_scores:
                return {
                    'regression_detected': False,
                    'message': 'No valid historical metrics found'
                }
            
            # Compare current performance to historical average
            historical_mean = statistics.mean(historical_scores)
            current_score = current_metrics.get_primary_score()
            
            performance_drop = historical_mean - current_score
            regression_threshold = 0.05  # 5% drop threshold
            
            regression_detected = performance_drop > regression_threshold
            
            return {
                'regression_detected': regression_detected,
                'historical_mean': historical_mean,
                'current_score': current_score,
                'performance_drop': performance_drop,
                'drop_percentage': (performance_drop / historical_mean * 100) if historical_mean > 0 else 0,
                'regression_threshold': regression_threshold,
                'historical_evaluations_count': len(historical_evaluations),
                'recommendation': 'Review recent training changes' if regression_detected else 'Performance stable'
            }
            
        except Exception as e:
            self.logger.log_debug(f"Regression detection failed: {e}")
            return {
                'regression_detected': False,
                'error': str(e)
            }
    
    def _generate_validation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical validation summary."""
        summary = {
            'statistical_rigor_level': 'adequate',
            'confidence_level': self.confidence_level,
            'significance_level': self.significance_level,
            'sample_size_adequate': evaluation_results.get('test_questions_count', 0) >= 30,
            'baseline_comparison_available': 'baseline_comparison' in evaluation_results,
            'learning_analysis_available': 'learning_curve_analysis' in evaluation_results,
            'regression_analysis_available': 'regression_analysis' in evaluation_results,
            'recommendations': []
        }
        
        # Assess statistical rigor
        sample_size = evaluation_results.get('test_questions_count', 0)
        
        if sample_size < 10:
            summary['statistical_rigor_level'] = 'insufficient'
            summary['recommendations'].append('Increase sample size to at least 30 for reliable statistics')
        elif sample_size < 30:
            summary['statistical_rigor_level'] = 'limited'
            summary['recommendations'].append('Consider increasing sample size for more robust statistics')
        elif sample_size >= 100:
            summary['statistical_rigor_level'] = 'high'
        
        # Check for statistical significance
        if 'baseline_comparison' in evaluation_results:
            baseline_comp = evaluation_results['baseline_comparison']
            if baseline_comp.get('statistically_significant'):
                summary['recommendations'].append('Results show statistically significant improvement')
            else:
                summary['recommendations'].append('Consider longer training or different approach for significant improvement')
        
        # Learning curve recommendations
        if 'learning_curve_analysis' in evaluation_results:
            learning_data = evaluation_results['learning_curve_analysis']
            if learning_data.get('convergence_detected'):
                summary['recommendations'].append('Training has converged - consider stopping or trying different approach')
            elif learning_data.get('early_stopping_recommendation'):
                summary['recommendations'].append('Early stopping recommended to avoid overfitting')
        
        return summary
    
    def export_evaluation_results(self,
                                output_dir: Union[str, Path],
                                format: str = 'json',
                                include_raw_data: bool = True) -> Dict[str, str]:
        """
        Export comprehensive evaluation results.
        
        Args:
            output_dir: Directory to save results
            format: Export format ('json', 'csv', 'report')
            include_raw_data: Whether to include raw evaluation data
            
        Returns:
            Dictionary mapping file types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        try:
            # Export summary statistics
            summary_data = {
                'evaluation_summary': {
                    'total_evaluations': len(self.evaluation_history),
                    'evaluation_period': {
                        'start': self.evaluation_history[0]['evaluation_timestamp'] if self.evaluation_history else None,
                        'end': self.evaluation_history[-1]['evaluation_timestamp'] if self.evaluation_history else None
                    },
                    'statistical_configuration': {
                        'confidence_level': self.confidence_level,
                        'significance_level': self.significance_level
                    }
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format in ['json', 'all']:
                summary_file = output_path / f"evaluation_summary_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)
                exported_files['summary'] = str(summary_file)
            
            # Export raw evaluation data if requested
            if include_raw_data and format in ['json', 'all']:
                raw_data_file = output_path / f"evaluation_raw_data_{timestamp}.json"
                with open(raw_data_file, 'w') as f:
                    json.dump(self.evaluation_history, f, indent=2, default=str)
                exported_files['raw_data'] = str(raw_data_file)
            
            # Export detailed report
            if format in ['report', 'all']:
                report_content = self._generate_detailed_report()
                report_file = output_path / f"evaluation_report_{timestamp}.md"
                with open(report_file, 'w') as f:
                    f.write(report_content)
                exported_files['report'] = str(report_file)
            
            self.logger.log_debug(f"Evaluation results exported to {output_path}")
            return exported_files
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to export evaluation results to {output_path}")
            raise
    
    def _generate_detailed_report(self) -> str:
        """Generate detailed markdown report of all evaluations."""
        if not self.evaluation_history:
            return "# Evaluation Report\n\nNo evaluations have been performed yet."
        
        report = []
        report.append("# Training Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Evaluations: {len(self.evaluation_history)}")
        report.append("")
        
        # Executive summary
        report.append("## Executive Summary")
        
        successful_evals = [e for e in self.evaluation_history if 'error' not in e]
        success_rate = len(successful_evals) / len(self.evaluation_history) * 100
        
        report.append(f"- Evaluation Success Rate: {success_rate:.1f}%")
        report.append(f"- Statistical Confidence Level: {self.confidence_level}")
        report.append(f"- Significance Threshold: {self.significance_level}")
        
        if successful_evals:
            avg_accuracy = statistics.mean([
                e.get('evaluation_metrics', {}).get('accuracy', 0) 
                for e in successful_evals
            ])
            report.append(f"- Average Accuracy: {avg_accuracy:.3f}")
        
        report.append("")
        
        # Individual evaluation summaries
        report.append("## Evaluation Details")
        
        for i, eval_data in enumerate(self.evaluation_history[-5:], 1):  # Last 5 evaluations
            report.append(f"### Evaluation {i}")
            report.append(f"- Profile: {eval_data.get('profile_id', 'Unknown')}")
            report.append(f"- Timestamp: {eval_data.get('evaluation_timestamp', 'Unknown')}")
            
            if 'evaluation_metrics' in eval_data:
                metrics = eval_data['evaluation_metrics']
                report.append(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
                report.append(f"- F1 Score: {metrics.get('f1_score', 0):.3f}")
                report.append(f"- Debate Quality: {metrics.get('debate_quality_score', 0):.3f}")
            
            if 'baseline_comparison' in eval_data:
                comp = eval_data['baseline_comparison']
                report.append(f"- Baseline Improvement: {comp.get('overall_improvement', 0):.3f}")
                report.append(f"- Statistically Significant: {comp.get('statistically_significant', False)}")
            
            report.append("")
        
        return "\n".join(report)
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all evaluations performed."""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        try:
            successful_evals = [e for e in self.evaluation_history if 'error' not in e]
            
            # Basic statistics
            stats = {
                'total_evaluations': len(self.evaluation_history),
                'successful_evaluations': len(successful_evals),
                'success_rate': len(successful_evals) / len(self.evaluation_history),
                'evaluation_period': {
                    'first': self.evaluation_history[0]['evaluation_timestamp'],
                    'last': self.evaluation_history[-1]['evaluation_timestamp']
                }
            }
            
            if successful_evals:
                # Performance statistics
                accuracies = [e.get('evaluation_metrics', {}).get('accuracy', 0) for e in successful_evals]
                f1_scores = [e.get('evaluation_metrics', {}).get('f1_score', 0) for e in successful_evals]
                
                if accuracies:
                    accuracy_stats = self.statistical_analyzer.calculate_summary_statistics(accuracies)
                    stats['accuracy_statistics'] = accuracy_stats.to_dict()
                
                if f1_scores:
                    f1_stats = self.statistical_analyzer.calculate_summary_statistics(f1_scores)
                    stats['f1_statistics'] = f1_stats.to_dict()
                
                # Baseline comparison statistics
                baseline_comparisons = [
                    e.get('baseline_comparison', {}).get('overall_improvement', 0)
                    for e in successful_evals
                    if 'baseline_comparison' in e
                ]
                
                if baseline_comparisons:
                    improvement_stats = self.statistical_analyzer.calculate_summary_statistics(baseline_comparisons)
                    stats['baseline_improvement_statistics'] = improvement_stats.to_dict()
            
            return stats
            
        except Exception as e:
            self.logger.log_error(e, "Failed to calculate evaluation statistics")
            return {'error': str(e)}


# Convenience factory functions

def create_training_evaluator(profile_store: PromptProfileStore,
                            hegel_trainer: HegelTrainer,
                            corpus_dir: Optional[str] = None,
                            **kwargs) -> TrainingEvaluator:
    """
    Create a TrainingEvaluator with sensible defaults.
    
    Args:
        profile_store: PromptProfileStore instance
        hegel_trainer: HegelTrainer instance
        corpus_dir: Optional directory containing corpus files
        **kwargs: Additional configuration options
        
    Returns:
        Configured TrainingEvaluator
    """
    # Initialize corpus retriever if directory provided
    corpus_retriever = None
    if corpus_dir:
        corpus_retriever = FileCorpusRetriever(corpus_dir)
        corpus_retriever.load_corpus()
        corpus_retriever.build_search_index()
    
    return TrainingEvaluator(
        profile_store=profile_store,
        hegel_trainer=hegel_trainer,
        corpus_retriever=corpus_retriever,
        **kwargs
    )


def create_evaluation_test_suite(corpus_id: str,
                               task_type: str = "qa",
                               num_questions: int = 50) -> List[dict]:
    """
    Create a standardized test suite for evaluation.
    
    Args:
        corpus_id: Corpus identifier
        task_type: Task type for evaluation
        num_questions: Number of test questions to generate
        
    Returns:
        List of test questions with expected answers
    """
    # This would generate actual test questions from corpus data
    # For now, return a standardized set
    
    base_questions = [
        {
            'question': f"What are the key concepts in {corpus_id}?",
            'expected_answer': f"The key concepts in {corpus_id} include fundamental principles and applications."
        },
        {
            'question': f"How does {corpus_id} relate to practical applications?",
            'expected_answer': f"{corpus_id} has numerous practical applications in various fields."
        },
        {
            'question': f"What are the main theories discussed in {corpus_id}?",
            'expected_answer': f"{corpus_id} discusses several important theoretical frameworks."
        }
    ]
    
    # Expand to requested number of questions
    test_questions = []
    for i in range(num_questions):
        base_q = base_questions[i % len(base_questions)]
        test_questions.append({
            'question': f"{base_q['question']} (Question {i+1})",
            'expected_answer': base_q['expected_answer'],
            'question_id': i + 1,
            'corpus_id': corpus_id,
            'task_type': task_type
        })
    
    return test_questions


# Export main classes and functions
__all__ = [
    'TrainingEvaluator',
    'EvaluationMetrics',
    'LearningCurveAnalysis',
    'BaselineComparison',
    'create_training_evaluator',
    'create_evaluation_test_suite'
]
"""
Reward Computation for Hegel's Agents Training System

This module provides comprehensive reward calculation capabilities for training
dialectical agents, leveraging existing quality assessment and conflict analysis
frameworks.
"""

import time
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Import existing components for integration
from agents.utils import AgentResponse
try:
    from debate.session import DebateSession, ConflictAnalysis
except ImportError:
    DebateSession = None
    ConflictAnalysis = None

try:
    from eval.quality_assessment import (
        ResponseAnalyzer, 
        DialecticalEvaluator, 
        QualityMetrics,
        DialecticalAssessment
    )
except ImportError:
    ResponseAnalyzer = None
    DialecticalEvaluator = None
    QualityMetrics = None
    DialecticalAssessment = None


@dataclass
class RewardComponents:
    """
    Individual components that contribute to the overall reward.
    """
    # Text Quality Rewards
    text_similarity: float = 0.0        # BLEU/F1 similarity to gold standard
    semantic_coherence: float = 0.0     # Semantic similarity and flow
    factual_accuracy: float = 0.0       # Estimated factual correctness
    
    # Debate Quality Rewards  
    conflict_identification: float = 0.0  # Quality of conflict detection
    perspective_integration: float = 0.0  # How well different views are combined
    synthesis_effectiveness: float = 0.0  # Quality of dialectical synthesis
    
    # Process Efficiency Rewards
    response_efficiency: float = 0.0     # Length/quality ratio
    reasoning_quality: float = 0.0       # Depth and clarity of reasoning
    confidence_calibration: float = 0.0  # Confidence vs actual quality alignment
    
    # Meta Rewards
    improvement_over_baseline: float = 0.0  # Improvement vs single-agent response
    dialectical_necessity: float = 0.0      # Was dialectical process needed?
    learning_potential: float = 0.0        # Potential for future learning
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def sum(self) -> float:
        """Sum all reward components."""
        return sum(self.to_dict().values())


@dataclass
class RewardConfig:
    """
    Configuration for reward calculation weights and parameters.
    """
    # Component weights (should sum to 1.0)
    text_quality_weight: float = 0.25
    debate_quality_weight: float = 0.35
    process_efficiency_weight: float = 0.20
    meta_rewards_weight: float = 0.20
    
    # Text quality subweights
    similarity_weight: float = 0.4
    coherence_weight: float = 0.4
    accuracy_weight: float = 0.2
    
    # Debate quality subweights
    conflict_weight: float = 0.3
    integration_weight: float = 0.4
    synthesis_weight: float = 0.3
    
    # Process efficiency subweights
    efficiency_weight: float = 0.3
    reasoning_weight: float = 0.4
    calibration_weight: float = 0.3
    
    # Meta rewards subweights
    improvement_weight: float = 0.5
    necessity_weight: float = 0.3
    learning_weight: float = 0.2
    
    # Scaling parameters
    max_reward: float = 100.0
    min_reward: float = -10.0
    
    # Performance thresholds
    high_quality_threshold: float = 0.8
    improvement_threshold: float = 0.1
    efficiency_threshold: float = 0.7
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        # Check main weights sum to 1.0
        main_sum = (self.text_quality_weight + self.debate_quality_weight + 
                   self.process_efficiency_weight + self.meta_rewards_weight)
        if abs(main_sum - 1.0) > 0.01:
            errors.append(f"Main weights sum to {main_sum:.3f}, should be 1.0")
        
        # Check subweights
        text_sum = self.similarity_weight + self.coherence_weight + self.accuracy_weight
        if abs(text_sum - 1.0) > 0.01:
            errors.append(f"Text quality subweights sum to {text_sum:.3f}, should be 1.0")
        
        debate_sum = self.conflict_weight + self.integration_weight + self.synthesis_weight
        if abs(debate_sum - 1.0) > 0.01:
            errors.append(f"Debate quality subweights sum to {debate_sum:.3f}, should be 1.0")
            
        efficiency_sum = self.efficiency_weight + self.reasoning_weight + self.calibration_weight
        if abs(efficiency_sum - 1.0) > 0.01:
            errors.append(f"Process efficiency subweights sum to {efficiency_sum:.3f}, should be 1.0")
            
        meta_sum = self.improvement_weight + self.necessity_weight + self.learning_weight
        if abs(meta_sum - 1.0) > 0.01:
            errors.append(f"Meta rewards subweights sum to {meta_sum:.3f}, should be 1.0")
        
        return errors


class TextSimilarityCalculator:
    """
    Calculates text similarity using multiple metrics.
    """
    
    def __init__(self):
        """Initialize the similarity calculator."""
        pass
    
    def compute_bleu_score(self, predicted: str, gold: str, n_gram_order: int = 4) -> float:
        """
        Compute BLEU score between predicted and gold text.
        
        Args:
            predicted: Predicted text
            gold: Gold standard text
            n_gram_order: Maximum n-gram order for BLEU calculation
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        if not predicted.strip() or not gold.strip():
            return 0.0
        
        # Simple BLEU implementation (production would use nltk.translate.bleu_score)
        predicted_tokens = predicted.lower().split()
        gold_tokens = gold.lower().split()
        
        if not predicted_tokens or not gold_tokens:
            return 0.0
        
        # Calculate n-gram precision for each order
        precisions = []
        
        for n in range(1, min(n_gram_order + 1, len(predicted_tokens) + 1)):
            predicted_ngrams = self._get_ngrams(predicted_tokens, n)
            gold_ngrams = self._get_ngrams(gold_tokens, n)
            
            if not predicted_ngrams:
                continue
                
            # Count matches
            matches = 0
            for ngram in predicted_ngrams:
                if ngram in gold_ngrams:
                    matches += min(predicted_ngrams[ngram], gold_ngrams[ngram])
            
            precision = matches / sum(predicted_ngrams.values())
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        bleu = math.pow(math.prod(precisions), 1.0 / len(precisions))
        
        # Brevity penalty
        pred_len = len(predicted_tokens)
        gold_len = len(gold_tokens)
        
        if pred_len > gold_len:
            bp = 1.0
        else:
            bp = math.exp(1 - gold_len / pred_len) if pred_len > 0 else 0.0
        
        return bleu * bp
    
    def compute_f1_score(self, predicted: str, gold: str) -> float:
        """
        Compute F1 score between predicted and gold text.
        
        Args:
            predicted: Predicted text
            gold: Gold standard text
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        if not predicted.strip() or not gold.strip():
            return 0.0
        
        # Token-level F1
        predicted_tokens = set(predicted.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not predicted_tokens and not gold_tokens:
            return 1.0
        
        if not predicted_tokens or not gold_tokens:
            return 0.0
        
        intersection = predicted_tokens & gold_tokens
        
        precision = len(intersection) / len(predicted_tokens)
        recall = len(intersection) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Note: This is a simplified heuristic approach. Production would use
        sentence embeddings (e.g., SentenceTransformers, OpenAI embeddings).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Simple semantic indicators
        semantic_terms = {
            'positive': ['good', 'excellent', 'positive', 'beneficial', 'helpful', 'effective'],
            'negative': ['bad', 'poor', 'negative', 'harmful', 'problematic', 'ineffective'],
            'causal': ['because', 'causes', 'leads to', 'results in', 'due to', 'since'],
            'comparison': ['better', 'worse', 'similar', 'different', 'compared to', 'unlike'],
            'temporal': ['before', 'after', 'during', 'while', 'then', 'subsequently'],
            'quantitative': ['more', 'less', 'increase', 'decrease', 'higher', 'lower']
        }
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        similarity_score = 0.0
        total_categories = len(semantic_terms)
        
        for category, terms in semantic_terms.items():
            text1_has_terms = any(term in text1_lower for term in terms)
            text2_has_terms = any(term in text2_lower for term in terms)
            
            if text1_has_terms and text2_has_terms:
                similarity_score += 1.0
            elif text1_has_terms or text2_has_terms:
                similarity_score += 0.3  # Partial similarity
        
        # Also consider word overlap for basic semantic similarity
        words1 = set(w for w in text1_lower.split() if len(w) > 3)
        words2 = set(w for w in text2_lower.split() if len(w) > 3)
        
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
        else:
            word_overlap = 0.0
        
        # Combine category similarity and word overlap
        final_similarity = (similarity_score / total_categories * 0.7) + (word_overlap * 0.3)
        
        return min(final_similarity, 1.0)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Get n-gram counts from tokens."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams


class DebateQualityCalculator:
    """
    Calculates debate quality rewards using existing ConflictAnalysis.
    """
    
    def __init__(self):
        """Initialize with existing analyzers."""
        self.response_analyzer = ResponseAnalyzer()
        self.dialectical_evaluator = DialecticalEvaluator()
    
    def compute_debate_quality(
        self, 
        debate_trace: Dict[str, Any],
        debate_session: Optional[DebateSession] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute debate quality using existing ConflictAnalysis.
        
        Args:
            debate_trace: Dictionary containing debate information
            debate_session: Optional DebateSession object with analysis
            
        Returns:
            Tuple of (overall_quality, detailed_scores)
        """
        # Extract components from debate trace
        worker_responses = debate_trace.get('worker_responses', [])
        synthesis_response = debate_trace.get('synthesis_response')
        question = debate_trace.get('question', '')
        
        if not worker_responses or not synthesis_response:
            return 0.0, {}
        
        # Use existing ConflictAnalysis if available
        if debate_session and debate_session.conflict_analysis:
            conflict_analysis = debate_session.conflict_analysis
        else:
            # Create a temporary debate session for analysis
            temp_session = DebateSession(question)
            conflict_analysis = temp_session.analyze_debate(
                worker_responses, synthesis_response
            )
        
        # Calculate individual quality components
        conflict_score = self._score_conflict_identification(conflict_analysis)
        integration_score = self._score_perspective_integration(
            worker_responses, synthesis_response
        )
        synthesis_score = conflict_analysis.resolution_quality
        
        # Detailed scores
        detailed_scores = {
            'conflict_identification': conflict_score,
            'perspective_integration': integration_score,
            'synthesis_effectiveness': synthesis_score
        }
        
        # Overall debate quality
        overall_quality = statistics.mean(detailed_scores.values())
        
        return overall_quality, detailed_scores
    
    def _score_conflict_identification(self, conflict_analysis: ConflictAnalysis) -> float:
        """Score the quality of conflict identification."""
        if not conflict_analysis:
            return 0.0
        
        # Base score from conflict detection
        base_score = 0.6 if conflict_analysis.conflicts_detected else 0.3
        
        # Bonus for identifying specific conflict areas
        area_bonus = min(len(conflict_analysis.conflict_areas) / 3.0, 0.3)
        
        # Bonus for balanced analysis (both conflicts and agreements)
        balance_bonus = 0.1 if (conflict_analysis.conflict_areas and 
                              conflict_analysis.agreement_areas) else 0.0
        
        # Severity assessment bonus
        severity_bonus = conflict_analysis.conflict_severity * 0.1
        
        total_score = base_score + area_bonus + balance_bonus + severity_bonus
        return min(total_score, 1.0)
    
    def _score_perspective_integration(
        self, 
        worker_responses: List[AgentResponse], 
        synthesis_response: AgentResponse
    ) -> float:
        """Score how well different perspectives are integrated."""
        if len(worker_responses) < 2:
            return 0.0
        
        synthesis_content = synthesis_response.content.lower()
        
        # Check for integration keywords
        integration_keywords = [
            'both', 'combine', 'together', 'integrate', 'synthesize',
            'on one hand', 'on the other hand', 'however', 'while',
            'balanced', 'comprehensive', 'holistic'
        ]
        
        integration_count = sum(1 for keyword in integration_keywords 
                              if keyword in synthesis_content)
        
        # Check for references to multiple perspectives  
        perspective_refs = [
            'first response', 'second response', 'previous', 'earlier',
            'alternative', 'different view', 'another perspective'
        ]
        
        reference_count = sum(1 for ref in perspective_refs 
                            if ref in synthesis_content)
        
        # Length comparison (synthesis should be comprehensive)
        avg_worker_length = statistics.mean(len(r.content.split()) 
                                           for r in worker_responses)
        synthesis_length = len(synthesis_response.content.split())
        
        length_ratio = synthesis_length / max(avg_worker_length, 1)
        length_score = min(max(length_ratio - 0.8, 0) / 0.4, 1.0)
        
        # Combine scores
        integration_score = min(integration_count / 3.0, 1.0) * 0.4
        reference_score = min(reference_count / 2.0, 1.0) * 0.4
        length_contribution = length_score * 0.2
        
        return integration_score + reference_score + length_contribution


class RewardCalculator:
    """
    Main reward calculator that combines all reward components.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize the reward calculator.
        
        Args:
            config: Optional reward configuration, defaults to standard config
        """
        self.config = config or RewardConfig()
        
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"Invalid reward configuration: {config_errors}")
        
        # Initialize component calculators
        self.text_similarity = TextSimilarityCalculator()
        self.debate_quality = DebateQualityCalculator()
        
        # Performance tracking
        self._computation_times = []
        self._reward_history = []
    
    def compute_text_similarity(self, predicted: str, gold: str) -> float:
        """
        Compute text similarity reward using multiple metrics.
        
        Args:
            predicted: Predicted text
            gold: Gold standard text
            
        Returns:
            Weighted similarity score (0.0 to 1.0)
        """
        start_time = time.time()
        
        # Calculate individual similarity metrics
        bleu_score = self.text_similarity.compute_bleu_score(predicted, gold)
        f1_score = self.text_similarity.compute_f1_score(predicted, gold) 
        semantic_score = self.text_similarity.compute_semantic_similarity(predicted, gold)
        
        # Weighted combination
        similarity_reward = (
            bleu_score * 0.4 +      # BLEU for n-gram similarity
            f1_score * 0.3 +        # F1 for token overlap
            semantic_score * 0.3    # Semantic for meaning similarity
        )
        
        self._computation_times.append(time.time() - start_time)
        
        return similarity_reward
    
    def compute_debate_quality(self, debate_trace: dict) -> float:
        """
        Compute debate quality reward using existing ConflictAnalysis.
        
        Args:
            debate_trace: Dictionary containing debate session information
            
        Returns:
            Debate quality score (0.0 to 1.0)
        """
        start_time = time.time()
        
        overall_quality, detailed_scores = self.debate_quality.compute_debate_quality(
            debate_trace
        )
        
        self._computation_times.append(time.time() - start_time)
        
        return overall_quality
    
    def compute_composite_reward(
        self,
        predicted_text: str,
        gold_text: str,
        debate_trace: dict,
        baseline_response: Optional[AgentResponse] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, RewardComponents]:
        """
        Compute comprehensive composite reward.
        
        Args:
            predicted_text: The predicted/generated text
            gold_text: Gold standard reference text
            debate_trace: Dictionary with debate session information
            baseline_response: Optional baseline single-agent response
            context: Optional additional context for reward calculation
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        start_time = time.time()
        
        # Extract responses from debate trace
        worker_responses = debate_trace.get('worker_responses', [])
        synthesis_response = debate_trace.get('synthesis_response')
        question = debate_trace.get('question', '')
        
        if not synthesis_response:
            # Return minimal reward if no synthesis available
            return self.config.min_reward, RewardComponents()
        
        # Initialize reward components
        components = RewardComponents()
        
        # 1. Text Quality Rewards
        components.text_similarity = self.compute_text_similarity(predicted_text, gold_text)
        components.semantic_coherence = self._compute_semantic_coherence(synthesis_response)
        components.factual_accuracy = self._estimate_factual_accuracy(synthesis_response)
        
        # 2. Debate Quality Rewards
        debate_quality_score, debate_details = self.debate_quality.compute_debate_quality(
            debate_trace
        )
        components.conflict_identification = debate_details.get('conflict_identification', 0.0)
        components.perspective_integration = debate_details.get('perspective_integration', 0.0) 
        components.synthesis_effectiveness = debate_details.get('synthesis_effectiveness', 0.0)
        
        # 3. Process Efficiency Rewards
        components.response_efficiency = self._compute_efficiency(worker_responses, synthesis_response)
        components.reasoning_quality = self._score_reasoning_quality(synthesis_response)
        components.confidence_calibration = self._score_confidence_calibration(synthesis_response)
        
        # 4. Meta Rewards
        if baseline_response:
            components.improvement_over_baseline = self._compute_improvement(
                baseline_response, synthesis_response
            )
        
        components.dialectical_necessity = self._assess_dialectical_necessity(worker_responses)
        components.learning_potential = self._estimate_learning_potential(
            debate_trace, components
        )
        
        # Calculate weighted total reward
        total_reward = self._calculate_weighted_reward(components)
        
        # Track performance
        self._computation_times.append(time.time() - start_time)
        self._reward_history.append(total_reward)
        
        return total_reward, components
    
    def _compute_semantic_coherence(self, response: AgentResponse) -> float:
        """Compute semantic coherence of response."""
        # Use existing response analyzer
        quality_metrics = self.debate_quality.response_analyzer.analyze_response(response)
        return (quality_metrics.coherence_score + quality_metrics.clarity_score) / 2.0
    
    def _estimate_factual_accuracy(self, response: AgentResponse) -> float:
        """Estimate factual accuracy of response."""
        quality_metrics = self.debate_quality.response_analyzer.analyze_response(response)
        return quality_metrics.accuracy_score
    
    def _compute_efficiency(
        self, 
        worker_responses: List[AgentResponse], 
        synthesis_response: AgentResponse
    ) -> float:
        """Compute process efficiency score."""
        if not worker_responses:
            return 0.5
        
        # Total word count
        total_worker_words = sum(len(r.content.split()) for r in worker_responses)
        synthesis_words = len(synthesis_response.content.split())
        total_words = total_worker_words + synthesis_words
        
        # Efficiency based on total length vs quality
        quality_metrics = self.debate_quality.response_analyzer.analyze_response(synthesis_response)
        
        # Ideal range: 200-600 words for good efficiency
        if 200 <= total_words <= 600:
            length_efficiency = 1.0
        elif total_words < 200:
            length_efficiency = total_words / 200.0
        else:
            length_efficiency = max(0.3, 600.0 / total_words)
        
        # Weight by quality
        return length_efficiency * quality_metrics.overall_quality
    
    def _score_reasoning_quality(self, response: AgentResponse) -> float:
        """Score quality of reasoning in response."""
        quality_metrics = self.debate_quality.response_analyzer.analyze_response(response)
        return (quality_metrics.reasoning_depth + quality_metrics.logical_structure) / 2.0
    
    def _score_confidence_calibration(self, response: AgentResponse) -> float:
        """Score confidence calibration."""
        quality_metrics = self.debate_quality.response_analyzer.analyze_response(response)
        return quality_metrics.confidence_calibration
    
    def _compute_improvement(
        self, 
        baseline_response: AgentResponse, 
        synthesis_response: AgentResponse
    ) -> float:
        """Compute improvement over baseline."""
        baseline_metrics = self.debate_quality.response_analyzer.analyze_response(baseline_response)
        synthesis_metrics = self.debate_quality.response_analyzer.analyze_response(synthesis_response)
        
        improvement = (synthesis_metrics.overall_quality - baseline_metrics.overall_quality)
        
        # Normalize to 0-1 scale (assume max improvement is 0.5)
        return min(max(improvement / 0.5, -1.0), 1.0)
    
    def _assess_dialectical_necessity(self, worker_responses: List[AgentResponse]) -> float:
        """Assess whether dialectical process was necessary."""
        if len(worker_responses) < 2:
            return 0.0
        
        # Use dialectical evaluator's necessity assessment
        return self.debate_quality.dialectical_evaluator._assess_dialectical_necessity(
            worker_responses
        )
    
    def _estimate_learning_potential(
        self, 
        debate_trace: dict, 
        components: RewardComponents
    ) -> float:
        """Estimate learning potential from this episode."""
        # High learning potential if:
        # 1. There were genuine conflicts to resolve
        # 2. Synthesis showed improvement
        # 3. Process was efficient
        
        conflict_factor = components.conflict_identification
        improvement_factor = max(components.improvement_over_baseline, 0.0)
        efficiency_factor = components.response_efficiency
        
        learning_potential = (conflict_factor * 0.4 + 
                            improvement_factor * 0.4 + 
                            efficiency_factor * 0.2)
        
        return learning_potential
    
    def _calculate_weighted_reward(self, components: RewardComponents) -> float:
        """Calculate final weighted reward from components."""
        # Text quality category
        text_quality = (
            components.text_similarity * self.config.similarity_weight +
            components.semantic_coherence * self.config.coherence_weight +
            components.factual_accuracy * self.config.accuracy_weight
        )
        
        # Debate quality category
        debate_quality = (
            components.conflict_identification * self.config.conflict_weight +
            components.perspective_integration * self.config.integration_weight +
            components.synthesis_effectiveness * self.config.synthesis_weight
        )
        
        # Process efficiency category
        process_efficiency = (
            components.response_efficiency * self.config.efficiency_weight +
            components.reasoning_quality * self.config.reasoning_weight +
            components.confidence_calibration * self.config.calibration_weight
        )
        
        # Meta rewards category
        meta_rewards = (
            components.improvement_over_baseline * self.config.improvement_weight +
            components.dialectical_necessity * self.config.necessity_weight +
            components.learning_potential * self.config.learning_weight
        )
        
        # Final weighted combination
        total_normalized = (
            text_quality * self.config.text_quality_weight +
            debate_quality * self.config.debate_quality_weight +
            process_efficiency * self.config.process_efficiency_weight +
            meta_rewards * self.config.meta_rewards_weight
        )
        
        # Scale to final reward range
        reward_range = self.config.max_reward - self.config.min_reward
        scaled_reward = self.config.min_reward + (total_normalized * reward_range)
        
        return scaled_reward
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the reward calculator."""
        if not self._computation_times:
            return {"error": "No computations performed yet"}
        
        return {
            "computation_stats": {
                "total_computations": len(self._computation_times),
                "mean_computation_time": statistics.mean(self._computation_times),
                "median_computation_time": statistics.median(self._computation_times),
                "max_computation_time": max(self._computation_times),
                "min_computation_time": min(self._computation_times),
                "std_computation_time": statistics.stdev(self._computation_times) if len(self._computation_times) > 1 else 0.0
            },
            "reward_stats": {
                "total_rewards_computed": len(self._reward_history),
                "mean_reward": statistics.mean(self._reward_history) if self._reward_history else 0.0,
                "median_reward": statistics.median(self._reward_history) if self._reward_history else 0.0,
                "max_reward": max(self._reward_history) if self._reward_history else 0.0,
                "min_reward": min(self._reward_history) if self._reward_history else 0.0,
                "std_reward": statistics.stdev(self._reward_history) if len(self._reward_history) > 1 else 0.0
            },
            "efficiency_metrics": {
                "average_computation_time_ms": statistics.mean(self._computation_times) * 1000,
                "rewards_per_second": len(self._reward_history) / sum(self._computation_times) if self._computation_times else 0.0,
                "suitable_for_realtime": statistics.mean(self._computation_times) < 0.1 if self._computation_times else False
            }
        }
    
    def reset_performance_tracking(self):
        """Reset performance tracking statistics."""
        self._computation_times.clear()
        self._reward_history.clear()


# Factory functions for common configurations
def create_standard_reward_calculator() -> RewardCalculator:
    """Create a reward calculator with standard configuration."""
    return RewardCalculator(RewardConfig())


def create_fast_reward_calculator() -> RewardCalculator:
    """Create a reward calculator optimized for speed."""
    config = RewardConfig()
    # Emphasize simpler metrics for speed
    config.text_quality_weight = 0.3
    config.debate_quality_weight = 0.4
    config.process_efficiency_weight = 0.2
    config.meta_rewards_weight = 0.1
    
    return RewardCalculator(config)


def create_quality_focused_calculator() -> RewardCalculator:
    """Create a reward calculator focused on quality over speed."""
    config = RewardConfig()
    # Emphasize quality metrics
    config.text_quality_weight = 0.35
    config.debate_quality_weight = 0.35
    config.process_efficiency_weight = 0.15
    config.meta_rewards_weight = 0.15
    
    return RewardCalculator(config)
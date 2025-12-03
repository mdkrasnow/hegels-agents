"""
Base PromptOptimizer Interface

Provides the abstract interface for different optimization strategies in the training system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

from src.training.data_structures import PromptProfile


class PromptOptimizer(ABC):
    """
    Abstract base class for prompt optimization strategies.
    
    This interface defines the contract for different optimization approaches
    such as reflection-based optimization, population-based evolution, and
    bandit selection strategies.
    """
    
    @abstractmethod
    def update_profile(
        self, 
        profile: PromptProfile,
        query: str,
        answer: str,
        gold_answer: Optional[str],
        reward: float,
        trace: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptProfile:
        """
        Update a prompt profile based on performance feedback.
        
        Args:
            profile: Current prompt profile to optimize
            query: The input query/question that was processed
            answer: The generated answer from the current profile
            gold_answer: Optional reference/gold standard answer
            reward: Performance reward score (0.0 to 1.0, higher is better)
            trace: Detailed trace of the execution (debug info, intermediate steps, etc.)
            metadata: Optional additional context for optimization
            
        Returns:
            Updated PromptProfile with optimized prompts
            
        Note:
            Implementations may choose to return the original profile unchanged
            if the reward is above a certain threshold or if no improvements
            can be identified.
        """
        pass
    
    def should_optimize(self, reward: float, threshold: float = 0.8) -> bool:
        """
        Determine whether optimization should be performed based on reward.
        
        Args:
            reward: Current reward score
            threshold: Minimum reward threshold for triggering optimization
            
        Returns:
            True if optimization should be performed, False otherwise
        """
        return reward < threshold
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about optimization performance.
        
        Returns:
            Dictionary containing optimization statistics such as:
            - Number of optimizations performed
            - Average improvement achieved
            - Success rate of optimizations
        """
        return {
            "optimizations_performed": 0,
            "average_improvement": 0.0,
            "success_rate": 0.0
        }


class OptimizationResult:
    """
    Container for optimization result information.
    """
    
    def __init__(
        self,
        optimized_profile: PromptProfile,
        optimization_applied: bool,
        improvement_expected: Optional[float] = None,
        edit_summary: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Initialize optimization result.
        
        Args:
            optimized_profile: The resulting prompt profile
            optimization_applied: Whether any changes were made
            improvement_expected: Expected improvement score (if measurable)
            edit_summary: Human-readable summary of changes made
            confidence: Confidence in the optimization quality (0.0 to 1.0)
        """
        self.optimized_profile = optimized_profile
        self.optimization_applied = optimization_applied
        self.improvement_expected = improvement_expected
        self.edit_summary = edit_summary
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "optimization_applied": self.optimization_applied,
            "improvement_expected": self.improvement_expected,
            "edit_summary": self.edit_summary,
            "confidence": self.confidence,
            "profile_id": self.optimized_profile.profile_id
        }
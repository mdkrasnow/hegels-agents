"""
Training Step Execution for Hegel's Agents

This module implements the core training loop execution, integrating reward computation,
profile optimization, and comprehensive training state management.

Key Features:
- End-to-end training step execution
- Integration with existing reward system
- Profile evolution tracking and rollback capabilities
- Comprehensive error handling and recovery
- Training state persistence and monitoring
"""

import time
import uuid
import copy
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field

# Import existing components
from agents.utils import AgentResponse, AgentLogger
from debate.session import DebateSession, TurnType, ConflictAnalysis
from training.data_structures import PromptProfile, TrainingStep
from training.rewards import RewardCalculator, RewardComponents, RewardConfig

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from training.hegel_trainer import HegelTrainer


@dataclass
class TrainingStepResult:
    """
    Comprehensive result of a single training step execution.
    """
    # Basic execution info
    step_id: str
    timestamp: datetime
    execution_time_ms: float
    
    # Training data
    query: str
    corpus_id: str
    task_type: str
    gold_answer: Optional[str] = None
    provided_reward: Optional[float] = None
    
    # Execution results
    debate_result: Optional[Dict[str, Any]] = None
    final_response: Optional[AgentResponse] = None
    
    # Reward computation
    computed_reward: Optional[float] = None
    reward_components: Optional[RewardComponents] = None
    
    # Profile evolution
    original_profile_id: Optional[str] = None
    updated_profile_id: Optional[str] = None
    profile_changed: bool = False
    
    # Training metadata
    success: bool = True
    error_message: Optional[str] = None
    rollback_performed: bool = False
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_id': self.step_id,
            'timestamp': self.timestamp.isoformat(),
            'execution_time_ms': self.execution_time_ms,
            'query': self.query,
            'corpus_id': self.corpus_id,
            'task_type': self.task_type,
            'gold_answer': self.gold_answer,
            'provided_reward': self.provided_reward,
            'debate_result': self.debate_result,
            'final_response': self.final_response.to_dict() if self.final_response else None,
            'computed_reward': self.computed_reward,
            'reward_components': self.reward_components.to_dict() if self.reward_components else None,
            'original_profile_id': self.original_profile_id,
            'updated_profile_id': self.updated_profile_id,
            'profile_changed': self.profile_changed,
            'success': self.success,
            'error_message': self.error_message,
            'rollback_performed': self.rollback_performed,
            'training_metadata': self.training_metadata
        }


class TrainingExecutor:
    """
    Core training executor that orchestrates the complete training loop.
    
    Responsible for:
    - Executing debates with current profiles
    - Computing reward signals
    - Triggering profile optimization when needed
    - Managing training state and rollback
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, 
                 profile_store=None,  # Will be PromptProfileStore once available
                 reward_config: Optional[RewardConfig] = None,
                 logger: Optional[AgentLogger] = None):
        """
        Initialize training executor.
        
        Args:
            profile_store: PromptProfileStore instance for profile management
            reward_config: Configuration for reward computation
            logger: Optional logger (creates new one if not provided)
        """
        self.profile_store = profile_store
        self.reward_calculator = RewardCalculator(config=reward_config or RewardConfig())
        self.logger = logger or AgentLogger("training_executor")
        
        # Training state tracking
        self._active_steps: Dict[str, TrainingStepResult] = {}
        self._performance_history: List[Dict[str, Any]] = []
        
        # Performance monitoring
        self._stats = {
            'steps_executed': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'rollbacks_performed': 0,
            'profile_updates': 0,
            'total_execution_time_ms': 0,
            'average_reward': 0.0
        }
        
        self.logger.log_debug("TrainingExecutor initialized")
    
    def execute_training_step(self,
                             trainer_instance,  # HegelTrainer instance
                             query: str,
                             corpus_id: str,
                             task_type: str = "qa",
                             gold_answer: Optional[str] = None,
                             provided_reward: Optional[float] = None,
                             **kwargs) -> TrainingStepResult:
        """
        Execute a complete training step with debate, reward computation, and optimization.
        
        Args:
            trainer_instance: HegelTrainer instance to use for debate execution
            query: Query to process
            corpus_id: Corpus identifier
            task_type: Type of task being performed
            gold_answer: Optional gold standard answer for reward computation
            provided_reward: Optional pre-computed reward (overrides computation)
            **kwargs: Additional arguments for debate execution
            
        Returns:
            TrainingStepResult with comprehensive execution information
        """
        step_id = f"training_step_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        result = TrainingStepResult(
            step_id=step_id,
            timestamp=datetime.utcnow(),
            execution_time_ms=0.0,
            query=query,
            corpus_id=corpus_id,
            task_type=task_type,
            gold_answer=gold_answer,
            provided_reward=provided_reward
        )
        
        # Track active step
        self._active_steps[step_id] = result
        
        try:
            self.logger.log_debug(f"Executing training step {step_id} for corpus '{corpus_id}'")
            
            # Step 1: Execute debate with current configuration (grad=False to get clean baseline)
            self.logger.log_debug(f"Step 1: Executing baseline debate for {step_id}")
            debate_result = self._execute_baseline_debate(trainer_instance, query, corpus_id, task_type, **kwargs)
            result.debate_result = debate_result
            result.final_response = debate_result.get('final_response')
            
            # Step 2: Compute reward
            self.logger.log_debug(f"Step 2: Computing reward for {step_id}")
            if provided_reward is not None:
                result.computed_reward = provided_reward
                result.training_metadata['reward_source'] = 'provided'
            else:
                reward_data = self._compute_reward(debate_result, gold_answer, query)
                result.computed_reward = reward_data['total_reward']
                result.reward_components = reward_data['components']
                result.training_metadata['reward_source'] = 'computed'
            
            # Step 3: Determine if optimization is needed
            optimization_needed = self._should_optimize(result.computed_reward, result.reward_components)
            result.training_metadata['optimization_triggered'] = optimization_needed
            
            if optimization_needed:
                self.logger.log_debug(f"Step 3: Profile optimization triggered for {step_id}")
                profile_updated = self._optimize_profile(trainer_instance, result)
                result.profile_changed = profile_updated
            else:
                self.logger.log_debug(f"Step 3: No optimization needed for {step_id}")
            
            # Step 4: Update statistics and cleanup
            self._update_statistics(result)
            result.success = True
            
            self.logger.log_debug(f"Training step {step_id} completed successfully. Reward: {result.computed_reward:.3f}")
            
        except Exception as e:
            self.logger.log_error(Exception(f"Training step {step_id} failed: {str(e)}"), f"training_step_{step_id}")
            self.logger.log_debug(f"Full traceback: {traceback.format_exc()}")
            
            result.success = False
            result.error_message = str(e)
            result.training_metadata['error_details'] = traceback.format_exc()
            
            # Attempt rollback if needed
            if result.profile_changed:
                try:
                    self._perform_rollback(result)
                    result.rollback_performed = True
                    self.logger.log_debug(f"Rollback performed for failed step {step_id}")
                except Exception as rollback_error:
                    self.logger.log_error(Exception(f"Rollback failed for step {step_id}: {rollback_error}"), f"rollback_{step_id}")
            
            self._stats['failed_steps'] += 1
        
        finally:
            # Calculate execution time and cleanup
            end_time = time.time()
            result.execution_time_ms = (end_time - start_time) * 1000
            self._stats['total_execution_time_ms'] += result.execution_time_ms
            
            # Remove from active steps
            self._active_steps.pop(step_id, None)
            
            # Store in performance history
            self._performance_history.append({
                'step_id': step_id,
                'timestamp': result.timestamp.isoformat(),
                'reward': result.computed_reward,
                'execution_time_ms': result.execution_time_ms,
                'success': result.success,
                'profile_changed': result.profile_changed
            })
            
            # Keep history bounded
            if len(self._performance_history) > 1000:
                self._performance_history = self._performance_history[-500:]
        
        return result
    
    def _execute_baseline_debate(self, trainer_instance, query: str, corpus_id: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a baseline debate without training mode to get clean results.
        
        This temporarily disables grad mode to get unmodified debate results.
        """
        # Store original grad state
        original_grad = trainer_instance.grad
        
        try:
            # Temporarily disable grad mode for clean baseline
            trainer_instance.grad = False
            
            # Create wrapped agents for this execution
            from agents.worker import BasicWorkerAgent
            from agents.reviewer import BasicReviewerAgent
            
            worker = BasicWorkerAgent(f"worker_training_{uuid.uuid4().hex[:8]}")
            reviewer = BasicReviewerAgent(f"reviewer_training_{uuid.uuid4().hex[:8]}")
            
            # Wrap agents with trainer
            wrapped_worker = trainer_instance.wrap_worker_agent(worker)
            wrapped_reviewer = trainer_instance.wrap_reviewer_agent(reviewer)
            
            # Execute debate using existing debate session logic
            session = DebateSession(query)
            
            # Get worker responses
            worker_response1 = wrapped_worker.respond(query)
            session.add_turn(wrapped_worker.agent_id, worker_response1, TurnType.WORKER_RESPONSE)
            
            worker_response2 = wrapped_worker.respond(query)
            session.add_turn(wrapped_worker.agent_id, worker_response2, TurnType.WORKER_RESPONSE)
            
            # Get reviewer synthesis
            worker_responses = [worker_response1, worker_response2]
            synthesis = wrapped_reviewer.synthesize_responses(query, worker_responses)
            session.add_turn(wrapped_reviewer.agent_id, synthesis, TurnType.REVIEWER_SYNTHESIS)
            
            # Analyze the debate
            conflict_analysis = session.analyze_debate(worker_responses, synthesis, wrapped_reviewer)
            session.end_session()
            
            return {
                'session': session,
                'worker_responses': worker_responses,
                'synthesis': synthesis,
                'final_response': synthesis,
                'conflict_analysis': conflict_analysis,
                'session_summary': session.get_summary()
            }
            
        finally:
            # Restore original grad state
            trainer_instance.grad = original_grad
    
    def _compute_reward(self, debate_result: Dict[str, Any], gold_answer: Optional[str], query: str) -> Dict[str, Any]:
        """
        Compute comprehensive reward for the debate result.
        
        Args:
            debate_result: Results from baseline debate execution
            gold_answer: Optional gold standard answer
            query: Original query
            
        Returns:
            Dictionary with total_reward and components
        """
        # Extract components for reward calculation
        final_response = debate_result.get('final_response')
        worker_responses = debate_result.get('worker_responses', [])
        conflict_analysis = debate_result.get('conflict_analysis')
        session_summary = debate_result.get('session_summary', {})
        
        if not final_response:
            raise ValueError("No final response found in debate result")
        
        # Use existing reward calculator
        if gold_answer:
            # Compute text similarity reward
            total_reward = self.reward_calculator.compute_text_similarity(
                predicted=final_response.content,
                gold=gold_answer
            )
            
            # Add debate quality assessment if available
            if conflict_analysis:
                debate_quality = self.reward_calculator.compute_debate_quality({
                    'conflict_analysis': conflict_analysis,
                    'worker_responses': worker_responses,
                    'final_response': final_response,
                    'session_summary': session_summary
                })
                
                # Weighted combination
                total_reward = (total_reward * 0.6) + (debate_quality * 0.4)
        
        else:
            # No gold answer - use debate quality assessment only
            if conflict_analysis:
                total_reward = self.reward_calculator.compute_debate_quality({
                    'conflict_analysis': conflict_analysis,
                    'worker_responses': worker_responses,
                    'final_response': final_response,
                    'session_summary': session_summary
                })
            else:
                # Fallback to basic response quality assessment
                total_reward = self._assess_response_quality(final_response)
        
        # Create reward components (simplified - could be extended)
        components = RewardComponents(
            text_similarity=total_reward if gold_answer else 0.0,
            synthesis_effectiveness=conflict_analysis.resolution_quality if conflict_analysis else 0.0,
            conflict_identification=1.0 if (conflict_analysis and conflict_analysis.conflicts_detected) else 0.0,
            response_efficiency=min(1.0, 100.0 / max(1, len(final_response.content))),
            confidence_calibration=final_response.confidence or 0.5
        )
        
        return {
            'total_reward': total_reward,
            'components': components,
            'computation_metadata': {
                'has_gold_answer': gold_answer is not None,
                'has_conflict_analysis': conflict_analysis is not None,
                'worker_response_count': len(worker_responses),
                'final_response_length': len(final_response.content)
            }
        }
    
    def _assess_response_quality(self, response: AgentResponse) -> float:
        """
        Basic response quality assessment when no gold answer is available.
        
        Args:
            response: AgentResponse to assess
            
        Returns:
            Quality score 0.0 - 1.0
        """
        score = 0.0
        
        # Length-based heuristics
        content_length = len(response.content)
        if 50 <= content_length <= 500:  # Reasonable length
            score += 0.2
        
        # Has reasoning
        if response.reasoning and len(response.reasoning) > 20:
            score += 0.2
        
        # Has confidence
        if response.confidence is not None:
            score += 0.1
            # Bonus for reasonable confidence
            if 0.3 <= response.confidence <= 0.9:
                score += 0.1
        
        # Has sources
        if response.sources:
            score += 0.1
        
        # Content quality heuristics
        content_lower = response.content.lower()
        
        # Positive indicators
        if any(word in content_lower for word in ['however', 'because', 'therefore', 'although']):
            score += 0.1
        
        if any(word in content_lower for word in ['evidence', 'research', 'study', 'analysis']):
            score += 0.1
        
        # Negative indicators
        if any(phrase in content_lower for phrase in ['i don\'t know', 'unclear', 'uncertain']):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _should_optimize(self, reward: float, components: Optional[RewardComponents]) -> bool:
        """
        Determine if profile optimization should be triggered.
        
        Args:
            reward: Computed reward value
            components: Reward components breakdown
            
        Returns:
            True if optimization should be triggered
        """
        # Simple threshold-based optimization for now
        # In a full implementation, this could be more sophisticated
        
        # Low reward threshold
        if reward < 0.3:
            return True
        
        # Check specific component thresholds
        if components:
            # Poor synthesis effectiveness
            if components.synthesis_effectiveness < 0.4:
                return True
            
            # Poor confidence calibration
            if components.confidence_calibration < 0.3:
                return True
        
        # Occasionally optimize even good responses for exploration
        import random
        if random.random() < 0.05:  # 5% exploration
            return True
        
        return False
    
    def _optimize_profile(self, trainer_instance, result: TrainingStepResult) -> bool:
        """
        Trigger profile optimization based on training results.
        
        Args:
            trainer_instance: HegelTrainer instance
            result: Current training step result
            
        Returns:
            True if profile was updated
        """
        # For now, this is a placeholder implementation
        # In a full system, this would integrate with:
        # - ReflectionOptimizer for prompt improvement
        # - PromptProfileStore for persistence
        # - Profile versioning and lineage tracking
        
        self.logger.log_debug(f"Profile optimization triggered for {result.step_id}")
        self.logger.log_debug(f"Current reward: {result.computed_reward:.3f}")
        
        # Store current profile info
        result.original_profile_id = "current_profile_placeholder"
        
        # Simulate optimization (in real implementation, would call ReflectionOptimizer)
        result.training_metadata['optimization_attempt'] = True
        result.training_metadata['optimization_reason'] = f"Low reward: {result.computed_reward:.3f}"
        
        # For now, just log the optimization attempt
        # Real implementation would:
        # 1. Extract current prompts from wrapped agents
        # 2. Generate improvement suggestions using ReflectionOptimizer
        # 3. Create new PromptProfile with improved prompts
        # 4. Save to PromptProfileStore
        # 5. Update trainer_instance configuration
        
        self.logger.log_debug(f"Profile optimization completed for {result.step_id} (placeholder)")
        result.updated_profile_id = f"optimized_profile_{uuid.uuid4().hex[:8]}"
        
        return True  # Simulated successful optimization
    
    def _perform_rollback(self, result: TrainingStepResult):
        """
        Perform rollback of profile changes if training step failed.
        
        Args:
            result: Training step result with rollback information
        """
        if result.original_profile_id and result.updated_profile_id:
            self.logger.log_debug(f"Rolling back profile from {result.updated_profile_id} to {result.original_profile_id}")
            # In real implementation, would restore from PromptProfileStore
            result.training_metadata['rollback_performed'] = True
        
        self._stats['rollbacks_performed'] += 1
    
    def _update_statistics(self, result: TrainingStepResult):
        """Update internal statistics with training step results."""
        self._stats['steps_executed'] += 1
        if result.success:
            self._stats['successful_steps'] += 1
        if result.profile_changed:
            self._stats['profile_updates'] += 1
        
        # Update average reward
        if result.computed_reward is not None:
            current_avg = self._stats['average_reward']
            step_count = self._stats['steps_executed']
            self._stats['average_reward'] = ((current_avg * (step_count - 1)) + result.computed_reward) / step_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training executor statistics."""
        return {
            'executor_stats': self._stats.copy(),
            'active_steps': len(self._active_steps),
            'history_length': len(self._performance_history),
            'performance_summary': {
                'recent_avg_reward': sum(h['reward'] for h in self._performance_history[-10:] if h['reward']) / max(1, len([h for h in self._performance_history[-10:] if h['reward']])),
                'recent_avg_time_ms': sum(h['execution_time_ms'] for h in self._performance_history[-10:]) / max(1, len(self._performance_history[-10:])),
                'recent_success_rate': sum(1 for h in self._performance_history[-10:] if h['success']) / max(1, len(self._performance_history[-10:]))
            } if self._performance_history else {}
        }
    
    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance history."""
        return self._performance_history[-limit:]


# Factory functions for different configurations

def create_standard_executor(profile_store=None, logger=None) -> TrainingExecutor:
    """Create training executor with standard configuration."""
    config = RewardConfig()  # Use defaults
    return TrainingExecutor(profile_store=profile_store, reward_config=config, logger=logger)


def create_research_executor(profile_store=None, logger=None) -> TrainingExecutor:
    """Create training executor optimized for research with detailed analysis."""
    config = RewardConfig(
        debate_quality_weight=0.5,  # Emphasize debate quality for research
        meta_rewards_weight=0.3,    # Include meta-learning signals
        text_quality_weight=0.2     # De-emphasize pure text similarity
    )
    return TrainingExecutor(profile_store=profile_store, reward_config=config, logger=logger)


def create_production_executor(profile_store=None, logger=None) -> TrainingExecutor:
    """Create training executor optimized for production with efficiency focus."""
    config = RewardConfig(
        text_quality_weight=0.4,      # Emphasize output quality
        process_efficiency_weight=0.3,  # Emphasize efficiency
        debate_quality_weight=0.3     # Basic dialectical quality
    )
    return TrainingExecutor(profile_store=profile_store, reward_config=config, logger=logger)
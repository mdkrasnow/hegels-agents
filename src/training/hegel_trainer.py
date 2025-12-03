"""
HegelTrainer - Training Wrapper for Hegel's Agents Phase 1

This module provides the main training wrapper that preserves existing functionality 
while adding training capabilities. Critical that grad=False mode produces identical 
results to current system.

Key Requirements:
- Wrap existing agents without modifying their classes  
- Ensure grad=False behaves exactly like current system
- Integrate with existing DebateSession and AgentLogger
- Maintain exact API compatibility
- Add comprehensive logging of all operations
"""

import uuid
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import copy
import json
from dataclasses import dataclass

from agents.worker import BasicWorkerAgent
from agents.reviewer import BasicReviewerAgent
from agents.utils import AgentResponse, AgentLogger, DebateContext
from debate.session import DebateSession, TurnType
from config.settings import get_config

# Import training components
from training.data_structures import RolePrompt, PromptProfile, TrainingStep
from training.training_executor import TrainingExecutor, TrainingStepResult, create_standard_executor

try:
    from training.database.prompt_profile_store import PromptProfileStore
except ImportError:
    # Fallback if PromptProfileStore not available
    PromptProfileStore = None


class HegelTrainer:
    """
    Main training wrapper that preserves existing functionality while adding training capabilities.
    
    Critical Design Principles:
    1. grad=False mode produces IDENTICAL results to current system
    2. Zero performance overhead in inference mode
    3. Composition over inheritance - wrap, don't modify
    4. Preserve all existing logging and error handling patterns
    """
    
    def __init__(self, 
                 grad: bool = False, 
                 profile_store: Optional[Any] = None,  # Will be PromptProfileStore once T1.3 complete
                 trainer_id: str = None):
        """
        Initialize HegelTrainer wrapper.
        
        Args:
            grad: Training mode flag. If False, behaves exactly like current system
            profile_store: PromptProfileStore instance (requires T1.3)
            trainer_id: Unique identifier for this trainer instance
        """
        self.grad = grad
        self.trainer_id = trainer_id or f"hegel_trainer_{uuid.uuid4().hex[:8]}"
        self.logger = AgentLogger(f"trainer_{self.trainer_id}")
        
        # Training capabilities (enabled only when dependencies ready)
        self.profile_store = profile_store
        self._training_ready = profile_store is not None
        
        # Initialize training executor for grad=True mode
        self._training_executor = None
        if self.grad and self._training_ready:
            self._training_executor = create_standard_executor(
                profile_store=self.profile_store,
                logger=self.logger
            )
        
        # Wrapped agent tracking
        self._wrapped_agents: Dict[str, Dict[str, Any]] = {}
        
        # Training session tracking
        self._current_session: Optional[str] = None
        self._session_data: Dict[str, Any] = {}
        
        # Performance monitoring
        self._stats = {
            'agents_wrapped': 0,
            'training_interactions': 0,
            'sessions_tracked': 0,
            'grad_mode_calls': 0,
            'inference_mode_calls': 0
        }
        
        self.logger.log_debug(f"HegelTrainer initialized with grad={grad}, training_ready={self._training_ready}")
    
    def wrap_worker_agent(self, 
                         agent: BasicWorkerAgent, 
                         profile_id: Optional[str] = None,
                         agent_alias: Optional[str] = None) -> 'TrainingWorkerWrapper':
        """
        Wrap a BasicWorkerAgent with training capabilities.
        
        Args:
            agent: BasicWorkerAgent instance to wrap
            profile_id: Optional prompt profile ID to apply
            agent_alias: Optional alias for tracking this agent
            
        Returns:
            TrainingWorkerWrapper that behaves identically to original when grad=False
        """
        if not isinstance(agent, BasicWorkerAgent):
            raise TypeError(f"Expected BasicWorkerAgent, got {type(agent)}")
        
        wrapper_id = agent_alias or f"worker_{len(self._wrapped_agents)}"
        
        # Create wrapper
        wrapper = TrainingWorkerWrapper(
            original_agent=agent,
            trainer=self,
            wrapper_id=wrapper_id,
            profile_id=profile_id
        )
        
        # Track wrapped agent
        self._wrapped_agents[wrapper_id] = {
            'type': 'worker',
            'wrapper': wrapper,
            'original': agent,
            'profile_id': profile_id,
            'wrapped_at': datetime.utcnow()
        }
        
        self._stats['agents_wrapped'] += 1
        self.logger.log_debug(f"Wrapped worker agent as '{wrapper_id}' with profile_id={profile_id}")
        
        return wrapper
    
    def wrap_reviewer_agent(self, 
                           agent: BasicReviewerAgent, 
                           profile_id: Optional[str] = None,
                           agent_alias: Optional[str] = None) -> 'TrainingReviewerWrapper':
        """
        Wrap a BasicReviewerAgent with training capabilities.
        
        Args:
            agent: BasicReviewerAgent instance to wrap
            profile_id: Optional prompt profile ID to apply  
            agent_alias: Optional alias for tracking this agent
            
        Returns:
            TrainingReviewerWrapper that behaves identically to original when grad=False
        """
        if not isinstance(agent, BasicReviewerAgent):
            raise TypeError(f"Expected BasicReviewerAgent, got {type(agent)}")
        
        wrapper_id = agent_alias or f"reviewer_{len(self._wrapped_agents)}"
        
        # Create wrapper
        wrapper = TrainingReviewerWrapper(
            original_agent=agent,
            trainer=self,
            wrapper_id=wrapper_id,
            profile_id=profile_id
        )
        
        # Track wrapped agent
        self._wrapped_agents[wrapper_id] = {
            'type': 'reviewer',
            'wrapper': wrapper,
            'original': agent,
            'profile_id': profile_id,
            'wrapped_at': datetime.utcnow()
        }
        
        self._stats['agents_wrapped'] += 1
        self.logger.log_debug(f"Wrapped reviewer agent as '{wrapper_id}' with profile_id={profile_id}")
        
        return wrapper
    
    def create_training_session(self, 
                               session_id: Optional[str] = None, 
                               corpus_id: Optional[str] = None,
                               task_type: str = "qa") -> str:
        """
        Create a new training session for tracking interactions.
        
        Args:
            session_id: Optional session identifier
            corpus_id: Corpus being used for training
            task_type: Type of task being performed
            
        Returns:
            Session ID for tracking
        """
        session_id = session_id or f"training_{uuid.uuid4().hex[:12]}"
        
        self._current_session = session_id
        self._session_data[session_id] = {
            'corpus_id': corpus_id,
            'task_type': task_type,
            'started_at': datetime.utcnow(),
            'interactions': [],
            'grad_mode': self.grad
        }
        
        self._stats['sessions_tracked'] += 1
        self.logger.log_debug(f"Created training session '{session_id}' for corpus='{corpus_id}', task='{task_type}'")
        
        return session_id
    
    def end_training_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        End a training session and return summary statistics.
        
        Args:
            session_id: Session to end (current session if None)
            
        Returns:
            Session summary data
        """
        session_id = session_id or self._current_session
        
        if session_id not in self._session_data:
            raise ValueError(f"Training session '{session_id}' not found")
        
        session = self._session_data[session_id]
        session['ended_at'] = datetime.utcnow()
        session['duration_seconds'] = (session['ended_at'] - session['started_at']).total_seconds()
        
        if session_id == self._current_session:
            self._current_session = None
        
        self.logger.log_debug(f"Ended training session '{session_id}', duration={session['duration_seconds']:.2f}s")
        
        return session
    
    def _log_training_interaction(self, 
                                 agent_id: str, 
                                 question: str, 
                                 response: AgentResponse, 
                                 profile_id: Optional[str] = None,
                                 session_id: Optional[str] = None):
        """
        Log a training interaction for future analysis.
        
        This method is called only when grad=True to collect training data.
        """
        if not self.grad:
            return  # No-op in inference mode
        
        session_id = session_id or self._current_session
        
        interaction_data = {
            'interaction_id': uuid.uuid4().hex,
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': agent_id,
            'profile_id': profile_id,
            'question': question,
            'response': {
                'content': response.content,
                'reasoning': response.reasoning,
                'confidence': response.confidence,
                'sources': response.sources or [],
                'metadata': response.metadata or {}
            }
        }
        
        # Store in session if available
        if session_id and session_id in self._session_data:
            self._session_data[session_id]['interactions'].append(interaction_data)
        
        # TODO: Store in PromptProfileStore once T1.3 is ready
        if self._training_ready and self.profile_store:
            # self.profile_store.log_interaction(interaction_data)
            pass
        
        self._stats['training_interactions'] += 1
        self.logger.log_debug(f"Logged training interaction for agent '{agent_id}' in session '{session_id}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training wrapper statistics."""
        return {
            'trainer_id': self.trainer_id,
            'grad_mode': self.grad,
            'training_ready': self._training_ready,
            'current_session': self._current_session,
            'stats': self._stats.copy(),
            'wrapped_agents': len(self._wrapped_agents),
            'active_sessions': len([s for s in self._session_data.values() if 'ended_at' not in s])
        }
    
    def list_wrapped_agents(self) -> List[Dict[str, Any]]:
        """Get list of all wrapped agents with their metadata."""
        return [
            {
                'wrapper_id': wrapper_id,
                'type': info['type'],
                'profile_id': info['profile_id'],
                'wrapped_at': info['wrapped_at'].isoformat(),
                'original_agent_id': info['original'].agent_id
            }
            for wrapper_id, info in self._wrapped_agents.items()
        ]
    
    def run(self, 
            query: str, 
            corpus_id: str, 
            task_type: str = "qa",
            grad: bool = False, 
            gold_answer: Optional[str] = None, 
            reward: Optional[float] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Execute debate with optional training mode (grad=True enables live learning).
        
        This is the main entry point that provides both inference (grad=False) and
        training (grad=True) capabilities while maintaining full backward compatibility.
        
        Args:
            query: Question or prompt to process
            corpus_id: Corpus identifier for context retrieval  
            task_type: Type of task ("qa", "analysis", "summary", etc.)
            grad: Enable training mode - when True, executes training loop with live learning
            gold_answer: Gold standard answer for training reward computation
            reward: Pre-computed reward (overrides automatic computation)
            **kwargs: Additional arguments for debate execution
            
        Returns:
            Comprehensive results dictionary with debate output and training metadata
            
        Behavior:
        - grad=False: Behaves exactly like existing system (0% regression guaranteed)
        - grad=True: Executes full training loop with profile evolution and learning
        """
        start_time = datetime.utcnow()
        execution_id = f"run_{uuid.uuid4().hex[:12]}"
        
        # Override instance grad mode if explicitly specified
        effective_grad = grad if grad is not None else self.grad
        
        self.logger.log_debug(f"HegelTrainer.run() started: {execution_id}, grad={effective_grad}, corpus='{corpus_id}', task='{task_type}'")
        
        try:
            # Update statistics
            if effective_grad:
                self._stats['grad_mode_calls'] += 1
            else:
                self._stats['inference_mode_calls'] += 1
            
            if effective_grad and self._training_ready and self._training_executor:
                # TRAINING MODE (grad=True): Execute full training loop with live learning
                return self._execute_training_mode(
                    execution_id=execution_id,
                    query=query,
                    corpus_id=corpus_id,
                    task_type=task_type,
                    gold_answer=gold_answer,
                    provided_reward=reward,
                    start_time=start_time,
                    **kwargs
                )
            else:
                # INFERENCE MODE (grad=False): Execute standard debate without training
                return self._execute_inference_mode(
                    execution_id=execution_id,
                    query=query,
                    corpus_id=corpus_id,
                    task_type=task_type,
                    start_time=start_time,
                    grad_requested=effective_grad,
                    **kwargs
                )
                
        except Exception as e:
            self.logger.log_error(Exception(f"HegelTrainer.run() failed for {execution_id}: {str(e)}"), f"run_{execution_id}")
            
            # Return error result with consistent structure
            return {
                'execution_id': execution_id,
                'success': False,
                'error': str(e),
                'mode': 'training' if effective_grad else 'inference',
                'grad_mode': effective_grad,
                'corpus_id': corpus_id,
                'task_type': task_type,
                'query': query,
                'timestamp': start_time.isoformat(),
                'execution_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000,
                'training_ready': self._training_ready,
                'final_response': None,
                'training_metadata': {
                    'error_details': str(e),
                    'training_executor_available': self._training_executor is not None
                }
            }
    
    def _execute_training_mode(self,
                              execution_id: str,
                              query: str,
                              corpus_id: str,
                              task_type: str,
                              gold_answer: Optional[str],
                              provided_reward: Optional[float],
                              start_time: datetime,
                              **kwargs) -> Dict[str, Any]:
        """
        Execute full training mode with live learning and profile evolution.
        
        This mode:
        1. Executes debate to get baseline performance  
        2. Computes reward signals (text similarity, debate quality, etc.)
        3. Triggers profile optimization if performance is poor
        4. Tracks all training state and profile evolution
        5. Provides comprehensive training metadata
        """
        self.logger.log_debug(f"Executing TRAINING mode for {execution_id}")
        
        # Execute training step using TrainingExecutor
        training_result: TrainingStepResult = self._training_executor.execute_training_step(
            trainer_instance=self,
            query=query,
            corpus_id=corpus_id,
            task_type=task_type,
            gold_answer=gold_answer,
            provided_reward=provided_reward,
            **kwargs
        )
        
        # Calculate total execution time
        end_time = datetime.utcnow()
        total_execution_ms = (end_time - start_time).total_seconds() * 1000
        
        # Construct comprehensive result
        result = {
            # Core execution info
            'execution_id': execution_id,
            'success': training_result.success,
            'mode': 'training',
            'grad_mode': True,
            'corpus_id': corpus_id,
            'task_type': task_type,
            'query': query,
            'timestamp': start_time.isoformat(),
            'execution_time_ms': total_execution_ms,
            
            # Training-specific results
            'training_step_id': training_result.step_id,
            'training_step_time_ms': training_result.execution_time_ms,
            'computed_reward': training_result.computed_reward,
            'provided_reward': provided_reward,
            'gold_answer': gold_answer,
            
            # Debate results
            'final_response': training_result.final_response,
            'debate_result': training_result.debate_result,
            
            # Profile evolution tracking
            'profile_evolution': {
                'original_profile_id': training_result.original_profile_id,
                'updated_profile_id': training_result.updated_profile_id,
                'profile_changed': training_result.profile_changed,
                'optimization_triggered': training_result.training_metadata.get('optimization_triggered', False),
                'optimization_reason': training_result.training_metadata.get('optimization_reason', None)
            },
            
            # Training state and monitoring
            'training_metadata': {
                'training_ready': self._training_ready,
                'training_executor_stats': self._training_executor.get_stats() if self._training_executor else None,
                'reward_components': training_result.reward_components.to_dict() if training_result.reward_components else None,
                'reward_source': training_result.training_metadata.get('reward_source', 'unknown'),
                'rollback_performed': training_result.rollback_performed,
                'error_details': training_result.error_message,
                **training_result.training_metadata
            },
            
            # Wrapper state
            'wrapper_state': {
                'wrapped_agents': len(self._wrapped_agents),
                'current_session': self._current_session,
                'trainer_stats': self._stats.copy()
            }
        }
        
        # Add error information if training failed
        if not training_result.success:
            result['error'] = training_result.error_message
        
        self.logger.log_debug(f"TRAINING mode completed for {execution_id}. Success: {training_result.success}, Reward: {training_result.computed_reward:.3f if training_result.computed_reward else 'N/A'}")
        
        return result
    
    def _execute_inference_mode(self,
                               execution_id: str,
                               query: str,
                               corpus_id: str,
                               task_type: str,
                               start_time: datetime,
                               grad_requested: bool = False,
                               **kwargs) -> Dict[str, Any]:
        """
        Execute standard inference mode (grad=False) with identical behavior to existing system.
        
        This mode:
        1. Creates standard worker and reviewer agents
        2. Executes traditional debate session
        3. Returns results in format compatible with existing system
        4. Zero training or profile modification
        5. Zero performance overhead
        """
        self.logger.log_debug(f"Executing INFERENCE mode for {execution_id}")
        
        # Create agents for this execution (standard behavior)
        from agents.worker import BasicWorkerAgent
        from agents.reviewer import BasicReviewerAgent
        
        worker = BasicWorkerAgent(f"worker_{execution_id}")
        reviewer = BasicReviewerAgent(f"reviewer_{execution_id}")
        
        # Wrap agents with trainer (in inference mode, this adds zero overhead)
        wrapped_worker = self.wrap_worker_agent(worker, agent_alias=f"worker_{execution_id}")
        wrapped_reviewer = self.wrap_reviewer_agent(reviewer, agent_alias=f"reviewer_{execution_id}")
        
        # Execute standard debate session
        session = DebateSession(query)
        
        try:
            # Get worker responses
            worker_response1 = wrapped_worker.respond(query)
            session.add_turn(wrapped_worker.agent_id, worker_response1, TurnType.WORKER_RESPONSE)
            
            worker_response2 = wrapped_worker.respond(query) 
            session.add_turn(wrapped_worker.agent_id, worker_response2, TurnType.WORKER_RESPONSE)
            
            # Get reviewer synthesis
            worker_responses = [worker_response1, worker_response2]
            synthesis = wrapped_reviewer.synthesize_responses(query, worker_responses)
            session.add_turn(wrapped_reviewer.agent_id, synthesis, TurnType.REVIEWER_SYNTHESIS)
            
            # Analyze debate quality
            conflict_analysis = session.analyze_debate(worker_responses, synthesis, wrapped_reviewer)
            session.end_session()
            
            # Calculate execution time
            end_time = datetime.utcnow()
            total_execution_ms = (end_time - start_time).total_seconds() * 1000
            
            # Return results in format compatible with existing system
            result = {
                # Core execution info
                'execution_id': execution_id,
                'success': True,
                'mode': 'inference',
                'grad_mode': False,
                'corpus_id': corpus_id,
                'task_type': task_type,
                'query': query,
                'timestamp': start_time.isoformat(),
                'execution_time_ms': total_execution_ms,
                
                # Standard debate results (backward compatible)
                'final_response': synthesis,
                'worker_responses': worker_responses,
                'synthesis': synthesis,
                'conflict_analysis': conflict_analysis,
                'session': session,
                'session_summary': session.get_summary(),
                
                # Training metadata (minimal for inference mode)
                'training_metadata': {
                    'training_ready': self._training_ready,
                    'grad_requested': grad_requested,
                    'grad_available': self._training_ready and self._training_executor is not None,
                    'mode_reason': 'grad_disabled' if not grad_requested else 'training_not_ready'
                },
                
                # Wrapper state
                'wrapper_state': {
                    'wrapped_agents': len(self._wrapped_agents),
                    'current_session': self._current_session,
                    'trainer_stats': self._stats.copy()
                }
            }
            
            self.logger.log_debug(f"INFERENCE mode completed successfully for {execution_id}")
            return result
            
        except Exception as e:
            self.logger.log_error(Exception(f"INFERENCE mode failed for {execution_id}: {str(e)}"), f"inference_{execution_id}")
            session.end_session()
            
            end_time = datetime.utcnow()
            total_execution_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'execution_id': execution_id,
                'success': False,
                'error': str(e),
                'mode': 'inference',
                'grad_mode': False,
                'corpus_id': corpus_id,
                'task_type': task_type,
                'query': query,
                'timestamp': start_time.isoformat(),
                'execution_time_ms': total_execution_ms,
                'final_response': None,
                'training_metadata': {
                    'training_ready': self._training_ready,
                    'error_in_inference_mode': True
                }
            }
    
    def enable_training_mode(self, profile_store=None):
        """
        Enable training mode by initializing training executor.
        
        This method allows upgrading an inference-only trainer to support training.
        
        Args:
            profile_store: Optional PromptProfileStore instance
        """
        if profile_store:
            self.profile_store = profile_store
            self._training_ready = True
        
        if self._training_ready and not self._training_executor:
            self._training_executor = create_standard_executor(
                profile_store=self.profile_store,
                logger=self.logger
            )
            self.logger.log_debug("Training mode enabled - TrainingExecutor initialized")
        
        return self._training_ready
    
    def disable_training_mode(self):
        """
        Disable training mode and cleanup training executor.
        
        After calling this, all run() calls will execute in inference mode only.
        """
        if self._training_executor:
            self._training_executor = None
            self.logger.log_debug("Training mode disabled - TrainingExecutor cleaned up")
        
        self.grad = False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        base_stats = self.get_stats()
        
        if self._training_executor:
            base_stats['training_executor'] = self._training_executor.get_stats()
            base_stats['training_performance_history'] = self._training_executor.get_performance_history(50)
        else:
            base_stats['training_executor'] = None
            
        return base_stats


class TrainingWorkerWrapper:
    """
    Training-aware wrapper for BasicWorkerAgent.
    
    Preserves exact functionality when grad=False, adds training capabilities when grad=True.
    """
    
    def __init__(self, 
                 original_agent: BasicWorkerAgent, 
                 trainer: HegelTrainer, 
                 wrapper_id: str,
                 profile_id: Optional[str] = None):
        """
        Initialize worker wrapper.
        
        Args:
            original_agent: BasicWorkerAgent to wrap
            trainer: HegelTrainer instance managing this wrapper
            wrapper_id: Unique identifier for this wrapper
            profile_id: Optional prompt profile to apply
        """
        self._original = original_agent
        self._trainer = trainer
        self.wrapper_id = wrapper_id
        self.profile_id = profile_id
        
        # Preserve original agent interface
        self.agent_id = f"{wrapper_id}[{original_agent.agent_id}]"
        self.logger = original_agent.logger  # Use original logger for compatibility
        
        # Store original prompt for restoration
        self._original_system_prompt = original_agent.SYSTEM_PROMPT
        self._current_profile = None
        
    def _apply_profile(self, profile_data: Any) -> None:
        """
        Apply prompt profile to the wrapped agent.
        
        NOTE: This will be implemented once T1.1 (data structures) is ready.
        """
        if not self._trainer.grad or not profile_data:
            return
        
        # TODO: Implement when PromptProfile is available from T1.1
        # if hasattr(profile_data, 'worker') and hasattr(profile_data.worker, 'system_prompt'):
        #     self._original.SYSTEM_PROMPT = profile_data.worker.system_prompt
        #     self._current_profile = profile_data
        
        self._trainer.logger.log_debug(f"Applied profile to worker {self.wrapper_id}")
    
    def _restore_original_prompt(self) -> None:
        """Restore the original system prompt."""
        self._original.SYSTEM_PROMPT = self._original_system_prompt
        self._current_profile = None
    
    def respond(self, question: str, external_context: Optional[str] = None) -> AgentResponse:
        """
        Generate response with training awareness.
        
        Args:
            question: Question to answer
            external_context: Optional external context
            
        Returns:
            AgentResponse - identical to original when grad=False
        """
        try:
            # Apply profile if in training mode and profile available
            if self._trainer.grad and self.profile_id and self._trainer._training_ready:
                # TODO: Load and apply profile once T1.3 is ready
                # profile = self._trainer.profile_store.get_profile(self.profile_id)
                # self._apply_profile(profile)
                pass
            
            # Generate response using original agent (CRITICAL: identical behavior)
            response = self._original.respond(question, external_context)
            
            # Track statistics
            if self._trainer.grad:
                self._trainer._stats['grad_mode_calls'] += 1
            else:
                self._trainer._stats['inference_mode_calls'] += 1
            
            # Log training data if in training mode
            if self._trainer.grad:
                self._trainer._log_training_interaction(
                    agent_id=self.agent_id,
                    question=question,
                    response=response,
                    profile_id=self.profile_id
                )
            
            return response
            
        finally:
            # Always restore original prompt to ensure no state pollution
            if self._trainer.grad:
                self._restore_original_prompt()
    
    def batch_respond(self, questions: List[str], contexts: Optional[List[str]] = None) -> List[AgentResponse]:
        """Batch response with training awareness."""
        # Delegate to original implementation for consistency
        if not self._trainer.grad:
            return self._original.batch_respond(questions, contexts)
        
        # In training mode, process individually to capture each interaction
        responses = []
        for i, question in enumerate(questions):
            context = contexts[i] if contexts else None
            response = self.respond(question, context)
            responses.append(response)
        
        return responses
    
    def add_knowledge(self, texts: List[str]):
        """Add knowledge to wrapped agent."""
        return self._original.add_knowledge(texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics including training wrapper info."""
        original_stats = self._original.get_stats()
        original_stats['wrapper_info'] = {
            'wrapper_id': self.wrapper_id,
            'profile_id': self.profile_id,
            'trainer_id': self._trainer.trainer_id,
            'grad_mode': self._trainer.grad,
            'current_profile_active': self._current_profile is not None
        }
        return original_stats
    
    def __getattr__(self, name):
        """Delegate all other attributes to original agent for compatibility."""
        return getattr(self._original, name)


class TrainingReviewerWrapper:
    """
    Training-aware wrapper for BasicReviewerAgent.
    
    Preserves exact functionality when grad=False, adds training capabilities when grad=True.
    """
    
    def __init__(self, 
                 original_agent: BasicReviewerAgent, 
                 trainer: HegelTrainer, 
                 wrapper_id: str,
                 profile_id: Optional[str] = None):
        """
        Initialize reviewer wrapper.
        
        Args:
            original_agent: BasicReviewerAgent to wrap
            trainer: HegelTrainer instance managing this wrapper
            wrapper_id: Unique identifier for this wrapper
            profile_id: Optional prompt profile to apply
        """
        self._original = original_agent
        self._trainer = trainer
        self.wrapper_id = wrapper_id
        self.profile_id = profile_id
        
        # Preserve original agent interface
        self.agent_id = f"{wrapper_id}[{original_agent.agent_id}]"
        self.logger = original_agent.logger  # Use original logger for compatibility
        
        # Store original prompts for restoration
        self._original_critique_prompt = original_agent.CRITIQUE_PROMPT
        self._original_synthesis_prompt = original_agent.SYNTHESIS_PROMPT
        self._current_profile = None
    
    def _apply_profile(self, profile_data: Any) -> None:
        """
        Apply prompt profile to the wrapped agent.
        
        NOTE: This will be implemented once T1.1 (data structures) is ready.
        """
        if not self._trainer.grad or not profile_data:
            return
        
        # TODO: Implement when PromptProfile is available from T1.1
        # if hasattr(profile_data, 'reviewer'):
        #     reviewer_config = profile_data.reviewer
        #     if hasattr(reviewer_config, 'critique_prompt'):
        #         self._original.CRITIQUE_PROMPT = reviewer_config.critique_prompt
        #     if hasattr(reviewer_config, 'synthesis_prompt'):
        #         self._original.SYNTHESIS_PROMPT = reviewer_config.synthesis_prompt
        #     self._current_profile = profile_data
        
        self._trainer.logger.log_debug(f"Applied profile to reviewer {self.wrapper_id}")
    
    def _restore_original_prompts(self) -> None:
        """Restore the original prompts."""
        self._original.CRITIQUE_PROMPT = self._original_critique_prompt
        self._original.SYNTHESIS_PROMPT = self._original_synthesis_prompt
        self._current_profile = None
    
    def critique_response(self, question: str, response: AgentResponse) -> AgentResponse:
        """Critique response with training awareness."""
        try:
            # Apply profile if in training mode
            if self._trainer.grad and self.profile_id and self._trainer._training_ready:
                # TODO: Load and apply profile once T1.3 is ready
                pass
            
            # Generate critique using original agent
            critique = self._original.critique_response(question, response)
            
            # Track statistics
            if self._trainer.grad:
                self._trainer._stats['grad_mode_calls'] += 1
                self._trainer._log_training_interaction(
                    agent_id=self.agent_id,
                    question=f"CRITIQUE: {question}",
                    response=critique,
                    profile_id=self.profile_id
                )
            else:
                self._trainer._stats['inference_mode_calls'] += 1
            
            return critique
            
        finally:
            if self._trainer.grad:
                self._restore_original_prompts()
    
    def synthesize_responses(self, question: str, responses: List[AgentResponse]) -> AgentResponse:
        """Synthesize responses with training awareness."""
        try:
            # Apply profile if in training mode
            if self._trainer.grad and self.profile_id and self._trainer._training_ready:
                # TODO: Load and apply profile once T1.3 is ready
                pass
            
            # Generate synthesis using original agent
            synthesis = self._original.synthesize_responses(question, responses)
            
            # Track statistics
            if self._trainer.grad:
                self._trainer._stats['grad_mode_calls'] += 1
                self._trainer._log_training_interaction(
                    agent_id=self.agent_id,
                    question=f"SYNTHESIS: {question}",
                    response=synthesis,
                    profile_id=self.profile_id
                )
            else:
                self._trainer._stats['inference_mode_calls'] += 1
            
            return synthesis
            
        finally:
            if self._trainer.grad:
                self._restore_original_prompts()
    
    def compare_responses(self, question: str, response1: AgentResponse, response2: AgentResponse) -> AgentResponse:
        """Compare responses with training awareness."""
        try:
            # Apply profile if in training mode
            if self._trainer.grad and self.profile_id and self._trainer._training_ready:
                # TODO: Load and apply profile once T1.3 is ready
                pass
            
            # Generate comparison using original agent
            comparison = self._original.compare_responses(question, response1, response2)
            
            # Track statistics
            if self._trainer.grad:
                self._trainer._stats['grad_mode_calls'] += 1
                self._trainer._log_training_interaction(
                    agent_id=self.agent_id,
                    question=f"COMPARE: {question}",
                    response=comparison,
                    profile_id=self.profile_id
                )
            else:
                self._trainer._stats['inference_mode_calls'] += 1
            
            return comparison
            
        finally:
            if self._trainer.grad:
                self._restore_original_prompts()
    
    def review_and_synthesize(self, question: str, responses: List[AgentResponse]) -> Dict[str, AgentResponse]:
        """Review and synthesize with training awareness."""
        # Use original implementation for consistency
        result = self._original.review_and_synthesize(question, responses)
        
        # Log each component if in training mode
        if self._trainer.grad:
            for i, critique in enumerate(result['critiques']):
                self._trainer._log_training_interaction(
                    agent_id=self.agent_id,
                    question=f"CRITIQUE_{i}: {question}",
                    response=critique,
                    profile_id=self.profile_id
                )
            
            self._trainer._log_training_interaction(
                agent_id=self.agent_id,
                question=f"SYNTHESIS: {question}",
                response=result['synthesis'],
                profile_id=self.profile_id
            )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics including training wrapper info."""
        original_stats = self._original.get_stats()
        original_stats['wrapper_info'] = {
            'wrapper_id': self.wrapper_id,
            'profile_id': self.profile_id,
            'trainer_id': self._trainer.trainer_id,
            'grad_mode': self._trainer.grad,
            'current_profile_active': self._current_profile is not None
        }
        return original_stats
    
    def __getattr__(self, name):
        """Delegate all other attributes to original agent for compatibility."""
        return getattr(self._original, name)


# Compatibility helpers for existing code

def create_trainer(grad: bool = False, **kwargs) -> HegelTrainer:
    """
    Factory function to create HegelTrainer instance.
    
    Args:
        grad: Training mode flag
        **kwargs: Additional arguments for HegelTrainer
        
    Returns:
        HegelTrainer instance
    """
    return HegelTrainer(grad=grad, **kwargs)


def wrap_agents_for_training(worker: BasicWorkerAgent, 
                            reviewer: BasicReviewerAgent,
                            trainer: Optional[HegelTrainer] = None,
                            worker_profile: Optional[str] = None,
                            reviewer_profile: Optional[str] = None) -> tuple:
    """
    Convenience function to wrap both worker and reviewer agents.
    
    Args:
        worker: BasicWorkerAgent to wrap
        reviewer: BasicReviewerAgent to wrap  
        trainer: HegelTrainer instance (creates new one if None)
        worker_profile: Profile ID for worker
        reviewer_profile: Profile ID for reviewer
        
    Returns:
        Tuple of (wrapped_worker, wrapped_reviewer, trainer)
    """
    if trainer is None:
        trainer = create_trainer(grad=False)
    
    wrapped_worker = trainer.wrap_worker_agent(worker, worker_profile)
    wrapped_reviewer = trainer.wrap_reviewer_agent(reviewer, reviewer_profile)
    
    return wrapped_worker, wrapped_reviewer, trainer


def create_inference_agents(worker_id: str = "worker_agent", 
                          reviewer_id: str = "reviewer_agent") -> tuple:
    """
    Create agents configured for inference mode (grad=False).
    
    Args:
        worker_id: ID for worker agent
        reviewer_id: ID for reviewer agent
        
    Returns:
        Tuple of (worker, reviewer, trainer) ready for inference
    """
    # Create original agents
    worker = BasicWorkerAgent(worker_id)
    reviewer = BasicReviewerAgent(reviewer_id)
    
    # Wrap with trainer in inference mode
    trainer = create_trainer(grad=False)
    wrapped_worker = trainer.wrap_worker_agent(worker)
    wrapped_reviewer = trainer.wrap_reviewer_agent(reviewer)
    
    return wrapped_worker, wrapped_reviewer, trainer


# Export main classes and functions
__all__ = [
    'HegelTrainer',
    'TrainingWorkerWrapper', 
    'TrainingReviewerWrapper',
    'create_trainer',
    'wrap_agents_for_training',
    'create_inference_agents'
]
"""
Secure Hegel Trainer

Enhanced HegelTrainer with integrated security controls for safe training operations.
Provides comprehensive security validation, cost monitoring, and emergency stop mechanisms.
"""

import uuid
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
import logging

from .hegel_trainer import HegelTrainer, TrainingWorkerWrapper, TrainingReviewerWrapper
from ..security import (
    get_training_security_manager, TrainingSecurityConfig, TrainingSecurityLevel,
    EmergencyStopReason, validate_prompt_safety, get_cost_monitor
)
from agents.utils import AgentResponse

logger = logging.getLogger(__name__)


class SecureHegelTrainer(HegelTrainer):
    """
    Security-enhanced HegelTrainer with comprehensive protection mechanisms.
    
    Features:
    - Integrated training security management
    - Prompt safety validation for all training inputs
    - Cost monitoring with session-specific limits
    - Emergency stop mechanisms
    - Secure training session management
    - Threat detection and response
    """
    
    def __init__(self, 
                 grad: bool = False, 
                 profile_store: Optional[Any] = None,
                 trainer_id: str = None,
                 security_config: Optional[TrainingSecurityConfig] = None,
                 security_level: TrainingSecurityLevel = TrainingSecurityLevel.STANDARD):
        """
        Initialize SecureHegelTrainer with security controls.
        
        Args:
            grad: Training mode flag
            profile_store: PromptProfileStore instance
            trainer_id: Unique identifier for this trainer
            security_config: Security configuration
            security_level: Security level for this trainer
        """
        super().__init__(grad, profile_store, trainer_id)
        
        # Initialize security
        self.security_level = security_level
        self.security_config = security_config or self._create_default_security_config(security_level)
        self.security_manager = get_training_security_manager(self.security_config)
        self.cost_monitor = get_cost_monitor()
        
        # Security state tracking
        self._current_security_session: Optional[str] = None
        self._security_enabled = True
        self._threat_count = 0
        self._security_violations = []
        
        # Register emergency stop callback
        self.security_manager.add_emergency_stop_callback(self._handle_emergency_stop)
        
        self.logger.log_debug(f"SecureHegelTrainer initialized with {security_level.value} security")
    
    def _create_default_security_config(self, security_level: TrainingSecurityLevel) -> TrainingSecurityConfig:
        """Create default security configuration based on security level."""
        base_config = TrainingSecurityConfig(security_level=security_level)
        
        if security_level == TrainingSecurityLevel.PERMISSIVE:
            base_config.max_session_cost = 200.0
            base_config.max_session_duration_hours = 12.0
            base_config.enable_prompt_safety = True
            base_config.prompt_safety_strict = False
            base_config.auto_stop_on_threat = False
        
        elif security_level == TrainingSecurityLevel.STANDARD:
            base_config.max_session_cost = 100.0
            base_config.max_session_duration_hours = 8.0
            base_config.enable_prompt_safety = True
            base_config.prompt_safety_strict = True
            base_config.auto_stop_on_threat = True
        
        elif security_level == TrainingSecurityLevel.STRICT:
            base_config.max_session_cost = 50.0
            base_config.max_session_duration_hours = 4.0
            base_config.enable_prompt_safety = True
            base_config.prompt_safety_strict = True
            base_config.auto_stop_on_threat = True
            base_config.validate_training_data = True
            base_config.require_data_signatures = True
        
        elif security_level == TrainingSecurityLevel.PARANOID:
            base_config.max_session_cost = 25.0
            base_config.max_session_duration_hours = 2.0
            base_config.enable_prompt_safety = True
            base_config.prompt_safety_strict = True
            base_config.auto_stop_on_threat = True
            base_config.validate_training_data = True
            base_config.require_data_signatures = True
            base_config.max_concurrent_sessions = 1
            base_config.security_check_interval = 10.0
        
        return base_config
    
    def create_secure_training_session(self, 
                                     session_id: Optional[str] = None, 
                                     corpus_id: Optional[str] = None,
                                     task_type: str = "qa",
                                     max_cost: Optional[float] = None,
                                     max_duration_hours: Optional[float] = None) -> str:
        """
        Create a secure training session with integrated security controls.
        
        Args:
            session_id: Optional session identifier
            corpus_id: Corpus being used for training
            task_type: Type of task being performed
            max_cost: Maximum cost for this session
            max_duration_hours: Maximum duration in hours
            
        Returns:
            Session ID for tracking
        """
        if not self._security_enabled:
            raise RuntimeError("Security is disabled - cannot create secure session")
        
        # Create base training session
        base_session_id = super().create_training_session(session_id, corpus_id, task_type)
        
        # Create security session with same ID
        try:
            self._current_security_session = self.security_manager.create_training_session(
                session_id=base_session_id,
                max_cost=max_cost or self.security_config.max_session_cost,
                max_duration_hours=max_duration_hours or self.security_config.max_session_duration_hours,
                security_level=self.security_level,
                metadata={
                    "corpus_id": corpus_id,
                    "task_type": task_type,
                    "trainer_id": self.trainer_id,
                    "grad_mode": self.grad
                }
            )
            
            self.logger.log_info(f"Created secure training session '{base_session_id}' with integrated security")
            
            return base_session_id
            
        except Exception as e:
            # Clean up base session if security session creation failed
            super().end_training_session(base_session_id)
            raise RuntimeError(f"Failed to create secure training session: {e}") from e
    
    def end_secure_training_session(self, session_id: Optional[str] = None, reason: str = "Normal completion") -> Dict[str, Any]:
        """
        End secure training session with security summary.
        
        Args:
            session_id: Session to end (current session if None)
            reason: Reason for ending session
            
        Returns:
            Combined session summary with security data
        """
        session_id = session_id or self._current_session
        
        if not session_id:
            raise ValueError("No active session to end")
        
        # End base training session
        base_summary = super().end_training_session(session_id, reason)
        
        # End security session
        security_summary = {}
        if session_id == self._current_security_session:
            try:
                security_summary = self.security_manager.end_training_session(session_id, reason)
                self._current_security_session = None
            except Exception as e:
                self.logger.log_error(f"Error ending security session: {e}")
                security_summary = {"error": str(e)}
        
        # Combine summaries
        combined_summary = {
            **base_summary,
            "security_summary": security_summary,
            "threat_count": self._threat_count,
            "security_violations": self._security_violations.copy(),
            "security_level": self.security_level.value,
            "security_enabled": self._security_enabled
        }
        
        return combined_summary
    
    def wrap_worker_agent(self, 
                         agent, 
                         profile_id: Optional[str] = None,
                         agent_alias: Optional[str] = None) -> 'SecureTrainingWorkerWrapper':
        """
        Wrap a BasicWorkerAgent with enhanced security.
        
        Args:
            agent: BasicWorkerAgent instance to wrap
            profile_id: Optional prompt profile ID to apply
            agent_alias: Optional alias for tracking this agent
            
        Returns:
            SecureTrainingWorkerWrapper with security controls
        """
        # Create base wrapper
        base_wrapper = super().wrap_worker_agent(agent, profile_id, agent_alias)
        
        # Create secure wrapper
        secure_wrapper = SecureTrainingWorkerWrapper(
            base_wrapper=base_wrapper,
            trainer=self,
            security_manager=self.security_manager
        )
        
        # Update tracking
        wrapper_id = base_wrapper.wrapper_id
        self._wrapped_agents[wrapper_id]['wrapper'] = secure_wrapper
        self._wrapped_agents[wrapper_id]['security_enabled'] = True
        
        return secure_wrapper
    
    def wrap_reviewer_agent(self, 
                           agent, 
                           profile_id: Optional[str] = None,
                           agent_alias: Optional[str] = None) -> 'SecureTrainingReviewerWrapper':
        """
        Wrap a BasicReviewerAgent with enhanced security.
        
        Args:
            agent: BasicReviewerAgent instance to wrap
            profile_id: Optional prompt profile ID to apply
            agent_alias: Optional alias for tracking this agent
            
        Returns:
            SecureTrainingReviewerWrapper with security controls
        """
        # Create base wrapper
        base_wrapper = super().wrap_reviewer_agent(agent, profile_id, agent_alias)
        
        # Create secure wrapper
        secure_wrapper = SecureTrainingReviewerWrapper(
            base_wrapper=base_wrapper,
            trainer=self,
            security_manager=self.security_manager
        )
        
        # Update tracking
        wrapper_id = base_wrapper.wrapper_id
        self._wrapped_agents[wrapper_id]['wrapper'] = secure_wrapper
        self._wrapped_agents[wrapper_id]['security_enabled'] = True
        
        return secure_wrapper
    
    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """
        Trigger emergency stop for all training operations.
        
        Args:
            reason: Reason for emergency stop
        """
        self.security_manager.emergency_stop_all_sessions(
            EmergencyStopReason.MANUAL_STOP, 
            reason
        )
        
        self._security_enabled = False
        self.logger.log_critical(f"Emergency stop activated: {reason}")
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop and re-enable security."""
        self.security_manager.reset_emergency_stop()
        self._security_enabled = True
        self.logger.log_info("Emergency stop reset - secure training can resume")
    
    def _handle_emergency_stop(self, reason: EmergencyStopReason, description: str) -> None:
        """Handle emergency stop callback."""
        self._security_enabled = False
        
        # End all training sessions
        if self._current_session:
            try:
                super().end_training_session(self._current_session, f"Emergency stop: {reason.value}")
            except Exception as e:
                self.logger.log_error(f"Error ending session during emergency stop: {e}")
        
        self.logger.log_critical(f"Training stopped due to security emergency: {reason.value} - {description}")
    
    def validate_training_input(self, text: str, context: str = "training") -> Tuple[bool, List[str]]:
        """
        Validate training input for security threats.
        
        Args:
            text: Text to validate
            context: Context of the validation
            
        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        if not self._security_enabled:
            return True, []
        
        issues = []
        
        # Validate with current session
        if self._current_security_session:
            is_safe, session_issues = self.security_manager.validate_training_prompt(
                text, self._current_security_session
            )
            if not is_safe:
                issues.extend(session_issues)
                self._threat_count += 1
                self._security_violations.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "context": context,
                    "issues": session_issues,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                })
        else:
            # Fallback to direct prompt safety validation
            is_safe, threats = validate_prompt_safety(text, context)
            if not is_safe:
                issues = [f"Threat: {threat.description}" for threat in threats]
        
        return len(issues) == 0, issues
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        base_stats = self.get_stats()
        
        security_status = {
            **base_stats,
            "security_level": self.security_level.value,
            "security_enabled": self._security_enabled,
            "current_security_session": self._current_security_session,
            "threat_count": self._threat_count,
            "security_violations_count": len(self._security_violations),
            "security_manager_status": self.security_manager.get_security_status(),
            "recent_violations": self._security_violations[-5:]  # Last 5 violations
        }
        
        return security_status


class SecureTrainingWorkerWrapper:
    """Security-enhanced wrapper for TrainingWorkerWrapper."""
    
    def __init__(self, base_wrapper: TrainingWorkerWrapper, trainer: SecureHegelTrainer, security_manager):
        self.base_wrapper = base_wrapper
        self.trainer = trainer
        self.security_manager = security_manager
        
        # Expose base wrapper attributes
        self.wrapper_id = base_wrapper.wrapper_id
        self.profile_id = base_wrapper.profile_id
        self.agent_id = base_wrapper.agent_id
        self.logger = base_wrapper.logger
    
    def respond(self, question: str, external_context: Optional[str] = None) -> AgentResponse:
        """
        Generate response with security validation.
        
        Args:
            question: Question to answer
            external_context: Optional external context
            
        Returns:
            AgentResponse with security validation
        """
        # Validate input for security threats
        if self.trainer.grad and self.trainer._security_enabled:
            is_safe, issues = self.trainer.validate_training_input(question, "worker_question")
            if not is_safe:
                raise ValueError(f"Security validation failed: {'; '.join(issues)}")
            
            if external_context:
                is_safe, issues = self.trainer.validate_training_input(external_context, "worker_context")
                if not is_safe:
                    raise ValueError(f"Context security validation failed: {'; '.join(issues)}")
        
        # Record iteration if in training mode
        if self.trainer.grad and self.trainer._current_security_session:
            estimated_cost = 0.01  # Estimate cost per iteration
            self.security_manager.record_training_iteration(
                self.trainer._current_security_session, 
                estimated_cost
            )
        
        # Generate response using base wrapper
        return self.base_wrapper.respond(question, external_context)
    
    def __getattr__(self, name):
        """Delegate all other attributes to base wrapper."""
        return getattr(self.base_wrapper, name)


class SecureTrainingReviewerWrapper:
    """Security-enhanced wrapper for TrainingReviewerWrapper."""
    
    def __init__(self, base_wrapper: TrainingReviewerWrapper, trainer: SecureHegelTrainer, security_manager):
        self.base_wrapper = base_wrapper
        self.trainer = trainer
        self.security_manager = security_manager
        
        # Expose base wrapper attributes
        self.wrapper_id = base_wrapper.wrapper_id
        self.profile_id = base_wrapper.profile_id
        self.agent_id = base_wrapper.agent_id
        self.logger = base_wrapper.logger
    
    def critique_response(self, question: str, response: AgentResponse) -> AgentResponse:
        """Critique response with security validation."""
        # Validate inputs
        if self.trainer.grad and self.trainer._security_enabled:
            is_safe, issues = self.trainer.validate_training_input(question, "review_question")
            if not is_safe:
                raise ValueError(f"Question security validation failed: {'; '.join(issues)}")
            
            is_safe, issues = self.trainer.validate_training_input(response.content, "review_response")
            if not is_safe:
                raise ValueError(f"Response security validation failed: {'; '.join(issues)}")
        
        # Record iteration
        if self.trainer.grad and self.trainer._current_security_session:
            estimated_cost = 0.015  # Slightly higher cost for review operations
            self.security_manager.record_training_iteration(
                self.trainer._current_security_session, 
                estimated_cost
            )
        
        return self.base_wrapper.critique_response(question, response)
    
    def synthesize_responses(self, question: str, responses: List[AgentResponse]) -> AgentResponse:
        """Synthesize responses with security validation."""
        # Validate inputs
        if self.trainer.grad and self.trainer._security_enabled:
            is_safe, issues = self.trainer.validate_training_input(question, "synthesis_question")
            if not is_safe:
                raise ValueError(f"Question security validation failed: {'; '.join(issues)}")
            
            for i, response in enumerate(responses):
                is_safe, issues = self.trainer.validate_training_input(response.content, f"synthesis_response_{i}")
                if not is_safe:
                    raise ValueError(f"Response {i} security validation failed: {'; '.join(issues)}")
        
        # Record iteration
        if self.trainer.grad and self.trainer._current_security_session:
            estimated_cost = 0.02  # Higher cost for synthesis
            self.security_manager.record_training_iteration(
                self.trainer._current_security_session, 
                estimated_cost
            )
        
        return self.base_wrapper.synthesize_responses(question, responses)
    
    def __getattr__(self, name):
        """Delegate all other attributes to base wrapper."""
        return getattr(self.base_wrapper, name)


# Factory functions for easy creation

def create_secure_trainer(grad: bool = False, 
                         security_level: TrainingSecurityLevel = TrainingSecurityLevel.STANDARD,
                         **kwargs) -> SecureHegelTrainer:
    """
    Factory function to create SecureHegelTrainer instance.
    
    Args:
        grad: Training mode flag
        security_level: Security level for the trainer
        **kwargs: Additional arguments for SecureHegelTrainer
        
    Returns:
        SecureHegelTrainer instance
    """
    return SecureHegelTrainer(grad=grad, security_level=security_level, **kwargs)


def create_secure_training_environment(security_level: TrainingSecurityLevel = TrainingSecurityLevel.STANDARD,
                                     max_cost: float = 100.0,
                                     max_duration_hours: float = 8.0) -> Tuple[SecureHegelTrainer, str]:
    """
    Create a complete secure training environment.
    
    Args:
        security_level: Security level for the environment
        max_cost: Maximum cost for the training session
        max_duration_hours: Maximum duration for the session
        
    Returns:
        Tuple of (trainer, session_id)
    """
    trainer = create_secure_trainer(grad=True, security_level=security_level)
    session_id = trainer.create_secure_training_session(
        max_cost=max_cost,
        max_duration_hours=max_duration_hours
    )
    
    return trainer, session_id


# Export main classes and functions
__all__ = [
    'SecureHegelTrainer',
    'SecureTrainingWorkerWrapper', 
    'SecureTrainingReviewerWrapper',
    'create_secure_trainer',
    'create_secure_training_environment'
]
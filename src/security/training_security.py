"""
Training Security Module

Provides specialized security controls for training operations including
emergency stop mechanisms, training data validation, and secure training session management.
"""

import time
import uuid
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from .cost_monitor import get_cost_monitor, CostMonitor
from .input_validator import get_validator, InputValidator
from .security_logger import get_security_logger, SecurityLogger, SecurityEventType
from .prompt_safety_validator import get_prompt_safety_validator, PromptSafetyValidator, PromptSafetyConfig
from .rate_limiter import get_rate_limiter, RateLimiter

logger = logging.getLogger(__name__)


class TrainingSecurityLevel(Enum):
    """Training security levels."""
    PERMISSIVE = "permissive"      # Basic security for development
    STANDARD = "standard"          # Normal production security
    STRICT = "strict"              # High security for sensitive training
    PARANOID = "paranoid"          # Maximum security for critical systems


class EmergencyStopReason(Enum):
    """Reasons for emergency stop."""
    COST_EXCEEDED = "cost_exceeded"
    THREAT_DETECTED = "threat_detected"
    MANUAL_STOP = "manual_stop"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT_BREACH = "rate_limit_breach"
    DATA_POISONING = "data_poisoning"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class TrainingSecurityConfig:
    """Configuration for training security."""
    security_level: TrainingSecurityLevel = TrainingSecurityLevel.STANDARD
    max_session_duration_hours: float = 8.0
    max_session_cost: float = 100.0
    max_training_iterations: int = 10000
    
    # Prompt safety settings
    enable_prompt_safety: bool = True
    prompt_safety_strict: bool = True
    
    # Emergency stop settings
    enable_emergency_stop: bool = True
    auto_stop_on_threat: bool = True
    auto_stop_on_cost_exceed: bool = True
    
    # Data validation settings
    validate_training_data: bool = True
    require_data_signatures: bool = False
    max_training_data_size: int = 100 * 1024 * 1024  # 100MB
    
    # Monitoring settings
    log_all_training_events: bool = True
    alert_on_suspicious_activity: bool = True
    store_training_artifacts: bool = True
    
    # Performance settings
    max_concurrent_sessions: int = 5
    security_check_interval: float = 30.0  # seconds


@dataclass
class TrainingSession:
    """Training session metadata."""
    session_id: str
    started_at: datetime
    security_level: TrainingSecurityLevel
    max_cost: float
    max_duration_hours: float
    cost_spent: float = 0.0
    iterations_completed: int = 0
    threat_detections: int = 0
    emergency_stopped: bool = False
    stop_reason: Optional[EmergencyStopReason] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingSecurityManager:
    """
    Comprehensive security manager for training operations.
    
    Features:
    - Emergency stop mechanisms
    - Training session cost and duration limits
    - Prompt safety validation for training data
    - Threat detection and response
    - Secure training data validation
    - Session isolation and monitoring
    """
    
    def __init__(self, config: TrainingSecurityConfig):
        self.config = config
        self._lock = threading.Lock()
        
        # Initialize security components
        self.cost_monitor = get_cost_monitor()
        self.input_validator = get_validator()
        self.security_logger = get_security_logger()
        self.prompt_safety_validator = get_prompt_safety_validator(
            PromptSafetyConfig(
                training_mode_strict=config.prompt_safety_strict,
                log_all_detections=config.log_all_training_events
            )
        )
        self.rate_limiter = get_rate_limiter()
        
        # Training session management
        self._active_sessions: Dict[str, TrainingSession] = {}
        self._emergency_stop_active = False
        self._global_stop_callbacks: List[Callable] = []
        
        # Statistics
        self._stats = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'emergency_stops_triggered': 0,
            'threats_detected': 0,
            'cost_violations': 0,
            'validation_failures': 0
        }
        
        # Start monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info(f"TrainingSecurityManager initialized with {config.security_level.value} security level")
    
    def create_training_session(self, 
                               session_id: Optional[str] = None,
                               max_cost: Optional[float] = None,
                               max_duration_hours: Optional[float] = None,
                               security_level: Optional[TrainingSecurityLevel] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new secure training session.
        
        Args:
            session_id: Session identifier (auto-generated if None)
            max_cost: Maximum cost for this session
            max_duration_hours: Maximum duration in hours
            security_level: Security level for this session
            metadata: Additional session metadata
            
        Returns:
            Session ID
            
        Raises:
            ValueError: If session creation fails
        """
        with self._lock:
            # Check if emergency stop is active
            if self._emergency_stop_active:
                raise ValueError("Cannot create training session: Emergency stop is active")
            
            # Check concurrent session limit
            if len(self._active_sessions) >= self.config.max_concurrent_sessions:
                raise ValueError(f"Maximum concurrent sessions ({self.config.max_concurrent_sessions}) reached")
            
            # Generate session ID
            session_id = session_id or f"training_{uuid.uuid4().hex[:12]}"
            
            if session_id in self._active_sessions:
                raise ValueError(f"Training session '{session_id}' already exists")
            
            # Create session
            session = TrainingSession(
                session_id=session_id,
                started_at=datetime.utcnow(),
                security_level=security_level or self.config.security_level,
                max_cost=max_cost or self.config.max_session_cost,
                max_duration_hours=max_duration_hours or self.config.max_session_duration_hours,
                metadata=metadata
            )
            
            self._active_sessions[session_id] = session
            
            # Initialize cost tracking for this session
            self.cost_monitor.start_training_session(session_id, session.max_cost)
            
            # Log session creation
            self.security_logger.log_event(
                event_type=SecurityEventType.INFO,
                severity="low",
                message=f"Training session '{session_id}' created with {session.security_level.value} security",
                metadata={
                    "session_id": session_id,
                    "max_cost": session.max_cost,
                    "max_duration": session.max_duration_hours,
                    "security_level": session.security_level.value
                }
            )
            
            self._stats['sessions_created'] += 1
            
            logger.info(f"Created training session '{session_id}' with max cost ${session.max_cost}")
            
            return session_id
    
    def end_training_session(self, session_id: str, reason: str = "Normal completion") -> Dict[str, Any]:
        """
        End a training session and return summary.
        
        Args:
            session_id: Session to end
            reason: Reason for ending session
            
        Returns:
            Session summary
        """
        with self._lock:
            if session_id not in self._active_sessions:
                raise ValueError(f"Training session '{session_id}' not found")
            
            session = self._active_sessions[session_id]
            
            # End cost tracking
            cost_summary = self.cost_monitor.end_training_session(session_id)
            session.cost_spent = cost_summary['total_cost']
            
            # Calculate session duration
            duration = (datetime.utcnow() - session.started_at).total_seconds() / 3600
            
            # Create summary
            summary = {
                "session_id": session_id,
                "started_at": session.started_at.isoformat(),
                "ended_at": datetime.utcnow().isoformat(),
                "duration_hours": duration,
                "cost_spent": session.cost_spent,
                "iterations_completed": session.iterations_completed,
                "threat_detections": session.threat_detections,
                "emergency_stopped": session.emergency_stopped,
                "stop_reason": session.stop_reason.value if session.stop_reason else None,
                "reason": reason,
                "cost_summary": cost_summary
            }
            
            # Remove from active sessions
            del self._active_sessions[session_id]
            
            # Log session completion
            self.security_logger.log_event(
                event_type=SecurityEventType.INFO,
                severity="low",
                message=f"Training session '{session_id}' ended: {reason}",
                metadata=summary
            )
            
            self._stats['sessions_completed'] += 1
            
            logger.info(f"Ended training session '{session_id}' - Duration: {duration:.2f}h, Cost: ${session.cost_spent:.4f}")
            
            return summary
    
    def emergency_stop_all_sessions(self, reason: EmergencyStopReason, description: str = "") -> None:
        """
        Trigger emergency stop for all active training sessions.
        
        Args:
            reason: Reason for emergency stop
            description: Additional description
        """
        with self._lock:
            self._emergency_stop_active = True
            
            # Stop all active sessions
            for session_id, session in self._active_sessions.items():
                session.emergency_stopped = True
                session.stop_reason = reason
                
                # Trigger cost monitor emergency stop
                self.cost_monitor.emergency_stop(f"Training emergency stop: {reason.value}")
            
            # Log critical event
            self.security_logger.log_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity="critical",
                message=f"EMERGENCY STOP: All training sessions stopped - {reason.value}",
                metadata={
                    "reason": reason.value,
                    "description": description,
                    "active_sessions": list(self._active_sessions.keys()),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Call emergency stop callbacks
            for callback in self._global_stop_callbacks:
                try:
                    callback(reason, description)
                except Exception as e:
                    logger.error(f"Error in emergency stop callback: {e}")
            
            self._stats['emergency_stops_triggered'] += 1
            
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason.value} - {description}")
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop state."""
        with self._lock:
            self._emergency_stop_active = False
            self.cost_monitor.reset_emergency_stop()
            
            self.security_logger.log_event(
                event_type=SecurityEventType.INFO,
                severity="medium",
                message="Emergency stop reset - training can resume"
            )
            
            logger.info("Emergency stop reset - training operations can resume")
    
    def validate_training_prompt(self, prompt: str, session_id: str) -> Tuple[bool, List[str]]:
        """
        Validate training prompt for security threats.
        
        Args:
            prompt: Prompt to validate
            session_id: Training session ID
            
        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        if self._emergency_stop_active:
            return False, ["Emergency stop is active"]
        
        session = self._active_sessions.get(session_id)
        if not session:
            return False, [f"Invalid session ID: {session_id}"]
        
        issues = []
        
        try:
            # Basic input validation
            self.input_validator.validate_text(prompt, "training_prompt")
        except Exception as e:
            issues.append(f"Input validation failed: {e}")
            self._stats['validation_failures'] += 1
        
        # Prompt safety validation
        if self.config.enable_prompt_safety:
            is_safe, threats = self.prompt_safety_validator.validate_prompt(prompt, "training")
            
            if not is_safe:
                session.threat_detections += 1
                self._stats['threats_detected'] += 1
                
                for threat in threats:
                    issues.append(f"Threat detected: {threat.description} (confidence: {threat.confidence:.2f})")
                
                # Auto-stop on high-confidence threats
                if (self.config.auto_stop_on_threat and 
                    any(threat.confidence >= 0.9 for threat in threats)):
                    self.emergency_stop_session(session_id, EmergencyStopReason.THREAT_DETECTED, 
                                              f"High-confidence threat detected in training prompt")
                
                # Log threat detection
                self.security_logger.log_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    severity="high" if any(threat.confidence >= 0.8 for threat in threats) else "medium",
                    message=f"Training prompt threat detected in session {session_id}",
                    metadata={
                        "session_id": session_id,
                        "threats": [threat.description for threat in threats],
                        "max_confidence": max(threat.confidence for threat in threats)
                    }
                )
        
        return len(issues) == 0, issues
    
    def emergency_stop_session(self, session_id: str, reason: EmergencyStopReason, description: str = "") -> None:
        """
        Emergency stop a specific training session.
        
        Args:
            session_id: Session to stop
            reason: Reason for stop
            description: Additional description
        """
        with self._lock:
            if session_id not in self._active_sessions:
                logger.warning(f"Attempted to emergency stop non-existent session: {session_id}")
                return
            
            session = self._active_sessions[session_id]
            session.emergency_stopped = True
            session.stop_reason = reason
            
            # Log emergency stop
            self.security_logger.log_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity="high",
                message=f"Emergency stop for training session {session_id}: {reason.value}",
                metadata={
                    "session_id": session_id,
                    "reason": reason.value,
                    "description": description,
                    "cost_spent": session.cost_spent,
                    "iterations": session.iterations_completed
                }
            )
            
            logger.warning(f"Emergency stopped training session '{session_id}': {reason.value}")
    
    def record_training_iteration(self, session_id: str, cost: float = 0.0) -> None:
        """
        Record completion of a training iteration.
        
        Args:
            session_id: Session ID
            cost: Cost of this iteration
        """
        with self._lock:
            if session_id not in self._active_sessions:
                logger.warning(f"Recording iteration for unknown session: {session_id}")
                return
            
            session = self._active_sessions[session_id]
            session.iterations_completed += 1
            session.cost_spent += cost
            
            # Check limits
            if session.iterations_completed >= self.config.max_training_iterations:
                self.emergency_stop_session(
                    session_id, 
                    EmergencyStopReason.RESOURCE_EXHAUSTION,
                    f"Maximum iterations reached: {session.iterations_completed}"
                )
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self._check_session_limits()
                time.sleep(self.config.security_check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_session_limits(self) -> None:
        """Check all active sessions for limit violations."""
        current_time = datetime.utcnow()
        
        with self._lock:
            sessions_to_stop = []
            
            for session_id, session in self._active_sessions.items():
                # Check duration limit
                duration_hours = (current_time - session.started_at).total_seconds() / 3600
                if duration_hours >= session.max_duration_hours:
                    sessions_to_stop.append((session_id, EmergencyStopReason.RESOURCE_EXHAUSTION, 
                                           f"Maximum duration exceeded: {duration_hours:.2f}h"))
                
                # Check cost limit
                current_cost = self.cost_monitor.get_training_session_cost(session_id)
                if current_cost >= session.max_cost:
                    sessions_to_stop.append((session_id, EmergencyStopReason.COST_EXCEEDED,
                                           f"Maximum cost exceeded: ${current_cost:.4f}"))
                    self._stats['cost_violations'] += 1
            
            # Stop sessions that exceeded limits
            for session_id, reason, description in sessions_to_stop:
                self.emergency_stop_session(session_id, reason, description)
    
    def add_emergency_stop_callback(self, callback: Callable) -> None:
        """Add callback for emergency stop events."""
        self._global_stop_callbacks.append(callback)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active training sessions."""
        with self._lock:
            sessions = []
            for session_id, session in self._active_sessions.items():
                duration = (datetime.utcnow() - session.started_at).total_seconds() / 3600
                current_cost = self.cost_monitor.get_training_session_cost(session_id)
                
                sessions.append({
                    "session_id": session_id,
                    "started_at": session.started_at.isoformat(),
                    "duration_hours": duration,
                    "cost_spent": current_cost,
                    "iterations_completed": session.iterations_completed,
                    "threat_detections": session.threat_detections,
                    "emergency_stopped": session.emergency_stopped,
                    "security_level": session.security_level.value,
                    "max_cost": session.max_cost,
                    "max_duration": session.max_duration_hours
                })
            
            return sessions
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        with self._lock:
            return {
                "emergency_stop_active": self._emergency_stop_active,
                "active_sessions": len(self._active_sessions),
                "max_concurrent_sessions": self.config.max_concurrent_sessions,
                "security_level": self.config.security_level.value,
                "statistics": self._stats.copy(),
                "threat_statistics": self.prompt_safety_validator.get_threat_statistics(),
                "cost_monitor_status": self.cost_monitor.get_spending_summary()
            }
    
    def shutdown(self) -> None:
        """Shutdown security manager."""
        self._monitoring_active = False
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        # End all active sessions
        active_sessions = list(self._active_sessions.keys())
        for session_id in active_sessions:
            try:
                self.end_training_session(session_id, "Security manager shutdown")
            except Exception as e:
                logger.error(f"Error ending session {session_id} during shutdown: {e}")
        
        logger.info("TrainingSecurityManager shutdown complete")


# Global training security manager
_training_security_manager: Optional[TrainingSecurityManager] = None


def get_training_security_manager(config: Optional[TrainingSecurityConfig] = None) -> TrainingSecurityManager:
    """
    Get or create the global training security manager.
    
    Args:
        config: Training security configuration
        
    Returns:
        TrainingSecurityManager instance
    """
    global _training_security_manager
    
    if _training_security_manager is None:
        _training_security_manager = TrainingSecurityManager(config or TrainingSecurityConfig())
    
    return _training_security_manager


def reset_training_security_manager() -> None:
    """Reset the global training security manager."""
    global _training_security_manager
    if _training_security_manager:
        _training_security_manager.shutdown()
    _training_security_manager = None


# Convenience functions
def create_secure_training_session(**kwargs) -> str:
    """Create secure training session using global manager."""
    return get_training_security_manager().create_training_session(**kwargs)


def emergency_stop_all_training(reason: EmergencyStopReason, description: str = "") -> None:
    """Emergency stop all training using global manager."""
    get_training_security_manager().emergency_stop_all_sessions(reason, description)


def validate_training_prompt_safety(prompt: str, session_id: str) -> Tuple[bool, List[str]]:
    """Validate training prompt safety using global manager."""
    return get_training_security_manager().validate_training_prompt(prompt, session_id)
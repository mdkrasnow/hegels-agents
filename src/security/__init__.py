"""
Security and Cost Control Module for Hegels Agents

This module provides comprehensive security controls and cost monitoring for API usage,
including rate limiting, budget monitoring, security validation, input sanitization,
security logging, secure API wrappers, prompt safety validation, training security,
emergency stop mechanisms, and comprehensive security testing.
"""

from .cost_monitor import CostMonitor, CostConfig, CostExceededError, get_cost_monitor
from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitError, get_rate_limiter
from .input_validator import InputValidator, ValidationConfig, ValidationError, get_validator
from .security_logger import SecurityLogger, SecurityLogConfig, SecurityEventType, get_security_logger
from .api_security import ApiSecurityWrapper, SecurityConfig, SecurityError, get_security_wrapper, create_secure_api_call
from .prompt_safety_validator import (
    PromptSafetyValidator, PromptSafetyConfig, ThreatType, ThreatLevel,
    ThreatDetection, get_prompt_safety_validator, validate_prompt_safety, is_prompt_safe
)
from .training_security import (
    TrainingSecurityManager, TrainingSecurityConfig, TrainingSecurityLevel,
    EmergencyStopReason, TrainingSession, get_training_security_manager,
    create_secure_training_session, emergency_stop_all_training, validate_training_prompt_safety
)
from .security_test_suite import (
    SecurityTestSuite, SecurityTestResult, TestSeverity, TestStatus,
    run_security_assessment, run_penetration_tests, validate_security_configuration
)

__all__ = [
    # Core classes
    'CostMonitor', 'RateLimiter', 'InputValidator', 'SecurityLogger', 'ApiSecurityWrapper',
    'PromptSafetyValidator', 'TrainingSecurityManager', 'SecurityTestSuite',
    
    # Configuration classes
    'CostConfig', 'RateLimitConfig', 'ValidationConfig', 'SecurityLogConfig', 'SecurityConfig',
    'PromptSafetyConfig', 'TrainingSecurityConfig',
    
    # Exceptions
    'ValidationError', 'RateLimitError', 'CostExceededError', 'SecurityError',
    
    # Enums
    'SecurityEventType', 'ThreatType', 'ThreatLevel', 'TrainingSecurityLevel',
    'EmergencyStopReason', 'TestSeverity', 'TestStatus',
    
    # Data classes
    'ThreatDetection', 'TrainingSession', 'SecurityTestResult',
    
    # Factory functions
    'get_cost_monitor', 'get_rate_limiter', 'get_validator', 'get_security_logger',
    'get_security_wrapper', 'create_secure_api_call', 'get_prompt_safety_validator',
    'get_training_security_manager',
    
    # Convenience functions
    'validate_prompt_safety', 'is_prompt_safe', 'create_secure_training_session',
    'emergency_stop_all_training', 'validate_training_prompt_safety',
    'run_security_assessment', 'run_penetration_tests', 'validate_security_configuration'
]
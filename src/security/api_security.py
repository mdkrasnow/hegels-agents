"""
API Security Wrapper Module

Provides secure API call wrapper that integrates rate limiting, cost monitoring,
input validation, and security logging for all external API calls.
"""

import time
import uuid
import functools
from typing import Any, Dict, Optional, Callable, Tuple, Union
from dataclasses import dataclass
import logging

from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitError, get_rate_limiter
from .cost_monitor import CostMonitor, CostConfig, CostExceededError, get_cost_monitor
from .input_validator import InputValidator, ValidationConfig, ValidationError, get_validator
from .security_logger import SecurityLogger, SecurityLogConfig, SecurityEventType, get_security_logger

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Complete security configuration."""
    rate_limit_config: RateLimitConfig
    cost_config: CostConfig
    validation_config: ValidationConfig
    security_log_config: SecurityLogConfig
    
    # Security features
    enable_request_logging: bool = True
    enable_response_logging: bool = False
    enable_cost_estimation: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_max_wait: float = 60.0


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class ApiSecurityWrapper:
    """
    Secure wrapper for API calls with comprehensive protection.
    
    Features:
    - Rate limiting with multiple strategies
    - Cost estimation and budget enforcement
    - Input validation and sanitization
    - Security logging and audit trails
    - Automatic retries with backoff
    - Request/response monitoring
    """
    
    def __init__(self, config: SecurityConfig, provider: str = "default"):
        self.config = config
        self.provider = provider
        
        # Initialize security components
        self.rate_limiter = get_rate_limiter("default", config.rate_limit_config)
        self.cost_monitor = get_cost_monitor(config.cost_config)
        self.validator = get_validator(config.validation_config)
        self.security_logger = get_security_logger(config.security_log_config)
        
        logger.info(f"ApiSecurityWrapper initialized for provider: {provider}")
    
    def secure_api_call(self,
                       endpoint: str,
                       api_function: Callable,
                       *args,
                       estimated_output_tokens: int = 500,
                       operation_type: str = "generate",
                       model_name: str = "default",
                       validate_input: bool = True,
                       **kwargs) -> Any:
        """
        Make a secure API call with full protection.
        
        Args:
            endpoint: API endpoint identifier
            api_function: Function to call for API request
            *args: Arguments for api_function
            estimated_output_tokens: Estimated tokens for cost calculation
            operation_type: Type of operation (generate, embed, etc.)
            model_name: Model name for cost calculation
            validate_input: Whether to validate input parameters
            **kwargs: Keyword arguments for api_function
            
        Returns:
            API response
            
        Raises:
            SecurityError: If security checks fail
            RateLimitError: If rate limited
            CostExceededError: If cost budget exceeded
            ValidationError: If input validation fails
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Step 1: Input validation
            if validate_input and self.config.enable_input_validation:
                self._validate_inputs(args, kwargs, request_id)
            
            # Step 2: Prepare input for cost estimation
            input_text = self._extract_text_for_estimation(args, kwargs)
            
            # Step 3: Cost estimation
            estimated_cost = 0.0
            if self.config.enable_cost_estimation:
                estimated_cost = self.cost_monitor.estimate_cost(
                    provider=self.provider,
                    model=model_name,
                    input_text=input_text,
                    estimated_output_tokens=estimated_output_tokens,
                    operation_type=operation_type
                )
            
            # Step 4: Security checks
            self._perform_security_checks(endpoint, estimated_cost, request_id)
            
            # Step 5: Make API call with retries
            response, actual_cost = self._make_api_call_with_retries(
                api_function, args, kwargs, request_id, start_time
            )
            
            # Step 6: Record successful usage
            self._record_successful_usage(
                endpoint, model_name, input_text, estimated_cost, 
                actual_cost, estimated_output_tokens, request_id, start_time
            )
            
            # Step 7: Log successful request
            if self.config.enable_request_logging:
                self.security_logger.log_api_request(
                    endpoint=endpoint,
                    method="POST",
                    status_code=200,
                    request_id=request_id,
                    response_time=time.time() - start_time,
                    request_data={"model": model_name, "estimated_cost": estimated_cost}
                )
            
            return response
            
        except Exception as e:
            # Log error
            self.security_logger.log_api_request(
                endpoint=endpoint,
                method="POST", 
                status_code=500,
                request_id=request_id,
                response_time=time.time() - start_time,
                error=str(e)
            )
            
            # Re-raise with context
            if isinstance(e, (RateLimitError, CostExceededError, ValidationError)):
                raise
            else:
                raise SecurityError(f"API call failed: {e}") from e
    
    def _validate_inputs(self, args: tuple, kwargs: dict, request_id: str) -> None:
        """Validate input parameters."""
        try:
            # Validate positional arguments
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    self.validator.validate_text(arg, f"arg_{i}")
                elif isinstance(arg, dict):
                    self.validator.validate_json(arg, f"arg_{i}")
            
            # Validate keyword arguments
            for key, value in kwargs.items():
                if isinstance(value, str):
                    self.validator.validate_text(value, key)
                elif isinstance(value, dict):
                    self.validator.validate_json(value, key)
                    
        except ValidationError as e:
            self.security_logger.log_event(
                event_type=SecurityEventType.INPUT_VALIDATION_ERROR,
                severity="medium",
                message=f"Input validation failed: {e}",
                request_id=request_id,
                metadata={"field": e.field}
            )
            raise
    
    def _extract_text_for_estimation(self, args: tuple, kwargs: dict) -> str:
        """Extract text content for cost estimation."""
        text_content = []
        
        # Extract from args
        for arg in args:
            if isinstance(arg, str):
                text_content.append(arg)
            elif isinstance(arg, dict):
                text_content.extend(self._extract_text_from_dict(arg))
        
        # Extract from kwargs
        for value in kwargs.values():
            if isinstance(value, str):
                text_content.append(value)
            elif isinstance(value, dict):
                text_content.extend(self._extract_text_from_dict(value))
        
        return " ".join(text_content)
    
    def _extract_text_from_dict(self, data: dict) -> list:
        """Recursively extract text from dictionary."""
        texts = []
        
        for value in data.values():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                texts.extend(self._extract_text_from_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        texts.extend(self._extract_text_from_dict(item))
        
        return texts
    
    def _perform_security_checks(self, endpoint: str, estimated_cost: float, request_id: str) -> None:
        """Perform security checks before API call."""
        try:
            # Rate limiting check
            if self.config.enable_rate_limiting:
                self.rate_limiter.check_and_consume(endpoint, estimated_cost)
            
            # Cost budget check
            if self.config.enable_cost_estimation:
                self.cost_monitor.check_budget_before_request(estimated_cost)
                
        except RateLimitError as e:
            self.security_logger.log_event(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                severity="medium",
                message=f"Rate limit exceeded for {endpoint}: {e}",
                request_id=request_id,
                endpoint=endpoint,
                metadata={"retry_after": e.retry_after}
            )
            raise
            
        except CostExceededError as e:
            self.security_logger.log_event(
                event_type=SecurityEventType.COST_BUDGET_EXCEEDED,
                severity="high",
                message=f"Cost budget exceeded: {e}",
                request_id=request_id,
                endpoint=endpoint,
                metadata={"current_spend": e.current_spend, "budget": e.budget}
            )
            raise
    
    def _make_api_call_with_retries(self,
                                   api_function: Callable,
                                   args: tuple,
                                   kwargs: dict,
                                   request_id: str,
                                   start_time: float) -> Tuple[Any, Optional[float]]:
        """Make API call with retry logic."""
        last_exception = None
        wait_time = 1.0
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = api_function(*args, **kwargs)
                
                # Try to extract cost information if available
                actual_cost = None
                if hasattr(response, 'usage') and hasattr(response.usage, 'cost'):
                    actual_cost = response.usage.cost
                
                return response, actual_cost
                
            except Exception as e:
                last_exception = e
                
                # Don't retry certain errors
                if isinstance(e, (ValidationError, SecurityError)):
                    raise
                
                # Don't retry on last attempt
                if attempt >= self.config.max_retries:
                    break
                
                # Calculate wait time with exponential backoff
                wait_time = min(wait_time * self.config.retry_backoff_multiplier,
                               self.config.retry_max_wait)
                
                logger.warning(f"API call attempt {attempt + 1} failed: {e}. "
                              f"Retrying in {wait_time:.1f}s")
                
                # Log retry attempt
                self.security_logger.log_event(
                    event_type=SecurityEventType.API_ERROR,
                    severity="low",
                    message=f"API call retry {attempt + 1}: {e}",
                    request_id=request_id,
                    metadata={"attempt": attempt + 1, "wait_time": wait_time}
                )
                
                time.sleep(wait_time)
        
        # All retries failed
        raise SecurityError(f"API call failed after {self.config.max_retries} retries") from last_exception
    
    def _record_successful_usage(self,
                               endpoint: str,
                               model_name: str,
                               input_text: str,
                               estimated_cost: float,
                               actual_cost: Optional[float],
                               estimated_output_tokens: int,
                               request_id: str,
                               start_time: float) -> None:
        """Record usage statistics for successful API call."""
        # Estimate tokens (rough approximation)
        input_tokens = max(1, len(input_text) // 4)
        output_tokens = estimated_output_tokens  # We don't have actual output tokens
        
        # Record in cost monitor
        self.cost_monitor.record_usage(
            provider=self.provider,
            model=model_name,
            endpoint=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            request_id=request_id
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "provider": self.provider,
            "rate_limiter": self.rate_limiter.get_status(),
            "cost_monitor": self.cost_monitor.get_spending_summary(),
            "security_events": self.security_logger.get_security_summary(),
            "configuration": {
                "rate_limiting_enabled": self.config.enable_rate_limiting,
                "cost_monitoring_enabled": self.config.enable_cost_estimation,
                "input_validation_enabled": self.config.enable_input_validation,
                "request_logging_enabled": self.config.enable_request_logging,
            }
        }


def secure_api_decorator(wrapper: ApiSecurityWrapper,
                        endpoint: str,
                        estimated_output_tokens: int = 500,
                        operation_type: str = "generate",
                        model_name: str = "default"):
    """
    Decorator to automatically apply security wrapper to API functions.
    
    Args:
        wrapper: ApiSecurityWrapper instance
        endpoint: API endpoint identifier
        estimated_output_tokens: Estimated output tokens
        operation_type: Type of operation
        model_name: Model name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            return wrapper.secure_api_call(
                endpoint=endpoint,
                api_function=func,
                *args,
                estimated_output_tokens=estimated_output_tokens,
                operation_type=operation_type,
                model_name=model_name,
                **kwargs
            )
        return wrapper_func
    return decorator


# Global security wrapper instances
_security_wrappers: Dict[str, ApiSecurityWrapper] = {}


def get_security_wrapper(provider: str = "default", 
                        config: Optional[SecurityConfig] = None) -> ApiSecurityWrapper:
    """
    Get or create security wrapper for provider.
    
    Args:
        provider: API provider name
        config: Security configuration
        
    Returns:
        ApiSecurityWrapper instance
    """
    global _security_wrappers
    
    if provider not in _security_wrappers:
        if config is None:
            # Create default configuration
            config = SecurityConfig(
                rate_limit_config=RateLimitConfig(),
                cost_config=CostConfig(),
                validation_config=ValidationConfig(),
                security_log_config=SecurityLogConfig()
            )
        
        _security_wrappers[provider] = ApiSecurityWrapper(config, provider)
    
    return _security_wrappers[provider]


def reset_security_wrappers() -> None:
    """Reset all security wrappers (mainly for testing)."""
    global _security_wrappers
    _security_wrappers.clear()


def create_secure_api_call(provider: str = "gemini") -> Callable:
    """
    Create a secure API call function for a specific provider.
    
    Args:
        provider: API provider name
        
    Returns:
        Function that makes secure API calls
    """
    wrapper = get_security_wrapper(provider)
    
    def secure_call(endpoint: str, api_function: Callable, *args, **kwargs):
        return wrapper.secure_api_call(endpoint, api_function, *args, **kwargs)
    
    return secure_call
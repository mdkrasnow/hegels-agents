"""
Secure Agent Wrapper

Provides secure wrappers for existing agent API calls with minimal code changes.
Integrates security controls into existing agent implementations.
"""

import time
import logging
from typing import Any, Optional
import google.generativeai as genai

try:
    from ..config.settings import get_config
except ImportError:
    from config.settings import get_config
from .api_security import get_security_wrapper, SecurityConfig
from .rate_limiter import RateLimitConfig
from .cost_monitor import CostConfig
from .input_validator import ValidationConfig
from .security_logger import SecurityLogConfig

logger = logging.getLogger(__name__)


class SecureGeminiClient:
    """
    Secure wrapper for Gemini API client that provides transparent security integration.
    
    This class can be used as a drop-in replacement for the Gemini client in existing
    agent code to add security controls without major refactoring.
    """
    
    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash'):
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize the original client
        self.client = genai.Client(api_key=api_key)
        
        # Initialize security wrapper with production-safe defaults
        security_config = SecurityConfig(
            rate_limit_config=RateLimitConfig(
                requests_per_minute=30,  # Conservative for production
                requests_per_hour=500,
                requests_per_day=5000,
                burst_allowance=5,
                cost_per_request=0.001,  # Estimated
                max_daily_cost=50.0
            ),
            cost_config=CostConfig(
                daily_budget=50.0,
                weekly_budget=200.0,
                monthly_budget=800.0,
                warning_threshold=0.8,
                critical_threshold=0.95
            ),
            validation_config=ValidationConfig(
                max_text_length=200000,  # Large for AI prompts
                allow_html=False,
                allow_scripts=False,
                block_sql_injection=True,
                block_xss_attempts=True,
                block_path_traversal=True
            ),
            security_log_config=SecurityLogConfig(
                log_file_path="logs/security_events.log",
                include_request_data=True,
                include_response_data=False,  # Don't log full responses
                sanitize_sensitive_data=True
            )
        )
        
        self.security_wrapper = get_security_wrapper("gemini", security_config)
        
        logger.info(f"SecureGeminiClient initialized for model: {model_name}")
    
    @property
    def models(self):
        """Provide access to models with security wrapper."""
        return SecureModelsInterface(self.client.models, self.security_wrapper, self.model_name)


class SecureModelsInterface:
    """Secure interface for Gemini models."""
    
    def __init__(self, original_models, security_wrapper, model_name):
        self._original_models = original_models
        self._security_wrapper = security_wrapper
        self._model_name = model_name
    
    def generate_content(self, model: str = None, contents: str = "", config: Any = None) -> Any:
        """
        Secure content generation with full security controls.
        
        Args:
            model: Model name (optional, uses default if not provided)
            contents: Content/prompt to send
            config: Generation configuration
            
        Returns:
            Response object with generated content
        """
        model = model or self._model_name
        
        # Estimate output tokens based on config or use default
        estimated_output_tokens = 2500  # Default
        if config and hasattr(config, 'max_output_tokens'):
            estimated_output_tokens = config.max_output_tokens
        
        # Create the actual API call function
        def _api_call():
            return self._original_models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        
        # Make secure API call
        return self._security_wrapper.secure_api_call(
            endpoint=f"models/{model}/generateContent",
            api_function=_api_call,
            estimated_output_tokens=estimated_output_tokens,
            operation_type="generate",
            model_name=model
        )


def create_secure_gemini_client(api_key: Optional[str] = None, 
                               model_name: str = 'gemini-2.5-flash') -> SecureGeminiClient:
    """
    Create a secure Gemini client with all security controls enabled.
    
    Args:
        api_key: Gemini API key (uses config if not provided)
        model_name: Model name to use
        
    Returns:
        SecureGeminiClient instance
    """
    if api_key is None:
        config = get_config()
        api_key = config.get_gemini_api_key()
    
    return SecureGeminiClient(api_key, model_name)


def secure_gemini_call(prompt: str, 
                      model_name: str = 'gemini-2.5-flash',
                      max_output_tokens: int = 2500,
                      temperature: float = 0.6,
                      api_key: Optional[str] = None) -> str:
    """
    Make a secure Gemini API call with default security settings.
    
    This is a convenience function for making single API calls with security.
    
    Args:
        prompt: Prompt to send to the model
        model_name: Model to use
        max_output_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        api_key: API key (uses config if not provided)
        
    Returns:
        Generated text response
        
    Raises:
        SecurityError: If security checks fail
        RateLimitError: If rate limited
        CostExceededError: If budget exceeded
        ValidationError: If input validation fails
    """
    client = create_secure_gemini_client(api_key, model_name)
    
    # Create generation config
    config = genai.types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature
    )
    
    # Make secure call
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config
    )
    
    return response.text


def patch_agent_for_security(agent_instance):
    """
    Patch an existing agent instance to use secure API calls.
    
    This function can be called on existing agent instances to add security
    without modifying their code.
    
    Args:
        agent_instance: Agent instance to patch
    """
    if hasattr(agent_instance, 'client') and hasattr(agent_instance, 'model_name'):
        # Replace the client with secure version
        secure_client = create_secure_gemini_client(
            api_key=agent_instance.client.api_key,
            model_name=agent_instance.model_name
        )
        
        # Store original for potential restoration
        agent_instance._original_client = agent_instance.client
        agent_instance.client = secure_client
        
        logger.info(f"Agent {type(agent_instance).__name__} patched for security")
    
    # Patch _make_gemini_call if it exists
    if hasattr(agent_instance, '_make_gemini_call'):
        original_call = agent_instance._make_gemini_call
        
        def secure_call_wrapper(prompt: str) -> str:
            """Secure wrapper for _make_gemini_call."""
            try:
                return secure_gemini_call(
                    prompt=prompt,
                    model_name=getattr(agent_instance, 'model_name', 'gemini-2.5-flash'),
                    max_output_tokens=2500,
                    temperature=0.6
                )
            except Exception as e:
                # Fall back to original implementation if security fails
                logger.warning(f"Security wrapper failed, using original: {e}")
                return original_call(prompt)
        
        agent_instance._original_make_gemini_call = original_call
        agent_instance._make_gemini_call = secure_call_wrapper


def restore_agent_from_security(agent_instance):
    """
    Restore an agent instance to use original (non-secure) API calls.
    
    Args:
        agent_instance: Agent instance to restore
    """
    if hasattr(agent_instance, '_original_client'):
        agent_instance.client = agent_instance._original_client
        delattr(agent_instance, '_original_client')
    
    if hasattr(agent_instance, '_original_make_gemini_call'):
        agent_instance._make_gemini_call = agent_instance._original_make_gemini_call
        delattr(agent_instance, '_original_make_gemini_call')
    
    logger.info(f"Agent {type(agent_instance).__name__} restored from security patches")


def get_security_status_for_all_wrappers() -> dict:
    """Get security status for all active security wrappers."""
    from .api_security import _security_wrappers
    
    status = {}
    for provider, wrapper in _security_wrappers.items():
        status[provider] = wrapper.get_security_status()
    
    return status


# Example usage functions for testing and validation

def test_security_integration():
    """Test function to validate security integration works correctly."""
    try:
        # Test basic secure call
        response = secure_gemini_call(
            prompt="What is 2+2?",
            max_output_tokens=50,
            temperature=0.1
        )
        
        logger.info(f"Security integration test successful. Response: {response[:100]}...")
        return True
        
    except Exception as e:
        logger.error(f"Security integration test failed: {e}")
        return False


def demonstrate_security_features():
    """Demonstrate various security features."""
    print("=== Security Features Demonstration ===")
    
    # Get security status
    status = get_security_status_for_all_wrappers()
    
    if status:
        for provider, provider_status in status.items():
            print(f"\nProvider: {provider}")
            print(f"Daily Cost: ${provider_status['cost_monitor']['current_spending']['daily']:.4f}")
            print(f"Rate Limit Status: {provider_status['rate_limiter']['current_status']}")
            print(f"Recent Events: {provider_status['security_events']['recent_events_count']}")
    else:
        print("No active security wrappers found.")
    
    return status
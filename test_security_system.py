#!/usr/bin/env python3
"""
Comprehensive Security System Test Suite

Tests all security components including rate limiting, cost monitoring,
input validation, security logging, and API integration.
"""

import os
import sys
import time
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import security components
from src.security.rate_limiter import (
    RateLimiter, RateLimitConfig, RateLimitError, TokenBucket, SlidingWindowCounter
)
from src.security.cost_monitor import (
    CostMonitor, CostConfig, CostExceededError, ApiUsage, CostAlert
)
from src.security.input_validator import (
    InputValidator, ValidationConfig, ValidationError
)
from src.security.security_logger import (
    SecurityLogger, SecurityLogConfig, SecurityEventType, SecurityEvent
)
from src.security.api_security import (
    ApiSecurityWrapper, SecurityConfig
)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting functionality."""
    
    def setUp(self):
        """Setup for rate limiter tests."""
        self.config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=20,
            requests_per_day=100,
            burst_allowance=3,
            max_daily_cost=10.0
        )
        self.limiter = RateLimiter(self.config)
    
    def test_token_bucket_basic(self):
        """Test basic token bucket functionality."""
        bucket = TokenBucket(capacity=3, refill_rate=1.0)  # 1 token per second
        
        # Should be able to consume initial tokens
        self.assertTrue(bucket.consume(1))
        self.assertTrue(bucket.consume(1))
        self.assertTrue(bucket.consume(1))
        
        # Should be empty now
        self.assertFalse(bucket.consume(1))
        
        # Wait for refill
        time.sleep(1.1)
        self.assertTrue(bucket.consume(1))
    
    def test_sliding_window_basic(self):
        """Test basic sliding window functionality."""
        window = SlidingWindowCounter(window_size=2, limit=3)  # 3 requests per 2 seconds
        
        # Should allow initial requests
        self.assertTrue(window.is_allowed())
        self.assertTrue(window.is_allowed())
        self.assertTrue(window.is_allowed())
        
        # Should reject 4th request
        self.assertFalse(window.is_allowed())
        
        # Wait for window to slide
        time.sleep(2.1)
        self.assertTrue(window.is_allowed())
    
    def test_rate_limiter_integration(self):
        """Test rate limiter with multiple constraints."""
        # Should allow initial requests
        self.assertTrue(self.limiter.check_and_consume())
        self.assertTrue(self.limiter.check_and_consume())
        
        # Test cost budget
        with self.assertRaises(RateLimitError):
            self.limiter.check_and_consume(estimated_cost=15.0)  # Exceeds daily budget
    
    def test_rate_limiter_status(self):
        """Test rate limiter status reporting."""
        status = self.limiter.get_status()
        
        self.assertIn('config', status)
        self.assertIn('current_status', status)
        self.assertIn('windows', status)
        
        self.assertEqual(status['config']['requests_per_minute'], 5)


class TestCostMonitor(unittest.TestCase):
    """Test cost monitoring functionality."""
    
    def setUp(self):
        """Setup for cost monitor tests."""
        self.config = CostConfig(
            daily_budget=10.0,
            weekly_budget=50.0,
            monthly_budget=200.0
        )
        self.monitor = CostMonitor(self.config)
    
    def test_cost_estimation(self):
        """Test cost estimation for different providers."""
        # Test Gemini estimation
        cost = self.monitor.estimate_cost("gemini", "gemini-pro", "Hello world", 100)
        self.assertGreater(cost, 0)
        self.assertLess(cost, 1.0)  # Should be small for short text
        
        # Test embedding estimation
        cost_embed = self.monitor.estimate_cost("gemini", "gemini-embed", "Hello world", 0, "embed")
        self.assertGreater(cost_embed, 0)
    
    def test_budget_enforcement(self):
        """Test budget constraint enforcement."""
        # Should allow request within budget
        self.assertTrue(self.monitor.check_budget_before_request(1.0))
        
        # Should reject request exceeding budget
        with self.assertRaises(CostExceededError):
            self.monitor.check_budget_before_request(15.0)
    
    def test_usage_recording(self):
        """Test usage recording and tracking."""
        # Record some usage
        self.monitor.record_usage(
            provider="gemini",
            model="gemini-pro",
            endpoint="generate",
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.05,
            actual_cost=0.048
        )
        
        summary = self.monitor.get_spending_summary()
        self.assertGreater(summary['current_spending']['daily'], 0)
        self.assertIn('gemini', summary['breakdown']['by_provider'])
    
    def test_cost_prediction(self):
        """Test cost prediction functionality."""
        # Record some usage first
        for _ in range(3):
            self.monitor.record_usage(
                provider="gemini",
                model="gemini-pro", 
                endpoint="generate",
                input_tokens=100,
                output_tokens=50,
                estimated_cost=0.01
            )
        
        prediction = self.monitor.get_cost_prediction(hours_ahead=24)
        self.assertIn('predicted_cost', prediction)
        self.assertIn('confidence', prediction)


class TestInputValidator(unittest.TestCase):
    """Test input validation functionality."""
    
    def setUp(self):
        """Setup for input validator tests."""
        self.config = ValidationConfig(
            max_text_length=1000,
            block_sql_injection=True,
            block_xss_attempts=True
        )
        self.validator = InputValidator(self.config)
    
    def test_text_validation_basic(self):
        """Test basic text validation."""
        # Should pass normal text
        result = self.validator.validate_text("Hello, world!")
        self.assertEqual(result, "Hello, world!")
        
        # Should reject overly long text
        with self.assertRaises(ValidationError):
            self.validator.validate_text("x" * 2000)
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'; DELETE FROM users WHERE '1'='1",
            "UNION SELECT * FROM passwords"
        ]
        
        for malicious_input in malicious_inputs:
            with self.assertRaises(ValidationError):
                self.validator.validate_text(malicious_input)
    
    def test_xss_detection(self):
        """Test XSS detection."""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ]
        
        for malicious_input in malicious_inputs:
            with self.assertRaises(ValidationError):
                self.validator.validate_text(malicious_input)
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f",
            "....//....//....//etc/passwd"
        ]
        
        for malicious_path in malicious_paths:
            with self.assertRaises(ValidationError):
                self.validator.validate_file_path(malicious_path)
    
    def test_json_validation(self):
        """Test JSON validation."""
        # Should pass valid JSON
        valid_json = {"key": "value", "number": 42}
        result = self.validator.validate_json(valid_json)
        self.assertEqual(result, valid_json)
        
        # Should reject deeply nested JSON
        deeply_nested = {"level1": {"level2": {}}}
        current = deeply_nested["level1"]["level2"]
        for i in range(25):  # Exceed max depth
            current[f"level{i+3}"] = {}
            current = current[f"level{i+3}"]
        
        with self.assertRaises(ValidationError):
            self.validator.validate_json(deeply_nested)
    
    def test_email_validation(self):
        """Test email validation."""
        # Valid emails
        valid_emails = [
            "test@example.com",
            "user.name+tag@example.co.uk",
            "user123@test-domain.com"
        ]
        
        for email in valid_emails:
            result = self.validator.validate_email(email)
            self.assertIsInstance(result, str)
        
        # Invalid emails
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "test@",
            "test..test@example.com"
        ]
        
        for email in invalid_emails:
            with self.assertRaises(ValidationError):
                self.validator.validate_email(email)


class TestSecurityLogger(unittest.TestCase):
    """Test security logging functionality."""
    
    def setUp(self):
        """Setup for security logger tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SecurityLogConfig(
            log_file_path=os.path.join(self.temp_dir, "security.log"),
            sanitize_sensitive_data=True
        )
        self.logger = SecurityLogger(self.config)
    
    def test_event_logging(self):
        """Test basic event logging."""
        self.logger.log_event(
            event_type=SecurityEventType.API_REQUEST,
            severity="low",
            message="Test API request",
            endpoint="/test",
            metadata={"test": "data"}
        )
        
        events = self.logger.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].message, "Test API request")
        self.assertEqual(events[0].endpoint, "/test")
    
    def test_sensitive_data_sanitization(self):
        """Test sensitive data sanitization."""
        sensitive_message = "API key: sk-1234567890abcdef1234567890abcdef"
        
        self.logger.log_event(
            event_type=SecurityEventType.API_REQUEST,
            severity="low", 
            message=sensitive_message
        )
        
        events = self.logger.get_events()
        self.assertNotIn("sk-1234567890abcdef1234567890abcdef", events[0].message)
        self.assertIn("sk-12345", events[0].message)  # Should keep prefix
    
    def test_event_filtering(self):
        """Test event filtering."""
        # Log different types of events
        self.logger.log_event(SecurityEventType.API_REQUEST, "low", "Request 1")
        self.logger.log_event(SecurityEventType.API_ERROR, "high", "Error 1")
        self.logger.log_event(SecurityEventType.RATE_LIMIT_EXCEEDED, "medium", "Rate limit")
        
        # Filter by event type
        api_events = self.logger.get_events(event_types=[SecurityEventType.API_REQUEST])
        self.assertEqual(len(api_events), 1)
        
        # Filter by severity
        high_events = self.logger.get_events(severity_levels=["high"])
        self.assertEqual(len(high_events), 1)
    
    def test_security_summary(self):
        """Test security summary generation."""
        # Log some events
        for i in range(5):
            self.logger.log_event(SecurityEventType.API_REQUEST, "low", f"Request {i}")
        
        summary = self.logger.get_security_summary()
        self.assertEqual(summary['total_events'], 5)
        self.assertIn('api_request', summary['events_by_type'])


class TestApiSecurityWrapper(unittest.TestCase):
    """Test API security wrapper integration."""
    
    def setUp(self):
        """Setup for API security wrapper tests."""
        self.config = SecurityConfig(
            rate_limit_config=RateLimitConfig(requests_per_minute=10, max_daily_cost=5.0),
            cost_config=CostConfig(daily_budget=5.0),
            validation_config=ValidationConfig(),
            security_log_config=SecurityLogConfig()
        )
        self.wrapper = ApiSecurityWrapper(self.config, "test-provider")
    
    def test_secure_api_call_success(self):
        """Test successful secure API call."""
        # Mock API function
        mock_api = Mock(return_value="API response")
        
        result = self.wrapper.secure_api_call(
            endpoint="test",
            api_function=mock_api,
            "test input"
        )
        
        self.assertEqual(result, "API response")
        mock_api.assert_called_once_with("test input")
    
    def test_secure_api_call_validation_error(self):
        """Test API call with validation error."""
        mock_api = Mock(return_value="API response")
        
        # This should trigger SQL injection detection
        with self.assertRaises(ValidationError):
            self.wrapper.secure_api_call(
                endpoint="test",
                api_function=mock_api,
                "'; DROP TABLE users; --"
            )
    
    def test_secure_api_call_cost_exceeded(self):
        """Test API call exceeding cost budget."""
        mock_api = Mock(return_value="API response")
        
        with self.assertRaises(CostExceededError):
            self.wrapper.secure_api_call(
                endpoint="test",
                api_function=mock_api,
                "test input",
                estimated_output_tokens=100000  # Very expensive
            )
    
    def test_security_status(self):
        """Test security status reporting."""
        status = self.wrapper.get_security_status()
        
        self.assertIn('provider', status)
        self.assertIn('rate_limiter', status)
        self.assertIn('cost_monitor', status)
        self.assertIn('security_events', status)
        self.assertIn('configuration', status)


class TestSecurityIntegration(unittest.TestCase):
    """Test end-to-end security integration."""
    
    def setUp(self):
        """Setup for integration tests."""
        # Mock environment variables
        os.environ.update({
            'GEMINI_API_KEY': 'test_api_key_1234567890',
            'SUPABASE_DB_URL': 'postgresql://test:test@localhost/test'
        })
    
    def test_config_integration(self):
        """Test security configuration integration."""
        from src.config.settings import ConfigManager
        
        config_manager = ConfigManager()
        config_manager.load_configuration()
        
        # Should have security config
        self.assertIsNotNone(config_manager.security)
        self.assertTrue(config_manager.security.enable_rate_limiting)
        self.assertTrue(config_manager.security.enable_cost_monitoring)
    
    @patch('google.generativeai.Client')
    def test_secure_agent_wrapper(self, mock_client_class):
        """Test secure agent wrapper integration."""
        from src.security.secure_agent_wrapper import create_secure_gemini_client
        
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.models.generate_content.return_value = mock_response
        
        # Create secure client
        secure_client = create_secure_gemini_client()
        
        # Make a call (should work with mocked API)
        result = secure_client.models.generate_content(
            contents="Test prompt",
            config=Mock(max_output_tokens=100)
        )
        
        self.assertEqual(result.text, "Test response")


def run_performance_tests():
    """Run performance tests for security components."""
    print("\n=== Performance Tests ===")
    
    # Rate limiter performance
    print("Testing rate limiter performance...")
    config = RateLimitConfig(requests_per_minute=1000)
    limiter = RateLimiter(config)
    
    start_time = time.time()
    for _ in range(1000):
        try:
            limiter.check_and_consume()
        except RateLimitError:
            pass
    
    elapsed = time.time() - start_time
    print(f"Rate limiter: {1000/elapsed:.0f} checks/second")
    
    # Input validator performance
    print("Testing input validator performance...")
    validator = InputValidator(ValidationConfig())
    test_text = "This is a test input with some content to validate. " * 20
    
    start_time = time.time()
    for _ in range(1000):
        try:
            validator.validate_text(test_text)
        except ValidationError:
            pass
    
    elapsed = time.time() - start_time
    print(f"Input validator: {1000/elapsed:.0f} validations/second")


def run_security_audit():
    """Run a basic security audit of the implementation."""
    print("\n=== Security Audit ===")
    
    # Check for potential security issues
    issues = []
    
    # Test SQL injection patterns
    validator = InputValidator(ValidationConfig())
    sql_test_cases = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'; DELETE FROM users",
        "UNION SELECT * FROM passwords"
    ]
    
    for test_case in sql_test_cases:
        try:
            validator.validate_text(test_case)
            issues.append(f"SQL injection not detected: {test_case}")
        except ValidationError:
            pass  # Good, should be detected
    
    # Test XSS patterns
    xss_test_cases = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert(1)>"
    ]
    
    for test_case in xss_test_cases:
        try:
            validator.validate_text(test_case)
            issues.append(f"XSS not detected: {test_case}")
        except ValidationError:
            pass  # Good, should be detected
    
    if issues:
        print("Security issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No security issues detected in basic audit.")
    
    return len(issues) == 0


def main():
    """Run comprehensive security system tests."""
    print("=== Security System Test Suite ===")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    # Run security audit
    audit_passed = run_security_audit()
    
    print(f"\n=== Test Summary ===")
    print(f"Security audit: {'PASSED' if audit_passed else 'FAILED'}")
    
    return audit_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
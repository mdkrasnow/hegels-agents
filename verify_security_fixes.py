#!/usr/bin/env python3
"""
Quick verification of security fixes applied.
"""

import sys
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_input_validation():
    """Test the enhanced input validation patterns."""
    print("Testing enhanced input validation...")
    
    from src.security import get_validator, ValidationConfig, ValidationError
    
    validator = get_validator(ValidationConfig(
        block_sql_injection=True,
        block_xss_attempts=True,
        block_path_traversal=True
    ))
    
    # Test enhanced SQL injection detection
    advanced_sql = "1; EXEC sp_configure 'show advanced options', 1--"
    try:
        validator.validate_text(advanced_sql)
        print("✗ Advanced SQL injection not blocked")
        sql_fixed = False
    except ValidationError:
        print("✓ Advanced SQL injection blocked")
        sql_fixed = True
    
    # Test enhanced XSS detection
    advanced_xss = "';alert('XSS');//"
    try:
        validator.validate_text(advanced_xss)
        print("✗ Advanced XSS not blocked")
        xss_fixed = False
    except ValidationError:
        print("✓ Advanced XSS blocked")
        xss_fixed = True
    
    # Test enhanced path traversal
    advanced_path = "....\\\\....\\\\....\\\\windows\\system32"
    try:
        validator.validate_file_path(advanced_path)
        print("✗ Advanced path traversal not blocked")
        path_fixed = False
    except ValidationError:
        print("✓ Advanced path traversal blocked")  
        path_fixed = True
    
    return sql_fixed and xss_fixed and path_fixed


def test_fixed_security_logging():
    """Test the fixed security logging JSON serialization."""
    print("Testing fixed security logging...")
    
    from src.security import SecurityLogger, SecurityLogConfig, SecurityEventType
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SecurityLogConfig(
            log_file_path=os.path.join(temp_dir, "security.log"),
            export_format="json"
        )
        
        try:
            logger = SecurityLogger(config)
            
            # This should not cause JSON serialization errors now
            logger.log_event(
                SecurityEventType.API_REQUEST,
                "low", 
                "Test JSON serialization fix"
            )
            
            print("✓ Security logging JSON serialization fixed")
            return True
            
        except Exception as e:
            print(f"✗ Security logging still has issues: {e}")
            return False


def test_fixed_cost_alerting():
    """Test the fixed cost alerting system."""
    print("Testing fixed cost alerting...")
    
    from src.security import CostMonitor, CostConfig
    
    config = CostConfig(
        daily_budget=10.0,
        warning_threshold=0.8,
        critical_threshold=0.95
    )
    
    monitor = CostMonitor(config)
    
    alerts_received = []
    def alert_callback(alert):
        alerts_received.append(alert)
    
    monitor.add_alert_callback(alert_callback)
    
    # Trigger warning (80% of $10 = $8)
    monitor.record_usage("test", "test", "test", 100, 50, 8.5, 8.5)
    
    # The alert should be triggered by _check_alert_thresholds during record_usage
    # Let's manually trigger it to see what happens
    monitor._check_alert_thresholds()
    
    # Small delay for alert processing
    time.sleep(0.1)
    
    print(f"Debug: Daily spend: {monitor.daily_spend}, Alerts received: {len(alerts_received)}")
    
    if alerts_received:
        print("✓ Cost alerting system fixed")
        return True
    else:
        # Check if alert would be triggered for the current spend level
        percentage = monitor.daily_spend / monitor.config.daily_budget
        print(f"Debug: Spend percentage: {percentage:.1%}, Warning threshold: {monitor.config.warning_threshold:.1%}")
        print("✗ Cost alerting still not working")
        return False


def test_api_wrapper_integration():
    """Test API wrapper integration after fixes."""
    print("Testing API wrapper integration...")
    
    from src.security import (
        ApiSecurityWrapper, SecurityConfig, RateLimitConfig, 
        CostConfig, ValidationConfig, SecurityLogConfig
    )
    
    try:
        config = SecurityConfig(
            rate_limit_config=RateLimitConfig(requests_per_minute=100),
            cost_config=CostConfig(daily_budget=50.0),
            validation_config=ValidationConfig(),
            security_log_config=SecurityLogConfig()
        )
        
        wrapper = ApiSecurityWrapper(config, "test-provider")
        
        # Mock API function
        mock_api = Mock(return_value="API response")
        
        # Test proper function call
        result = wrapper.secure_api_call(
            "test_endpoint",
            mock_api,
            "test input"
        )
        
        if result == "API response":
            print("✓ API wrapper integration working")
            return True
        else:
            print("✗ API wrapper integration failed")
            return False
            
    except Exception as e:
        print(f"✗ API wrapper integration error: {e}")
        return False


def main():
    """Run verification tests for security fixes."""
    print("=" * 50)
    print("VERIFYING SECURITY FIXES")
    print("=" * 50)
    
    tests = [
        ("Enhanced Input Validation", test_enhanced_input_validation),
        ("Fixed Security Logging", test_fixed_security_logging),
        ("Fixed Cost Alerting", test_fixed_cost_alerting),
        ("API Wrapper Integration", test_api_wrapper_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print(f"\n{'=' * 50}")
    print(f"VERIFICATION RESULTS: {passed}/{total} PASSED")
    print(f"{'=' * 50}")
    
    if passed == total:
        print("🎉 All security fixes verified successfully!")
        return True
    else:
        print("❌ Some security fixes still need work")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
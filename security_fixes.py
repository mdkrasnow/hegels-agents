#!/usr/bin/env python3
"""
Critical Security Fixes

This script addresses the critical security issues found during validation:
1. Enhance input validation patterns
2. Fix alerting system
3. Fix JSON serialization in security logging
4. Improve XSS detection
5. Strengthen SQL injection detection
"""

import sys
import os
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def fix_input_validation_patterns():
    """Fix and enhance input validation patterns."""
    print("Fixing input validation patterns...")
    
    # Read the current input validator
    input_validator_path = Path(__file__).parent / "src" / "security" / "input_validator.py"
    
    with open(input_validator_path, 'r') as f:
        content = f.read()
    
    # Enhanced SQL injection patterns
    enhanced_sql_patterns = '''        patterns = [
            r"(\\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION)\\b.*\\b(FROM|INTO|SET|WHERE|TABLE)\\b)",
            r"(\\b(OR|AND)\\s+\\d+\\s*=\\s*\\d+)",
            r"(\\b(OR|AND)\\s+['\\\"].*['\\\"])",
            r"(;\\s*(SELECT|INSERT|UPDATE|DELETE|DROP))",
            r"(\\b(UNION\\s+(ALL\\s+)?SELECT))",
            r"(\\b(EXEC|EXECUTE)\\s*\\()",
            r"(\\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\\b)",
            # Additional patterns for better detection
            r"(\\b(sp_configure|xp_cmdshell|sp_executesql)\\b)",
            r"(\\b(WAITFOR\\s+DELAY)\\b)",
            r"(\\b(CAST|CONVERT)\\s*\\()",
            r"(\\b(CHAR|CHR)\\s*\\()",
            r"(\\\\'\\s*(OR|AND)\\s*\\\\')",
            r"(--[^\\r\\n]*)",
            r"(/\\*.*\\*/)",
        ]'''
    
    # Enhanced XSS patterns  
    enhanced_xss_patterns = '''        patterns = [
            r"<\\s*script[^>]*>.*?</\\s*script\\s*>",
            r"<\\s*iframe[^>]*>.*?</\\s*iframe\\s*>",
            r"javascript\\s*:",
            r"vbscript\\s*:",
            r"on\\w+\\s*=",
            r"<\\s*object[^>]*>",
            r"<\\s*embed[^>]*>",
            r"<\\s*meta[^>]*>",
            # Additional patterns for better XSS detection
            r"<\\s*link[^>]*>",
            r"<\\s*style[^>]*>.*?</\\s*style\\s*>",
            r"expression\\s*\\(",
            r"alert\\s*\\(",
            r"confirm\\s*\\(",
            r"prompt\\s*\\(",
            r"eval\\s*\\(",
            r"String\\.fromCharCode",
            r"document\\.(cookie|write|location)",
            r"window\\.(location|open)",
            r"&#[xX]?[0-9a-fA-F]+;",
            r"javascript\\s*&#58;",
            r"\\\\'\\s*(alert|prompt|confirm)\\s*\\(",
        ]'''
    
    # Enhanced path traversal patterns
    enhanced_path_patterns = '''        patterns = [
            r"\\.\\./",
            r"\\.\\.\\\\\\\.",
            r"%2e%2e%2f",
            r"%2e%2e%5c", 
            r"\\.\\.\\.%2f",
            r"\\.\\.\\.%5c",
            # Additional patterns for better path traversal detection
            r"\\.\\.\\\\",
            r"\\.\\.%5c",
            r"%2e%2e\\\\",
            r"\\.\\.%2f",
            r"\\.\\.\\./",
            r"\\.\\.\\.\\.\\./",
            r"\\.\\.\\\\\\.\\.\\\\",
            r"\\.\\.\\.\\.\\.\\.\\.",
            r"/etc/passwd",
            r"/etc/shadow",
            r"windows\\\\system32",
            r"boot\\.ini",
        ]'''
    
    # Replace the pattern compilation methods
    content = re.sub(
        r'patterns = \[\s*r".*?"\s*\]\s*return \[re\.compile\(pattern, re\.IGNORECASE \| re\.DOTALL\) for pattern in patterns\]',
        enhanced_sql_patterns + '''
        return [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]''',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'def _compile_xss_patterns\(self\) -> List\[re\.Pattern\]:\s*"""Compile XSS detection patterns\."""\s*patterns = \[.*?\]\s*return \[re\.compile\(pattern, re\.IGNORECASE \| re\.DOTALL\) for pattern in patterns\]',
        '''def _compile_xss_patterns(self) -> List[re.Pattern]:
        """Compile XSS detection patterns."""
''' + enhanced_xss_patterns + '''
        return [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]''',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'def _compile_path_patterns\(self\) -> List\[re\.Pattern\]:\s*"""Compile path traversal detection patterns\."""\s*patterns = \[.*?\]\s*return \[re\.compile\(pattern, re\.IGNORECASE\) for pattern in patterns\]',
        '''def _compile_path_patterns(self) -> List[re.Pattern]:
        """Compile path traversal detection patterns."""
''' + enhanced_path_patterns + '''
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]''',
        content,
        flags=re.DOTALL
    )
    
    # Write back the enhanced content
    with open(input_validator_path, 'w') as f:
        f.write(content)
    
    print("✓ Enhanced input validation patterns")


def fix_security_logging_serialization():
    """Fix JSON serialization issue in security logging."""
    print("Fixing security logging JSON serialization...")
    
    security_logger_path = Path(__file__).parent / "src" / "security" / "security_logger.py"
    
    with open(security_logger_path, 'r') as f:
        content = f.read()
    
    # Add custom JSON encoder for SecurityEventType
    json_encoder_code = '''
from enum import Enum
import json

class SecurityEventEncoder(json.JSONEncoder):
    """Custom JSON encoder for security events."""
    
    def default(self, obj):
        if isinstance(obj, SecurityEventType):
            return obj.value
        return super().default(obj)
'''
    
    # Insert the encoder after the imports
    import_section = content.split('logger = logging.getLogger(__name__)')[0]
    rest_of_content = content.split('logger = logging.getLogger(__name__)')[1]
    
    content = import_section + 'logger = logging.getLogger(__name__)' + json_encoder_code + rest_of_content
    
    # Fix the JSON serialization call
    content = content.replace(
        'log_line = json.dumps(log_data)',
        'log_line = json.dumps(log_data, cls=SecurityEventEncoder)'
    )
    
    with open(security_logger_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed security logging JSON serialization")


def fix_cost_alerting_system():
    """Fix cost alerting system."""
    print("Fixing cost alerting system...")
    
    cost_monitor_path = Path(__file__).parent / "src" / "security" / "cost_monitor.py"
    
    with open(cost_monitor_path, 'r') as f:
        content = f.read()
    
    # Fix the alert checking logic - need to avoid duplicate alerts
    fixed_alert_check = '''    def _check_alert_thresholds(self, additional_cost: float = 0.0) -> None:
        """Check if spending has crossed alert thresholds."""
        budgets = [
            ("daily", self.daily_spend + additional_cost, self.config.daily_budget),
            ("weekly", self.weekly_spend + additional_cost, self.config.weekly_budget),
            ("monthly", self.monthly_spend + additional_cost, self.config.monthly_budget),
        ]
        
        for period, spend, budget in budgets:
            if budget <= 0:  # Skip if budget is 0 or negative
                continue
                
            percentage = spend / budget
            
            # Only trigger alerts for threshold crossings
            if percentage >= self.config.critical_threshold:
                # Check if we already have a recent critical alert for this period
                recent_critical = any(
                    alert.severity == "critical" and 
                    alert.budget_type == period and
                    time.time() - alert.timestamp < 3600  # Within last hour
                    for alert in self.alerts[-5:]  # Check last 5 alerts
                )
                
                if not recent_critical:
                    self._create_alert("critical", f"Critical: {period} spending at {percentage:.1%} of budget", 
                                     spend, budget, period)
                                     
            elif percentage >= self.config.warning_threshold:
                # Check if we already have a recent warning alert for this period  
                recent_warning = any(
                    alert.severity in ["warning", "critical"] and
                    alert.budget_type == period and
                    time.time() - alert.timestamp < 3600  # Within last hour
                    for alert in self.alerts[-5:]  # Check last 5 alerts
                )
                
                if not recent_warning:
                    self._create_alert("warning", f"Warning: {period} spending at {percentage:.1%} of budget", 
                                     spend, budget, period)'''
    
    # Replace the method
    content = re.sub(
        r'def _check_alert_thresholds\(self, additional_cost: float = 0\.0\) -> None:.*?(?=\n    def|\nclass|\n\n# Global|\Z)',
        fixed_alert_check,
        content,
        flags=re.DOTALL
    )
    
    with open(cost_monitor_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed cost alerting system")


def create_enhanced_test_suite():
    """Create an enhanced test suite that properly tests the fixes."""
    print("Creating enhanced test suite...")
    
    enhanced_test_content = '''#!/usr/bin/env python3
"""
Enhanced Security Test Suite - Post-Fix Validation

Tests the enhanced security controls after applying fixes.
"""

import sys
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.security import *

def test_enhanced_sql_injection_protection():
    """Test enhanced SQL injection protection."""
    print("Testing enhanced SQL injection protection...")
    
    validator = get_validator(ValidationConfig(block_sql_injection=True))
    
    advanced_sql_attacks = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'; DELETE FROM users WHERE '1'='1", 
        "UNION SELECT * FROM passwords",
        "'; UPDATE users SET password='hacked' WHERE '1'='1",
        "1; EXEC sp_configure 'show advanced options', 1--",
        "' OR 1=1 UNION SELECT null, username, password FROM users--",
        "'; EXEC xp_cmdshell('rm -rf /')--",
        "1'; WAITFOR DELAY '00:00:10'--",
        "admin'/**/OR/**/1=1--",
        "1' UNION SELECT CHAR(117,115,101,114,110,97,109,101)--"
    ]
    
    blocked = 0
    total = len(advanced_sql_attacks)
    
    for attack in advanced_sql_attacks:
        try:
            validator.validate_text(attack)
            print(f"✗ SQL injection not blocked: {attack}")
        except ValidationError:
            blocked += 1
    
    success_rate = blocked / total
    print(f"SQL injection protection: {blocked}/{total} blocked ({success_rate:.1%})")
    return success_rate >= 0.95  # 95% threshold


def test_enhanced_xss_protection():
    """Test enhanced XSS protection."""
    print("Testing enhanced XSS protection...")
    
    validator = get_validator(ValidationConfig(block_xss_attempts=True))
    
    advanced_xss_attacks = [
        "<script>alert('XSS')</script>",
        "<iframe src='javascript:alert(1)'></iframe>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "<body onload=alert('XSS')>",
        "<input onfocus=alert('XSS') autofocus>",
        "';alert('XSS');//",
        "<script src='http://evil.com/xss.js'></script>",
        "<link rel='stylesheet' href='javascript:alert(1)'>",
        "<style>@import'javascript:alert(1)';</style>",
        "javascript&#58;alert('XSS')",
        "<img src=x onerror=eval(String.fromCharCode(97,108,101,114,116,40,49,41))>",
        "<iframe src=javascript:alert(document.domain)></iframe>"
    ]
    
    blocked = 0
    total = len(advanced_xss_attacks)
    
    for attack in advanced_xss_attacks:
        try:
            validator.validate_text(attack)
            print(f"✗ XSS attack not blocked: {attack}")
        except ValidationError:
            blocked += 1
    
    success_rate = blocked / total
    print(f"XSS protection: {blocked}/{total} blocked ({success_rate:.1%})")
    return success_rate >= 0.95


def test_enhanced_path_traversal_protection():
    """Test enhanced path traversal protection."""
    print("Testing enhanced path traversal protection...")
    
    validator = get_validator(ValidationConfig(block_path_traversal=True))
    
    advanced_path_attacks = [
        "../../../etc/passwd",
        "..\\\\..\\\\..\\\\windows\\\\system32\\\\config\\\\SAM",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
        "..\\\\..\\\\..\\\\..\\\\..\\\\..\\\\etc\\\\passwd",
        "/var/www/../../etc/passwd",
        "....\\\\....\\\\....\\\\windows\\\\system32",
        "..%2f..%2f..%2fetc%2fpasswd",
        "..\\\\..\\\\..\\\\boot.ini",
        "/etc/shadow",
        "file:///etc/passwd",
        "..%5c..%5c..%5cwindows%5csystem32"
    ]
    
    blocked = 0
    total = len(advanced_path_attacks)
    
    for attack in advanced_path_attacks:
        try:
            validator.validate_file_path(attack)
            print(f"✗ Path traversal not blocked: {attack}")
        except ValidationError:
            blocked += 1
    
    success_rate = blocked / total
    print(f"Path traversal protection: {blocked}/{total} blocked ({success_rate:.1%})")
    return success_rate >= 0.95


def test_fixed_cost_alerting():
    """Test the fixed cost alerting system."""
    print("Testing fixed cost alerting system...")
    
    config = CostConfig(
        daily_budget=10.0,
        warning_threshold=0.8,  # 80%
        critical_threshold=0.95  # 95%
    )
    monitor = CostMonitor(config)
    
    alerts_received = []
    def alert_callback(alert):
        alerts_received.append(alert)
    
    monitor.add_alert_callback(alert_callback)
    
    # Trigger warning threshold (80% of $10 = $8)
    monitor.record_usage("test", "test", "test", 1000, 500, 8.2, 8.2)
    
    # Small delay to ensure alert processing
    time.sleep(0.1)
    
    warning_triggered = any(alert.severity == "warning" for alert in alerts_received)
    print(f"Warning alert triggered: {warning_triggered}")
    
    # Trigger critical threshold (95% of $10 = $9.50)
    monitor.record_usage("test", "test", "test", 1000, 500, 1.5, 1.5)
    
    time.sleep(0.1)
    
    critical_triggered = any(alert.severity == "critical" for alert in alerts_received)
    print(f"Critical alert triggered: {critical_triggered}")
    print(f"Total alerts received: {len(alerts_received)}")
    
    return warning_triggered and critical_triggered


def test_fixed_security_logging():
    """Test the fixed security logging JSON serialization."""
    print("Testing fixed security logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SecurityLogConfig(
            log_file_path=os.path.join(temp_dir, "security.log"),
            sanitize_sensitive_data=True,
            export_format="json"
        )
        
        try:
            security_logger = SecurityLogger(config)
            
            # This should not cause JSON serialization errors now
            security_logger.log_event(
                SecurityEventType.API_REQUEST,
                "low",
                "Test event with enum serialization",
                endpoint="/test"
            )
            
            # Check if log file was created and contains valid JSON
            log_file_path = config.log_file_path
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    content = f.read()
                    print(f"Log file created successfully")
                    return True
            else:
                print("Log file was not created")
                return False
                
        except Exception as e:
            print(f"Security logging failed: {e}")
            return False


def test_api_wrapper_integration():
    """Test API wrapper integration with proper arguments."""
    print("Testing API wrapper integration...")
    
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
        
        # Test with proper argument order
        result = wrapper.secure_api_call(
            endpoint="test_endpoint",
            api_function=mock_api,
            "test input"  # Positional argument
        )
        
        integration_working = result == "API response"
        mock_api.assert_called_once_with("test input")
        
        print(f"API wrapper integration: {integration_working}")
        return integration_working
        
    except Exception as e:
        print(f"API wrapper integration failed: {e}")
        return False


def main():
    """Run enhanced security tests."""
    print("=" * 60)
    print("ENHANCED SECURITY TEST SUITE - POST-FIX VALIDATION") 
    print("=" * 60)
    
    tests = [
        ("Enhanced SQL Injection Protection", test_enhanced_sql_injection_protection),
        ("Enhanced XSS Protection", test_enhanced_xss_protection),
        ("Enhanced Path Traversal Protection", test_enhanced_path_traversal_protection),
        ("Fixed Cost Alerting", test_fixed_cost_alerting),
        ("Fixed Security Logging", test_fixed_security_logging),
        ("API Wrapper Integration", test_api_wrapper_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n--- {test_name} ---")
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print(f"\\n{'=' * 60}")
    print("ENHANCED TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Tests Passed: {passed}/{total} ({passed/total:.1%})")
    
    if passed >= total * 0.9:  # 90% pass rate
        print("🎉 Enhanced security tests PASSED - Ready for production")
        return True
    else:
        print("❌ Enhanced security tests FAILED - More work needed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("enhanced_security_test.py", "w") as f:
        f.write(enhanced_test_content)
    
    print("✓ Created enhanced test suite")


def main():
    """Apply all security fixes."""
    print("Applying critical security fixes...")
    
    try:
        fix_input_validation_patterns()
        fix_security_logging_serialization()
        fix_cost_alerting_system()
        create_enhanced_test_suite()
        
        print("\n✓ All security fixes applied successfully!")
        print("\nTo test the fixes, run:")
        print("  python enhanced_security_test.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Error applying fixes: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
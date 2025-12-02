"""
Input Validation and Sanitization Module

Provides comprehensive input validation, sanitization, and security checks
for all user inputs and API requests.
"""

import re
import html
import json
import base64
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    max_text_length: int = 100000  # Maximum text length
    max_json_size: int = 1000000   # Maximum JSON payload size
    max_list_items: int = 10000    # Maximum items in lists
    max_dict_depth: int = 20       # Maximum nested dict depth
    
    # Text content restrictions
    allow_html: bool = False
    allow_scripts: bool = False
    allow_external_links: bool = True
    
    # File restrictions
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: List[str] = None
    
    # Security patterns to block
    block_sql_injection: bool = True
    block_xss_attempts: bool = True
    block_path_traversal: bool = True


class ValidationError(Exception):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    
    Provides:
    - Text length and content validation
    - HTML/XSS sanitization
    - SQL injection detection
    - Path traversal prevention
    - JSON structure validation
    - File upload validation
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Compile security patterns for performance
        self._sql_injection_patterns = self._compile_sql_patterns()
        self._xss_patterns = self._compile_xss_patterns()
        self._path_traversal_patterns = self._compile_path_patterns()
        
        logger.info("InputValidator initialized with security patterns")
    
    def _compile_sql_patterns(self) -> List[re.Pattern]:
        """Compile SQL injection detection patterns."""
        patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION)\b.*\b(FROM|INTO|SET|WHERE|TABLE)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"].*['\"])",
            r"(;\s*(SELECT|INSERT|UPDATE|DELETE|DROP))",
            r"(\b(UNION\s+(ALL\s+)?SELECT))",
            r"(\b(EXEC|EXECUTE)\s*\()",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)",
        ]
        return [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]
    
    def _compile_xss_patterns(self) -> List[re.Pattern]:
        """Compile XSS detection patterns."""
        patterns = [
            r"<\s*script[^>]*>.*?</\s*script\s*>",
            r"<\s*iframe[^>]*>.*?</\s*iframe\s*>",
            r"javascript\s*:",
            r"vbscript\s*:",
            r"on\w+\s*=",
            r"<\s*object[^>]*>",
            r"<\s*embed[^>]*>",
            r"<\s*meta[^>]*>",
        ]
        return [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]
    
    def _compile_path_patterns(self) -> List[re.Pattern]:
        """Compile path traversal detection patterns."""
        patterns = [
            r"\.\./",
            r"\.\.\\\.",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"\.\.%2f",
            r"\.\.%5c",
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def validate_text(self, text: str, field_name: str = "text", 
                     max_length: Optional[int] = None) -> str:
        """
        Validate and sanitize text input.
        
        Args:
            text: Text to validate
            field_name: Name of field for error reporting
            max_length: Override default max length
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be a string", field_name, text)
        
        # Length validation
        max_len = max_length or self.config.max_text_length
        if len(text) > max_len:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {max_len} characters",
                field_name, len(text)
            )
        
        # Security checks
        if self.config.block_sql_injection and self._detect_sql_injection(text):
            logger.warning(f"SQL injection attempt detected in {field_name}")
            raise ValidationError(f"Potential SQL injection detected in {field_name}", field_name)
        
        if self.config.block_xss_attempts and self._detect_xss(text):
            logger.warning(f"XSS attempt detected in {field_name}")
            raise ValidationError(f"Potential XSS attack detected in {field_name}", field_name)
        
        if self.config.block_path_traversal and self._detect_path_traversal(text):
            logger.warning(f"Path traversal attempt detected in {field_name}")
            raise ValidationError(f"Potential path traversal detected in {field_name}", field_name)
        
        # Sanitize if needed
        if not self.config.allow_html:
            text = self._sanitize_html(text)
        
        return text
    
    def validate_json(self, data: Union[str, Dict, List], field_name: str = "json") -> Union[Dict, List]:
        """
        Validate JSON structure and content.
        
        Args:
            data: JSON data to validate (string or parsed)
            field_name: Name of field for error reporting
            
        Returns:
            Validated JSON data
            
        Raises:
            ValidationError: If validation fails
        """
        # Parse if string
        if isinstance(data, str):
            if len(data) > self.config.max_json_size:
                raise ValidationError(f"{field_name} JSON too large", field_name, len(data))
            
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON in {field_name}: {e}", field_name)
        
        # Validate structure
        self._validate_json_structure(data, field_name, depth=0)
        
        return data
    
    def _validate_json_structure(self, data: Any, field_name: str, depth: int) -> None:
        """Recursively validate JSON structure."""
        if depth > self.config.max_dict_depth:
            raise ValidationError(f"{field_name} JSON too deeply nested", field_name, depth)
        
        if isinstance(data, dict):
            if len(data) > self.config.max_list_items:
                raise ValidationError(f"{field_name} dictionary too large", field_name, len(data))
            
            for key, value in data.items():
                if not isinstance(key, str):
                    raise ValidationError(f"Non-string key in {field_name}", field_name, key)
                
                # Validate key
                self.validate_text(key, f"{field_name}.{key}", max_length=1000)
                
                # Recursively validate value
                self._validate_json_structure(value, f"{field_name}.{key}", depth + 1)
        
        elif isinstance(data, list):
            if len(data) > self.config.max_list_items:
                raise ValidationError(f"{field_name} list too large", field_name, len(data))
            
            for i, item in enumerate(data):
                self._validate_json_structure(item, f"{field_name}[{i}]", depth + 1)
        
        elif isinstance(data, str):
            self.validate_text(data, field_name, max_length=self.config.max_text_length)
    
    def validate_file_path(self, path: str, field_name: str = "path") -> str:
        """
        Validate file path for security.
        
        Args:
            path: File path to validate
            field_name: Name of field for error reporting
            
        Returns:
            Validated path
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(path, str):
            raise ValidationError(f"{field_name} must be a string", field_name, path)
        
        # Basic validation
        if not path.strip():
            raise ValidationError(f"{field_name} cannot be empty", field_name)
        
        # Path traversal check
        if self._detect_path_traversal(path):
            raise ValidationError(f"Path traversal detected in {field_name}", field_name, path)
        
        # Null byte check
        if '\x00' in path:
            raise ValidationError(f"Null byte detected in {field_name}", field_name, path)
        
        # Extension check if configured
        if self.config.allowed_file_extensions:
            extension = path.lower().split('.')[-1] if '.' in path else ''
            if extension not in self.config.allowed_file_extensions:
                raise ValidationError(
                    f"File extension '{extension}' not allowed in {field_name}",
                    field_name, extension
                )
        
        return path.strip()
    
    def validate_email(self, email: str, field_name: str = "email") -> str:
        """
        Validate email address format.
        
        Args:
            email: Email to validate
            field_name: Name of field for error reporting
            
        Returns:
            Validated email
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(email, str):
            raise ValidationError(f"{field_name} must be a string", field_name, email)
        
        email = email.strip().lower()
        
        # Basic length check
        if len(email) > 254:  # RFC 5321 limit
            raise ValidationError(f"{field_name} too long", field_name, len(email))
        
        # Basic format check
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        if not email_pattern.match(email):
            raise ValidationError(f"Invalid email format in {field_name}", field_name, email)
        
        return email
    
    def validate_url(self, url: str, field_name: str = "url", 
                    allow_relative: bool = False) -> str:
        """
        Validate URL format and security.
        
        Args:
            url: URL to validate
            field_name: Name of field for error reporting
            allow_relative: Whether to allow relative URLs
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(url, str):
            raise ValidationError(f"{field_name} must be a string", field_name, url)
        
        url = url.strip()
        
        # Length check
        if len(url) > 2083:  # IE URL limit
            raise ValidationError(f"{field_name} URL too long", field_name, len(url))
        
        # Security checks
        url_lower = url.lower()
        
        # Block dangerous schemes
        dangerous_schemes = ['javascript:', 'data:', 'vbscript:', 'file:']
        for scheme in dangerous_schemes:
            if url_lower.startswith(scheme):
                raise ValidationError(f"Dangerous URL scheme in {field_name}", field_name, scheme)
        
        # Basic format check
        if not allow_relative:
            url_pattern = re.compile(
                r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
            )
            if not url_pattern.match(url):
                raise ValidationError(f"Invalid URL format in {field_name}", field_name, url)
        
        return url
    
    def _detect_sql_injection(self, text: str) -> bool:
        """Detect potential SQL injection attempts."""
        text_lower = text.lower()
        return any(pattern.search(text_lower) for pattern in self._sql_injection_patterns)
    
    def _detect_xss(self, text: str) -> bool:
        """Detect potential XSS attempts."""
        return any(pattern.search(text) for pattern in self._xss_patterns)
    
    def _detect_path_traversal(self, text: str) -> bool:
        """Detect potential path traversal attempts."""
        return any(pattern.search(text) for pattern in self._path_traversal_patterns)
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content."""
        # HTML escape
        text = html.escape(text)
        
        # Remove any remaining script-like content
        for pattern in self._xss_patterns:
            text = pattern.sub('', text)
        
        return text
    
    def validate_api_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete API request.
        
        Args:
            data: Request data to validate
            
        Returns:
            Validated request data
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError("Request data must be a dictionary", "request")
        
        validated = {}
        
        for key, value in data.items():
            # Validate key
            key = self.validate_text(key, f"request.{key}", max_length=100)
            
            # Validate value based on type
            if isinstance(value, str):
                value = self.validate_text(value, f"request.{key}")
            elif isinstance(value, (dict, list)):
                value = self.validate_json(value, f"request.{key}")
            elif isinstance(value, (int, float, bool)) or value is None:
                pass  # These types are safe
            else:
                raise ValidationError(f"Unsupported data type in request.{key}", key, type(value))
            
            validated[key] = value
        
        return validated


# Global validator instance
_validator: Optional[InputValidator] = None


def get_validator(config: Optional[ValidationConfig] = None) -> InputValidator:
    """
    Get or create the global input validator instance.
    
    Args:
        config: Validation configuration (uses default if None)
        
    Returns:
        InputValidator instance
    """
    global _validator
    
    if _validator is None:
        _validator = InputValidator(config or ValidationConfig())
    
    return _validator


def reset_validator() -> None:
    """Reset the global validator (mainly for testing)."""
    global _validator
    _validator = None


# Convenience functions
def validate_text(text: str, field_name: str = "text") -> str:
    """Validate text using global validator."""
    return get_validator().validate_text(text, field_name)


def validate_json(data: Union[str, Dict, List], field_name: str = "json") -> Union[Dict, List]:
    """Validate JSON using global validator."""
    return get_validator().validate_json(data, field_name)


def validate_api_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate API request using global validator."""
    return get_validator().validate_api_request(data)
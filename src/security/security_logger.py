"""
Security Logging and Monitoring Module

Provides specialized logging for security events, audit trails, and
security monitoring with proper sanitization and alerting.
"""

import json
import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    INPUT_VALIDATION_ERROR = "input_validation_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    API_REQUEST = "api_request"
    API_ERROR = "api_error"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "config_change"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: SecurityEventType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    sensitive_data_hash: Optional[str] = None


@dataclass
class SecurityLogConfig:
    """Configuration for security logging."""
    log_file_path: Optional[str] = None
    max_log_size: int = 100 * 1024 * 1024  # 100MB
    max_log_files: int = 5
    log_level: str = "INFO"
    include_request_data: bool = True
    include_response_data: bool = False
    sanitize_sensitive_data: bool = True
    alert_on_critical: bool = True
    export_format: str = "json"  # "json" or "csv"


class SecurityLogger:
    """
    Comprehensive security logging system.
    
    Features:
    - Structured security event logging
    - Sensitive data sanitization
    - Audit trail maintenance
    - Real-time alerting for critical events
    - Log rotation and management
    - Export capabilities for SIEM systems
    """
    
    def __init__(self, config: SecurityLogConfig):
        self.config = config
        self._lock = threading.Lock()
        
        # Event storage
        self.events: List[SecurityEvent] = []
        self.max_events_in_memory = 1000
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
        
        # Sensitive data patterns for sanitization
        self._sensitive_patterns = self._compile_sensitive_patterns()
        
        # Setup file logging if configured
        self.file_logger = None
        if self.config.log_file_path:
            self._setup_file_logging()
        
        logger.info("SecurityLogger initialized")
    
    def _compile_sensitive_patterns(self) -> List[tuple]:
        """Compile patterns for detecting sensitive data."""
        import re
        
        patterns = [
            # API keys and tokens
            (re.compile(r'(api[_-]?key|token|secret)["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})', re.IGNORECASE), 'API_KEY'),
            # Credit card numbers
            (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), 'CREDIT_CARD'),
            # Social Security Numbers
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'SSN'),
            # Email addresses (in some contexts)
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 'EMAIL'),
            # Phone numbers
            (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), 'PHONE'),
            # URLs with credentials
            (re.compile(r'https?://[^:]+:[^@]+@[^/]+'), 'URL_WITH_CREDS'),
        ]
        
        return patterns
    
    def _setup_file_logging(self) -> None:
        """Setup file-based logging with rotation."""
        try:
            from logging.handlers import RotatingFileHandler
            
            log_dir = Path(self.config.log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.file_logger = logging.getLogger("security_events")
            self.file_logger.setLevel(getattr(logging, self.config.log_level.upper()))
            
            handler = RotatingFileHandler(
                self.config.log_file_path,
                maxBytes=self.config.max_log_size,
                backupCount=self.config.max_log_files
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S UTC'
            )
            handler.setFormatter(formatter)
            
            self.file_logger.addHandler(handler)
            
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")
    
    def log_event(self,
                  event_type: SecurityEventType,
                  severity: str,
                  message: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  endpoint: Optional[str] = None,
                  request_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity (low, medium, high, critical)
            message: Human-readable message
            user_id: User identifier (if applicable)
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent
            endpoint: API endpoint or resource accessed
            request_id: Unique request identifier
            metadata: Additional event metadata
        """
        # Sanitize sensitive data
        if self.config.sanitize_sensitive_data:
            message = self._sanitize_sensitive_data(message)
            if metadata:
                metadata = self._sanitize_metadata(metadata)
        
        # Create event
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            request_id=request_id,
            metadata=metadata,
            sensitive_data_hash=self._hash_sensitive_data(message, metadata)
        )
        
        with self._lock:
            self.events.append(event)
            
            # Maintain memory limit
            if len(self.events) > self.max_events_in_memory:
                self.events.pop(0)
        
        # Log to file if configured
        if self.file_logger:
            self._log_to_file(event)
        
        # Log to standard logger
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL,
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"Security Event [{event_type.value}]: {message}")
        
        # Trigger alerts for critical events
        if severity == "critical" and self.config.alert_on_critical:
            self._trigger_alerts(event)
    
    def _sanitize_sensitive_data(self, text: str) -> str:
        """Sanitize sensitive data in text."""
        if not text:
            return text
        
        sanitized = text
        
        for pattern, data_type in self._sensitive_patterns:
            def replace_match(match):
                original = match.group(0)
                if data_type == 'API_KEY':
                    # Keep first few characters for debugging
                    return f"{original[:8]}{'*' * (len(original) - 8)}"
                elif data_type == 'CREDIT_CARD':
                    return f"****-****-****-{original[-4:]}"
                elif data_type == 'SSN':
                    return "***-**-****"
                elif data_type == 'EMAIL':
                    parts = original.split('@')
                    if len(parts) == 2:
                        return f"{parts[0][:2]}***@{parts[1]}"
                    return "***@***.***"
                elif data_type == 'PHONE':
                    return "***-***-****"
                elif data_type == 'URL_WITH_CREDS':
                    return original.split('@')[0].split('://')[0] + "://***:***@" + original.split('@')[1]
                else:
                    return "*" * len(original)
            
            sanitized = pattern.sub(replace_match, sanitized)
        
        return sanitized
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize metadata."""
        sanitized = {}
        
        for key, value in metadata.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_sensitive_data(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_metadata(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_sensitive_data(item) if isinstance(item, str)
                    else self._sanitize_metadata(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _hash_sensitive_data(self, message: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Create hash of potentially sensitive data for correlation."""
        data_to_hash = message
        if metadata:
            data_to_hash += json.dumps(metadata, sort_keys=True)
        
        return hashlib.sha256(data_to_hash.encode()).hexdigest()[:16]
    
    def _log_to_file(self, event: SecurityEvent) -> None:
        """Log event to file."""
        try:
            if self.config.export_format == "json":
                log_data = asdict(event)
                log_data['timestamp_iso'] = datetime.fromtimestamp(
                    event.timestamp, timezone.utc
                ).isoformat()
                log_line = json.dumps(log_data)
            else:
                # CSV format
                log_line = f"{event.timestamp},{event.event_type.value},{event.severity},{event.message}"
            
            self.file_logger.info(log_line)
            
        except Exception as e:
            logger.error(f"Failed to log to file: {e}")
    
    def _trigger_alerts(self, event: SecurityEvent) -> None:
        """Trigger alerts for critical events."""
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for critical event alerts."""
        self.alert_callbacks.append(callback)
    
    def log_api_request(self,
                       endpoint: str,
                       method: str,
                       status_code: int,
                       user_id: Optional[str] = None,
                       request_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       response_time: Optional[float] = None,
                       request_data: Optional[Dict] = None,
                       error: Optional[str] = None) -> None:
        """
        Log an API request for audit purposes.
        
        Args:
            endpoint: API endpoint accessed
            method: HTTP method
            status_code: Response status code
            user_id: User making the request
            request_id: Unique request identifier
            ip_address: Client IP address
            response_time: Request processing time
            request_data: Request payload (if logging enabled)
            error: Error message if request failed
        """
        severity = "low"
        event_type = SecurityEventType.API_REQUEST
        
        if status_code >= 400:
            severity = "medium"
            event_type = SecurityEventType.API_ERROR
        
        if status_code >= 500:
            severity = "high"
        
        # Build metadata
        metadata = {
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
        }
        
        if request_data and self.config.include_request_data:
            metadata["request_data"] = request_data
        
        if error:
            metadata["error"] = error
        
        message = f"{method} {endpoint} -> {status_code}"
        if error:
            message += f" ({error})"
        
        self.log_event(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            request_id=request_id,
            ip_address=ip_address,
            endpoint=endpoint,
            metadata=metadata
        )
    
    def log_authentication_event(self,
                                success: bool,
                                user_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                reason: Optional[str] = None) -> None:
        """
        Log authentication attempt.
        
        Args:
            success: Whether authentication succeeded
            user_id: User attempting authentication
            ip_address: Client IP address
            user_agent: Client user agent
            reason: Reason for failure (if applicable)
        """
        if success:
            self.log_event(
                event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
                severity="low",
                message=f"Authentication succeeded for user {user_id}",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
        else:
            self.log_event(
                event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                severity="medium",
                message=f"Authentication failed for user {user_id}: {reason}",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={"reason": reason}
            )
    
    def get_events(self,
                  event_types: Optional[List[SecurityEventType]] = None,
                  severity_levels: Optional[List[str]] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: Optional[int] = None) -> List[SecurityEvent]:
        """
        Retrieve security events with filtering.
        
        Args:
            event_types: Filter by event types
            severity_levels: Filter by severity levels
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            limit: Maximum number of events to return
            
        Returns:
            List of filtered security events
        """
        with self._lock:
            filtered_events = self.events.copy()
        
        # Apply filters
        if event_types:
            filtered_events = [e for e in filtered_events if e.event_type in event_types]
        
        if severity_levels:
            filtered_events = [e for e in filtered_events if e.severity in severity_levels]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            filtered_events = filtered_events[:limit]
        
        return filtered_events
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security event summary and statistics."""
        with self._lock:
            events = self.events.copy()
        
        if not events:
            return {"total_events": 0}
        
        # Count by type
        type_counts = {}
        for event in events:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for event in events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        # Recent events (last hour)
        recent_events = [e for e in events if time.time() - e.timestamp <= 3600]
        
        return {
            "total_events": len(events),
            "recent_events_count": len(recent_events),
            "events_by_type": type_counts,
            "events_by_severity": severity_counts,
            "oldest_event": min(e.timestamp for e in events),
            "newest_event": max(e.timestamp for e in events),
        }


# Global security logger instance
_security_logger: Optional[SecurityLogger] = None


def get_security_logger(config: Optional[SecurityLogConfig] = None) -> SecurityLogger:
    """
    Get or create the global security logger instance.
    
    Args:
        config: Security logging configuration
        
    Returns:
        SecurityLogger instance
    """
    global _security_logger
    
    if _security_logger is None:
        _security_logger = SecurityLogger(config or SecurityLogConfig())
    
    return _security_logger


def reset_security_logger() -> None:
    """Reset the global security logger (mainly for testing)."""
    global _security_logger
    _security_logger = None


# Convenience functions
def log_security_event(event_type: SecurityEventType, severity: str, message: str, **kwargs) -> None:
    """Log security event using global logger."""
    get_security_logger().log_event(event_type, severity, message, **kwargs)


def log_api_request(endpoint: str, method: str, status_code: int, **kwargs) -> None:
    """Log API request using global logger."""
    get_security_logger().log_api_request(endpoint, method, status_code, **kwargs)
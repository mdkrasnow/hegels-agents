"""
Logging configuration and utilities for Hegels Agents project.

This module provides a comprehensive logging system designed for research and debugging
of multi-agent conversations, with structured logging capabilities for experiment tracking.
"""

import logging
import logging.handlers
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import traceback


class LogLevel(Enum):
    """Log levels for different types of information."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categories for different types of logs in the Hegels Agents system."""
    SYSTEM = "system"            # System operations and setup
    AGENT = "agent"              # Agent interactions and responses
    DEBATE = "debate"            # Debate orchestration and turns
    CORPUS = "corpus"            # Corpus retrieval and management
    EXPERIMENT = "experiment"    # Experiment tracking and metrics
    TOOL = "tool"               # Tool usage and function calls
    USER = "user"               # User interactions
    PERFORMANCE = "performance" # Performance metrics and timing


@dataclass
class StructuredLogEvent:
    """Structured log event for research and debugging."""
    timestamp: datetime
    level: str
    category: str
    event_type: str
    message: str
    data: Dict[str, Any]
    experiment_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    turn_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredLogger:
    """
    Structured logger for research and experiment tracking.
    
    This logger provides both traditional text logging and structured JSON logging
    for detailed analysis of agent interactions and system behavior.
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize the structured logger.
        
        Args:
            name: Logger name (typically the module name)
            log_dir: Directory for log files (defaults to logs/ in project root)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Determine log directory
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.log_dir = project_root / "logs"
        else:
            self.log_dir = Path(log_dir)
        
        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)
        
        # Session and experiment tracking
        self.current_session_id: Optional[str] = None
        self.current_experiment_id: Optional[str] = None
        
        # Set up JSON log file for structured data
        self.json_log_file = self.log_dir / f"{name}_structured.jsonl"
        
        # Don't add handlers multiple times
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger handlers and formatters."""
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(filename)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def set_session(self, session_id: str):
        """Set the current session ID for all subsequent logs."""
        self.current_session_id = session_id
        self.info(f"Started logging session: {session_id}", category=LogCategory.SYSTEM)
    
    def set_experiment(self, experiment_id: str):
        """Set the current experiment ID for all subsequent logs."""
        self.current_experiment_id = experiment_id
        self.info(f"Started logging experiment: {experiment_id}", category=LogCategory.EXPERIMENT)
    
    def _log_structured(self, event: StructuredLogEvent):
        """Write structured log event to JSON file."""
        try:
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + '\n')
        except Exception as e:
            # If structured logging fails, at least log to regular logger
            self.logger.error(f"Failed to write structured log: {e}")
    
    def log(self, 
            level: LogLevel, 
            message: str, 
            category: LogCategory = LogCategory.SYSTEM,
            event_type: str = "general",
            data: Optional[Dict[str, Any]] = None,
            agent_id: Optional[str] = None,
            turn_id: Optional[str] = None,
            **kwargs):
        """
        Log a message with structured data.
        
        Args:
            level: Log level
            message: Human-readable message
            category: Log category
            event_type: Specific type of event
            data: Additional structured data
            agent_id: Agent identifier (if applicable)
            turn_id: Turn identifier (if applicable)
            **kwargs: Additional data to include in structured log
        """
        # Merge kwargs into data
        if data is None:
            data = {}
        data.update(kwargs)
        
        # Log to standard logger
        getattr(self.logger, level.value.lower())(message)
        
        # Create structured log event
        event = StructuredLogEvent(
            timestamp=datetime.utcnow(),
            level=level.value,
            category=category.value,
            event_type=event_type,
            message=message,
            data=data,
            experiment_id=self.current_experiment_id,
            session_id=self.current_session_id,
            agent_id=agent_id,
            turn_id=turn_id
        )
        
        # Write structured log
        self._log_structured(event)
    
    # Convenience methods for different log levels
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    # Specialized logging methods for agent system
    def log_agent_response(self, 
                          agent_id: str, 
                          message: str, 
                          response: Any, 
                          turn_id: Optional[str] = None,
                          tool_calls: Optional[List[Dict]] = None,
                          **kwargs):
        """Log an agent response with full context."""
        self.log(
            LogLevel.INFO,
            f"Agent {agent_id}: {message}",
            category=LogCategory.AGENT,
            event_type="response",
            agent_id=agent_id,
            turn_id=turn_id,
            response=str(response)[:1000] if response else None,  # Truncate long responses
            tool_calls=tool_calls,
            **kwargs
        )
    
    def log_debate_event(self, 
                        event_type: str, 
                        message: str, 
                        debate_id: str,
                        participants: List[str],
                        turn_index: Optional[int] = None,
                        **kwargs):
        """Log a debate-related event."""
        self.log(
            LogLevel.INFO,
            message,
            category=LogCategory.DEBATE,
            event_type=event_type,
            debate_id=debate_id,
            participants=participants,
            turn_index=turn_index,
            **kwargs
        )
    
    def log_corpus_retrieval(self, 
                           query: str, 
                           num_results: int, 
                           retrieval_time: float,
                           **kwargs):
        """Log corpus retrieval operation."""
        self.log(
            LogLevel.INFO,
            f"Retrieved {num_results} results for query: '{query[:100]}...'",
            category=LogCategory.CORPUS,
            event_type="retrieval",
            query=query,
            num_results=num_results,
            retrieval_time=retrieval_time,
            **kwargs
        )
    
    def log_experiment_metric(self, 
                            metric_name: str, 
                            value: Union[float, int, str], 
                            **kwargs):
        """Log an experiment metric."""
        self.log(
            LogLevel.INFO,
            f"Metric {metric_name}: {value}",
            category=LogCategory.EXPERIMENT,
            event_type="metric",
            metric_name=metric_name,
            metric_value=value,
            **kwargs
        )
    
    def log_exception(self, 
                     exception: Exception, 
                     context: str = "",
                     **kwargs):
        """Log an exception with full traceback."""
        tb = traceback.format_exc()
        self.log(
            LogLevel.ERROR,
            f"Exception in {context}: {str(exception)}",
            category=LogCategory.SYSTEM,
            event_type="exception",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback=tb,
            context=context,
            **kwargs
        )


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class LoggerManager:
    """Manages loggers across the application."""
    
    _loggers: Dict[str, StructuredLogger] = {}
    _global_log_dir: Optional[Path] = None
    
    @classmethod
    def set_log_directory(cls, log_dir: Path):
        """Set global log directory for all loggers."""
        cls._global_log_dir = log_dir
    
    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """Get or create a logger with the given name."""
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(name, cls._global_log_dir)
        return cls._loggers[name]
    
    @classmethod
    def set_global_session(cls, session_id: str):
        """Set session ID for all existing loggers."""
        for logger in cls._loggers.values():
            logger.set_session(session_id)
    
    @classmethod
    def set_global_experiment(cls, experiment_id: str):
        """Set experiment ID for all existing loggers."""
        for logger in cls._loggers.values():
            logger.set_experiment(experiment_id)
    
    @classmethod
    def configure_logging(cls, 
                         level: str = "INFO",
                         log_dir: Optional[Path] = None,
                         disable_existing: bool = True):
        """
        Configure global logging settings.
        
        Args:
            level: Global log level
            log_dir: Directory for log files
            disable_existing: Whether to disable existing loggers
        """
        if log_dir:
            cls.set_log_directory(log_dir)
        
        # Set global level
        logging.getLogger().setLevel(getattr(logging, level.upper()))
        
        # Disable existing loggers if requested (prevents duplicate logs)
        if disable_existing:
            logging.getLogger().handlers.clear()


# Convenience function to get logger
def get_logger(name: str) -> StructuredLogger:
    """Get a logger instance. Convenience function for LoggerManager.get_logger."""
    return LoggerManager.get_logger(name)


# Module-level logger for logging utilities themselves
_logger = get_logger("hegels_agents.logging")


def setup_logging(log_level: str = "INFO", 
                 log_dir: Optional[str] = None,
                 session_id: Optional[str] = None,
                 experiment_id: Optional[str] = None) -> StructuredLogger:
    """
    Set up logging for the entire application.
    
    Args:
        log_level: Minimum log level to display
        log_dir: Directory for log files 
        session_id: Session identifier
        experiment_id: Experiment identifier
        
    Returns:
        Main application logger
    """
    # Configure global logging
    log_dir_path = Path(log_dir) if log_dir else None
    LoggerManager.configure_logging(log_level, log_dir_path)
    
    # Set global session and experiment if provided
    if session_id:
        LoggerManager.set_global_session(session_id)
    if experiment_id:
        LoggerManager.set_global_experiment(experiment_id)
    
    # Get main logger
    main_logger = get_logger("hegels_agents.main")
    main_logger.info("Logging system initialized", 
                    category=LogCategory.SYSTEM,
                    event_type="initialization",
                    log_level=log_level,
                    log_dir=str(log_dir_path) if log_dir_path else None)
    
    return main_logger


# Export key classes and functions
__all__ = [
    'StructuredLogger',
    'LoggerManager', 
    'get_logger',
    'setup_logging',
    'LogLevel',
    'LogCategory',
    'StructuredLogEvent'
]
"""
Configuration Management System for Hegels Agents

This module provides a flexible configuration system that can evolve from
simple environment variable loading to sophisticated parameter management.

Design Features:
- Environment variable loading with validation
- Extensible architecture for future enhancements
- Development and production configuration support
- Graceful error handling for missing variables
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import yaml


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


@dataclass
class DatabaseConfig:
    """Database configuration container."""
    url: str
    
    def validate(self) -> None:
        """Validate database configuration."""
        if not self.url:
            raise ConfigurationError("Database URL cannot be empty")
        if not self.url.startswith(('postgresql://', 'postgres://')):
            raise ConfigurationError("Database URL must be a valid PostgreSQL connection string")


@dataclass
class APIConfig:
    """API configuration container."""
    gemini_api_key: str
    
    def validate(self) -> None:
        """Validate API configuration."""
        if not self.gemini_api_key:
            raise ConfigurationError("Gemini API key cannot be empty")
        if len(self.gemini_api_key) < 10:  # Basic length validation
            raise ConfigurationError("Gemini API key appears to be invalid")


@dataclass
class AppConfig:
    """Application configuration container."""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    def validate(self) -> None:
        """Validate application configuration."""
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            raise ConfigurationError(f"Environment must be one of: {valid_environments}")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(f"Log level must be one of: {valid_log_levels}")


class ConfigManager:
    """
    Central configuration manager with extensible design.
    
    This class provides a foundation that can evolve from simple environment
    variable loading to sophisticated parameter management systems.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config_cache: Dict[str, Any] = {}
        self._loaded = False
        
        # Initialize configuration containers
        self.database: Optional[DatabaseConfig] = None
        self.api: Optional[APIConfig] = None
        self.app: Optional[AppConfig] = None
    
    def load_configuration(self, validate: bool = True) -> None:
        """
        Load configuration from environment variables and optional config file.
        
        Args:
            validate: Whether to validate configuration after loading
        """
        try:
            logger.info("Loading configuration...")
            
            # Load from environment variables
            self._load_from_environment()
            
            # Load from config file if provided
            if self.config_file and self.config_file.exists():
                self._load_from_file()
            
            # Initialize configuration objects
            self._initialize_config_objects()
            
            # Validate if requested
            if validate:
                self.validate_configuration()
            
            self._loaded = True
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Required environment variables
        required_vars = {
            'GEMINI_API_KEY': 'Gemini API key for AI model access',
            'SUPABASE_DB_URL': 'Supabase database connection URL'
        }
        
        # Check for required variables
        missing_vars = []
        for var_name, description in required_vars.items():
            value = os.getenv(var_name)
            if not value:
                missing_vars.append(f"{var_name} ({description})")
            else:
                self._config_cache[var_name.lower()] = value
        
        if missing_vars:
            error_msg = "Missing required environment variables:\n" + "\n".join(f"  - {var}" for var in missing_vars)
            error_msg += "\n\nPlease check your .env file or environment setup."
            raise ConfigurationError(error_msg)
        
        # Optional environment variables with defaults
        optional_vars = {
            'ENVIRONMENT': 'development',
            'DEBUG': 'false',
            'LOG_LEVEL': 'INFO'
        }
        
        for var_name, default_value in optional_vars.items():
            value = os.getenv(var_name, default_value)
            self._config_cache[var_name.lower()] = value
    
    def _load_from_file(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Merge file config with environment config (env takes precedence)
                    for key, value in file_config.items():
                        if key.lower() not in self._config_cache:
                            self._config_cache[key.lower()] = value
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file {self.config_file}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading config file {self.config_file}: {e}")
    
    def _initialize_config_objects(self) -> None:
        """Initialize typed configuration objects."""
        # Database configuration
        self.database = DatabaseConfig(
            url=self._config_cache['supabase_db_url']
        )
        
        # API configuration
        self.api = APIConfig(
            gemini_api_key=self._config_cache['gemini_api_key']
        )
        
        # Application configuration
        self.app = AppConfig(
            environment=self._config_cache.get('environment', 'development'),
            debug=self._config_cache.get('debug', 'false').lower() == 'true',
            log_level=self._config_cache.get('log_level', 'INFO').upper()
        )
    
    def validate_configuration(self) -> None:
        """Validate all configuration objects."""
        if not self.database or not self.api or not self.app:
            raise ConfigurationError("Configuration objects must be initialized before validation")
        
        try:
            self.database.validate()
            self.api.validate()
            self.app.validate()
            logger.info("Configuration validation successful")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self._loaded:
            raise ConfigurationError("Configuration must be loaded first")
        
        return self._config_cache.get(key.lower(), default)
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app.environment == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.environment == 'production'
    
    def get_database_url(self) -> str:
        """Get database URL."""
        return self.database.url
    
    def get_gemini_api_key(self) -> str:
        """Get Gemini API key."""
        return self.api.gemini_api_key
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary (excluding sensitive data).
        
        Returns:
            Dictionary representation of configuration
        """
        if not self._loaded:
            return {}
        
        return {
            'environment': self.app.environment,
            'debug': self.app.debug,
            'log_level': self.app.log_level,
            'database_configured': bool(self.database.url),
            'api_key_configured': bool(self.api.gemini_api_key),
        }


# Global configuration instance
config = ConfigManager()


def load_config(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Load and return configuration manager instance.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured ConfigManager instance
    """
    global config
    
    if config_file:
        config = ConfigManager(config_file)
    
    config.load_configuration()
    return config


def get_config() -> ConfigManager:
    """
    Get the global configuration instance.
    
    Returns:
        Global ConfigManager instance
        
    Raises:
        ConfigurationError: If configuration has not been loaded
    """
    if not config._loaded:
        raise ConfigurationError("Configuration has not been loaded. Call load_config() first.")
    
    return config


# Convenience functions for common configuration access
def get_database_url() -> str:
    """Get database URL from global config."""
    return get_config().get_database_url()


def get_gemini_api_key() -> str:
    """Get Gemini API key from global config."""
    return get_config().get_gemini_api_key()


def is_development() -> bool:
    """Check if running in development mode."""
    return get_config().is_development()


def is_production() -> bool:
    """Check if running in production mode."""
    return get_config().is_production()
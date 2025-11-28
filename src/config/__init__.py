"""
Configuration package for Hegels Agents.

This package provides a comprehensive configuration management system that
supports environment variables, YAML files, and extensible parameter management.

Usage:
    from src.config import load_config, get_config
    
    # Load configuration
    config = load_config()
    
    # Access configuration values
    db_url = config.get_database_url()
    api_key = config.get_gemini_api_key()
    
    # Check environment
    if config.is_development():
        print("Running in development mode")

Classes:
    ConfigManager: Main configuration management class
    ConfigurationError: Custom exception for configuration errors
    DatabaseConfig: Database configuration container
    APIConfig: API configuration container
    AppConfig: Application configuration container

Functions:
    load_config: Load configuration from environment and files
    get_config: Get the global configuration instance
    get_database_url: Get database URL
    get_gemini_api_key: Get Gemini API key
    is_development: Check if in development mode
    is_production: Check if in production mode
"""

from .settings import (
    ConfigManager,
    ConfigurationError,
    DatabaseConfig,
    APIConfig,
    AppConfig,
    config,
    load_config,
    get_config,
    get_database_url,
    get_gemini_api_key,
    is_development,
    is_production
)

__all__ = [
    'ConfigManager',
    'ConfigurationError',
    'DatabaseConfig',
    'APIConfig',
    'AppConfig',
    'config',
    'load_config',
    'get_config',
    'get_database_url',
    'get_gemini_api_key',
    'is_development',
    'is_production'
]

__version__ = '1.0.0'
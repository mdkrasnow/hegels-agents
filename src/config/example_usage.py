#!/usr/bin/env python3
"""
Example usage of the Hegels Agents configuration management system.

This script demonstrates how to use the configuration system in your application.
"""

import os
import sys
from pathlib import Path

# Example of how to import the configuration system
try:
    from . import load_config, get_config, get_database_url, get_gemini_api_key
except ImportError:
    # For running as standalone script
    sys.path.append(str(Path(__file__).parent.parent))
    from config import load_config, get_config, get_database_url, get_gemini_api_key


def main():
    """Example of configuration usage."""
    print("Hegels Agents Configuration System Example")
    print("=" * 50)
    
    # Method 1: Basic usage with environment variables
    print("\n1. Loading configuration from environment variables:")
    
    try:
        # Load configuration
        config = load_config()
        
        print(f"   Environment: {config.app.environment}")
        print(f"   Debug mode: {config.app.debug}")
        print(f"   Log level: {config.app.log_level}")
        print(f"   Database configured: {'Yes' if config.database.url else 'No'}")
        print(f"   API key configured: {'Yes' if config.api.gemini_api_key else 'No'}")
        
        # Access specific values using convenience functions
        print(f"\n2. Accessing specific configuration values:")
        print(f"   Database URL: {get_database_url()[:30]}..." if get_database_url() else "   Database URL: Not configured")
        print(f"   API Key: {get_gemini_api_key()[:10]}..." if get_gemini_api_key() else "   API Key: Not configured")
        
        # Check environment type
        print(f"\n3. Environment checks:")
        print(f"   Is development: {config.is_development()}")
        print(f"   Is production: {config.is_production()}")
        
        # Export configuration (safe export without sensitive data)
        print(f"\n4. Configuration export:")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"   Error: {e}")
        print("\n   To fix this, make sure you have set the required environment variables:")
        print("   - GEMINI_API_KEY: Your Gemini API key")
        print("   - SUPABASE_DB_URL: Your Supabase database URL")
        print("\n   You can copy .env.template to .env and fill in your values.")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Environment validation script for Hegels Agents project.
Run this after setting up your Python environment to verify dependencies.
"""

import sys
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)"


def check_imports() -> List[Tuple[str, bool, str]]:
    """Check if required packages can be imported."""
    packages = [
        ("google.genai", "Google GenAI SDK"),
        ("psycopg2", "PostgreSQL adapter"),
        ("pydantic", "Data validation"),
        ("sqlalchemy", "SQL toolkit"),
        ("pytest", "Testing framework"),
        ("tqdm", "Progress bars"),
        ("dotenv", "Environment variables"),
        ("structlog", "Structured logging"),
    ]
    
    results = []
    for package, description in packages:
        try:
            __import__(package)
            results.append((package, True, description))
        except ImportError as e:
            results.append((package, False, f"{description} - Error: {e}"))
    
    return results


def check_optional_imports() -> List[Tuple[str, bool, str]]:
    """Check optional dependencies."""
    optional_packages = [
        ("numpy", "Numerical computing"),
        ("pandas", "Data analysis"),
        ("matplotlib", "Plotting"),
        ("supabase", "Supabase client"),
        ("jupyter", "Interactive notebooks"),
    ]
    
    results = []
    for package, description in optional_packages:
        try:
            __import__(package)
            results.append((package, True, description))
        except ImportError:
            results.append((package, False, f"{description} (optional)"))
    
    return results


def main():
    """Main validation function."""
    print("🔍 Hegels Agents Environment Validation")
    print("=" * 50)
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    status = "✅" if python_ok else "❌"
    print(f"{status} Python Version: {python_msg}")
    
    if not python_ok:
        print("\n❌ Python version requirement not met. Please upgrade to Python 3.10+")
        sys.exit(1)
    
    # Check required imports
    print("\n📦 Required Dependencies:")
    required_results = check_imports()
    all_required_ok = True
    
    for package, success, description in required_results:
        status = "✅" if success else "❌"
        print(f"  {status} {package:15} - {description}")
        if not success:
            all_required_ok = False
    
    # Check optional imports
    print("\n📦 Optional Dependencies:")
    optional_results = check_optional_imports()
    
    for package, success, description in optional_results:
        status = "✅" if success else "ℹ️ "
        print(f"  {status} {package:15} - {description}")
    
    # Summary
    print("\n" + "=" * 50)
    if all_required_ok:
        print("🎉 Environment validation successful!")
        print("All required dependencies are available.")
        
        optional_count = sum(1 for _, success, _ in optional_results if success)
        total_optional = len(optional_results)
        print(f"Optional dependencies: {optional_count}/{total_optional} available")
        
        print("\nNext steps:")
        print("1. Create a .env file with your API keys")
        print("2. Review setup-python-env.md for configuration details")
        print("3. Start implementing the Hegels Agents system!")
    else:
        print("❌ Environment validation failed!")
        print("Please install missing dependencies using:")
        print("  uv pip install -r requirements.txt")
        print("  # or")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
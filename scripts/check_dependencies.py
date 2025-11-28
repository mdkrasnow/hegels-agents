#!/usr/bin/env python3
"""
Dependency checker for Hegels Agents project.

This script verifies that all required dependencies are properly installed
and can be imported correctly. It's designed to catch issues early in the
development process.
"""

import sys
import importlib
import subprocess
import platform
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Use importlib.metadata (Python 3.8+) instead of deprecated pkg_resources
try:
    from importlib.metadata import version as get_version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8
    import pkg_resources
    def get_version(package_name: str) -> str:
        return pkg_resources.get_distribution(package_name).version
    
    class PackageNotFoundError(Exception):
        pass

from packaging import version


class DependencyChecker:
    """Comprehensive dependency verification for the Hegels Agents project."""
    
    def __init__(self):
        """Initialize the dependency checker."""
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Core dependencies with minimum versions
        self.core_dependencies = {
            "google.genai": {"package": "google-genai", "min_version": "0.1.0"},
            "psycopg2": {"package": "psycopg2-binary", "min_version": "2.9.0", "alternative": "psycopg"},
            "pydantic": {"package": "pydantic", "min_version": "2.0.0"},
            "pytest": {"package": "pytest", "min_version": "7.0.0"},
            "tqdm": {"package": "tqdm", "min_version": "4.0.0"},
        }
        
        # Optional dependencies that enhance functionality
        self.optional_dependencies = {
            "sqlalchemy": {"package": "sqlalchemy", "min_version": "2.0.0"},
            "asyncpg": {"package": "asyncpg", "min_version": "0.27.0"},
            "numpy": {"package": "numpy", "min_version": "1.20.0"},
            "pandas": {"package": "pandas", "min_version": "1.3.0"},
            "requests": {"package": "requests", "min_version": "2.25.0"},
        }
        
        # Development dependencies
        self.dev_dependencies = {
            "black": {"package": "black", "min_version": "22.0.0"},
            "flake8": {"package": "flake8", "min_version": "4.0.0"},
            "mypy": {"package": "mypy", "min_version": "1.0.0"},
            "pre_commit": {"package": "pre-commit", "min_version": "2.15.0", "import_name": "pre_commit"},
        }
    
    def get_installed_version(self, package_name: str) -> Optional[str]:
        """
        Get the installed version of a package.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            Version string if installed, None if not found
        """
        try:
            return get_version(package_name)
        except PackageNotFoundError:
            return None
    
    def can_import(self, module_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a module can be imported.
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            importlib.import_module(module_name)
            return True, None
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def check_version_compatibility(self, installed: str, required: str) -> bool:
        """
        Check if installed version meets minimum requirement.
        
        Args:
            installed: Installed version string
            required: Required minimum version string
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            return version.parse(installed) >= version.parse(required)
        except Exception:
            # If version parsing fails, assume compatibility
            return True
    
    def check_dependency_group(self, dependencies: Dict[str, Dict], group_name: str, required: bool = True) -> Dict[str, Any]:
        """
        Check a group of dependencies.
        
        Args:
            dependencies: Dictionary of dependencies to check
            group_name: Name of the dependency group
            required: Whether these dependencies are required
            
        Returns:
            Dictionary with check results
        """
        results = {
            "group_name": group_name,
            "required": required,
            "total": len(dependencies),
            "passed": 0,
            "failed": 0,
            "details": {}
        }
        
        for module_name, dep_info in dependencies.items():
            package_name = dep_info["package"]
            min_version = dep_info.get("min_version")
            import_name = dep_info.get("import_name", module_name)
            alternative = dep_info.get("alternative")
            
            # Check installation
            installed_version = self.get_installed_version(package_name)
            
            # Check import capability
            can_import_main, import_error = self.can_import(import_name)
            can_import_alt = False
            
            # Check alternative if main fails and alternative exists
            if not can_import_main and alternative:
                can_import_alt, _ = self.can_import(alternative)
            
            # Determine if this dependency passes
            is_installed = installed_version is not None
            can_import_any = can_import_main or can_import_alt
            version_ok = True
            
            if is_installed and min_version and installed_version:
                version_ok = self.check_version_compatibility(installed_version, min_version)
            
            passed = is_installed and can_import_any and version_ok
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            # Store detailed results
            results["details"][module_name] = {
                "package": package_name,
                "installed": is_installed,
                "version": installed_version,
                "min_version": min_version,
                "version_ok": version_ok,
                "can_import": can_import_main,
                "import_error": import_error if not can_import_main else None,
                "alternative_import": can_import_alt if alternative else None,
                "passed": passed,
                "status": "✅" if passed else "❌"
            }
        
        return results
    
    def check_python_environment(self) -> Dict[str, Any]:
        """
        Check Python environment details.
        
        Returns:
            Dictionary with Python environment information
        """
        return {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "in_virtualenv": hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            ),
            "executable": sys.executable,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro
            },
            "compatible": sys.version_info >= (3, 9)
        }
    
    def check_project_structure(self) -> Dict[str, Any]:
        """
        Verify project structure is correct.
        
        Returns:
            Dictionary with project structure verification results
        """
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            "src",
            "src/config",
            "src/agents", 
            "src/corpus",
            "src/debate",
            "src/eval",
            "scripts"
        ]
        
        required_files = [
            "src/__init__.py",
            "src/config/__init__.py",
            "src/agents/__init__.py",
            "src/corpus/__init__.py"
        ]
        
        structure_results = {
            "project_root": str(project_root),
            "directories": {},
            "files": {},
            "all_present": True
        }
        
        # Check directories
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            structure_results["directories"][dir_path] = {
                "exists": exists,
                "path": str(full_path),
                "status": "✅" if exists else "❌"
            }
            if not exists:
                structure_results["all_present"] = False
        
        # Check files
        for file_path in required_files:
            full_path = project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            structure_results["files"][file_path] = {
                "exists": exists,
                "path": str(full_path),
                "status": "✅" if exists else "❌"
            }
            if not exists:
                structure_results["all_present"] = False
        
        return structure_results
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """
        Run comprehensive dependency and environment check.
        
        Returns:
            Complete results dictionary
        """
        print("🔍 Running comprehensive dependency check...")
        
        results = {
            "python_env": self.check_python_environment(),
            "core_deps": self.check_dependency_group(self.core_dependencies, "Core Dependencies"),
            "optional_deps": self.check_dependency_group(self.optional_dependencies, "Optional Dependencies", required=False),
            "dev_deps": self.check_dependency_group(self.dev_dependencies, "Development Dependencies", required=False),
            "project_structure": self.check_project_structure()
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> bool:
        """
        Print formatted results of dependency check.
        
        Args:
            results: Results dictionary from run_comprehensive_check
            
        Returns:
            True if all critical checks passed, False otherwise
        """
        all_critical_passed = True
        
        # Python Environment
        env = results["python_env"]
        print(f"\n🐍 Python Environment:")
        print(f"  Version: {env['python_version']} ({env['python_implementation']})")
        print(f"  Platform: {env['platform']}")
        print(f"  Virtual Environment: {'✅ Yes' if env['in_virtualenv'] else '⚠️  No'}")
        print(f"  Compatible (3.9+): {'✅ Yes' if env['compatible'] else '❌ No'}")
        
        if not env['compatible']:
            all_critical_passed = False
        
        # Core Dependencies
        core = results["core_deps"]
        print(f"\n📦 {core['group_name']}: {core['passed']}/{core['total']} passed")
        for dep_name, details in core["details"].items():
            version_info = f" (v{details['version']})" if details['version'] else ""
            print(f"  {details['status']} {dep_name}{version_info}")
            if details['import_error'] and not details['passed']:
                print(f"    Import Error: {details['import_error']}")
        
        if core["failed"] > 0:
            all_critical_passed = False
        
        # Optional Dependencies
        opt = results["optional_deps"]
        print(f"\n🔧 {opt['group_name']}: {opt['passed']}/{opt['total']} available")
        for dep_name, details in opt["details"].items():
            version_info = f" (v{details['version']})" if details['version'] else ""
            status = "✅" if details['passed'] else "⚠️ "
            print(f"  {status} {dep_name}{version_info}")
        
        # Development Dependencies  
        dev = results["dev_deps"]
        print(f"\n🛠️  {dev['group_name']}: {dev['passed']}/{dev['total']} available")
        for dep_name, details in dev["details"].items():
            version_info = f" (v{details['version']})" if details['version'] else ""
            status = "✅" if details['passed'] else "⚠️ "
            print(f"  {status} {dep_name}{version_info}")
        
        # Project Structure
        structure = results["project_structure"]
        print(f"\n📁 Project Structure: {'✅ Valid' if structure['all_present'] else '❌ Issues Found'}")
        
        if not structure['all_present']:
            print("  Missing directories:")
            for dir_name, dir_info in structure["directories"].items():
                if not dir_info['exists']:
                    print(f"    ❌ {dir_name}")
            
            print("  Missing files:")
            for file_name, file_info in structure["files"].items():
                if not file_info['exists']:
                    print(f"    ❌ {file_name}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Critical Issues: {'✅ None' if all_critical_passed else '❌ Found'}")
        print(f"  Core Dependencies: {core['passed']}/{core['total']}")
        print(f"  Optional Features: {opt['passed']}/{opt['total']}")
        print(f"  Development Tools: {dev['passed']}/{dev['total']}")
        
        if not all_critical_passed:
            print(f"\n⚠️  Critical issues found. Please resolve them before proceeding.")
            print(f"   Run 'python scripts/setup_environment.py' to set up missing dependencies.")
        else:
            print(f"\n🎉 All critical dependencies are satisfied!")
        
        return all_critical_passed
    
    def get_installation_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate installation suggestions for missing dependencies.
        
        Args:
            results: Results dictionary from run_comprehensive_check
            
        Returns:
            List of installation command suggestions
        """
        suggestions = []
        
        # Core dependencies
        for dep_name, details in results["core_deps"]["details"].items():
            if not details["passed"]:
                package = details["package"]
                suggestions.append(f"pip install {package}")
        
        # Optional dependencies that might be useful
        missing_optional = []
        for dep_name, details in results["optional_deps"]["details"].items():
            if not details["passed"]:
                missing_optional.append(details["package"])
        
        if missing_optional:
            suggestions.append(f"# Optional: pip install {' '.join(missing_optional)}")
        
        return suggestions


def main():
    """Main entry point for the dependency checker."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check dependencies for Hegels Agents project"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--suggest-install",
        action="store_true", 
        help="Show installation commands for missing dependencies"
    )
    
    args = parser.parse_args()
    
    try:
        checker = DependencyChecker()
        results = checker.run_comprehensive_check()
        
        if args.json:
            import json
            print(json.dumps(results, indent=2, default=str))
        else:
            all_passed = checker.print_results(results)
            
            if args.suggest_install:
                suggestions = checker.get_installation_suggestions(results)
                if suggestions:
                    print(f"\n💡 Installation Suggestions:")
                    for suggestion in suggestions:
                        print(f"  {suggestion}")
            
            # Exit with appropriate code
            sys.exit(0 if all_passed else 1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during dependency check: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
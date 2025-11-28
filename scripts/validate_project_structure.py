#!/usr/bin/env python3
"""
Project structure validation script for Hegels Agents.

This script validates that the project follows the expected directory structure
and contains all necessary files for the progressive enhancement architecture.
It's designed to catch structural issues early and ensure consistency.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import yaml


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "ERROR"      # Critical issues that prevent functionality
    WARNING = "WARNING"  # Issues that should be addressed but don't break functionality
    INFO = "INFO"        # Informational messages about structure


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    level: ValidationLevel
    passed: bool
    message: str
    path: Optional[Path] = None
    suggestion: Optional[str] = None


class ProjectStructureValidator:
    """Validates the Hegels Agents project structure."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the project structure validator.
        
        Args:
            project_root: Path to project root directory. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (directory containing this script's parent)
            script_dir = Path(__file__).parent
            self.project_root = script_dir.parent
        else:
            self.project_root = Path(project_root)
        
        self.results: List[ValidationResult] = []
    
    def get_expected_structure(self) -> Dict[str, Any]:
        """
        Get the expected project structure definition.
        
        Returns:
            Dictionary defining the expected structure
        """
        return {
            "directories": {
                "required": [
                    "src",
                    "src/config",
                    "src/agents", 
                    "src/corpus",
                    "src/debate",
                    "src/eval",
                    "scripts",
                    "documentation"
                ],
                "optional": [
                    "tests",
                    "src/tests",
                    "logs",
                    "data",
                    "examples",
                    "docs",
                    "htmlcov",
                    "test_reports",
                    ".pytest_cache",
                    ".mypy_cache",
                    "__pycache__",
                    "venv",
                    ".venv",
                    "env",
                    ".env",
                    "node_modules"
                ]
            },
            "files": {
                "required": [
                    "src/__init__.py",
                    "src/config/__init__.py",
                    "src/agents/__init__.py",
                    "src/corpus/__init__.py",
                    "README.md"
                ],
                "optional": [
                    "src/debate/__init__.py",
                    "src/eval/__init__.py",
                    "src/config/settings.py",
                    "src/config/logging.py",
                    "pyproject.toml",
                    "requirements.txt",
                    "Pipfile",
                    ".gitignore",
                    ".env.template",
                    ".env",
                    "LICENSE",
                    "CONTRIBUTING.md",
                    "CHANGELOG.md",
                    "setup.py",
                    "setup.cfg",
                    "tox.ini",
                    ".pre-commit-config.yaml",
                    "pytest.ini",
                    ".coverage",
                    ".coveragerc"
                ],
                "scripts": [
                    "scripts/setup_environment.py",
                    "scripts/check_dependencies.py",
                    "scripts/run_tests.py",
                    "scripts/validate_project_structure.py"
                ]
            },
            "patterns": {
                "python_modules": ["*.py", "**/*.py"],
                "config_files": ["*.yml", "*.yaml", "*.toml", "*.ini", "*.cfg"],
                "documentation": ["*.md", "*.rst", "*.txt"],
                "ignore_dirs": [
                    "__pycache__", 
                    ".pytest_cache", 
                    ".mypy_cache",
                    "htmlcov",
                    ".git",
                    "venv",
                    ".venv", 
                    "env",
                    ".env",
                    "node_modules"
                ]
            }
        }
    
    def validate_directory_structure(self) -> List[ValidationResult]:
        """
        Validate the directory structure.
        
        Returns:
            List of validation results
        """
        results = []
        structure = self.get_expected_structure()
        
        # Check required directories
        for dir_path in structure["directories"]["required"]:
            full_path = self.project_root / dir_path
            
            if full_path.exists() and full_path.is_dir():
                results.append(ValidationResult(
                    name=f"required_dir_{dir_path.replace('/', '_')}",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message=f"Required directory exists: {dir_path}",
                    path=full_path
                ))
            else:
                results.append(ValidationResult(
                    name=f"required_dir_{dir_path.replace('/', '_')}",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Required directory missing: {dir_path}",
                    path=full_path,
                    suggestion=f"Create with: mkdir -p {full_path}"
                ))
        
        # Check for unexpected directories (not exhaustive, just common issues)
        potential_issues = [".git", "node_modules", "__pycache__"]
        for issue_dir in potential_issues:
            issue_path = self.project_root / issue_dir
            if issue_path.exists():
                if issue_dir == ".git":
                    results.append(ValidationResult(
                        name="git_repo",
                        level=ValidationLevel.INFO,
                        passed=True,
                        message="Git repository detected",
                        path=issue_path
                    ))
                elif issue_dir in ["__pycache__", "node_modules"]:
                    results.append(ValidationResult(
                        name=f"cache_dir_{issue_dir}",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Cache/build directory present: {issue_dir}",
                        path=issue_path,
                        suggestion=f"Consider adding {issue_dir} to .gitignore"
                    ))
        
        return results
    
    def validate_python_packages(self) -> List[ValidationResult]:
        """
        Validate Python package structure (presence of __init__.py files).
        
        Returns:
            List of validation results
        """
        results = []
        structure = self.get_expected_structure()
        
        # Check required __init__.py files
        required_files = [f for f in structure["files"]["required"] if f.endswith("__init__.py")]
        
        for init_file in required_files:
            full_path = self.project_root / init_file
            
            if full_path.exists() and full_path.is_file():
                # Check if it's a valid Python file
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Basic syntax check (compile but don't execute)
                    compile(content, str(full_path), 'exec')
                    
                    results.append(ValidationResult(
                        name=f"init_file_{init_file.replace('/', '_').replace('.py', '')}",
                        level=ValidationLevel.INFO,
                        passed=True,
                        message=f"Valid Python package file: {init_file}",
                        path=full_path
                    ))
                    
                except SyntaxError as e:
                    results.append(ValidationResult(
                        name=f"init_file_{init_file.replace('/', '_').replace('.py', '')}",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Syntax error in {init_file}: {e}",
                        path=full_path,
                        suggestion="Fix Python syntax errors"
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        name=f"init_file_{init_file.replace('/', '_').replace('.py', '')}",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Error reading {init_file}: {e}",
                        path=full_path
                    ))
            else:
                results.append(ValidationResult(
                    name=f"init_file_{init_file.replace('/', '_').replace('.py', '')}",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Missing Python package file: {init_file}",
                    path=full_path,
                    suggestion=f"Create with: touch {full_path}"
                ))
        
        return results
    
    def validate_configuration_files(self) -> List[ValidationResult]:
        """
        Validate configuration files and project metadata.
        
        Returns:
            List of validation results
        """
        results = []
        
        # Check for dependency management file
        dependency_files = ["pyproject.toml", "requirements.txt", "Pipfile"]
        has_dependency_file = False
        
        for dep_file in dependency_files:
            dep_path = self.project_root / dep_file
            if dep_path.exists():
                has_dependency_file = True
                results.append(ValidationResult(
                    name=f"dependency_file_{dep_file.replace('.', '_')}",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message=f"Found dependency file: {dep_file}",
                    path=dep_path
                ))
                
                # Validate content for specific files
                if dep_file == "pyproject.toml":
                    self._validate_pyproject_toml(dep_path, results)
                elif dep_file == "requirements.txt":
                    self._validate_requirements_txt(dep_path, results)
        
        if not has_dependency_file:
            results.append(ValidationResult(
                name="dependency_management",
                level=ValidationLevel.ERROR,
                passed=False,
                message="No dependency management file found",
                suggestion="Create pyproject.toml, requirements.txt, or Pipfile"
            ))
        
        # Check for README
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            results.append(ValidationResult(
                name="readme_file",
                level=ValidationLevel.INFO,
                passed=True,
                message="README.md file exists",
                path=readme_path
            ))
            
            # Check README content
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 50:
                    results.append(ValidationResult(
                        name="readme_content",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message="README.md appears to be very short or empty",
                        path=readme_path,
                        suggestion="Add project description and usage instructions"
                    ))
            except Exception:
                pass
        else:
            results.append(ValidationResult(
                name="readme_file",
                level=ValidationLevel.ERROR,
                passed=False,
                message="README.md file missing",
                path=readme_path,
                suggestion="Create README.md with project description"
            ))
        
        # Check for .gitignore
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            results.append(ValidationResult(
                name="gitignore_file",
                level=ValidationLevel.INFO,
                passed=True,
                message=".gitignore file exists",
                path=gitignore_path
            ))
        else:
            results.append(ValidationResult(
                name="gitignore_file",
                level=ValidationLevel.WARNING,
                passed=False,
                message=".gitignore file missing",
                path=gitignore_path,
                suggestion="Create .gitignore to exclude temporary files"
            ))
        
        return results
    
    def _validate_pyproject_toml(self, file_path: Path, results: List[ValidationResult]):
        """Validate pyproject.toml content."""
        try:
            import tomli
        except ImportError:
            try:
                import tomllib as tomli  # Python 3.11+
            except ImportError:
                results.append(ValidationResult(
                    name="pyproject_validation",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message="Cannot validate pyproject.toml (tomli not available)",
                    path=file_path,
                    suggestion="Install tomli for pyproject.toml validation"
                ))
                return
        
        try:
            with open(file_path, 'rb') as f:
                data = tomli.load(f)
            
            # Check for required sections
            if 'project' in data:
                project = data['project']
                if 'name' in project and 'version' in project:
                    results.append(ValidationResult(
                        name="pyproject_metadata",
                        level=ValidationLevel.INFO,
                        passed=True,
                        message="pyproject.toml has valid project metadata",
                        path=file_path
                    ))
                else:
                    results.append(ValidationResult(
                        name="pyproject_metadata",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message="pyproject.toml missing project name or version",
                        path=file_path
                    ))
            
        except Exception as e:
            results.append(ValidationResult(
                name="pyproject_syntax",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Error parsing pyproject.toml: {e}",
                path=file_path
            ))
    
    def _validate_requirements_txt(self, file_path: Path, results: List[ValidationResult]):
        """Validate requirements.txt content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Basic validation - check for common packages
            expected_packages = ["google-genai", "psycopg2-binary", "pydantic", "pytest"]
            found_packages = set()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before version specifiers)
                    package_name = line.split('>=')[0].split('==')[0].split('<=')[0].strip()
                    found_packages.add(package_name)
            
            missing_packages = set(expected_packages) - found_packages
            if missing_packages:
                results.append(ValidationResult(
                    name="requirements_packages",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"requirements.txt missing expected packages: {', '.join(missing_packages)}",
                    path=file_path,
                    suggestion=f"Add missing packages: {' '.join(missing_packages)}"
                ))
            else:
                results.append(ValidationResult(
                    name="requirements_packages",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="requirements.txt contains expected packages",
                    path=file_path
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                name="requirements_syntax",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Error reading requirements.txt: {e}",
                path=file_path
            ))
    
    def validate_script_files(self) -> List[ValidationResult]:
        """
        Validate utility scripts.
        
        Returns:
            List of validation results
        """
        results = []
        structure = self.get_expected_structure()
        
        # Check required scripts
        for script_path in structure["files"]["scripts"]:
            full_path = self.project_root / script_path
            
            if full_path.exists() and full_path.is_file():
                # Check if it's executable and has proper shebang
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    
                    if first_line.startswith('#!') and 'python' in first_line:
                        results.append(ValidationResult(
                            name=f"script_{full_path.name.replace('.py', '')}",
                            level=ValidationLevel.INFO,
                            passed=True,
                            message=f"Valid Python script: {script_path}",
                            path=full_path
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"script_{full_path.name.replace('.py', '')}",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"Script missing shebang line: {script_path}",
                            path=full_path,
                            suggestion="Add #!/usr/bin/env python3 as first line"
                        ))
                        
                    # Check if script is executable
                    if not os.access(full_path, os.X_OK):
                        results.append(ValidationResult(
                            name=f"script_executable_{full_path.name.replace('.py', '')}",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"Script not executable: {script_path}",
                            path=full_path,
                            suggestion=f"Make executable with: chmod +x {full_path}"
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        name=f"script_{full_path.name.replace('.py', '')}",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Error reading script {script_path}: {e}",
                        path=full_path
                    ))
            else:
                results.append(ValidationResult(
                    name=f"script_{script_path.replace('/', '_').replace('.py', '')}",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Required script missing: {script_path}",
                    path=full_path,
                    suggestion=f"Create script: {script_path}"
                ))
        
        return results
    
    def validate_project_consistency(self) -> List[ValidationResult]:
        """
        Validate overall project consistency and best practices.
        
        Returns:
            List of validation results
        """
        results = []
        
        # Check if project follows naming conventions
        src_modules = list((self.project_root / "src").rglob("*.py")) if (self.project_root / "src").exists() else []
        
        naming_issues = []
        for module_path in src_modules:
            if module_path.name == "__init__.py":
                continue
                
            # Check for lowercase with underscores (PEP 8)
            module_name = module_path.stem
            if not module_name.islower() or ' ' in module_name or '-' in module_name:
                naming_issues.append(str(module_path.relative_to(self.project_root)))
        
        if naming_issues:
            results.append(ValidationResult(
                name="module_naming",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Modules with non-PEP8 names: {', '.join(naming_issues[:3])}{'...' if len(naming_issues) > 3 else ''}",
                suggestion="Use lowercase with underscores for Python module names"
            ))
        else:
            results.append(ValidationResult(
                name="module_naming",
                level=ValidationLevel.INFO,
                passed=True,
                message="Module names follow PEP 8 conventions"
            ))
        
        # Check for common files that should be ignored
        temp_files = list(self.project_root.rglob("*.pyc")) + list(self.project_root.rglob("*.pyo"))
        if temp_files:
            results.append(ValidationResult(
                name="temp_files",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Found {len(temp_files)} temporary Python files (.pyc/.pyo)",
                suggestion="Add __pycache__/ and *.pyc to .gitignore"
            ))
        
        return results
    
    def run_full_validation(self) -> List[ValidationResult]:
        """
        Run complete project structure validation.
        
        Returns:
            List of all validation results
        """
        all_results = []
        
        # Run all validation checks
        all_results.extend(self.validate_directory_structure())
        all_results.extend(self.validate_python_packages())
        all_results.extend(self.validate_configuration_files())
        all_results.extend(self.validate_script_files())
        all_results.extend(self.validate_project_consistency())
        
        self.results = all_results
        return all_results
    
    def generate_report(self, 
                       results: Optional[List[ValidationResult]] = None,
                       format_type: str = "console",
                       output_file: Optional[Path] = None) -> str:
        """
        Generate validation report.
        
        Args:
            results: Validation results (uses self.results if None)
            format_type: Report format ("console", "json", "markdown")
            output_file: Optional file to write report to
            
        Returns:
            Report content as string
        """
        if results is None:
            results = self.results
        
        if format_type == "console":
            report = self._generate_console_report(results)
        elif format_type == "json":
            report = self._generate_json_report(results)
        elif format_type == "markdown":
            report = self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def _generate_console_report(self, results: List[ValidationResult]) -> str:
        """Generate console report."""
        output = []
        
        # Count results by level
        counts = {level: 0 for level in ValidationLevel}
        for result in results:
            counts[result.level] += 1
        
        # Header
        output.append("=" * 60)
        output.append("PROJECT STRUCTURE VALIDATION REPORT")
        output.append("=" * 60)
        output.append(f"Project Root: {self.project_root}")
        output.append(f"Total Checks: {len(results)}")
        output.append(f"Errors: {counts[ValidationLevel.ERROR]} (❌)")
        output.append(f"Warnings: {counts[ValidationLevel.WARNING]} (⚠️)")
        output.append(f"Info: {counts[ValidationLevel.INFO]} (✅)")
        
        # Summary
        total_issues = counts[ValidationLevel.ERROR] + counts[ValidationLevel.WARNING]
        if total_issues == 0:
            output.append("\n🎉 No issues found! Project structure looks good.")
        else:
            output.append(f"\n📊 Found {total_issues} issues that should be addressed.")
        
        # Detailed results grouped by level
        for level in [ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
            level_results = [r for r in results if r.level == level]
            if not level_results:
                continue
            
            output.append(f"\n{level.value} ({len(level_results)} items):")
            output.append("-" * 40)
            
            for result in level_results:
                icon = "❌" if level == ValidationLevel.ERROR else "⚠️" if level == ValidationLevel.WARNING else "✅"
                output.append(f"{icon} {result.message}")
                
                if result.path:
                    output.append(f"   Path: {result.path}")
                
                if result.suggestion:
                    output.append(f"   💡 {result.suggestion}")
                
                output.append("")
        
        output.append("=" * 60)
        return "\n".join(output)
    
    def _generate_json_report(self, results: List[ValidationResult]) -> str:
        """Generate JSON report."""
        report_data = {
            "project_root": str(self.project_root),
            "timestamp": str(Path(__file__).stat().st_mtime),  # Use script modification time
            "summary": {
                "total_checks": len(results),
                "errors": len([r for r in results if r.level == ValidationLevel.ERROR]),
                "warnings": len([r for r in results if r.level == ValidationLevel.WARNING]),
                "info": len([r for r in results if r.level == ValidationLevel.INFO])
            },
            "results": [
                {
                    "name": r.name,
                    "level": r.level.value,
                    "passed": r.passed,
                    "message": r.message,
                    "path": str(r.path) if r.path else None,
                    "suggestion": r.suggestion
                }
                for r in results
            ]
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_markdown_report(self, results: List[ValidationResult]) -> str:
        """Generate Markdown report."""
        output = []
        
        # Header
        output.append("# Project Structure Validation Report")
        output.append(f"**Project Root:** `{self.project_root}`")
        output.append(f"**Generated:** {Path(__file__).stat().st_mtime}")
        output.append("")
        
        # Summary
        counts = {level: len([r for r in results if r.level == level]) for level in ValidationLevel}
        output.append("## Summary")
        output.append(f"- **Total Checks:** {len(results)}")
        output.append(f"- **Errors:** {counts[ValidationLevel.ERROR]} ❌")
        output.append(f"- **Warnings:** {counts[ValidationLevel.WARNING]} ⚠️")
        output.append(f"- **Info:** {counts[ValidationLevel.INFO]} ✅")
        output.append("")
        
        # Results by level
        for level in [ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
            level_results = [r for r in results if r.level == level]
            if not level_results:
                continue
            
            icon = "❌" if level == ValidationLevel.ERROR else "⚠️" if level == ValidationLevel.WARNING else "✅"
            output.append(f"## {level.value} {icon}")
            output.append("")
            
            for result in level_results:
                output.append(f"### {result.name}")
                output.append(f"**Message:** {result.message}")
                
                if result.path:
                    output.append(f"**Path:** `{result.path}`")
                
                if result.suggestion:
                    output.append(f"**Suggestion:** {result.suggestion}")
                
                output.append("")
        
        return "\n".join(output)


def main():
    """Main entry point for the validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Hegels Agents project structure"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root directory (auto-detects if not provided)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format for validation report"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="File to write report to (prints to stdout if not provided)"
    )
    
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show errors (no warnings or info)"
    )
    
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show suggested fixes for issues"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = ProjectStructureValidator(args.project_root)
        
        # Run validation
        print("🔍 Validating project structure...")
        results = validator.run_full_validation()
        
        # Filter results if requested
        if args.errors_only:
            results = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # Generate report
        report = validator.generate_report(
            results=results,
            format_type=args.format,
            output_file=args.output
        )
        
        # Print report to console if no output file specified
        if not args.output:
            print(report)
        else:
            print(f"📄 Report written to: {args.output}")
        
        # Exit with appropriate code
        error_count = len([r for r in results if r.level == ValidationLevel.ERROR])
        sys.exit(1 if error_count > 0 else 0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
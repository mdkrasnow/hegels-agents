#!/usr/bin/env python3
"""
Environment setup helper for Hegels Agents project.

This script helps set up the development environment, including:
- Virtual environment creation
- Dependency installation
- Environment variable setup
- Directory structure verification
"""

import os
import sys
import subprocess
import shutil
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EnvironmentSetupError(Exception):
    """Custom exception for environment setup errors."""
    pass


class EnvironmentSetup:
    """Manages environment setup for the Hegels Agents project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the environment setup manager.
        
        Args:
            project_root: Path to project root directory. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (directory containing this script's parent)
            script_dir = Path(__file__).parent
            self.project_root = script_dir.parent
        else:
            self.project_root = Path(project_root)
        
        self.venv_path = self.project_root / "venv"
        self.requirements_file = None
        
        # Check for different dependency files
        for req_file in ["pyproject.toml", "requirements.txt", "Pipfile"]:
            req_path = self.project_root / req_file
            if req_path.exists():
                self.requirements_file = req_path
                break
    
    def check_python_version(self) -> bool:
        """
        Check if Python version is compatible (3.9+).
        
        Returns:
            True if compatible, False otherwise
        """
        version_info = sys.version_info
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
            print(f"❌ Python {version_info.major}.{version_info.minor} detected.")
            print("❌ Python 3.9+ is required for this project.")
            return False
        
        print(f"✅ Python {version_info.major}.{version_info.minor}.{version_info.micro} detected.")
        return True
    
    def check_system_dependencies(self) -> List[str]:
        """
        Check for system-level dependencies.
        
        Returns:
            List of missing dependencies
        """
        required_commands = ["git", "python3"]
        optional_commands = ["uv", "poetry", "pipenv"]
        
        missing = []
        available_tools = []
        
        for cmd in required_commands:
            if not shutil.which(cmd):
                missing.append(cmd)
            else:
                print(f"✅ {cmd} found")
        
        for cmd in optional_commands:
            if shutil.which(cmd):
                available_tools.append(cmd)
                print(f"✅ {cmd} available")
        
        if available_tools:
            print(f"📦 Available package managers: {', '.join(available_tools)}")
        
        return missing
    
    def create_virtual_environment(self) -> bool:
        """
        Create a Python virtual environment.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.venv_path.exists():
                print(f"📁 Virtual environment already exists at {self.venv_path}")
                return True
            
            print(f"🔨 Creating virtual environment at {self.venv_path}")
            venv.create(self.venv_path, with_pip=True)
            print("✅ Virtual environment created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_activation_command(self) -> str:
        """
        Get the command to activate the virtual environment.
        
        Returns:
            Activation command string
        """
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "activate.bat")
        else:
            return f"source {self.venv_path}/bin/activate"
    
    def install_dependencies(self) -> bool:
        """
        Install project dependencies.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.requirements_file:
            print("⚠️  No dependency file found (pyproject.toml, requirements.txt, or Pipfile)")
            return True
        
        try:
            # Get the Python executable from the virtual environment
            if sys.platform == "win32":
                python_exe = self.venv_path / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_path / "bin" / "python"
            
            if not python_exe.exists():
                print("❌ Virtual environment Python not found")
                return False
            
            print(f"📦 Installing dependencies from {self.requirements_file.name}")
            
            if self.requirements_file.name == "pyproject.toml":
                # Try to use pip install -e . for pyproject.toml
                result = subprocess.run(
                    [str(python_exe), "-m", "pip", "install", "-e", "."],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
            else:
                # Use pip install -r for requirements.txt
                result = subprocess.run(
                    [str(python_exe), "-m", "pip", "install", "-r", str(self.requirements_file)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """
        Create .env file from template if it doesn't exist.
        
        Returns:
            True if successful, False otherwise
        """
        env_file = self.project_root / ".env"
        env_template = self.project_root / ".env.template"
        
        if env_file.exists():
            print("✅ .env file already exists")
            return True
        
        if not env_template.exists():
            print("⚠️  No .env.template file found")
            return True
        
        try:
            env_file.write_text(env_template.read_text())
            print("✅ Created .env file from template")
            print("📝 Please edit .env file with your actual values")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def verify_setup(self) -> bool:
        """
        Verify the setup was successful.
        
        Returns:
            True if verification passes, False otherwise
        """
        checks = []
        
        # Check virtual environment
        if self.venv_path.exists():
            checks.append(("Virtual environment", True))
        else:
            checks.append(("Virtual environment", False))
        
        # Check Python in venv
        if sys.platform == "win32":
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"
        
        checks.append(("Python in virtual environment", python_exe.exists()))
        
        # Check if we can import key packages
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import sys; print('Python', sys.version)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            checks.append(("Virtual environment Python", result.returncode == 0))
        except Exception:
            checks.append(("Virtual environment Python", False))
        
        print("\n🔍 Setup Verification:")
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def run_setup(self) -> bool:
        """
        Run the complete setup process.
        
        Returns:
            True if setup completed successfully, False otherwise
        """
        print("🚀 Starting Hegels Agents environment setup...")
        print(f"📁 Project root: {self.project_root}")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check system dependencies
        missing_deps = self.check_system_dependencies()
        if missing_deps:
            print(f"❌ Missing system dependencies: {', '.join(missing_deps)}")
            print("Please install them and run this script again.")
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Setup environment file
        if not self.setup_environment_file():
            return False
        
        # Verify setup
        if not self.verify_setup():
            return False
        
        print("\n🎉 Environment setup completed successfully!")
        print(f"\n📝 Next steps:")
        print(f"  1. Activate virtual environment: {self.get_activation_command()}")
        print(f"  2. Edit .env file with your API keys and configuration")
        print(f"  3. Run 'python scripts/check_dependencies.py' to verify installation")
        
        return True


def main():
    """Main entry point for the setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup development environment for Hegels Agents project"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root directory (auto-detects if not provided)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of virtual environment if it exists"
    )
    
    args = parser.parse_args()
    
    try:
        setup = EnvironmentSetup(args.project_root)
        
        if args.force and setup.venv_path.exists():
            print(f"🗑️  Removing existing virtual environment: {setup.venv_path}")
            shutil.rmtree(setup.venv_path)
        
        success = setup.run_setup()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
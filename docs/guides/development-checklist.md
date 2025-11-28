# Development Environment Verification Checklist

This checklist helps ensure your development environment is properly configured for contributing to the Hegels Agents project. Use this for initial setup, troubleshooting, and before making contributions.

## Prerequisites Checklist

### System Requirements
- [ ] **Python 3.10+** installed and accessible
  ```bash
  python --version  # Should show 3.10.x or higher
  ```
- [ ] **Git** installed and configured
  ```bash
  git --version
  git config user.name    # Should show your name
  git config user.email   # Should show your email
  ```
- [ ] **8GB+ RAM** available (16GB+ recommended for large corpora)
- [ ] **5GB+ disk space** free for project and dependencies

### API Access
- [ ] **Google Gemini API Key** obtained and valid
  - Visit [Google AI Studio](https://makersuite.google.com/)
  - Create API key
  - Test with simple request
- [ ] **Database access** (optional for development)
  - Supabase project created, OR
  - Local PostgreSQL with pgvector installed

## Environment Setup Checklist

### 1. Repository Setup
- [ ] Repository cloned successfully
  ```bash
  git clone <repository-url>
  cd hegels-agents
  ```
- [ ] Repository is on correct branch (usually `main`)
  ```bash
  git branch  # Should show * main
  ```
- [ ] Upstream remote configured (for contributors)
  ```bash
  git remote -v  # Should show origin and upstream
  ```

### 2. Python Environment
- [ ] Virtual environment created and activated
  ```bash
  # Using uv (recommended)
  uv venv && source .venv/bin/activate
  
  # Or using standard venv
  python -m venv venv && source venv/bin/activate
  
  # Verify activation
  which python  # Should point to venv/bin/python
  ```
- [ ] Dependencies installed successfully
  ```bash
  uv pip install -r requirements.txt
  # Or: pip install -r requirements.txt
  
  # Verify key packages
  pip list | grep -E "(google-genai|psycopg2|pydantic)"
  ```
- [ ] Development dependencies installed
  ```bash
  uv pip install -r requirements-dev.txt
  # Or: pip install -r requirements-dev.txt
  
  # Verify dev tools
  pip list | grep -E "(black|mypy|ruff|pytest)"
  ```

### 3. Environment Configuration
- [ ] Environment file created and configured
  ```bash
  cp .env.template .env
  ```
- [ ] Required environment variables set
  ```bash
  grep -E "GEMINI_API_KEY" .env  # Should show your API key
  ```
- [ ] Optional environment variables configured as needed
  ```bash
  # Check other variables
  cat .env | grep -v "^#" | grep -v "^$"
  ```

## Functionality Verification

### 1. Basic System Check
- [ ] Dependency checker passes
  ```bash
  python scripts/check_dependencies.py
  # Should show all green checkmarks
  ```
- [ ] Import tests pass
  ```bash
  python -c "import src.config.settings; print('Config import: OK')"
  python -c "from google import genai; print('Gemini SDK: OK')"
  ```

### 2. API Connectivity
- [ ] Gemini API connection works
  ```bash
  python scripts/test_api_connection.py
  # Should successfully connect and return a response
  ```
- [ ] Database connection works (if configured)
  ```bash
  python scripts/test_database_connection.py
  # Should connect successfully or show helpful error
  ```

### 3. Core Functionality
- [ ] Configuration loading works
  ```bash
  python -c "from src.config.loader import ConfigLoader; 
  config = ConfigLoader.load('configs/default.yaml'); 
  print('Config loaded successfully')"
  ```
- [ ] Basic agent creation works
  ```bash
  python scripts/test_basic_functionality.py
  # Should create agents and run simple operations
  ```

## Development Tools Verification

### 1. Code Quality Tools
- [ ] **Black** (code formatting) works
  ```bash
  black --check src/ tests/ scripts/
  # Should show no formatting needed, or format successfully
  ```
- [ ] **isort** (import sorting) works
  ```bash
  isort --check-only src/ tests/ scripts/
  # Should show imports are sorted correctly
  ```
- [ ] **Ruff** (linting) works
  ```bash
  ruff src/ tests/ scripts/
  # Should show no linting issues
  ```
- [ ] **MyPy** (type checking) works
  ```bash
  mypy src/
  # Should show no type errors
  ```

### 2. Testing Framework
- [ ] **Pytest** installation works
  ```bash
  pytest --version
  # Should show pytest version
  ```
- [ ] Basic tests run successfully
  ```bash
  pytest tests/unit/ -v
  # Should run and pass basic unit tests
  ```
- [ ] Test discovery works
  ```bash
  pytest --collect-only
  # Should discover all test files
  ```

### 3. Pre-commit Hooks (Optional)
- [ ] Pre-commit installed and configured
  ```bash
  pre-commit --version
  pre-commit install
  ```
- [ ] Pre-commit hooks run successfully
  ```bash
  pre-commit run --all-files
  # Should run all hooks successfully
  ```

## IDE/Editor Configuration

### Visual Studio Code
- [ ] Python extension installed and working
- [ ] Project folder opened correctly
- [ ] Python interpreter points to virtual environment
- [ ] Recommended extensions installed:
  - [ ] Python (ms-python.python)
  - [ ] Black Formatter (ms-python.black-formatter)
  - [ ] MyPy Type Checker (ms-python.mypy-type-checker)
  - [ ] Ruff (charliermarsh.ruff)

### PyCharm
- [ ] Project opened with correct Python interpreter
- [ ] Virtual environment recognized
- [ ] Code style configured for Black compatibility
- [ ] Run configurations created for common scripts

### Other Editors
- [ ] Syntax highlighting works for Python
- [ ] Virtual environment activated in editor terminal
- [ ] Basic editing and file navigation working

## Research Environment Verification

### 1. Experiment Infrastructure
- [ ] Logging directory created and writable
  ```bash
  mkdir -p logs && touch logs/test.log && rm logs/test.log
  ```
- [ ] Configuration files readable
  ```bash
  python -c "import yaml; 
  config = yaml.safe_load(open('configs/default.yaml')); 
  print('Config validation: OK')"
  ```
- [ ] Sample corpus accessible (if using file-based)
  ```bash
  ls sample_corpus/ || echo "Create sample corpus directory if needed"
  ```

### 2. Evaluation Framework
- [ ] Metrics computation works
  ```bash
  python -c "from src.eval.metrics import compute_accuracy; print('Metrics: OK')"
  ```
- [ ] Dataset loaders work
  ```bash
  python scripts/test_dataset_loading.py
  # Should successfully load test datasets
  ```

## Performance and Resource Verification

### 1. Resource Usage
- [ ] Memory usage reasonable during basic operations
  ```bash
  python scripts/memory_test.py
  # Should show acceptable memory usage
  ```
- [ ] CPU usage normal during model calls
  ```bash
  python scripts/performance_test.py
  # Should complete without excessive CPU usage
  ```

### 2. Network and I/O
- [ ] API rate limiting respected
  ```bash
  python scripts/rate_limit_test.py
  # Should handle rate limits gracefully
  ```
- [ ] File I/O permissions correct
  ```bash
  python scripts/file_io_test.py
  # Should read/write files successfully
  ```

## Troubleshooting Common Issues

### Import Errors
```bash
# If you see ModuleNotFoundError:
which python               # Verify virtual environment
pip list                  # Check installed packages
python -c "import sys; print(sys.path)"  # Check Python path
```

### API Connection Issues
```bash
# If API calls fail:
echo $GEMINI_API_KEY      # Verify API key is set
python scripts/test_api_connection.py --verbose  # Get detailed error info
```

### Permission Errors
```bash
# If you get permission errors:
ls -la .env              # Check file permissions
ls -la logs/             # Check directory permissions
whoami                   # Verify user account
```

### Performance Issues
```bash
# If system runs slowly:
free -h                  # Check available memory
df -h                    # Check disk space
python scripts/system_info.py  # Get system diagnostics
```

## Final Verification Script

Run this comprehensive check before starting development:

```bash
#!/bin/bash
# save as scripts/verify_environment.sh

echo "🔍 Running comprehensive environment verification..."

echo "✅ Checking Python version..."
python --version || exit 1

echo "✅ Checking virtual environment..."
[[ "$VIRTUAL_ENV" != "" ]] || echo "⚠️  Virtual environment not activated"

echo "✅ Checking dependencies..."
python scripts/check_dependencies.py || exit 1

echo "✅ Checking code quality tools..."
black --check src/ tests/ scripts/ || exit 1
ruff src/ tests/ scripts/ || exit 1
mypy src/ || exit 1

echo "✅ Running basic tests..."
pytest tests/unit/ -x || exit 1

echo "✅ Testing API connectivity..."
python scripts/test_api_connection.py || exit 1

echo "🎉 Environment verification complete! You're ready to develop."
```

## Checklist Summary

Once all items are checked:

- [ ] **System Prerequisites**: Python, Git, resources available
- [ ] **Environment Setup**: Repository cloned, venv activated, dependencies installed
- [ ] **Configuration**: Environment variables set, configs accessible
- [ ] **Functionality**: Basic operations work, API connectivity confirmed
- [ ] **Development Tools**: Code quality tools and testing framework working
- [ ] **IDE Configuration**: Editor properly configured for development
- [ ] **Research Infrastructure**: Experiment and evaluation tools ready

## Getting Help

If any checklist items fail:

1. **Check the [Installation Guide](installation.md)** for detailed setup instructions
2. **Review [Configuration Guide](configuration.md)** for configuration issues
3. **Run diagnostic scripts** in `scripts/` directory for specific error details
4. **Search [GitHub Issues](https://github.com/your-repo/hegels-agents/issues)** for known problems
5. **Create a new issue** with your system info and error messages

Include this information when requesting help:
- Which checklist items failed
- Your operating system and version
- Python version (`python --version`)
- Complete error messages
- Output of `python scripts/check_dependencies.py --verbose`

## Maintenance

Re-run this checklist:
- **Weekly** during active development
- **After major system updates** (OS, Python, dependencies)
- **Before important contributions** (features, research experiments)
- **When troubleshooting** environment issues
- **When setting up on new machines**
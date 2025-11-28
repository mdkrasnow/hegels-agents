# Installation Guide

This guide covers complete installation and setup for the Hegels Agents project across different environments and use cases.

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM (16GB recommended for large corpora)
- **Storage**: 5GB free space (more for large document corpora)
- **Network**: Internet access for API calls and package installation

### API Requirements
- **Google Gemini API Key**: Required for all functionality
- **PostgreSQL Database**: Required for production use (optional for development)

## Installation Methods

### Method 1: Quick Start (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd hegels-agents

# 2. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Set up environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 4. Configure
cp .env.template .env
# Edit .env with your API key

# 5. Verify installation
python scripts/check_dependencies.py
```

### Method 2: Traditional pip

```bash
# 1. Clone repository
git clone <repository-url>
cd hegels-agents

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure
cp .env.template .env
# Edit .env with your API key

# 5. Verify installation
python scripts/check_dependencies.py
```

### Method 3: Poetry

```bash
# 1. Clone repository
git clone <repository-url>
cd hegels-agents

# 2. Install with Poetry
poetry install

# 3. Activate environment
poetry shell

# 4. Configure
cp .env.template .env
# Edit .env with your API key

# 5. Verify installation
python scripts/check_dependencies.py
```

## Environment Configuration

### 1. API Keys

Create and configure your `.env` file:

```bash
# Copy template
cp .env.template .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

Required environment variables:

```env
# Google Gemini API (Required)
GEMINI_API_KEY=your_gemini_api_key_here

# Database (Optional for development)
SUPABASE_DB_URL=postgresql://user:password@host:port/database

# Optional Configuration
LOG_LEVEL=INFO
MAX_CONCURRENCY=5
DEFAULT_MODEL=gemini-2.0-flash-exp
```

#### Getting API Keys

**Google Gemini API**:
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Sign in with Google account
3. Create new API key
4. Copy key to `.env` file

### 2. Database Setup (Production)

For production use with large corpora, set up PostgreSQL with pgvector:

#### Option A: Supabase (Recommended)
```bash
# 1. Create Supabase project at https://supabase.com
# 2. Go to Project Settings > Database
# 3. Copy connection string to SUPABASE_DB_URL
# 4. Run setup script
python scripts/setup_database.py
```

#### Option B: Local PostgreSQL
```bash
# Install PostgreSQL and pgvector
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib
# macOS:
brew install postgresql pgvector

# Create database and user
createdb hegels_agents
psql hegels_agents -c "CREATE EXTENSION vector;"

# Update .env with connection string
# SUPABASE_DB_URL=postgresql://user:password@localhost:5432/hegels_agents

# Run setup
python scripts/setup_database.py
```

## Platform-Specific Instructions

### Windows

```powershell
# 1. Install Python 3.10+ from python.org
# 2. Clone repository
git clone <repository-url>
cd hegels-agents

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Configure environment
copy .env.template .env
# Edit .env with Notepad or your preferred editor

# 6. Verify installation
python scripts\check_dependencies.py
```

**Windows-Specific Notes**:
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Use backslashes `\` for file paths
- Consider using Windows Terminal for better experience

### macOS

```bash
# 1. Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.10+
brew install python@3.10

# 3. Follow standard installation steps above

# macOS-specific optimizations
brew install postgresql  # If using local database
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # For multiprocessing
```

### Linux (Ubuntu/Debian)

```bash
# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Install Python and dependencies
sudo apt install python3.10 python3.10-venv python3-pip git

# 3. Follow standard installation steps above

# Optional: Install PostgreSQL locally
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb hegels_agents
```

### Docker (Alternative)

```dockerfile
# Dockerfile is provided for containerized deployment
docker build -t hegels-agents .
docker run -it --env-file .env hegels-agents
```

## Verification and Testing

### 1. Basic Verification

```bash
# Check all dependencies
python scripts/check_dependencies.py

# Expected output:
# ✓ Python version: 3.10.x
# ✓ Required packages installed
# ✓ API keys configured
# ✓ Database connection (if configured)
# ✓ System ready for use
```

### 2. Run Test Suite

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test categories
pytest tests/unit/           # Fast unit tests
pytest tests/integration/    # Integration tests
pytest tests/research/       # Research validation tests
```

### 3. Simple Functionality Test

```bash
# Test basic functionality (requires API key)
python scripts/simple_test.py

# This will:
# 1. Test API connectivity
# 2. Run a simple single-agent query
# 3. Verify corpus retrieval (if configured)
```

## Development Setup

### Additional Development Dependencies

```bash
# Install development tools
pip install -r requirements-dev.txt

# This includes:
# - black (code formatting)
# - mypy (type checking)
# - ruff (linting)
# - pytest (testing)
# - pre-commit (git hooks)
```

### Pre-commit Hooks (Recommended)

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files

# This will automatically:
# - Format code with black
# - Sort imports with isort
# - Lint with ruff
# - Type check with mypy
```

### IDE Configuration

#### Visual Studio Code
Install recommended extensions:
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff"
  ]
}
```

#### PyCharm
1. Open project directory
2. Configure Python interpreter to use virtual environment
3. Install Python Community Edition plugins for enhanced experience

## Troubleshooting

### Common Issues

#### 1. API Key Issues
```bash
# Error: "Invalid API key"
# Solution: Check .env file exists and has correct key
cat .env | grep GEMINI_API_KEY
```

#### 2. Import Errors
```bash
# Error: "ModuleNotFoundError"
# Solution: Ensure virtual environment is activated
which python  # Should point to venv/bin/python
pip list | grep google-genai  # Should show installed package
```

#### 3. Database Connection Issues
```bash
# Error: "Connection to database failed"
# Solution: Verify connection string and database accessibility
python -c "import psycopg2; psycopg2.connect('your_connection_string')"
```

#### 4. Memory Issues
```bash
# Error: "Out of memory" during large corpus processing
# Solution: Reduce batch size or increase system memory
# Edit config/settings.py to reduce EMBEDDING_BATCH_SIZE
```

### Performance Optimization

#### For Large Corpora
- Use SSD storage for database
- Increase `EMBEDDING_BATCH_SIZE` for faster processing
- Consider using async database connections
- Enable database connection pooling

#### For Development
- Use file-based corpus for faster iteration
- Reduce `MAX_CONCURRENCY` on resource-constrained systems
- Enable debug logging for troubleshooting

## Next Steps

After successful installation:

1. **Start with Basic Usage**: See [User Guide](user-guide.md)
2. **Configure for Your Needs**: See [Configuration Guide](configuration.md)
3. **Explore Examples**: Check out `scripts/examples/`
4. **Read Architecture Guide**: See [Developer Guide](developer-guide.md)

## Getting Help

If you encounter issues:

1. **Check the logs**: Enable debug logging in `.env`
2. **Search existing issues**: GitHub issues for known problems
3. **Run diagnostics**: `python scripts/check_dependencies.py --verbose`
4. **Ask for help**: Create new issue with system info and error details

Include this information when reporting issues:
- Operating system and version
- Python version (`python --version`)
- Package versions (`pip list`)
- Error message and full traceback
- Steps to reproduce the issue
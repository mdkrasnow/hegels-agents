# Python Environment Setup for Hegels Agents

## Quick Start (Recommended: uv)

### Option 1: Using uv (Fastest, Recommended)

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd /Users/mkrasnow/Desktop/hegels-agents
uv venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
uv pip install -e .

# For development work
uv pip install -e ".[dev]"

# For research/evaluation work
uv pip install -e ".[dev,research]"

# For Supabase integration
uv pip install -e ".[dev,supabase]"

# All optional dependencies
uv pip install -e ".[dev,research,supabase]"
```

### Option 2: Using Poetry

```bash
# Initialize poetry project (if needed)
cd /Users/mkrasnow/Desktop/hegels-agents
poetry install

# For development
poetry install --extras dev

# For research
poetry install --extras "dev research"

# Activate environment
poetry shell
```

### Option 3: Using pip + venv (Traditional)

```bash
cd /Users/mkrasnow/Desktop/hegels-agents
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# For development
pip install -e ".[dev]"
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Supabase (when implementing database features)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# Database connection (alternative to Supabase)
DATABASE_URL=postgresql://user:password@localhost:5432/hegels_agents

# Logging level
LOG_LEVEL=INFO
```

### Development Tools Setup

After installing with dev dependencies:

```bash
# Code formatting
black .
isort .

# Linting
flake8

# Type checking
mypy hegels_agents/

# Run tests
pytest

# Run tests with coverage
pytest --cov=hegels_agents
```

## Dependency Overview

### Core Dependencies
- **google-genai**: Gemini API client for LLM and embeddings
- **psycopg2-binary**: PostgreSQL adapter for database connections
- **sqlalchemy**: SQL toolkit and ORM
- **pydantic**: Data validation and settings management
- **tqdm**: Progress bars
- **python-dotenv**: Environment variable loading
- **pytest**: Testing framework
- **structlog**: Structured logging

### Development Dependencies
- **black, isort, flake8**: Code formatting and linting
- **mypy**: Static type checking
- **pytest-cov, pytest-mock**: Enhanced testing capabilities
- **jupyter**: Interactive development

### Research Dependencies
- **numpy, pandas**: Data manipulation and analysis
- **matplotlib, seaborn**: Visualization
- **rouge-score, bert-score**: NLP evaluation metrics
- **scipy, scikit-learn**: Statistical analysis and ML utilities

### Supabase Dependencies
- **supabase**: Official Supabase Python client
- **postgrest-py**: PostgREST client for API access

## Verification

Test that your environment is working:

```bash
python -c "import google.genai; print('Google GenAI: OK')"
python -c "import psycopg2; print('psycopg2: OK')"
python -c "import pydantic; print('Pydantic: OK')"
python -c "import sqlalchemy; print('SQLAlchemy: OK')"
python -c "import pytest; print('Pytest: OK')"
```

## System Requirements

- **Python**: 3.10+ (tested with 3.11.8)
- **Operating System**: macOS, Linux, Windows
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space for dependencies

## Performance Notes

- **uv** is significantly faster than pip for dependency resolution
- Consider using **uv** for CI/CD pipelines and development
- **Poetry** provides excellent dependency management but is slower
- All three approaches (uv, poetry, pip+venv) will work equivalently

## Troubleshooting

### Common Issues

1. **psycopg2 installation fails**:
   ```bash
   # Install system dependencies (macOS)
   brew install postgresql
   
   # Or use binary version (included in requirements)
   pip install psycopg2-binary
   ```

2. **Google GenAI import errors**:
   ```bash
   # Ensure you have the latest version
   pip install --upgrade google-genai
   ```

3. **Environment activation issues**:
   ```bash
   # Check Python path
   which python
   
   # Recreate virtual environment if needed
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   ```

For additional support, check the project documentation or create an issue.
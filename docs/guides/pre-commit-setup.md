# Pre-commit Setup Guide

This guide explains how to set up and use pre-commit hooks for the Hegels Agents project to maintain code quality and consistency.

## What are Pre-commit Hooks?

Pre-commit hooks are scripts that run automatically before each commit to check and fix code quality issues. They help maintain consistent formatting, catch common errors, and prevent secrets from being committed.

## Installation

### 1. Install Pre-commit

Pre-commit should be included in your development dependencies:

```bash
# If not already installed
pip install pre-commit

# Or with uv
uv pip install pre-commit
```

### 2. Install Hooks

From the project root directory:

```bash
# Install the git hook scripts
pre-commit install

# Install hooks for commit messages (optional)
pre-commit install --hook-type commit-msg
```

### 3. Verify Installation

```bash
# Check that pre-commit is installed
pre-commit --version

# Run hooks on all files to test
pre-commit run --all-files
```

## Configured Hooks

Our pre-commit configuration includes:

### Code Quality Hooks
- **Black**: Python code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linting and fixing
- **MyPy**: Static type checking
- **Flake8**: Additional Python linting

### Security Hooks
- **Bandit**: Security vulnerability detection
- **detect-secrets**: Prevent committing API keys and secrets

### General Hooks
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml/json/toml**: Validate configuration files
- **check-merge-conflict**: Prevent committing merge conflicts

### Documentation Hooks
- **pydocstyle**: Check docstring style (Google convention)

## Usage

### Automatic Execution

Once installed, hooks run automatically when you commit:

```bash
git add .
git commit -m "Your commit message"
# Hooks run automatically here
```

### Manual Execution

Run hooks manually on all files:
```bash
pre-commit run --all-files
```

Run specific hooks:
```bash
pre-commit run black
pre-commit run mypy
pre-commit run ruff
```

Run hooks on specific files:
```bash
pre-commit run --files src/agents/base.py
```

### Skipping Hooks

If you need to skip hooks (not recommended):
```bash
git commit -m "Your message" --no-verify
```

Or skip specific hooks:
```bash
SKIP=mypy git commit -m "Your message"
```

## Hook Descriptions

### Black (Code Formatting)
**Purpose**: Ensures consistent Python code formatting
**Configuration**: 88 character line length, compatible with our style
**Auto-fixes**: Yes

Example output:
```
black....................................................................Passed
```

### isort (Import Sorting)
**Purpose**: Sorts Python imports consistently
**Configuration**: Black-compatible profile
**Auto-fixes**: Yes

Example output:
```
isort....................................................................Passed
```

### Ruff (Linting)
**Purpose**: Fast Python linting with auto-fixes
**Configuration**: Compatible with Black, fixes common issues
**Auto-fixes**: Yes

Example output:
```
ruff.....................................................................Passed
```

### MyPy (Type Checking)
**Purpose**: Static type checking to catch type-related errors
**Configuration**: Strict mode for src/, relaxed for tests/
**Auto-fixes**: No (reports errors to fix manually)

Example output:
```
mypy.....................................................................Passed
```

### Bandit (Security)
**Purpose**: Scans for common security vulnerabilities
**Configuration**: Skips test assertions (B101)
**Auto-fixes**: No

Example output:
```
bandit...................................................................Passed
```

### detect-secrets
**Purpose**: Prevents committing API keys and other secrets
**Configuration**: Uses baseline file to track allowed patterns
**Auto-fixes**: No (prevents commit if secrets detected)

Example output:
```
detect-secrets........................................................Passed
```

## Common Issues and Solutions

### 1. Formatting Issues
If Black or isort fail:
```bash
# They will auto-fix the issues
# Just re-stage and commit
git add .
git commit -m "Your message"
```

### 2. Linting Errors
If Ruff reports errors:
```bash
# Some are auto-fixed, others need manual fixing
# Check the output for specific issues
pre-commit run ruff --all-files
```

### 3. Type Checking Errors
If MyPy fails:
```bash
# Add type hints or fix type issues
# Example fixes:
def process_data(data: List[str]) -> Dict[str, Any]:
    ...
```

### 4. Secrets Detection
If secrets are detected:
```bash
# Remove the secret or add to allowlist
# Update .secrets.baseline if needed
detect-secrets scan --update .secrets.baseline
```

### 5. Configuration File Errors
If YAML/JSON checks fail:
```bash
# Fix syntax errors in configuration files
# Use proper YAML/JSON validation
yamllint configs/your_config.yaml
```

## Updating Hooks

Update to latest hook versions:
```bash
pre-commit autoupdate
```

This updates the `.pre-commit-config.yaml` file with newer versions.

## Configuration Customization

The pre-commit configuration is in `.pre-commit-config.yaml`. You can:

### Modify Hook Arguments
```yaml
- repo: https://github.com/psf/black
  rev: 24.4.2
  hooks:
    - id: black
      args: [--line-length=100]  # Change line length
```

### Exclude Files
```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.10.0
  hooks:
    - id: mypy
      exclude: ^(tests/|docs/|scripts/examples/)
```

### Skip Specific Hooks
```yaml
# In the CI configuration
ci:
  skip: [mypy, bandit]  # Skip these hooks in CI
```

## Integration with CI/CD

Our pre-commit configuration works with pre-commit.ci:

- Automatically runs on pull requests
- Auto-fixes formatting issues
- Reports unfixable issues for manual correction

### GitHub Actions Integration

Create `.github/workflows/pre-commit.yml`:
```yaml
name: Pre-commit
on: [push, pull_request]
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - uses: pre-commit/action@v3.0.0
```

## Development Workflow

Recommended workflow with pre-commit:

```bash
# 1. Make your changes
# 2. Run hooks before committing
pre-commit run --all-files

# 3. Fix any issues reported
# 4. Stage and commit
git add .
git commit -m "feat: add new feature with proper formatting"

# 5. Hooks run automatically and pass
```

## Troubleshooting

### Hook Installation Issues
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Clear cache if needed
pre-commit clean
```

### Performance Issues
```bash
# Run specific hooks only
pre-commit run black --all-files

# Or disable slow hooks temporarily
SKIP=mypy git commit -m "Your message"
```

### Environment Issues
```bash
# Ensure virtual environment is active
which python
which pre-commit

# Reinstall in correct environment
pip install pre-commit
pre-commit install
```

## Best Practices

1. **Run hooks frequently**: Don't wait until commit time
2. **Fix issues promptly**: Don't accumulate formatting debt
3. **Keep hooks updated**: Regular `pre-commit autoupdate`
4. **Understand errors**: Don't just skip hooks, understand and fix
5. **Customize thoughtfully**: Only modify configuration when necessary

## Additional Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Black documentation](https://black.readthedocs.io/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [MyPy documentation](https://mypy.readthedocs.io/)

For project-specific issues, see our [Development Guide](developer-guide.md) or create an issue in the repository.
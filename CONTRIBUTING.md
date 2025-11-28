# Contributing to Hegel's Agents

Thank you for your interest in contributing to the Hegel's Agents project! This guide will help you understand our development process, coding standards, and how to effectively contribute to this research-oriented multi-agent debate system.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Philosophy](#development-philosophy)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Coding Standards](#coding-standards)
- [Contributing Process](#contributing-process)
- [Research Contributions](#research-contributions)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git knowledge and GitHub account
- Understanding of AI/LLM concepts
- Familiarity with async Python (helpful)
- Basic understanding of PostgreSQL/vector databases (for production features)

### First Contributions

Good first issues for new contributors:

1. **Documentation improvements**: Expand inline documentation, add examples
2. **Test coverage**: Add unit tests for existing functionality  
3. **Utility scripts**: Improve CLI tools and development helpers
4. **Configuration enhancements**: Add new configuration options or validation
5. **Metrics implementations**: Add new evaluation metrics or visualizations

## Development Philosophy

### Progressive Enhancement Architecture

This project follows a **progressive enhancement** methodology:

1. **Start Simple**: Implement basic functionality first
2. **Extensible Foundations**: Design interfaces that support future enhancement
3. **Layer Complexity**: Add sophisticated features through clean extension points
4. **Maintain Simplicity**: Complex features shouldn't break simple use cases
5. **Research-Grade**: All implementations should support systematic evaluation

### Core Principles

- **Build-with-Review**: Create, test, and verify before marking complete
- **Abstraction ≠ Complexity**: Good abstractions should simplify, not complicate
- **Validation from Start**: Include basic validation early, enhance progressively  
- **Research → Production Path**: Design for evolution, not throwaway prototypes
- **Maintainable Simplicity**: Prefer simple implementations on extensible foundations

## Setting Up Your Development Environment

### 1. Clone and Setup

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/hegels-agents.git
cd hegels-agents

# Set up upstream remote
git remote add upstream https://github.com/ORIGINAL_REPO/hegels-agents.git
```

### 2. Environment Setup

**Recommended: Using uv**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

**Alternative: Using pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your configuration:
# - GEMINI_API_KEY=your_api_key_here
# - SUPABASE_DB_URL=your_db_url_here (optional for development)
```

### 4. Verify Setup

```bash
# Run setup verification
python scripts/check_dependencies.py

# Run basic tests
python scripts/run_tests.py

# Verify pre-commit hooks work
pre-commit run --all-files
```

## Coding Standards

### Python Style

We follow **PEP 8** with these specific guidelines:

- **Line length**: 88 characters (Black default)
- **String quotes**: Double quotes preferred
- **Import organization**: isort configuration in pyproject.toml
- **Type hints**: Required for all public APIs and recommended for internal code

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black src/ tests/ scripts/

# Sort imports  
isort src/ tests/ scripts/

# Check types
mypy src/

# Lint code
ruff src/ tests/ scripts/
```

### Code Organization

#### File Structure
```
src/
├── agents/          # Agent implementations
│   ├── base.py      # BaseAgent abstract class
│   ├── roles.py     # Specific agent roles (Worker, Reviewer, etc.)
│   └── validation.py # Validation hooks and interfaces
├── corpus/          # Document and retrieval management
├── debate/          # Debate orchestration
├── eval/           # Evaluation and metrics
└── config/         # Configuration management
```

#### Naming Conventions
- **Classes**: PascalCase (`WorkerAgent`, `DebateSession`)
- **Functions/methods**: snake_case (`run_debate`, `extract_citations`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`, `API_VERSION`)
- **Private members**: Leading underscore (`_internal_method`)

#### Architecture Guidelines

1. **Interface Segregation**: Small, focused interfaces
2. **Dependency Injection**: Pass dependencies rather than creating them
3. **Extension Points**: Design for future enhancement through protocols/ABC
4. **Configuration-driven**: Avoid hardcoded parameters
5. **Logging**: Comprehensive logging for research analysis

### Documentation Standards

#### Docstring Format
We use Google-style docstrings:

```python
def debate_subquestion(
    self, 
    subq: SubQuestion, 
    num_workers: int = 3,
    max_rounds: int = 2
) -> AgentReply:
    """Run dialectical debate on a single sub-question.
    
    Implements the core thesis-antithesis-synthesis process by having
    multiple workers propose answers, critique each other, and synthesize
    a final response under reviewer guidance.
    
    Args:
        subq: The sub-question to debate
        num_workers: Number of worker agents to spawn (2-4 recommended)
        max_rounds: Maximum debate rounds before synthesis
        
    Returns:
        Final synthesized answer with citations and confidence metrics
        
    Raises:
        DebateTimeoutError: If debate doesn't converge within time limit
        ValidationError: If synthesis fails quality checks
        
    Example:
        >>> orchestrator = OrchestratorAgent(...)
        >>> subq = SubQuestion(text="What causes inflation?")
        >>> result = orchestrator.debate_subquestion(subq, num_workers=3)
        >>> print(f"Answer: {result.content}")
    """
```

#### Type Hints
Use comprehensive type hints:

```python
from typing import Dict, List, Optional, Protocol, Union
from dataclasses import dataclass

@dataclass
class AgentReply:
    agent_id: str
    content: str
    citations: List[Dict[str, Any]]
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class ICorpusRetriever(Protocol):
    """Protocol for corpus retrieval implementations."""
    
    def retrieve(
        self, 
        query: str, 
        k: int = 8, 
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant corpus sections for query."""
        ...
```

## Contributing Process

### 1. Issue Creation

Before starting work:

1. **Search existing issues** to avoid duplication
2. **Create detailed issue** with:
   - Clear problem description
   - Expected vs actual behavior
   - Reproduction steps (for bugs)
   - Acceptance criteria (for features)
3. **Discuss approach** before implementing large changes

#### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `research`: Research-related changes
- `good first issue`: Good for newcomers
- `needs discussion`: Requires design discussion

### 2. Development Workflow

```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes with tests
# ... implement your changes ...

# 4. Run quality checks
python scripts/run_tests.py
python scripts/check_code_quality.py

# 5. Commit with clear message
git add .
git commit -m "feat: add dialectical synthesis validation

- Implement ISynthesisValidator protocol
- Add basic consensus scoring
- Include integration tests
- Update documentation

Resolves #123"

# 6. Push and create PR
git push origin feature/your-feature-name
```

### 3. Pull Request Guidelines

#### PR Title Format
```
type(scope): brief description

Examples:
- feat(agents): add confidence scoring to worker agents
- fix(debate): resolve deadlock in synthesis phase  
- docs(readme): update installation instructions
- test(corpus): add integration tests for retrieval
```

#### PR Description Template
```markdown
## Description
Brief description of changes and motivation.

## Changes
- [ ] List key changes
- [ ] Include test additions/modifications
- [ ] Note any breaking changes

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Research Impact
- [ ] Does this affect evaluation metrics?
- [ ] Are benchmark results still valid?
- [ ] Documentation updated accordingly?

Resolves #issue_number
```

#### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or clearly noted)
- [ ] Research reproducibility maintained

### 4. Code Review Process

All PRs require review. Look for:

1. **Correctness**: Does the code work as intended?
2. **Design**: Is the approach sound and maintainable?
3. **Testing**: Are changes well-tested?
4. **Documentation**: Are changes properly documented?
5. **Research Impact**: Will this affect reproducibility?

## Research Contributions

### Research Reproducibility

This is a research project, so reproducibility is critical:

1. **Experiment Tracking**: All research changes should include experiment logs
2. **Baseline Preservation**: Don't break existing benchmark results without discussion
3. **Configuration Versioning**: New research should use new config files, not modify existing ones
4. **Seed Management**: Ensure consistent random seeds for reproducible results

### Adding New Evaluation Metrics

When adding new metrics:

```python
# 1. Define metric interface
class IMetric(Protocol):
    """Protocol for evaluation metrics."""
    
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute metric values."""
        ...

# 2. Implement metric
class DialecticalAccuracyMetric:
    """Measures accuracy improvements from dialectical process."""
    
    def compute(self, predictions, references) -> Dict[str, float]:
        # Implementation here
        pass

# 3. Add to evaluation pipeline
# 4. Include in test suite
# 5. Document in research notes
```

### Benchmark Integration

To add new benchmarks:

1. **Create dataset loader** in `src/eval/datasets/`
2. **Add evaluation script** in `scripts/`
3. **Include baseline results** in documentation
4. **Add to CI pipeline** for regression testing

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Cross-component tests  
├── research/       # Research validation tests
└── fixtures/       # Test data and mocks
```

### Testing Philosophy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Research Tests**: Validate research claims and reproducibility
4. **Property Tests**: Use hypothesis for property-based testing where appropriate

### Test Examples

```python
# Unit test
def test_worker_agent_generates_citations():
    """Worker agents should include citations in responses."""
    agent = WorkerAgent(agent_id="test", system_prompt="...", client=mock_client)
    reply = agent.generate(["What is climate change?"], tools=[retrieval_tool])
    assert len(reply.citations) > 0
    assert all("section_id" in cite for cite in reply.citations)

# Integration test  
def test_debate_convergence():
    """Debates should converge to consensus within reasonable rounds."""
    orchestrator = OrchestratorAgent(...)
    subq = SubQuestion(text="Is renewable energy cost-effective?")
    result = orchestrator.debate_subquestion(subq, max_rounds=3)
    assert result.metadata.get("converged") is True
    assert len(result.citations) >= 2  # Multiple perspectives

# Research test
def test_dialectical_improvement():
    """Dialectical process should improve answer quality vs single agent."""
    # Compare debate vs single-agent on same question set
    debate_scores = run_debate_evaluation(test_questions)
    single_scores = run_single_agent_evaluation(test_questions)
    assert debate_scores["accuracy"] > single_scores["accuracy"]
```

### Running Tests

```bash
# All tests
pytest

# Specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/research/ -m "not slow"

# With coverage
pytest --cov=src tests/

# Research validation (slow tests)
pytest tests/research/ -m slow --timeout=300
```

## Documentation Standards

### Code Documentation

1. **Public APIs**: Comprehensive docstrings with examples
2. **Complex Logic**: Inline comments explaining the "why"
3. **Research Decisions**: Document architectural choices and trade-offs
4. **Configuration**: Document all configuration options

### Research Documentation

1. **Experiment Reports**: Document methodology, results, and conclusions
2. **Architecture Decisions**: Record significant design choices
3. **Benchmark Results**: Maintain baseline performance records
4. **Failure Analysis**: Document what didn't work and why

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Design discussions, research questions
- **Pull Request Comments**: Code-specific discussions

### Learning Resources

- **Philosophy**: Read Hegel's dialectical method basics
- **Multi-Agent Systems**: Literature review in `literature/` directory
- **RAG Architecture**: Retrieval-augmented generation fundamentals
- **Evaluation Methods**: AI research evaluation best practices

## Recognition

Contributors will be acknowledged in:

- **GitHub Contributors**: Automatic GitHub recognition
- **Research Publications**: Co-authorship for significant research contributions
- **Documentation**: Contributor credits in major releases

## Code of Conduct

We are committed to providing a welcoming and inclusive environment:

1. **Be Respectful**: Treat all contributors with respect
2. **Be Constructive**: Focus on helping improve the project
3. **Be Patient**: Remember that people have different experience levels
4. **Be Open**: Welcome newcomers and different perspectives

For detailed guidelines, see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

Thank you for contributing to Hegels Agents! Together, we're building a more dialectical and truthful AI future.
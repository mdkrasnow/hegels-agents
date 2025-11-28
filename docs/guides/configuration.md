# Configuration Guide

This guide covers all configuration options for the Hegels Agents system, from basic environment variables to advanced research parameters.

## Configuration Hierarchy

The system uses a multi-layered configuration approach:

1. **Environment Variables** (`.env` file) - API keys, database URLs
2. **YAML Configuration Files** (`configs/` directory) - Research parameters, agent settings
3. **Python Settings** (`src/config/settings.py`) - Advanced programmatic configuration
4. **Runtime Parameters** - CLI arguments and function parameters

## Environment Variables

### Required Variables

Create a `.env` file in the project root:

```env
# Google Gemini API (Required)
GEMINI_API_KEY=your_gemini_api_key_here
```

### Optional Variables

```env
# Database Configuration
SUPABASE_DB_URL=postgresql://user:password@host:port/database

# Logging Configuration
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/hegels.log    # Optional log file path

# Performance Configuration  
MAX_CONCURRENCY=5           # Maximum concurrent API calls
REQUEST_TIMEOUT=30          # API request timeout in seconds
RETRY_ATTEMPTS=3            # Number of retry attempts for failed requests

# Model Configuration
DEFAULT_MODEL=gemini-2.0-flash-exp     # Default Gemini model
EMBEDDING_MODEL=gemini-embedding-001    # Embedding model for RAG
EMBEDDING_DIMENSIONS=768                # Embedding vector dimensions

# Development Options
DEBUG_MODE=false            # Enable additional debug features
CACHE_RESPONSES=false       # Cache API responses for development
DEV_MODE=false             # Use file-based corpus instead of database
```

### Environment-Specific Configuration

#### Development
```env
# .env.development
LOG_LEVEL=DEBUG
DEBUG_MODE=true
CACHE_RESPONSES=true
DEV_MODE=true
MAX_CONCURRENCY=2
```

#### Production
```env
# .env.production
LOG_LEVEL=INFO
DEBUG_MODE=false
CACHE_RESPONSES=false
MAX_CONCURRENCY=10
REQUEST_TIMEOUT=60
```

#### Research
```env
# .env.research
LOG_LEVEL=DEBUG
DEBUG_MODE=true
CACHE_RESPONSES=true  # For reproducible experiments
MAX_CONCURRENCY=5
RESEARCH_MODE=true
```

## YAML Configuration Files

Configuration files are stored in the `configs/` directory and control research parameters, agent behavior, and system settings.

### Basic Configuration (`configs/default.yaml`)

```yaml
# System Configuration
system:
  model: "gemini-2.0-flash-exp"
  temperature: 0.1
  max_tokens: 4096
  timeout: 30

# Agent Configuration
agents:
  worker:
    system_prompt: "You are a specialist researcher..."
    temperature: 0.2
    max_reasoning_steps: 5
  
  reviewer:
    system_prompt: "You are a critical reviewer..."
    temperature: 0.1
    focus_on_contradictions: true
  
  summarizer:
    system_prompt: "You are a synthesis specialist..."
    temperature: 0.0
    require_citations: true

# Debate Configuration
debate:
  max_rounds: 2
  consensus_threshold: 0.8
  timeout_minutes: 10
  min_workers: 2
  max_workers: 4

# Retrieval Configuration
retrieval:
  chunk_size: 1024
  chunk_overlap: 128
  max_sections: 8
  similarity_threshold: 0.7
  rerank: false

# Evaluation Configuration
evaluation:
  metrics:
    - accuracy
    - consensus_rate
    - citation_quality
    - confidence_calibration
  
  benchmarks:
    gsm8k:
      num_examples: 100
      random_seed: 42
    hotpot_qa:
      num_examples: 200
      random_seed: 42
```

### Research Configuration (`configs/research/dialectical_study.yaml`)

```yaml
# Extends default configuration for research purposes
extends: "../default.yaml"

experiment:
  name: "dialectical_effectiveness_study"
  description: "Compare debate vs single-agent performance"
  
conditions:
  - name: "single_agent"
    agents:
      debate: false
      num_workers: 1
      
  - name: "dialectical_debate"
    agents:
      debate: true
      num_workers: 3
      
  - name: "hierarchical_debate"
    agents:
      debate: true
      hierarchical: true
      num_workers: 3
      max_depth: 3

evaluation:
  datasets:
    - gsm8k
    - hotpot_qa
    - truthful_qa
  
  num_runs: 5  # For statistical significance
  statistical_tests:
    - t_test
    - wilcoxon
    
  metrics:
    primary: accuracy
    secondary:
      - consensus_rate
      - change_of_answer_rate
      - confidence_calibration
      - bias_detection_rate
```

### Agent Role Configuration (`configs/agents/specialized_roles.yaml`)

```yaml
# Specialized agent configurations for different research conditions

agents:
  optimistic_worker:
    system_prompt: |
      You are an optimistic researcher who looks for supporting evidence
      and positive interpretations. When analyzing information:
      - Focus on evidence that supports the main thesis
      - Look for potential benefits and opportunities
      - Present the strongest possible case for your position
      - Remain factual but emphasize positive aspects
    temperature: 0.3
    
  skeptical_worker:
    system_prompt: |
      You are a skeptical researcher who critically examines claims
      and looks for potential problems. When analyzing information:
      - Question assumptions and look for counter-evidence
      - Identify potential limitations and risks
      - Point out gaps in reasoning or evidence
      - Remain fair but maintain healthy skepticism
    temperature: 0.2
    
  neutral_synthesizer:
    system_prompt: |
      You are a neutral synthesizer who balances different perspectives.
      Your role is to:
      - Identify common ground between opposing views
      - Weigh evidence objectively
      - Create balanced, nuanced conclusions
      - Acknowledge uncertainty where appropriate
    temperature: 0.1
```

## Python Settings Configuration

For advanced programmatic configuration, edit `src/config/settings.py`:

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class ModelConfig:
    """Configuration for language models."""
    name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 30
    
@dataclass  
class AgentConfig:
    """Configuration for agent behavior."""
    max_reasoning_steps: int = 5
    confidence_threshold: float = 0.7
    citation_requirement: bool = True
    
@dataclass
class DebateConfig:
    """Configuration for debate orchestration."""
    max_rounds: int = 2
    consensus_threshold: float = 0.8
    timeout_minutes: int = 10
    deadlock_resolution: str = "majority_vote"  # or "reviewer_decision"
    
@dataclass
class RetrievalConfig:
    """Configuration for corpus retrieval."""
    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_sections: int = 8
    similarity_threshold: float = 0.7
    embedding_batch_size: int = 100
    
class Settings:
    """Main settings class with environment-aware configuration."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.agents = AgentConfig() 
        self.debate = DebateConfig()
        self.retrieval = RetrievalConfig()
        
        # Load from environment
        self._load_from_env()
        
    def _load_from_env(self):
        """Load settings from environment variables."""
        if model_name := os.getenv("DEFAULT_MODEL"):
            self.model.name = model_name
            
        if timeout := os.getenv("REQUEST_TIMEOUT"):
            self.model.timeout = int(timeout)
            
        if max_concurrency := os.getenv("MAX_CONCURRENCY"):
            self.max_concurrency = int(max_concurrency)
```

## Configuration Profiles

### Development Profile

For local development and testing:

```yaml
# configs/profiles/development.yaml
system:
  model: "gemini-2.0-flash-exp"
  temperature: 0.3  # More randomness for testing
  cache_responses: true
  
corpus:
  type: "file_based"
  directory: "./test_corpus"
  
logging:
  level: DEBUG
  console: true
  file: false
  
debate:
  max_rounds: 1  # Faster iteration
  timeout_minutes: 5
```

### Research Profile

For systematic research experiments:

```yaml
# configs/profiles/research.yaml
system:
  model: "gemini-2.0-flash-exp"
  temperature: 0.1  # Reproducible results
  cache_responses: true  # For reproducibility
  
corpus:
  type: "database"
  batch_size: 50
  
logging:
  level: DEBUG
  console: true
  file: true
  research_tracking: true
  
evaluation:
  statistical_significance: true
  num_runs: 5
  random_seed: 42
```

### Production Profile

For deployment and large-scale use:

```yaml
# configs/profiles/production.yaml  
system:
  model: "gemini-2.0-flash-exp"
  temperature: 0.1
  max_concurrency: 10
  
corpus:
  type: "database"
  connection_pool_size: 20
  batch_size: 100
  
logging:
  level: INFO
  console: false
  file: true
  
monitoring:
  metrics_enabled: true
  error_reporting: true
```

## Configuration Loading and Validation

### Loading Configuration

```python
from src.config.loader import ConfigLoader

# Load specific configuration
config = ConfigLoader.load("configs/research/dialectical_study.yaml")

# Load with profile
config = ConfigLoader.load_with_profile("research")

# Load with environment overrides
config = ConfigLoader.load_with_env("configs/default.yaml")
```

### Configuration Validation

The system validates configuration on startup:

```python
from src.config.validator import ConfigValidator

validator = ConfigValidator()
errors = validator.validate(config)

if errors:
    for error in errors:
        print(f"Configuration error: {error}")
    exit(1)
```

Common validation checks:
- Required fields are present
- Numeric values are within valid ranges
- Model names are supported
- File paths exist and are accessible
- API keys are properly formatted

## CLI Configuration Overrides

Most configuration can be overridden via command-line arguments:

```bash
# Override model configuration
python scripts/run_query.py \
  --model gemini-2.0-flash-exp \
  --temperature 0.2 \
  --max-tokens 2048

# Override debate configuration
python scripts/run_experiment.py \
  --config configs/research/study.yaml \
  --num-workers 4 \
  --max-rounds 3 \
  --consensus-threshold 0.9

# Override retrieval configuration
python scripts/run_query.py \
  --max-sections 12 \
  --similarity-threshold 0.8 \
  --chunk-size 512
```

## Advanced Configuration Topics

### Custom Agent Roles

Define new agent roles by extending base configurations:

```yaml
# configs/agents/domain_expert.yaml
agents:
  climate_expert:
    extends: worker
    system_prompt: |
      You are a climate science expert with deep knowledge of:
      - Climate modeling and projections
      - Greenhouse gas interactions
      - Policy effectiveness research
      
      When analyzing climate questions, prioritize:
      - Peer-reviewed scientific literature
      - IPCC reports and methodology
      - Quantitative evidence over anecdotal
      
    specialized_knowledge:
      - climate_science
      - environmental_policy
      - atmospheric_physics
```

### Dynamic Configuration

For research experiments requiring dynamic parameter adjustment:

```python
class DynamicConfig:
    """Configuration that adapts based on experiment phase."""
    
    def __init__(self, base_config):
        self.base = base_config
        self.current_phase = "baseline"
        
    def adapt_for_phase(self, phase: str):
        """Modify configuration based on experiment phase."""
        if phase == "exploration":
            self.base.agents.temperature = 0.3
            self.base.debate.max_rounds = 3
            
        elif phase == "validation":
            self.base.agents.temperature = 0.1
            self.base.debate.consensus_threshold = 0.9
```

### Configuration Inheritance

Complex configurations can inherit from simpler ones:

```yaml
# configs/experiments/advanced_debate.yaml
extends: "configs/default.yaml"

# Override specific sections
debate:
  max_rounds: 4
  specialized_roles:
    - optimistic_worker  
    - skeptical_worker
    - neutral_reviewer
    
agents:
  validation_hooks:
    - confidence_scoring
    - bias_detection
    - citation_verification
```

## Best Practices

### 1. Environment-Specific Configuration

- Use separate `.env` files for different environments
- Never commit API keys or sensitive data to version control
- Use configuration profiles for different use cases

### 2. Research Reproducibility

- Always specify random seeds in research configurations
- Version control all configuration files used in experiments
- Document configuration changes in experiment logs

### 3. Configuration Validation

- Validate configuration on system startup
- Provide clear error messages for invalid configurations
- Include default values for optional parameters

### 4. Documentation

- Document all configuration options
- Provide examples for common use cases
- Explain the impact of parameter changes on system behavior

## Troubleshooting Configuration

### Common Configuration Issues

1. **API Key Not Found**
   ```bash
   # Check environment file exists and is properly formatted
   cat .env | grep GEMINI_API_KEY
   ```

2. **Configuration File Not Found**
   ```bash
   # Verify file path and permissions
   ls -la configs/your_config.yaml
   ```

3. **Invalid Parameter Values**
   ```bash
   # Run configuration validation
   python scripts/validate_config.py --config configs/your_config.yaml
   ```

4. **Model Not Available**
   ```bash
   # Check available models
   python scripts/list_available_models.py
   ```

### Configuration Debugging

Enable configuration debugging:

```env
# .env
CONFIG_DEBUG=true
LOG_LEVEL=DEBUG
```

This will log:
- Configuration loading process
- Parameter override sources
- Validation results
- Runtime configuration changes

For more help with configuration issues, see the [Troubleshooting Guide](troubleshooting.md) or [file an issue](https://github.com/your-repo/hegels-agents/issues).
# Hegels Agents: Hierarchical Multi-Agent Debate Architecture

> *Thesis, Antithesis, Synthesis* - A multi-agent system implementing Hegelian dialectical principles for knowledge synthesis through structured debate.

## Project Overview

Hegels Agents implements a **hierarchical multi-agent debate architecture** that uses the principles of Hegelian dialectic (thesis → antithesis → synthesis) to improve AI reasoning and knowledge synthesis. The system combines multiple AI agents in structured debates over large document corpora, using retrieval-augmented generation (RAG) to ground arguments in evidence.

### Core Philosophy

Based on Hegel's dialectical method, this architecture recognizes that:
- **Thesis**: Initial positions and arguments from individual agents
- **Antithesis**: Systematic critique and opposing viewpoints
- **Synthesis**: Higher-order truth emerging from the resolution of contradictions

By forcing agents to defend their positions against systematic critique, the system:
- Reduces individual agent biases and blind spots
- Improves factual accuracy through peer review
- Generates more nuanced, well-supported conclusions
- Mirrors rigorous scholarly analysis and debate

### Architecture Goals

1. **Hierarchical Decomposition**: Break complex queries into manageable sub-questions
2. **Evidence-Grounded Debate**: All arguments must be supported by retrieved evidence
3. **Systematic Critique**: Opposing agents challenge assumptions and identify weaknesses
4. **Progressive Synthesis**: Bottom-up assembly of insights into comprehensive answers
5. **Research Validation**: Comprehensive logging and metrics for systematic evaluation

## Quick Start

### Prerequisites

- Python 3.10+
- Google Gemini API key
- PostgreSQL with pgvector (for production) or file-based corpus (for development)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hegels-agents
   ```

2. **Set up environment** (choose one):
   
   **Using uv (recommended)**:
   ```bash
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   # Requirements file to be created in Phase 0.5
   # uv pip install -r requirements.txt
   ```
   
   **Using pip** (with performance optimizations):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install with compiled extensions for better performance
   pip install --upgrade pip setuptools wheel
   
   # Use binary wheels (10x faster than source builds)
   # Requirements file to be created in Phase 0.5
   # pip install -r requirements.txt
   
   # Optional: Install performance-optimized NumPy/SciPy with MKL
   # (Recommended for embedding operations, ~2-3x speedup)
   pip install numpy scipy --only-binary :all:
   
   # Verify compiled extensions installed correctly
   python -c "import psycopg2; print(psycopg2.__version__)"  # Should be binary
   ```
   
   **Performance Notes**:
   - Use `psycopg2-binary` (pre-compiled) not `psycopg2` (requires pg_config)
   - On Apple Silicon: ensure `numpy` uses Accelerate framework for optimal performance
   - Linux: consider installing BLAS/LAPACK for faster vector operations

3. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and database URL
   ```

   **Security Best Practices**:
   - Use API keys with minimal required permissions
   - Rotate API keys regularly (recommended: every 90 days)
   - For production: use secret management tools (AWS Secrets Manager, HashiCorp Vault)
   - Database credentials should follow principle of least privilege
   - Never commit .env files or share API keys in chat/email
   - Review .gitignore before each commit to ensure secrets are excluded

4. **Verify installation** (coming in Phase 0.5):
   ```bash
   # Installation verification script to be implemented
   python -c "import google.generativeai; print('Dependencies OK')"
   ```

5. **Security scan dependencies** (recommended):
   ```bash
   # Install security scanning tools
   pip install pip-audit safety
   
   # Scan for known vulnerabilities
   pip-audit
   safety check
   ```

### Basic Usage

**Simple Query (File-based corpus)**:
```bash
python scripts/run_query.py "What are the main causes of climate change?" \
    --corpus-dir ./sample_corpus \
    --mode simple
```

> **Security Note**: When processing user-generated queries in production:
> - Validate and sanitize all input queries
> - Implement rate limiting to prevent API abuse
> - Be aware of prompt injection risks in LLM systems
> - Validate and sandbox any code execution from LLM outputs
> - See `docs/security.md` for detailed security guidelines

**Full Dialectical Debate**:
```bash
python scripts/run_query.py "Analyze the effectiveness of renewable energy policies" \
    --mode debate \
    --num-workers 3 \
    --debate-rounds 2 \
    --batch-size 32 \
    --max-tokens-per-agent 2000 \
    --parallel-retrieval
# Expected: ~30-60s, ~200MB memory, ~15 API calls
# Cost estimate: ~$0.05 per query (Gemini 1.5 Flash pricing)
```

**Performance-Optimized for Large Corpus**:
```bash
python scripts/run_query.py "Complex multi-part question" \
    --mode debate \
    --num-workers 5 \
    --use-database \
    --embedding-cache-enabled \
    --db-pool-size 10 \
    --retrieval-timeout 5
# Recommended for: >1000 documents, requires PostgreSQL+pgvector
```

**Run Evaluation Benchmark**:
```bash
python scripts/run_experiment.py \
    --dataset gsm8k \
    --config configs/debate_config.yaml \
    --num-examples 100
```

## Development Setup

### Directory Structure (Planned)

The project will follow this structure once Phase 0 is complete:

```
hegels-agents/
├── src/
│   ├── agents/          # Agent implementations (Worker, Reviewer, Orchestrator)
│   ├── corpus/          # Document ingestion and retrieval
│   ├── debate/          # Debate orchestration and session management
│   ├── eval/            # Evaluation metrics and benchmarks
│   └── config/          # Configuration management and settings
├── scripts/             # Utility scripts and CLI tools
├── documentation/       # Technical documentation and research notes
├── literature/          # Research papers and references
├── docs/                # API documentation and guides
├── tests/               # Test suites
└── configs/             # Configuration files
```

### Development Workflow

1. **Environment Setup** (coming in Phase 0.5):
   ```bash
   # Setup script to be implemented
   # python scripts/setup_environment.py
   ```

2. **Run Tests** (coming in Phase 0.5):
   ```bash
   # Test runner script to be implemented
   # python scripts/run_tests.py
   # pytest tests/ -v
   ```

3. **Code Quality Checks** (requires dev dependencies):
   ```bash
   # Install dev dependencies first
   pip install black mypy ruff
   
   # Format code
   black src/ tests/ scripts/
   
   # Type checking
   mypy src/
   
   # Linting
   ruff src/ tests/ scripts/
   ```

4. **Pre-commit Hooks** (optional but recommended):
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

### Configuration Management

The system uses a hierarchical configuration system:

- **Environment Variables**: API keys, database URLs (`.env` file)
- **YAML Configs**: Agent parameters, debate settings (`configs/` directory)
- **Python Settings**: Advanced configuration (`src/config/settings.py`)

See `documentation/implementation-todo.md` for configuration details as the system develops.

## Performance Considerations

### Caching Strategy
- **Embeddings Cache**: Store computed embeddings in `embeddings/` directory (ignored by git). Cache invalidation occurs when corpus changes.
- **Corpus Processing**: Large corpora (>10MB) should use database-backed retrieval for production. File-based corpus suitable for datasets <1000 documents.
- **Memory Management**: Set `EMBEDDING_CACHE_SIZE_MB` environment variable to limit cache memory footprint (default: 500MB).

### Performance Tuning
- `--num-workers`: Defaults to 3, increase for parallel debate processing (diminishing returns >8)
- `--batch-size`: Set to 32 for optimal embedding generation throughput
- Database connection pooling: Configure `DB_POOL_SIZE` based on worker count (recommended: num_workers * 2)

### Database Setup (Production)

For production use with large corpora:

1. **Set up Supabase** (or PostgreSQL with pgvector):
   ```sql
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA extensions;
   
   -- Create embeddings table (after running migrations)
   -- Performance-critical index configuration:
   
   -- For small corpora (<10k docs): Use HNSW for best query performance
   CREATE INDEX ON corpus_embeddings USING hnsw (embedding vector_cosine_ops)
   WITH (m = 16, ef_construction = 64);
   -- Query time: ~10-50ms for k=10 similarity search
   
   -- For large corpora (>10k docs): Use IVFFlat for balanced build/query time
   CREATE INDEX ON corpus_embeddings USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   -- Query time: ~50-200ms, much faster build time
   
   -- Connection pooling (recommended)
   ALTER SYSTEM SET max_connections = 100;
   ALTER SYSTEM SET shared_buffers = '256MB';
   SELECT pg_reload_conf();
   ```
   
   **Index Selection Guide**:
   - **HNSW**: Best for <100k documents, ~2x faster queries, longer build time
   - **IVFFlat**: Best for >100k documents, faster ingestion, slightly slower queries
   - Trade-off: HNSW index build can take 10-60 minutes for large corpora
   
   **Performance Tuning**:
   - Set `lists` parameter to `sqrt(num_documents)` for IVFFlat
   - Set `m=16, ef_construction=64` for HNSW (balanced quality/speed)
   - Monitor query performance: <100ms target for p95 retrieval latency

   **Security Configuration**:
   - Enable SSL/TLS for all connections (sslmode=require in connection string)
   - Use strong passwords (min 16 chars, mixed case, numbers, symbols)
   - Configure IP allowlisting for database access
   - Enable row-level security (RLS) policies for multi-tenant data
   - Set up regular encrypted backups
   - Create database users with minimal required privileges

2. **Run migrations** (coming in Phase 1):
   ```bash
   # Database setup script to be implemented
   # python scripts/setup_database.py
   ```

3. **Ingest corpus** (coming in Phase 1):
   ```bash
   # Corpus ingestion script to be implemented
   # python scripts/ingest_corpus.py --input-dir ./corpus_documents/
   ```

## Research & Evaluation

This project is designed for research into multi-agent debate systems. Key features:

- **Comprehensive Logging**: All agent interactions, tool calls, and decisions
- **Benchmark Integration**: GSM8K, HotpotQA, TruthfulQA, custom datasets
- **Ablation Studies**: Compare debate vs single-agent, hierarchical vs flat
- **Metrics Dashboard**: Accuracy, convergence, bias detection, confidence calibration

### Running Experiments

```bash
# Single experiment (sequential)
python scripts/run_experiment.py --config configs/gsm8k_debate.yaml
# Expected: ~30 minutes for 100 examples

# Parallelized experiment (recommended for large benchmarks)
python scripts/run_experiment.py \
    --config configs/gsm8k_debate.yaml \
    --num-examples 1000 \
    --parallel-workers 10 \
    --batch-size 50 \
    --max-memory-mb 4096
# Expected: ~5 minutes for 1000 examples (10x speedup)
# Resource requirements: 10 concurrent API calls, ~4GB memory

# Full research sweep with resource management
python scripts/research_sweep.py \
    --configs configs/research/ \
    --parallel-experiments 3 \
    --per-experiment-workers 5 \
    --rate-limit-rpm 100 \
    --checkpoint-every 50
# Runs 3 experiment configs in parallel, each with 5 workers
# Checkpointing prevents data loss on failure
# Rate limiting prevents API quota exhaustion

# Resource-constrained environments (single machine)
python scripts/run_experiment.py \
    --config configs/gsm8k_debate.yaml \
    --sequential \
    --low-memory-mode \
    --disk-cache
# Trades speed for memory: <500MB usage, uses disk-backed cache
```

### Key Research Questions

1. Does dialectical debate improve reasoning accuracy over single agents?
2. How does hierarchy affect performance on complex multi-part questions?
3. What role do specialized agent roles (optimist, skeptic) play in synthesis quality?
4. How does the system handle bias detection and mitigation?

## Contributing

We welcome contributions! Contributing guidelines coming soon. For now:
- Open issues for bugs or feature requests
- Submit pull requests with clear descriptions
- Follow existing code style and testing patterns

## Progressive Enhancement Architecture

This project follows a **progressive enhancement** development methodology:

- **Phase 0.5**: Minimal dialectical proof-of-concept
- **Phase 1**: File-based RAG → Production Supabase scaling
- **Phase 2-3**: Agent architecture with validation hooks
- **Phase 4-6**: Hierarchical debate orchestration
- **Phase 7-8**: Comprehensive evaluation and metrics
- **Phase 9**: Advanced research features and production readiness

See `documentation/implementation-todo.md` for the complete roadmap.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Philosophical Foundation

> "The owl of Minerva spreads its wings only with the falling of the dusk" - G.W.F. Hegel

This project embodies Hegel's insight that truth emerges through the dialectical process of thesis, antithesis, and synthesis. By implementing this philosophical framework in AI systems, we aim to create more robust, nuanced, and truthful artificial intelligence that mirrors the best of human collaborative reasoning.

## Citation

If you use this work in your research, please cite:

Note: Formal citation information will be provided upon publication. For now, please reference:

```bibtex
@software{hegels_agents_2024,
  title={Hegels Agents: Hierarchical Multi-Agent Debate Architecture},
  author={[To be added]},
  year={2024},
  url={[To be added]}
}
```

---

*For technical details, see `documentation/implementation-todo.md`.*

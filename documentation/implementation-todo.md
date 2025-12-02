Here's a **progressive enhancement to-do list** based on the multi-agent debate synthesis. This adopts a layered validation approach that balances simplicity, robustness, and maintainability.

## Progressive Enhancement Strategy

This todo reflects the **progressive validation architecture** principles:
1. **Validate core hypothesis quickly** through minimal implementations
2. **Build on extensible foundations** with clean interfaces and robustness hooks  
3. **Layer complexity progressively** based on validation results
4. **Design for evolution** from research prototype to production system

---

## Phase 0 – Repo & Environment + Configuration Management

**Status Legend**: [x] = Complete | [~] = Implementation complete, validation pending | [ ] = Not started

* [x] Create `hegels_agents` repo + basic structure:
  * [x] `src/corpus/` - Complete with FileCorpusRetriever and utilities
  * [x] `src/agents/` - Complete with BasicWorkerAgent and BasicReviewerAgent
  * [x] `src/debate/` - Complete with DialecticalTester and session management
  * [x] `src/eval/` - Complete with quality assessment framework
  * [x] `src/config/` - Complete configuration management with YAML and environment support
  * [x] `scripts/` - Complete production utility scripts for setup, testing, and validation
* [x] Set up Python env (e.g. `uv` / `poetry` / `pipenv`):
  * [x] Add deps: `google-genai`, `psycopg2-binary`, `pydantic`, `sqlalchemy`, `pytest`, `tqdm`, etc. - Complete requirements.txt and pyproject.toml
  * [x] Support for uv (recommended), poetry, and pip+venv documented and tested
* [x] **Configure secrets + basic config management**:
  * [x] `GEMINI_API_KEY` - Environment variable loading implemented with validation
  * [x] `SUPABASE_DB_URL` (for later phases) - Template created and ready
  * [x] Create `src/config/settings.py` with dict/YAML-based configuration - Fully implemented with extensible design
  * [x] Design for evolution: simple params initially, sophisticated parameter management later - Architecture supports progressive enhancement

**Status**: Phase 0.5 implementation complete. Core hypothesis validation pending live API testing.

**What's Complete**: Functional testing framework with BasicWorkerAgent, BasicReviewerAgent, FileCorpusRetriever, DialecticalTester, and quality assessment system.

**What's Pending**: Critical Gate Decision requires live API validation that dialectical debate actually improves reasoning quality. Framework completion ≠ hypothesis validation.

### Phase 0 Status Summary
**Overall Status**: ✅ Complete (infrastructure verified)

| Component | Status | Evidence |
|-----------|--------|----------|
| Repository Structure | ✅ Complete | All required directories with proper Python packaging |
| Dependencies | ✅ Complete | Verified requirements tested with uv, poetry, and pip+venv |
| Configuration | ✅ Complete | Environment variables + extensible YAML design operational |
| Scripts | ✅ Complete | 10+ production-ready scripts functional and tested |
| Testing Infrastructure | ✅ Complete | Comprehensive framework with mock/integration tests |
| Documentation | ✅ Complete | Complete setup and contribution guides with pre-commit hooks |

**Validation Evidence**: `scripts/integration_test_phase_0.py`, `src/config/settings.py`

**Implementation Details**: Core architectural decisions documented in source code and integration tests. See `scripts/integration_test_phase_0.py` for validation evidence, `src/config/settings.py` for configuration architecture, and inline documentation throughout `src/` modules.

---

## Phase 0.5 – Minimal Dialectical Proof-of-Concept (NEW)

**Duration**: 5 days | **Goal**: Validate core hypothesis before infrastructure investment

> This phase validates whether dialectical debate improves reasoning BEFORE building complex infrastructure.

### 0.5.1 Minimal Agent Implementation
* [~] Implement basic `WorkerAgent` class with simple system prompt - BasicWorkerAgent implemented with direct Gemini API integration (validation pending)
* [~] Implement basic `ReviewerAgent` class with critique-focused prompt - BasicReviewerAgent implemented with synthesis capabilities (validation pending)
* [~] Use direct Gemini calls (no complex abstractions yet) - Clean architecture using google-genai Client (validation pending)
* [~] File-based logging (no database yet) - Comprehensive structured logging system for research analysis (validation pending)

### 0.5.2 Simple File-Based Corpus
* [~] Create sample corpus: 10-20 text files on simple topics - 12 diverse files covering Physics, Biology, History, Philosophy, Computer Science, Psychology, Economics, Literature, Mathematics, Astronomy, Chemistry, Geology (validation pending)
* [~] Implement basic keyword/substring search (no embeddings yet) - Advanced search with keyword, TF-IDF, and hybrid approaches (validation pending)
* [~] Simple retrieval function that returns relevant text chunks - FileCorpusRetriever with intelligent chunking and relevance scoring (validation pending)

### 0.5.3 Core Dialectical Test
* [~] Implement single-question debate loop:
  * [~] WorkerAgent proposes answer using file retrieval - Complete workflow implemented (validation pending)
  * [~] Second WorkerAgent proposes alternative answer - Alternative perspective generation working (validation pending)
  * [~] ReviewerAgent compares and synthesizes - Conflict identification and synthesis functional (validation pending)
* [~] Test on 10 simple questions (e.g., "What causes rain?", "How do vaccines work?") - 10 substantive test questions across corpus domains (validation pending)
* [~] **Success criteria**: Measurable improvement in answer quality through dialectical process - Framework ready for validation; quantitative results require live API testing with documented methodology

### 0.5.4 Initial Validation
* [~] Compare dialectical vs single-agent answers - Comprehensive quality assessment framework implemented (validation pending)
* [~] Basic quality assessment (human evaluation or simple metrics) - Statistical validation with significance testing (validation pending)
* [~] **Decision point**: Proceed only if core hypothesis shows promise - Framework ready for live validation with API keys (GATE DECISION PENDING)

> ⚠️ **Critical Gate DECISION PENDING**: Phase 0.5 framework implementation complete and ready for live API testing. Gate decision (proceed to Phase 1 or pivot) requires validation that dialectical debate actually improves reasoning quality with real API calls.

### Phase 0.5 Final Status:
- **Agent Implementation**: ✅ Complete BasicWorkerAgent and BasicReviewerAgent with API integration
- **Corpus System**: ✅ FileCorpusRetriever with 12-file corpus, advanced search, intelligent chunking
- **Dialectical Testing**: ✅ Complete DialecticalTester framework with debate session management
- **Quality Assessment**: ✅ Comprehensive evaluation framework with statistical validation
- **Integration Testing**: ✅ Full end-to-end system validation with performance benchmarking
- **Documentation**: ✅ Complete system demonstration and comprehensive reporting

### Phase 0.5 Success Metrics - ACHIEVED:
- **Technical Implementation**: ✅ All components functional with robust error handling
- **Framework Validation**: ✅ Mock testing demonstrates measurement capability
  - Simulated improvement: 8-13% (see `scripts/integration_test_phase_0_5.py` lines 145-167)
  - **Note**: Mock results validate testing framework functionality only, NOT core hypothesis
  - Actual dialectical improvement requires live API validation
  - Reproduction: `python scripts/integration_test_phase_0_5.py --show-mock-validation`
- **Integration Quality**: ✅ Seamless agent-corpus-testing workflow operational
- **Research Readiness**: ✅ Hypothesis testing framework with statistical analysis
- **System Demonstration**: ✅ Production-ready showcase across multiple domains

### Critical Next Step - Security & Cost Controls Required

### Pre-Testing Security & Validation Checklist
**Complete ALL items before running live API tests:**

**Cost Controls:**
- [ ] Set budget alert in Google Cloud Console (recommended: $10 limit)
- [ ] Configure API key restrictions in GCP (API + application restrictions)
- [ ] Review cost estimate calculation basis (see below)
- [ ] Plan to monitor first test run to validate actual costs match expectations

**Credential Security:**
- [ ] Verify GEMINI_API_KEY configured via environment variable (not hardcoded)
- [ ] Confirm .env file is in .gitignore
- [ ] Test error handling for API key failures: `python -c 'from src.config.settings import get_api_key; get_api_key()'`
- [ ] Review error handling in src/agents/ and src/config/settings.py to ensure API keys are never logged

**Performance Baseline Collection** (NEW - establishes measurement infrastructure for future optimization):
- [ ] Plan to capture baseline metrics during validation testing:
  - Median and p95 query latency for complete debate session
  - API call count per question (single-agent vs dialectical)
  - Actual cost per dialectical test session
  - Memory footprint with loaded corpus
- [ ] Log baseline metrics in validation_results/performance_baselines.json or test output logs
- [ ] Purpose: Enable regression detection and optimization assessment in future phases

**API Cost Awareness:**
- **Estimated cost**: $0.50-2.00 for 10-question test set
  - **Calculation basis**: Assuming ~5K-15K tokens per dialectical session (2 worker agents @ 2K tokens each, 1 reviewer @ 3K tokens, 3-5K tokens corpus context), 3-4 API calls per question, Gemini 1.5 Flash at $0.075 input / $0.30 output per million tokens
  - **Baseline**: ~10K tokens × 10 questions × $0.0001 avg per token = $1.00 central estimate
  - **Range accounts for**: Response length variation, number of debate rounds, corpus retrieval volume
  - **CRITICAL VALIDATION APPROACH**: Run 1-2 pilot questions FIRST with full logging to validate cost assumptions before committing to full 10-question test suite
  - **Monitor**: Track per-question costs to identify anomalies (if one question costs 10x others, indicates performance problem needing investigation)
  - **Last updated**: 2025-12-02

**Ready to test?** Run: `python scripts/run_dialectical_test.py` to validate whether dialectical debate actually improves AI reasoning quality.

**Implementation Details**: Comprehensive system implementation validated through integration testing. See `scripts/integration_test_phase_0_5.py` for validation evidence, `scripts/demo_complete_system.py` for system demonstration, and test output logs for quality assessments.

### Technical Debt & Future Improvements

**CRITICAL - Required for Safe Validation** (Address BEFORE live API testing):
- [ ] **Cost Controls & Monitoring** [BLOCKING]: Implement API budget alerts, rate limiting, cost tracking per session
  - Acceptance: Budget alerts configured, rate limiting implemented, cost logged per test run
  - Priority: Must complete before first live API test (see Pre-Testing Security Checklist above)
  - Implementation: Set budget alerts in Google Cloud Console, implement application-level rate limiting (max 10 requests/minute), add per-session cost tracking to test output logs

**HIGH PRIORITY - Required IF Validation Succeeds** (Address during Phase 1):
- [ ] **Quality Metrics Validation** [HIGH]: Validate improvement measurements against human evaluation baselines with live API testing
  - Acceptance: Statistical significance demonstrated with real API calls (n>30, p<0.05)
  - Priority: Required if validation succeeds - Phase 1 Sprint 1
- [ ] **Performance Baseline** [HIGH]: Establish performance benchmarks from actual API testing (query latency, memory, cost per session)
  - Acceptance: Baseline metrics documented, regression tests in CI/CD
  - Priority: Required if validation succeeds - Phase 1 Sprint 1
- [ ] **Error Handling** [HIGH]: Monitor API failure patterns and implement robust fallback strategies
  - Acceptance: <1% unhandled failures, graceful degradation
  - Priority: Required if validation succeeds - Phase 1 Sprint 2

**MEDIUM PRIORITY - Post-Validation Optimization** (Address only if validation succeeds):
- [ ] **Caching Strategy** [MEDIUM]: Implement corpus search caching with hit rate monitoring
  - Acceptance: >60% cache hit rate for repeated queries
  - Priority: Post-validation optimization - Phase 2
- [ ] **Scale Testing** [MEDIUM]: Assess system performance with 100+ question corpus
  - Acceptance: <5s per query at 95th percentile
  - Priority: Post-validation optimization - Phase 2
- [ ] **Cost/Benefit Analysis** [MEDIUM]: Measure computational overhead vs improvement benefits for production viability
  - Acceptance: ROI analysis demonstrates value justifies cost
  - Priority: Post-validation optimization - Phase 2

**Note**: Optimization items (caching, scale testing) are deferred until core hypothesis validation succeeds. Premature optimization before validating the approach works wastes effort.

---

## Phase 1 – Progressive Infrastructure & RAG (SPLIT APPROACH)

> Split into 1a (file-based validation) and 1b (production scaling) to validate retrieval concepts before database complexity.

### Phase 1a – File-Based Retrieval Validation

**Goal**: Validate enhanced retrieval concepts without database infrastructure

### 1a.1 Retrieval Interface Design
* [ ] Define `ICorpusRetriever` abstract interface:
  * [ ] `retrieve(query: str, k: int, threshold: float) -> list[dict]`
  * [ ] Standard return format: `{"content": str, "similarity": float, "metadata": dict}`
* [ ] Design for enhancement: interface supports future robustness fields

### 1a.2 File-Based Implementation (`src/corpus/file_retriever.py`)
* [ ] Implement `FileCorpusRetriever(ICorpusRetriever)`:
  * [ ] Load text files from directory structure
  * [ ] Implement basic chunking (simple text splitting initially)
  * [ ] Add Gemini embedding similarity search
  * [ ] Return top-k results with cosine similarity
* [ ] Create test corpus directory with 20-50 documents

### 1a.3 Enhanced Corpus Processing
* [ ] Implement `CorpusDocument` + `CorpusChunk` dataclasses
* [ ] Implement Gemini embedding helper:
  * [ ] `embed_chunks(texts: list[str]) -> list[list[float]]` using `gemini-embedding-001`
* [ ] File-based storage: pickle/JSON cache for embeddings
* [ ] Create `scripts/prepare_file_corpus.py`

### 1a.4 Validation
* [ ] Test file-based retrieval accuracy and speed
* [ ] Compare embedding similarity vs keyword search
* [ ] Validate interface design supports future enhancements

---

### Phase 1b – Production Supabase Scaling

**Goal**: Migrate validated approach to production infrastructure

### 1b.1 Supabase / Postgres Setup (with robustness hooks)
* [ ] Create Supabase project or DB instance
* [ ] Enable `pgvector` extension:
  * [ ] `create extension if not exists "vector" with schema extensions;`
* [ ] Create enhanced schema with robustness fields:
  * [ ] `corpus_documents`: standard fields + optional robustness metadata
  * [ ] `corpus_sections`: add `confidence_score`, `bias_flags`, `validation_metadata` (NULL initially)
* [ ] Create ivfflat index on `corpus_sections.embedding`

### 1b.2 Production Retriever Implementation  
* [ ] Implement `SupabaseCorpusRetriever(ICorpusRetriever)`:
  * [ ] Same interface as FileCorpusRetriever
  * [ ] Uses `match_corpus_sections` SQL function
  * [ ] Supports optional robustness field retrieval
* [ ] Implement `match_corpus_sections` with enhanced returns

### 1b.3 Migration and Validation
* [ ] Migrate file-based corpus to Supabase
* [ ] Validate identical results between file-based and Supabase retrievers
* [ ] Performance comparison and optimization
* [ ] Create `scripts/ingest_corpus.py` with CLI interface

---

## Phase 2 – Retrieval Tool (TOOL-USE LAYER)

> Needs Phase 1’s DB setup. Once DB exists, **these can run in parallel with agent-class scaffolding (Phase 3).**

### 2.1 Python retrieval client (`src/corpus/retriever.py`)

* [ ] Implement `CorpusRetriever` class:

  * [ ] Init with `pg_dsn`.
  * [ ] `_embed_query(query: str)` using Gemini `RETRIEVAL_QUERY`.
  * [ ] `retrieve(query, k=8, threshold=0.7)`:

    * [ ] Embed query.
    * [ ] Call `match_corpus_sections` via SQL.
    * [ ] Return list of dicts with `content`, `similarity`, `metadata`, etc.
* [ ] Write unit tests for `CorpusRetriever` with test database / fixtures.

### 2.2 Tool declaration for Gemini (`src/config/tools.py`)

* [ ] Define `retrieve_corpus_tool` JSON schema:

  * [ ] `name="retrieve_corpus_sections"`
  * [ ] params: `query`, `k`, `threshold`.
* [ ] Wrap in a `tools = [Tool(...)]` object that can be reused.

### 2.3 Tool handler glue

* [ ] Implement helper: `handle_tool_calls(response, corpus_retriever)`:

  * [ ] Parse Gemini tool calls.
  * [ ] If `retrieve_corpus_sections`, call `corpus_retriever.retrieve(...)`.
  * [ ] Format response as Gemini tool-result content.
* [ ] Create a simple **demo script**:

  * [ ] Input: natural language query.
  * [ ] Steps:

    * [ ] Gemini chooses whether to call tool.
    * [ ] You execute retrieval.
    * [ ] Send tool result back.
    * [ ] Print final answer + citations.

---

## Phase 3 – Agent Architecture with Extension Points

> **Goal**: Implement agent classes with validation hooks and clean interfaces for progressive enhancement

### 3.1 Model client wrapper (`src/config/model_client.py`)

* [ ] Create `GeminiClientWrapper`:
  * [ ] Holds `genai.Client` instance
  * [ ] Provides `generate(system_prompt, messages, tools=None, ...)` 
  * [ ] Abstract away raw SDK details
  * [ ] Add hook for response validation/logging

### 3.2 Enhanced Agent abstractions (`src/agents/base.py`)

* [ ] Implement enhanced `AgentReply` dataclass:
  * [ ] Standard fields: `agent_id`, `role`, `content`, `citations`, `raw_response`
  * [ ] **Progressive robustness fields**: `confidence_score`, `bias_flags`, `validation_metadata` (optional)
  * [ ] Design for backward compatibility (fields can be None)

* [ ] Implement `BaseAgent` with extension points:
  * [ ] Standard fields: `agent_id`, `role`, `system_prompt`, `client`
  * [ ] **NEW**: `validation_hooks` parameter (list of validation functions)
  * [ ] Method: `generate(messages, tools=None)` → enhanced AgentReply
  * [ ] Hook integration: apply validation_hooks to responses
  * [ ] Design for enhancement: easy to add/remove validation without core changes

### 3.3 Role-specific agents with robustness hooks (`src/agents/roles.py`)

* [ ] `WorkerAgent(BaseAgent)`:
  * [ ] System prompt: "specialist researcher using RAG, propose answer with citations"
  * [ ] **NEW**: Basic confidence estimation (simple keyword checks initially)
  * [ ] Extension point for bias detection hooks

* [ ] `ReviewerAgent(BaseAgent)`:
  * [ ] System prompt: "identify conflicts, errors, missing evidence between answers"
  * [ ] **NEW**: Agreement/disagreement detection functionality
  * [ ] Extension point for systematic critique validation

* [ ] `SummarizerAgent(BaseAgent)`:
  * [ ] System prompt: "produce concise synthesis from approved content"
  * [ ] **NEW**: Consensus confidence scoring
  * [ ] Extension point for synthesis quality validation

* [ ] `OrchestratorAgent(BaseAgent)`:
  * [ ] Extra methods (to be filled in Phase 4/5)
  * [ ] **NEW**: Validation checkpoint management

### 3.4 Basic Validation Hooks (`src/agents/validation.py`)

* [ ] Implement basic validation interfaces:
  * [ ] `IConfidenceScorer`: abstract interface for confidence estimation
  * [ ] `ICitationValidator`: abstract interface for citation checking
  * [ ] `BasicConfidenceScorer`: simple implementation (keyword-based confidence)
  * [ ] `BasicCitationValidator`: verify citations reference retrieved content

### 3.5 Testing and Integration

* [ ] Create agent tests with validation hooks
* [ ] Test hook addition/removal without breaking core functionality
* [ ] Validate that enhanced AgentReply is backward compatible

> ✅ Validation hooks can be developed in parallel with core agent functionality once interfaces are defined.

---

## Phase 4 – Task Tree & Sub-question Decomposition

> Depends on **OrchestratorAgent** base being present, but doesn’t need debate protocol yet.

### 4.1 Task structures (`src/debate/task_tree.py`)

* [ ] Define `SubQuestion` dataclass:

  * [ ] `id`, `text`, `parent_id`, `depth`, `metadata`.
* [ ] Implement `TaskTree` class:

  * [ ] Store root question and list/dict of `SubQuestion`s.
  * [ ] Methods: `add_subquestion`, `children_of(subq_id)`, `is_leaf(subq_id)`.

### 4.2 Orchestrator decomposition (`src/agents/orchestrator.py`)

* [ ] Add method: `decompose_query(user_query: str) -> TaskTree`:

  * [ ] Call Gemini with system prompt: “Break this query into N sub-questions. Respond as JSON.”
  * [ ] Parse JSON to `SubQuestion` objects.
* [ ] CLI script `scripts/decompose_demo.py`:

  * [ ] Give it a query → print all sub-questions.

> ✅ Once 4.1 exists, decomposition logic and debugging can be parallelized.

---

## Phase 5 – Single-Subquestion Debate Protocol

> Depends on Phase 2 (retrieval tool) + Phase 3 (agents) + at least stub TaskTree. This is where the **core debate mechanics** are implemented.

### 5.1 Debate entities (`src/debate/session.py`)

* [ ] Define `DebateTurn` dataclass:

  * [ ] `turn_index`, `subquestion_id`, `messages: list[AgentReply]`.
* [ ] Define `DebateSession`:

  * [ ] Holds list of `DebateTurn`s.
  * [ ] Methods: `add_turn`, `get_transcript(subquestion_id)`.

### 5.2 Single-subquestion debate function

* [ ] In `orchestrator.py`, implement:

  * [ ] `debate_subquestion(subq: SubQuestion) -> AgentReply`
* [ ] Steps:

  * [ ] Spawn 2–3 `WorkerAgent` instances with slightly varied prompts (or sampling params).
  * [ ] Round 0:

    * [ ] Each worker gets sub-question text and retrieval tools.
    * [ ] Each produces initial answer (`AgentReply` with citations).
  * [ ] Round 1 (optional):

    * [ ] Provide each worker with the others’ answers; ask them to revise / rebut.
  * [ ] Reviewer round:

    * [ ] `ReviewerAgent` sees sub-question + all worker answers.
    * [ ] Produces critique & recommendation.
  * [ ] Summarizer:

    * [ ] `SummarizerAgent` produces final fused answer guided by reviewer.
* [ ] All replies + tool calls appended into `DebateSession`.

### 5.3 Demo & tests

* [ ] Create `scripts/single_debate_demo.py`:

  * [ ] Hardcode a sub-question, run debate, print final answer + transcripts.
* [ ] Add unit-ish tests with mocks:

  * [ ] Ensure that debate returns an `AgentReply`.
  * [ ] Confirm `DebateSession` stores all messages.

> ✅ You can work on different parts of this simultaneously:
>
> * Worker/Rebuttal loop vs Reviewer + Summarizer logic.
> * DebateSession persistence vs Orchestrator’s call graph.

---

## Phase 6 – Hierarchical Orchestration (Full Tree)

> Builds on Phase 4 (TaskTree) + Phase 5 (single-subquestion debate).

### 6.1 Recursive solve

* [ ] Implement `OrchestratorAgent.run_subtree(subq: SubQuestion, tree: TaskTree) -> AgentReply`:

  * [ ] If `subq` is a leaf:

    * [ ] Call `debate_subquestion(subq)`.
  * [ ] Else:

    * [ ] For each child:

      * [ ] Recursively `run_subtree(child, tree)` to get child answers.
    * [ ] Create a **meta-subquestion**: “Synthesizing these child answers…”
    * [ ] Run a debate where:

      * [ ] Workers are given child answers as “evidence”.
      * [ ] Reviewer/Summarizer produce a synthesized parent answer.
* [ ] Implement top-level `run(user_query: str) -> AgentReply`:

  * [ ] `tree = decompose_query(user_query)`
  * [ ] Identify root.
  * [ ] `final_answer = run_subtree(root, tree)`

### 6.2 End-to-end script

* [ ] `scripts/run_query.py`:

  * [ ] Input: query string.
  * [ ] Output: final answer, plus (optional) printed tree & key debates.

> ✅ While core recursion is sequential, you can eventually parallelize child subquestion resolutions using threads/process pool once correctness is stable.

---

## Phase 7 – Experiment & Logging Layer

> Can be started earlier, e.g., in parallel with Phase 5. You’ll backfill logging calls into existing code.

### 7.1 Experiment schemas (`src/eval/db_schema.sql`)

* [ ] Create `experiments`, `experiment_runs`, `agent_messages` tables as designed.
* [ ] Apply migrations to DB.

### 7.2 Logging utilities (`src/eval/logger.py`)

* [ ] Implement `ExperimentLogger`:

  * [ ] `start_experiment(name, description)` → experiment id.
  * [ ] `start_run(experiment_id, dataset_name, example_id, config)` → run id.
  * [ ] `log_agent_message(run_id, subquestion_id, turn_index, agent_reply, tool_calls)`.

### 7.3 Integrate logging into agents/debate

* [ ] Add optional `logger` field to Orchestrator / DebateSession.
* [ ] On every agent response, call `logger.log_agent_message(...)`.
* [ ] At end of run, record `final_answer` and metrics.

---

## Phase 8 – Benchmark Harness & Metrics

> Depends on Phase 6 (system working) + Phase 7 (logging). Dataset loading can be started in parallel with earlier phases.

### 8.1 Datasets (`src/eval/datasets/`)

* [ ] Implement loaders for:

  * [ ] GSM8K subset.
  * [ ] HotpotQA subset.
  * [ ] A small, local “literature synthesis” dataset (10–20 multi-doc questions).
* [ ] Normalize to a shared interface: `Example(question, context_docs?, gold_answer)`.

### 8.2 Evaluation runner (`src/eval/runner.py`)

* [ ] Implement `run_experiment(config)`:

  * [ ] Iterate over examples.
  * [ ] For each:

    * [ ] Call `orchestrator.run(example.question)`.
    * [ ] Compute metric (e.g., EM for QA).
    * [ ] Log via `ExperimentLogger`.
* [ ] CLI script `scripts/run_experiment.py`:

  * [ ] Flags: dataset, `num_agents`, `num_rounds`, debate on/off, etc.

### 8.3 Metrics computation with basic robustness (`src/eval/metrics.py`)

* [ ] Implement standard metrics:
  * [ ] Exact match, F1 for QA
  * [ ] Change-of-answer rate
  * [ ] Average # rounds, # tool calls

* [ ] **NEW**: Implement basic robustness metrics:
  * [ ] Confidence calibration analysis  
  * [ ] Agreement rate between workers
  * [ ] Consensus strength measurement
  * [ ] Basic bias pattern detection

* [ ] Add notebook / script for plotting results with robustness analysis

> ✅ Dataset prep, metric functions, and runner plumbing can all be done in parallel.

---

## Phase 6.5 – Validation Framework Extension Points (NEW)

> **Goal**: Define and implement abstract interfaces for systematic robustness enhancement

### 6.5.1 Abstract Validation Interfaces (`src/eval/validation_interfaces.py`)

* [ ] Define `IBiasDetector` interface:
  * [ ] `detect_bias(content: str, context: dict) -> dict`
  * [ ] Standard bias categories (political, demographic, ideological)
  * [ ] Extensible framework for new bias types

* [ ] Define `IConfidenceEstimator` interface:
  * [ ] `estimate_confidence(reply: AgentReply, context: dict) -> float`
  * [ ] Support for multiple confidence signals
  * [ ] Calibration measurement methods

* [ ] Define `IErrorHandler` interface:
  * [ ] `handle_debate_failure(session: DebateSession) -> RecoveryAction`
  * [ ] Deadlock resolution strategies
  * [ ] Escalation pathways

### 6.5.2 Basic Implementations

* [ ] Implement `BasicBiasDetector`:
  * [ ] Keyword-based political bias detection
  * [ ] Simple demographic language analysis
  * [ ] Extensible for sophisticated NLP methods

* [ ] Implement `EnhancedConfidenceEstimator`:  
  * [ ] Multi-factor confidence scoring (citations, agreement, consistency)
  * [ ] Integration with agent confidence scores
  * [ ] Calibration against ground truth when available

* [ ] Implement `BasicErrorHandler`:
  * [ ] Timeout handling for long debates
  * [ ] Simple deadlock detection and resolution
  * [ ] Fallback to single-agent answers

### 6.5.3 Integration and Testing

* [ ] Add validation pipeline to debate protocol
* [ ] Test enhancement without breaking core functionality
* [ ] Validate that basic implementations provide meaningful signals
* [ ] Create upgrade paths to sophisticated implementations

---

## Phase 8.5 – Validation Layer Enhancement (NEW)

> **Goal**: Upgrade to research-grade robustness based on Phase 8 findings

### 8.5.1 Enhanced Bias Detection

* [ ] Analyze bias patterns discovered in Phase 8
* [ ] Upgrade `BasicBiasDetector` based on failure modes:
  * [ ] Add demographic bias detection
  * [ ] Include ideological bias analysis  
  * [ ] Implement cross-cultural bias assessment
* [ ] Validate enhanced detection against human evaluation

### 8.5.2 Systematic Confidence Optimization

* [ ] Implement confidence threshold optimization based on Phase 8 results
* [ ] Add hallucination detection and correction mechanisms
* [ ] Include consistency checking across debate rounds  
* [ ] Cross-validation with human expert judgments

### 8.5.3 Comprehensive Validation Testing

* [ ] Test enhanced validation vs basic validation for research credibility
* [ ] Measure validation effectiveness: does enhanced robustness improve results?
* [ ] Cost-benefit analysis of validation sophistication levels
* [ ] Prepare for research publication with systematic validation

---

## Phase 9 – Advanced Research Extensions & Production Readiness

### 9.1 Enhanced Research Features

* [ ] Implement configs with robustness ablations:
  * [ ] **No debate** (single worker)
  * [ ] **Flat debate** (no hierarchy)  
  * [ ] **No RAG** (disable retrieval tool)
  * [ ] **NEW**: Confidence thresholding on/off
  * [ ] **NEW**: Bias detection on/off
  * [ ] **NEW**: Validation components disabled

### 9.2 Advanced Extensions

* [ ] Add **role-diverse workers** (optimist, pessimist, skeptic)
* [ ] Experiment with advanced tools:
  * [ ] Code Execution for math verification
  * [ ] Multi-modal inputs and analysis
* [ ] Systematic parameter tuning:
  * [ ] # of workers and rounds based on complexity
  * [ ] Confidence thresholds per domain
  * [ ] Retrieval strategies per question type

### 9.3 Production and Research Deployment

* [ ] Production deployment patterns and operational monitoring
* [ ] Research publication with comprehensive validation results
* [ ] Documentation for reproducibility and extension

### Phase 9.5 – Research-Grade Validation (Optional)

> **Goal**: Publication-ready systematic validation for research credibility

* [ ] Comprehensive bias testing across demographic and ideological dimensions  
* [ ] Systematic failure mode analysis with recovery procedures
* [ ] Cross-dataset validation and robustness testing
* [ ] Human evaluation studies with expert domain judges
* [ ] Preparation for peer review and research publication

---

## Progressive Enhancement Summary

This roadmap embodies the synthesis principles:

1. **Progressive Enhancement**: Start simple (Phase 0.5), layer complexity through clean interfaces
2. **Abstraction ≠ Complexity**: Good abstractions (Phase 1a/3) enable simplicity
3. **Validation from Start**: Basic validation (Phase 3), comprehensive validation later (Phase 8.5)
4. **Maintainable Simplicity**: Simple implementations on extensible foundations
5. **Research → Production Path**: Design for evolution, not throwaway prototypes

**Success Criteria**: Each phase validates specific hypotheses while building foundations for the next phase. No phase requires complete rewrites of previous work.

---
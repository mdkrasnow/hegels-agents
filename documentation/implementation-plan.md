Here's a concrete, research-grade implementation plan that uses **progressive enhancement architecture**:

* Uses **Gemini via the Google GenAI SDK (Python)**
* Starts with **file-based corpus** then scales to **Supabase (pgvector)**
* Exposes corpus retrieval as a **Gemini tool (function calling)**
* Implements a **hierarchical multi-agent debate** with extensible foundations
* Includes **validation hooks from the start** with progressive robustness enhancement
* Logs everything so you can run proper **experiments & metrics**

## Progressive Enhancement Strategy

This implementation adopts a **progressive validation architecture** that:

1. **Validates core hypotheses quickly** through minimal implementations
2. **Builds on extensible foundations** with clean interfaces and robustness hooks
3. **Layers complexity progressively** based on validation results
4. **Enables evolution** from research prototype to production system

**Key Principle**: Start with **minimal viable abstractions** (not throwaway prototypes) that support both simple initial implementations AND future robustness enhancements through extension points.

---

## 0. High-level architecture

> **Note**: This architecture emphasizes progressive enhancement - we start simple but build on foundations that support systematic validation and production scaling.

**Goal:** Given a query and a fixed corpus, run a hierarchical multi-agent debate over RAG, then produce a synthesized answer + citations, and log all steps for research.

### Main components

1. **Corpus & Retrieval Layer (Supabase + embeddings)**

   * Tables: `corpus_documents`, `corpus_sections` with `vector` embeddings.
   * Gemini `gemini-embedding-001` for document + query embeddings.([Google AI for Developers][1])
   * A Postgres function (`match_corpus_sections`) to do pgvector similarity search.([Supabase][2])
   * A Python function that calls Supabase and is exposed to Gemini as a **tool** (function calling).([Google AI for Developers][3])

2. **Model Layer (Gemini API via google-genai)**

   * Use `google-genai` Python SDK + `gemini-2.5-flash` (or later) for agents.([Google APIs][4])
   * Shared client, different **system prompts** per agent role (Orchestrator, Worker, Reviewer, Summarizer).

3. **Agent & Debate Orchestration**

   * OOP hierarchy:

     * `BaseAgent` → `OrchestratorAgent`, `WorkerAgent`, `ReviewerAgent`, `SummarizerAgent`.
   * `DebateSession` and `DebateTurn` to track rounds, participants, and messages.
   * `TaskTree` structure for hierarchical decomposition (sub-questions).

4. **Experiment & Metrics Layer**

   * Tables: `experiments`, `experiment_runs`, `agent_messages`, `tool_calls`, etc.
   * Scripts to run benchmarks (GSM8K, HotpotQA, TruthfulQA, etc.) and compute accuracy/F1, change-of-answer rate, # rounds, etc.

---

## 1. Corpus & Supabase implementation

### 1.1. Supabase schema for corpus

Enable `pgvector` and create tables (adapted from Supabase RAG docs).([Supabase][5])

```sql
-- Enable pgvector extension once per DB
create extension if not exists "vector" with schema extensions;

-- High-level document metadata
create table corpus_documents (
  id           bigserial primary key,
  external_id  text,                -- e.g., filename, paper id
  title        text not null,
  metadata     jsonb default '{}'::jsonb,
  created_at   timestamptz not null default now()
);

-- Chunked sections ready for RAG
create table corpus_sections (
  id            bigserial primary key,
  document_id   bigint not null references corpus_documents (id) on delete cascade,
  section_index integer not null,
  content       text not null,
  token_count   integer,
  embedding     extensions.vector(768),  -- using 768-dim truncation
  metadata      jsonb default '{}'::jsonb
);

create index on corpus_sections (document_id);
create index on corpus_sections using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```

> 768 dimensions is fully supported by `gemini-embedding-001` via `output_dimensionality` and is a good storage/latency trade-off.([Google AI for Developers][1])

### 1.2. Matching function for tool-style retrieval

```sql
create or replace function match_corpus_sections (
  query_embedding extensions.vector(768),
  match_threshold float,
  match_count     int
)
returns table (
  section_id    bigint,
  document_id   bigint,
  section_index integer,
  content       text,
  similarity    float,
  metadata      jsonb
)
language sql stable as $$
  select
    cs.id            as section_id,
    cs.document_id,
    cs.section_index,
    cs.content,
    1 - (cs.embedding <=> query_embedding) as similarity,
    cs.metadata
  from corpus_sections cs
  where 1 - (cs.embedding <=> query_embedding) > match_threshold
  order by cs.embedding <=> query_embedding
  limit match_count;
$$;
```

You’ll call this via a direct Postgres connection (for research, no RLS needed), or, if using the Supabase client, via `rpc('match_corpus_sections', ...)`.([Supabase][2])

### 1.3. Ingestion & embedding pipeline

OOP module: `corpus/ingestion.py`

Key classes:

```python
class CorpusDocument:
    id: int | None
    external_id: str | None
    title: str
    text: str
    metadata: dict[str, Any]

class CorpusChunk:
    document_id: int
    section_index: int
    content: str
    token_count: int
    embedding: list[float]
    metadata: dict[str, Any]
```

Pipeline steps:

1. **Load raw docs** (PDFs, txt, whatever → normalized text).

2. **Chunk** into ~512–1024 tokens with overlap by simple rules or a library.

3. **Embed** chunks using Gemini embeddings:

   ```python
   from google import genai
   from google.genai import types

   client = genai.Client()  # uses GEMINI_API_KEY by default:contentReference[oaicite:7]{index=7}

   def embed_chunks(texts: list[str]) -> list[list[float]]:
       resp = client.models.embed_content(
           model="gemini-embedding-001",
           contents=texts,
           config=types.EmbedContentConfig(
               task_type="RETRIEVAL_DOCUMENT",
               output_dimensionality=768,
           ),
       )
       return [e.values for e in resp.embeddings]  # list[list[float]]
   ```

4. **Insert** `corpus_documents`, then `corpus_sections` with embeddings (as Python arrays → pgvector).

---

## 2. Retrieval as a Gemini tool

You’ll define a **custom tool** that Gemini can call via function calling to retrieve sections from Supabase.([Google AI for Developers][3])

### 2.1. Tool declaration (Python)

```python
from google.genai import types

retrieve_corpus_tool = {
    "name": "retrieve_corpus_sections",
    "description": "Retrieve the most relevant corpus sections for a query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search the corpus."
            },
            "k": {
                "type": "integer",
                "description": "Number of sections to retrieve (<= 20).",
                "default": 8
            },
            "threshold": {
                "type": "number",
                "description": "Minimum cosine similarity [0,1] for results.",
                "default": 0.7
            }
        },
        "required": ["query"]
    },
}
tools = [types.Tool(function_declarations=[retrieve_corpus_tool])]
```

### 2.2. Tool implementation in Python

```python
import psycopg2
from google import genai
from google.genai import types

class CorpusRetriever:
    def __init__(self, pg_dsn: str):
        self.conn = psycopg2.connect(pg_dsn)
        self.client = genai.Client()

    def _embed_query(self, query: str) -> list[float]:
        resp = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        return resp.embeddings[0].values

    def retrieve(self, query: str, k: int = 8, threshold: float = 0.7) -> list[dict]:
        embedding = self._embed_query(query)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                select section_id, document_id, section_index, content, similarity, metadata
                from match_corpus_sections(%s::vector, %s::float, %s::int)
                """,
                (embedding, threshold, k),
            )
            rows = cur.fetchall()
        return [
            {
                "section_id": r[0],
                "document_id": r[1],
                "section_index": r[2],
                "content": r[3],
                "similarity": float(r[4]),
                "metadata": r[5] or {},
            }
            for r in rows
        ]
```

Your agent loop will:

1. Call Gemini with the tool declaration.
2. If Gemini returns a `function_call` to `retrieve_corpus_sections`, you execute `CorpusRetriever.retrieve(...)`.
3. Send the results back to Gemini as a **tool response content** and let it continue generating.

---

## 3. Agent & Debate OOP design

### 3.1. Core entities

**Domain objects:**

```python
@dataclass
class SubQuestion:
    id: str
    text: str
    parent_id: str | None
    depth: int
    metadata: dict[str, Any]

@dataclass
class AgentReply:
    agent_id: str
    role: str        # "worker", "reviewer", "orchestrator"
    content: str
    citations: list[dict]  # corpus sections used
    raw_response: Any      # raw Gemini response for logging

@dataclass
class DebateTurn:
    turn_index: int
    subquestion_id: str
    messages: list[AgentReply]
```

**Agent base class:**

```python
class BaseAgent:
    def __init__(self, agent_id: str, system_prompt: str, client: genai.Client):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.client = client

    def generate(self, messages: list[str], tools=None) -> Any:
        contents = [
            f"System: {self.system_prompt}",
            *messages,
        ]
        config = types.GenerateContentConfig(
            tools=tools or [],
            # you can tweak safety / temperature here
        )
        return self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
```

Then extend:

* `OrchestratorAgent(BaseAgent)`
* `WorkerAgent(BaseAgent)`
* `ReviewerAgent(BaseAgent)`
* `SummarizerAgent(BaseAgent)`

Each gets a different system prompt enforcing behavior (planner vs evidence-gatherer vs critic vs synthesizer).

### 3.2. Orchestrator responsibilities

1. **Decompose** top-level query → `SubQuestion` tree.
2. **Assign** sub-questions to workers.
3. **Initiate debates** between conflicting answers.
4. **Bubble up** synthesized answers up the tree.

Pseudo-interface:

```python
class OrchestratorAgent(BaseAgent):
    def decompose_query(self, user_query: str) -> list[SubQuestion]:
        # single Gemini call: "Break this into N sub-questions, output JSON"
        ...

    def run_subtree(self, subq: SubQuestion) -> AgentReply:
        # recursively solve sub-question:
        # 1. if leaf: ask workers directly
        # 2. else: solve children, then debate & synthesize
        ...

    def run(self, user_query: str) -> AgentReply:
        subquestions = self.decompose_query(user_query)
        # build tree, then solve root
        ...
```

### 3.3. Worker & Reviewer pattern

* **WorkerAgent**: “propose the best answer you can; call tools as needed; include citations from retrieved sections.”
* **ReviewerAgent**: “critique two or more answers; point out logical/factual conflicts; request more retrieval if needed.”

Debate algorithm for one `SubQuestion`:

1. Orchestrator spawns **2–3 WorkerAgents** with slightly different prompts (e.g., one “optimistic synthesizer”, one “devil’s advocate”) plus randomness.
2. Each worker:

   * Calls Gemini with `retrieve_corpus_sections` tool available.
   * Produces an answer + inline evidence.
3. Orchestrator examines answers; if they differ materially:

   * Starts a **DebateSession**:

     * Turn 1: Ask each worker to read the others’ answers and respond.
     * Turn 2: ReviewerAgent gets full transcript, produces final critique & recommends a synthesis.
4. Orchestrator calls **SummarizerAgent** to generate the **final synthesized answer** for this sub-question, guided by reviewer’s comments.

All intermediate calls (prompts, responses, tool calls) are stored in the `agent_messages` table for later analysis.

---

## 4. Research logging and metrics

### 4.1. DB tables for experiments

Minimal schema (in Supabase or a separate Postgres):

```sql
create table experiments (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  description text,
  created_at  timestamptz not null default now()
);

create table experiment_runs (
  id             uuid primary key default gen_random_uuid(),
  experiment_id  uuid not null references experiments (id),
  dataset_name   text not null,
  example_id     text not null,
  config         jsonb not null, -- #agents, #rounds, model, etc.
  gold_answer    text,
  final_answer   text,
  is_correct     boolean,
  metrics        jsonb,
  created_at     timestamptz not null default now()
);

create table agent_messages (
  id             bigserial primary key,
  experiment_run_id uuid not null references experiment_runs (id),
  subquestion_id text,
  turn_index     int,
  agent_id       text,
  role           text,
  content        text,
  tool_calls     jsonb,
  tool_results   jsonb,
  created_at     timestamptz not null default now()
);
```

### 4.2. Metrics to compute

For each dataset / configuration:

* **Task-level metrics**

  * QA / reasoning datasets: exact match / F1.
  * Summarization: ROUGE-L, BERTScore (or human Likert ratings).
  * Truthful QA: TruthfulQA score (fraction of answers that avoid “common falsehoods”).

* **Debate-process metrics**

  * Change-of-answer rate: % examples where final answer ≠ any initial worker answer.
  * Convergence rate: % examples where all workers agree after N rounds (Du et al. show near-universal convergence with debate).([ACM Digital Library][6])
  * Average # of:

    * Tool calls per example.
    * Debate rounds per sub-question.
    * Distinct corpus sections used in the final answer.

* **Efficiency metrics**

  * Tokens in/out (per agent role).
  * Latency per example.
  * Cost per example vs baseline.

* **Ablations**

  * **No debate**: single worker, same retrieval tool.
  * **Non-hierarchical debate**: all agents debate on the full problem.
  * **No RAG**: debate only on model’s parametric knowledge.

---

## 5. Implementation phases (progressive roadmap)

### Phase 0.5 — Minimal Dialectical Proof-of-Concept (NEW)

**Duration**: 5 days | **Goal**: Validate core hypothesis before infrastructure investment

1. Implement basic `WorkerAgent` vs `ReviewerAgent` debate on single questions
2. File-based corpus with simple text search (no embeddings initially)
3. Test on 10 simple questions: does debate improve reasoning?
4. Success criteria: measurable improvement in answer quality through dialectical process

### Phase 1 — Progressive Infrastructure & RAG

**Phase 1a** (File-based validation):
1. Implement `ICorpusRetriever` interface and `FileCorpusRetriever`
2. Add Gemini embeddings for file-based similarity search
3. Validate retrieval concept without database complexity

**Phase 1b** (Production scaling):
1. Set up Supabase project, enable `vector` extension.([Supabase][2])
2. Implement `SupabaseCorpusRetriever` with same interface
3. Migrate validated approach to production infrastructure
4. Implement **single-agent RAG** baseline for comparison

### Phase 2 — Agent Architecture with Extension Points

1. Implement `BaseAgent` with optional `validation_hooks` parameter
2. Implement `WorkerAgent`, `OrchestratorAgent`, `ReviewerAgent`, `SummarizerAgent`
3. Include confidence scoring and citation verification interfaces (basic implementations)
4. Implement `TaskTree` + `SubQuestion` decomposition via Orchestrator
5. Wire retrieval tools with clean abstraction layers
6. Add logging with robustness-ready schemas (optional fields)

### Phase 3 — Single-Subquestion Debate with Validation Hooks

1. Implement debate protocol (2–3 workers + reviewer) with basic validation
2. Include agreement detection and simple confidence thresholds
3. Test manually and validate that debates converge reliably
4. Add basic robustness metrics (consensus measures, confidence distribution)

### Phase 4 — Hierarchical Debate Orchestration

1. Generalize to **hierarchical** debate (sub-questions → final synthesis)
2. Add validation checkpoints where robustness measures can be enhanced
3. Test on complex multi-part questions

### Phase 5 — Basic Evaluation & Metrics

1. Implement evaluation harness for:
   * GSM8K (math), HotpotQA (multi-doc QA), simple literature synthesis
2. Include basic robustness metrics alongside accuracy:
   * Confidence calibration, agreement rates, change-of-answer analysis
3. Run baselines vs debate architecture with statistical significance testing

### Phase 6.5 — Validation Framework Extension Points (NEW)

**Goal**: Define and implement abstract interfaces for systematic robustness

1. Define abstract interfaces: `IBiasDetector`, `IConfidenceEstimator`, `IErrorHandler`
2. Implement basic versions with clear upgrade paths
3. Add validation pipeline that can be enhanced without core logic changes
4. Test enhancement mechanisms (add/remove validation components)

### Phase 7 — Research Logging Enhancement

1. Enhance logging system to capture progressive validation data
2. Add experiment tracking with robustness metadata
3. Implement analysis tools for validation effectiveness

### Phase 8 — Comprehensive Evaluation with Basic Robustness

1. Run systematic experiments with basic robustness validation
2. Compare debate vs single-agent across multiple benchmarks
3. Analyze convergence, bias patterns, confidence calibration
4. Iterate on agent roles, debate rounds, retrieval strategies

### Phase 8.5 — Validation Layer Enhancement (NEW)

**Goal**: Upgrade to research-grade robustness based on Phase 8 findings

1. Enhance bias detection based on observed failure modes
2. Add comprehensive hallucination detection and correction
3. Implement systematic confidence threshold optimization
4. Add cross-validation with human expert judgments
5. Test enhanced validation vs basic validation for research credibility

### Phase 9 — Advanced Research Extensions & Production Readiness

1. Add **role-diverse workers** (optimist, pessimist, skeptic)
2. Include robustness ablations: confidence thresholding, bias detection, validation components
3. Experiment with advanced tools (Code Execution, multi-modal inputs)
4. Production deployment patterns and operational monitoring
5. Research publication with comprehensive validation results

### Phase 9.5 — Research-Grade Validation (Optional)

**Goal**: Publication-ready systematic validation for research credibility

1. Comprehensive bias testing across demographic and ideological dimensions
2. Systematic failure mode analysis with recovery procedures
3. Cross-dataset validation and robustness testing
4. Human evaluation studies with expert domain judges

## Progressive Enhancement Principles

This roadmap embodies key design principles:

1. **Progressive Enhancement**: Start simple, layer complexity through clean interfaces
2. **Abstraction ≠ Complexity**: Good abstractions enable simplicity
3. **Validation from Start**: Basic validation early, comprehensive validation later
4. **Maintainable Simplicity**: Simple implementations on extensible foundations  
5. **Research → Production Path**: Design for evolution, not throwaway prototypes

**Success Criteria**: Each phase should validate specific hypotheses while building foundations for the next phase. No phase should require complete rewrites of previous work.

---

[1]: https://ai.google.dev/gemini-api/docs/embeddings "Embeddings  |  Gemini API  |  Google AI for Developers"
[2]: https://supabase.com/docs/guides/ai/vector-columns "Vector columns | Supabase Docs"
[3]: https://ai.google.dev/gemini-api/docs/tools "Using tools with Gemini API  |  Google AI for Developers"
[4]: https://googleapis.github.io/python-genai/ "Google Gen AI SDK documentation"
[5]: https://supabase.com/docs/guides/ai/rag-with-permissions "RAG with Permissions | Supabase Docs"
[6]: https://dl.acm.org/doi/10.5555/3692070.3692537?utm_source=chatgpt.com "Improving factuality and reasoning in language models ..."

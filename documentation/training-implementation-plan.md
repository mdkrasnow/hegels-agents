# Hegel's Agents Training Layer Implementation Plan

## Executive Summary

This document outlines the implementation plan for wrapping the existing Hegel's Agents debate-and-retrieval system with a lightweight training layer. The training layer enables the system to learn from each question it answers when a `grad=True` flag is enabled, without modifying underlying Gemini weights. Instead, it automatically updates and version-controls prompts, role definitions, and agent configuration based on feedback through reflection-based edits and evolutionary/bandit selection across prompt variants.

## Problem Analysis

### Core Issues with Current System

1. **Static Configuration**: The current Hegel's agents system uses fixed prompts and hyperparameters defined in the agent classes (`BasicWorkerAgent.SYSTEM_PROMPT`, `BasicReviewerAgent.CRITIQUE_PROMPT`, etc.). These cannot adapt to different domains or improve based on experience.

2. **No Learning Mechanism**: There is no feedback loop to improve performance. Each question is answered independently without leveraging past successes or failures.

3. **Domain-Agnostic Approach**: The same prompts are used across all corpora (philosophy, mathematics, history, etc.), missing opportunities for domain-specific optimization.

4. **Performance Plateau**: Without adaptation, the system cannot improve beyond its initial configuration, limiting long-term effectiveness.

### Root Cause Analysis

The fundamental issue is that the current architecture treats prompts and configuration as static code rather than learnable parameters. The system has excellent foundations for:
- Multi-agent dialectical reasoning (`DebateSession` class)
- Structured logging and metadata tracking (`AgentLogger`)
- Response quality assessment (`ConflictAnalysis`)
- Extensible agent architecture (`BasicWorkerAgent`, `BasicReviewerAgent`)

But lacks the abstraction layer needed to:
- Version and evolve prompts based on performance
- Specialize configuration per corpus/domain
- Learn from feedback systematically
- Maintain reproducibility across experiments

## Training Layer Architecture

### 1. Conceptual Design

The training layer acts as a wrapper around the existing Hegel system, introducing prompt optimization without changing the core debate logic. When `grad=True`, each query becomes a training step that can update the system's "parameters" (prompts and configuration).

```
Current Flow:
Question → HegelDebateEngine → Answer

Enhanced Flow:
Question → HegelTrainer → [Load Profile] → HegelDebateEngine → Answer → [Update Profile if grad=True]
```

### 2. Core Components

#### 2.1 PromptProfile System
**Purpose**: Version-controlled storage of all prompts and hyperparameters per corpus/task combination.

```python
@dataclass
class RolePrompt:
    system_prompt: str
    style_hint: str | None = None
    temperature: float = 0.4
    max_tokens: int = 2000

@dataclass
class PromptProfile:
    id: str                       # uuid
    base_profile_id: str | None   # for evolution tracking
    corpus_id: str               # e.g., "mathematics", "philosophy"
    task_type: str               # e.g., "qa", "synthesis", "evaluation"
    
    # Agent role configurations
    orchestrator: RolePrompt
    worker: RolePrompt
    reviewer: RolePrompt
    summarizer: RolePrompt
    
    # Debate hyperparameters
    max_debate_rounds: int = 2
    num_workers: int = 2
    consensus_threshold: float = 0.8
    
    # Metadata
    created_at: datetime
    performance_stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### 2.2 PromptProfileStore
**Purpose**: Database persistence and retrieval of prompt profiles with versioning.

```python
class PromptProfileStore:
    def load_latest(self, corpus_id: str, task_type: str) -> PromptProfile
    def load_by_id(self, profile_id: str) -> PromptProfile
    def save_profile(self, profile: PromptProfile) -> str
    def save_training_step(self, step: TrainingStep) -> str
    def get_profile_history(self, corpus_id: str, task_type: str) -> List[PromptProfile]
    def get_performance_stats(self, corpus_id: str, task_type: str) -> dict
```

#### 2.3 PromptOptimizer Interface
**Purpose**: Abstract interface for different optimization strategies.

```python
class PromptOptimizer:
    def update_profile(self, 
                      profile: PromptProfile,
                      query: str,
                      answer: str,
                      gold_answer: str | None,
                      reward: float,
                      trace: dict,
                      metadata: dict) -> PromptProfile
```

#### 2.4 HegelTrainer Wrapper
**Purpose**: Main API that wraps existing functionality with training capabilities.

```python
class HegelTrainer:
    def __init__(self, profile_store: PromptProfileStore, 
                 optimizer: PromptOptimizer):
        self.profile_store = profile_store
        self.optimizer = optimizer
    
    def run(self, query: str, corpus_id: str, task_type: str = "qa",
            grad: bool = False, gold_answer: str | None = None,
            reward: float | None = None) -> dict
```

### 3. Data Persistence Strategy

#### 3.1 Database Schema
```sql
-- Version-controlled prompt profiles
CREATE TABLE hegel_prompt_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  base_profile_id UUID REFERENCES hegel_prompt_profiles(id),
  corpus_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  profile JSONB NOT NULL,
  performance_stats JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  INDEX(corpus_id, task_type, created_at DESC)
);

-- Training step logging
CREATE TABLE hegel_training_steps (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  old_profile_id UUID NOT NULL REFERENCES hegel_prompt_profiles(id),
  new_profile_id UUID NOT NULL REFERENCES hegel_prompt_profiles(id),
  corpus_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  query TEXT NOT NULL,
  gold_answer TEXT,
  predicted_answer TEXT NOT NULL,
  reward NUMERIC,
  metrics JSONB,
  debate_trace JSONB,
  optimization_strategy TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Population-based optimization state
CREATE TABLE hegel_profile_populations (
  corpus_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  profile_id UUID NOT NULL REFERENCES hegel_prompt_profiles(id),
  fitness_score NUMERIC,
  selection_count INTEGER DEFAULT 0,
  generation INTEGER DEFAULT 1,
  last_updated TIMESTAMPTZ DEFAULT NOW(),
  
  PRIMARY KEY(corpus_id, task_type, profile_id)
);
```

## Optimization Strategies

### 1. Reflection-Based Local Optimization (Stage 1)

**Philosophy**: Use the LLM itself to propose targeted improvements when performance is poor.

**Implementation**: `ReflectionOptimizer`

```python
class ReflectionOptimizer(PromptOptimizer):
    def update_profile(self, profile, query, answer, gold_answer, reward, trace, metadata):
        if reward > self.no_change_threshold:  # e.g., 0.9
            return profile
        
        # Generate reflection prompt
        reflection_context = f"""
        CONTEXT:
        - Question: {query}
        - System Answer: {answer}
        - Expected Answer: {gold_answer}
        - Current Reward: {reward}
        - Debate Trace: {trace}
        
        CURRENT PROMPTS:
        - Worker: {profile.worker.system_prompt}
        - Reviewer: {profile.reviewer.system_prompt}
        
        TASK:
        Analyze the failure and suggest specific, generalizable edits to the prompts.
        Focus on improving reasoning, accuracy, and dialectical quality.
        Return structured JSON with edit suggestions.
        """
        
        # Use Gemini to generate prompt edits
        edit_suggestions = self._generate_edits(reflection_context)
        
        # Apply edits to create new profile
        new_profile = self._apply_edits(profile, edit_suggestions)
        new_profile.metadata.update({
            'optimization_strategy': 'reflection',
            'parent_reward': reward,
            'edit_reason': edit_suggestions.get('reasoning')
        })
        
        return new_profile
```

**Advantages**:
- Fast implementation using existing Gemini API
- Provides qualitative insights embedded in prompts
- Good for research traceability and explainability
- Works with single examples (online learning)

**Disadvantages**:
- No performance guarantees
- Risk of prompt bloat over time
- May not find optimal solutions

### 2. Population-Based Evolution (Stage 2)

**Philosophy**: Maintain multiple prompt variants and evolve them based on performance.

**Implementation**: `PopulationOptimizer`

```python
class PopulationOptimizer(PromptOptimizer):
    def __init__(self, population_size: int = 8, selection_strategy: str = "thompson"):
        self.population_size = population_size
        self.selection_strategy = selection_strategy
    
    def select_profile(self, corpus_id: str, task_type: str) -> PromptProfile:
        # Multi-armed bandit selection (epsilon-greedy, Thompson sampling, etc.)
        population = self.profile_store.get_population(corpus_id, task_type)
        if self.selection_strategy == "thompson":
            return self._thompson_sampling(population)
        else:
            return self._epsilon_greedy(population)
    
    def update_population(self, corpus_id: str, task_type: str):
        # Periodic evolution step
        population = self.profile_store.get_population(corpus_id, task_type)
        
        # Rank by fitness
        ranked = sorted(population, key=lambda p: p.fitness_score, reverse=True)
        
        # Keep top performers
        survivors = ranked[:self.population_size // 2]
        
        # Generate offspring via reflection from top performers
        offspring = []
        for parent in survivors:
            child = self.reflection_optimizer.mutate(parent)
            offspring.append(child)
        
        # Replace population
        new_population = survivors + offspring
        self.profile_store.save_population(corpus_id, task_type, new_population)
```

**Advantages**:
- Systematic exploration of prompt space
- Robust to local optima
- Balances exploration vs exploitation
- Scalable to large numbers of training examples

**Disadvantages**:
- Requires batch processing for evolution
- More complex implementation
- Higher computational overhead

### 3. Hybrid Approach (Recommended)

**Implementation Strategy**:
1. Use `ReflectionOptimizer` for immediate improvements (online learning)
2. Use `PopulationOptimizer` for systematic exploration (batch evolution)
3. Enable seamless switching between strategies based on context

## Implementation Phases

### Phase 1: Foundation Layer (1-2 weeks)
**Goal**: Implement basic training wrapper without changing existing code.

**Deliverables**:
1. `PromptProfile` data structures and serialization
2. `PromptProfileStore` with basic database operations
3. `HegelTrainer` wrapper with `grad=False` mode (read-only)
4. Database schema and basic migration scripts
5. Integration tests with existing `BasicWorkerAgent` and `BasicReviewerAgent`

**Success Criteria**:
- All existing functionality works unchanged
- Can load/save prompt profiles from database
- `HegelTrainer.run()` with `grad=False` produces identical results to current system

### Phase 2: Reflection-Based Optimization (2-3 weeks)
**Goal**: Implement online learning through reflection-based prompt edits.

**Deliverables**:
1. `ReflectionOptimizer` implementation
2. Prompt edit generation and application logic
3. Training step logging and persistence
4. `grad=True` mode with gold answer evaluation
5. Basic reward computation (F1, BLEU, semantic similarity)

**Success Criteria**:
- System can update prompts based on poor performance
- Training steps are logged for reproducibility
- Performance improvements observable on simple test cases

### Phase 3: Per-Corpus Specialization (2 weeks)
**Goal**: Enable domain-specific prompt optimization.

**Deliverables**:
1. Corpus-aware profile management
2. Base profile inheritance and branching
3. Performance tracking per corpus/task type
4. Migration tools for existing corpora
5. Analysis tools for prompt evolution visualization

**Success Criteria**:
- Different corpora can have specialized prompts
- Can track performance improvements per domain
- Easy rollback to base profiles if needed

### Phase 4: Population-Based Evolution (3-4 weeks)
**Goal**: Implement systematic prompt exploration and optimization.

**Deliverables**:
1. `PopulationOptimizer` implementation
2. Multi-armed bandit selection strategies
3. Batch evolution algorithms
4. Fitness scoring and ranking systems
5. Population management and lifecycle tools

**Success Criteria**:
- Can maintain and evolve populations of prompt variants
- Systematic exploration finds better solutions than reflection alone
- Scalable to large training datasets

### Phase 5: Evaluation and Validation (2-3 weeks)
**Goal**: Comprehensive evaluation of training effectiveness.

**Deliverables**:
1. Benchmark datasets for evaluation
2. Automated evaluation pipelines
3. A/B testing framework for prompt comparisons
4. Statistical significance testing
5. Performance regression detection

**Success Criteria**:
- Quantitative evidence of training effectiveness
- Robust evaluation methodology
- Clear performance improvement over baseline

### Phase 6: Production Features (2-3 weeks)
**Goal**: Features needed for production deployment.

**Deliverables**:
1. Configuration management and environment separation
2. Model serving and caching optimizations
3. Monitoring and alerting for training quality
4. Rollback and safety mechanisms
5. Documentation and deployment guides

**Success Criteria**:
- Ready for production deployment
- Monitoring covers all critical metrics
- Safe rollback procedures tested

## Technical Considerations

### 1. Backward Compatibility
**Challenge**: Adding training layer without breaking existing functionality.

**Solution**:
- `HegelTrainer` wraps existing agents rather than replacing them
- Factory pattern for creating configured agents from profiles
- Default profiles that replicate current hardcoded prompts
- All changes are additive, no modification to core debate logic

### 2. Prompt Versioning and Reproducibility
**Challenge**: Managing prompt evolution while maintaining scientific rigor.

**Solution**:
- Immutable prompt profiles with unique IDs
- Complete lineage tracking through `base_profile_id`
- Training step logging captures all inputs/outputs
- Deterministic profile loading by ID for exact reproduction

### 3. Performance and Scaling
**Challenge**: Training overhead should not significantly impact inference time.

**Solution**:
- Lazy loading of profiles with caching
- Async training step persistence
- Batch evolution runs during off-peak hours
- Profile serving optimized for read performance

### 4. Safety and Quality Control
**Challenge**: Preventing prompt degradation or harmful optimization.

**Solution**:
- Human review triggers for significant prompt changes
- Automatic rollback on performance regression
- Prompt content filtering and safety checks
- Conservative optimization with small edit suggestions

### 5. Multi-Tenancy and Isolation
**Challenge**: Supporting multiple researchers/experiments simultaneously.

**Solution**:
- Experiment namespacing in database schema
- Profile inheritance allows shared base configurations
- Per-experiment optimization state isolation
- Resource quotas and rate limiting

## Evaluation Methodology

### 1. Baseline Establishment
- Run current system on diverse benchmark datasets
- Measure accuracy, F1, BLEU, and debate quality metrics
- Establish statistical baselines with confidence intervals

### 2. Training Effectiveness Evaluation
- Compare trained vs untrained profiles on held-out test sets
- Measure learning curves over training iterations
- Analyze prompt evolution patterns and convergence

### 3. Domain Specialization Assessment
- Evaluate cross-domain transfer vs domain-specific optimization
- Measure performance improvement per corpus type
- Analyze prompt specialization patterns

### 4. Optimization Strategy Comparison
- Compare reflection vs population-based optimization
- Measure sample efficiency and convergence rates
- Evaluate robustness to different reward signals

### 5. Production Readiness Validation
- Load testing with concurrent training and inference
- Reliability testing with fault injection
- Security review of prompt modification mechanisms

## Expected Outcomes

### Short-term (3 months)
- Functional training layer integrated with existing system
- Demonstrable improvement on 2-3 benchmark datasets
- Per-corpus specialization working for major domains
- Research-ready with comprehensive logging and evaluation

### Medium-term (6 months)
- Production deployment with monitoring and safety features
- Systematic evaluation across 10+ diverse corpora
- Publication-quality results demonstrating training effectiveness
- Advanced optimization strategies (hybrid approaches)

### Long-term (12 months)
- Multi-modal prompt optimization (text + examples)
- Integration with human feedback collection systems
- Advanced meta-learning across corpora
- Open-source release with community contributions

## Risk Mitigation

### Technical Risks
1. **Prompt degradation**: Implement conservative optimization bounds and automatic rollback
2. **Performance regression**: Continuous monitoring with alerting and circuit breakers
3. **Scalability issues**: Profile caching, async operations, and resource management
4. **Reproducibility challenges**: Immutable versioning and complete audit logs

### Research Risks
1. **Limited training effectiveness**: Multiple optimization strategies and fallback to manual tuning
2. **Overfitting to benchmarks**: Diverse evaluation datasets and domain transfer testing
3. **Prompt brittleness**: Regularization techniques and robustness testing
4. **Bias amplification**: Bias detection and mitigation in prompt evolution

### Operational Risks
1. **Production instability**: Thorough testing, gradual rollout, and monitoring
2. **Data privacy concerns**: Secure handling of training data and prompt content
3. **Cost overruns**: Resource budgeting and optimization strategy efficiency
4. **Team coordination**: Clear interfaces, documentation, and testing protocols

## Success Metrics

### Quantitative Metrics
- **Performance Improvement**: >15% accuracy increase on benchmark datasets
- **Sample Efficiency**: Meaningful improvement within 100 training examples per corpus
- **Convergence Speed**: Stable performance within 50 optimization iterations
- **Robustness**: <5% performance variance across random seeds

### Qualitative Metrics
- **Prompt Quality**: Human evaluation of prompt clarity and specificity
- **Domain Adaptation**: Expert assessment of domain-specific prompt relevance
- **System Usability**: Researcher feedback on training interface and tools
- **Research Impact**: Publications, citations, and community adoption

## Conclusion

The proposed training layer represents a significant enhancement to the Hegel's Agents system, enabling continuous improvement and domain specialization while preserving the robust dialectical reasoning foundation. The phased implementation approach balances research innovation with engineering rigor, ensuring both scientific validity and production readiness.

The key innovation lies in treating prompts as learnable parameters rather than static code, enabling the system to evolve and improve through experience while maintaining full reproducibility and version control. This approach opens new research directions in prompt optimization, multi-agent learning, and adaptive reasoning systems.

Success of this implementation will demonstrate the viability of gradient-free optimization for complex multi-agent systems and provide a foundation for future research in adaptive AI reasoning frameworks.
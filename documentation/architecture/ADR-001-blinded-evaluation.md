# ADR-001: Blinded Evaluation with Fresh Evaluator Instances

## Status
**Accepted** - December 2, 2025

## Context
The Hegel's Agents system evaluates response quality by comparing single-agent vs dialectical multi-agent approaches. To ensure scientific rigor, we must prevent evaluation bias that could favor one approach over another.

### Problem
Evaluator bias can manifest in several ways:
1. **Order effects**: Earlier evaluations influence later ones
2. **State leakage**: Previous evaluations contaminate current evaluation
3. **Pattern recognition**: Evaluator learns to distinguish response sources
4. **Anchoring bias**: First response influences judgment of second response

### Investigation Results
During code review, a performance optimization was proposed (PERF-001) to reuse evaluator instances across multiple evaluations to save 50-100ms of initialization overhead per evaluation. This sparked a multi-round debate among security, correctness, performance, and maintainability reviewers.

**Key Findings**:
- Fresh evaluator creation represents 2-5% of total evaluation time (API calls dominate at 500-2000ms)
- Gemini API calls may have non-obvious state retention mechanisms
- Anonymization system depends on complete isolation between evaluations
- Risk of optimization undermining scientific validity outweighs performance benefit

## Decision
**We will create fresh evaluator instances for each blinded evaluation.**

Specifically:
```python
def _create_fresh_evaluator(self) -> BasicReviewerAgent:
    """
    Create a fresh evaluator instance for each evaluation.
    
    ARCHITECTURAL DECISION: This is intentional bias prevention, not inefficiency.
    Do NOT optimize this to reuse instances without careful analysis.
    """
    return BasicReviewerAgent(agent_id=f"blinded_evaluator_{self.evaluation_count}")
```

## Rationale

### Scientific Integrity (Primary)
- **Complete isolation**: Each evaluation starts with clean state
- **No cross-contamination**: Previous responses cannot influence current evaluation
- **Bias prevention**: Evaluator cannot learn patterns or develop preferences
- **Reproducibility**: Each evaluation is independent and reproducible

### Security Benefits
- **State isolation**: Prevents potential information leakage between evaluations
- **Attack surface reduction**: Limits ability to manipulate evaluator state
- **Anonymization integrity**: Supports blinded evaluation guarantees

### Performance Trade-offs Accepted
- **Cost**: 50-100ms per evaluation (2-5% of total time)
- **Benefit**: Scientific validity and bias prevention
- **Conclusion**: Acceptable overhead for research correctness

### Alternative Approaches Considered

**Option A: Reuse evaluator instances with state reset**
- **Rejected**: Cannot guarantee complete state reset
- **Risk**: Subtle state leakage could invalidate results
- **Complexity**: Requires deep understanding of Gemini API internals

**Option B: Shared evaluator pool with rotation**
- **Rejected**: Still allows potential cross-contamination
- **Risk**: Pattern recognition across pool instances
- **Complexity**: Adds pooling logic without eliminating core risk

**Option C: Fresh instances (Accepted)**
- **Advantages**: Simplest, most robust, scientifically sound
- **Disadvantages**: 50-100ms overhead per evaluation
- **Verification**: Easy to verify - just inspect instance creation

## Consequences

### Positive
- **Scientific validity**: Results are defensible in research publications
- **Code simplicity**: No complex state management required
- **Future-proof**: Works regardless of API implementation changes
- **Debuggability**: Each evaluation is independent and traceable

### Negative
- **Performance cost**: 2-5% overhead from instance creation
- **Memory churn**: More object allocation and garbage collection
- **Optimization limitation**: Cannot apply instance pooling optimizations

### Mitigation
For performance-critical scenarios:
1. **Parallelize evaluations**: 40-50% speedup available through concurrent API calls
2. **Batch processing**: Amortize setup costs across multiple questions
3. **Profile before optimizing**: Measure actual impact in production workload

## Compliance
This decision ensures compliance with:
- Scientific evaluation best practices
- Blinded study methodology requirements
- Research reproducibility standards

## Review
- **Approved by**: Security, Correctness, Maintainability reviewers (unanimous)
- **Contested by**: Performance reviewer (withdrew objection after analysis)
- **Re-evaluation trigger**: Evidence of measurable performance impact >10% in production workload

## Related Decisions
- ADR-002: Anonymization strategy (to be written)
- ADR-003: Prompt injection defenses (to be written)
- ADR-004: Database schema versioning (to be written)

## References
- Code Review Debate: `.claude/work/review-debate/round-{1,2,3}/`
- Performance Analysis: PERF-001 finding
- Security Analysis: SEC-011 finding  
- Maintainability Analysis: MAIN-007 finding

---
**Keywords**: blinded evaluation, bias prevention, scientific rigor, performance trade-offs, architectural decision
EOF < /dev/null
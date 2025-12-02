# Training Layer Implementation Subagent Specifications

## Mission Overview
Implement the foundational training layer for Hegel's Agents system following the training-implementation-todo.md plan. Focus on Phase 1 Foundation Layer tasks that can be executed in parallel to maximize development velocity.

## Deployment Strategy
Deploy 5 specialized subagents to work on independent Phase 1 tasks concurrently. Each subagent follows a build-with-review cycle and reports structured results to the orchestrator.

---

## Subagent 1: Data Structures Architect
**Task**: T1.1 - Define Core Data Structures  
**Dependencies**: `[]` (No dependencies, can start immediately)  
**File**: `src/training/data_structures.py`

### Mission
Create the foundational data structures for the training system, ensuring backward compatibility with existing AgentResponse while adding training capabilities.

### Implementation Requirements
- Create `RolePrompt`, `PromptProfile`, `TrainingStep` dataclasses
- Extend existing `AgentResponse` with training metadata compatibility
- Implement JSON serialization/deserialization
- Use UUIDs for all profile IDs
- Follow existing naming conventions (snake_case)

### Success Criteria
- [ ] All data structures properly typed with dataclasses
- [ ] Serialization/deserialization to/from JSON works
- [ ] Validates against existing AgentResponse interface (src/agents/utils.py:18)
- [ ] Comprehensive unit tests with edge cases
- [ ] Zero impact on existing AgentResponse usage

### Critical Integration Points
- Must work with existing `AgentResponse` in `src/agents/utils.py:18`
- Should leverage `ConfigManager` pattern from `src/config/settings.py`
- Integrate with `AgentLogger.log_response()` method

### Build-With-Review Cycle
1. **Analysis**: Read existing AgentResponse and related code
2. **Design**: Create data structure specifications
3. **Implementation**: Write dataclasses with full typing
4. **Testing**: Create comprehensive unit tests
5. **Integration**: Validate compatibility with existing code
6. **Review**: Self-assess completeness and quality

### Confidence Scoring
Report confidence (0.0-1.0) on:
- Backward compatibility preservation
- JSON serialization robustness
- Integration with existing logging
- Test coverage completeness

---

## Subagent 2: Database Schema Designer
**Task**: T1.2 - Create Database Schema  
**Dependencies**: `[]` (No dependencies, can start immediately)  
**File**: `src/training/schema.sql` and migration scripts

### Mission
Design and implement the PostgreSQL/Supabase database schema for training data persistence, ensuring optimal performance and compatibility with existing infrastructure.

### Implementation Requirements
- Create training tables following the plan specifications
- Add proper indexes for expected query patterns
- Include JSONB indexes for prompt content searches
- Create migration and rollback scripts
- Use timestamptz consistently with existing patterns

### Success Criteria
- [ ] Schema creates successfully on clean database
- [ ] All foreign key relationships properly defined
- [ ] Indexes optimized for expected query patterns
- [ ] Migration scripts handle existing data gracefully
- [ ] Rollback scripts tested and working

### Critical Integration Points
- Leverage `DatabaseConfig` class in `src/config/settings.py:33`
- Use existing environment variable `SUPABASE_DB_URL`
- Plan for existing Supabase setup compatibility

### Build-With-Review Cycle
1. **Analysis**: Study existing database patterns and configuration
2. **Design**: Create schema with performance considerations
3. **Implementation**: Write SQL schema and migration scripts
4. **Testing**: Test on clean database instance
5. **Integration**: Validate with existing database config
6. **Review**: Performance analysis and rollback testing

### Confidence Scoring
Report confidence (0.0-1.0) on:
- Schema completeness and correctness
- Index optimization effectiveness
- Migration script reliability
- Rollback procedure safety

---

## Subagent 3: Profile Store Engineer
**Task**: T1.3 - Implement PromptProfileStore  
**Dependencies**: `[T1.1, T1.2]` (Requires data structures and schema)  
**File**: `src/training/profile_store.py`

### Mission
Build the critical database persistence layer for prompt profiles with proper error handling, connection pooling, and transaction management.

### Implementation Requirements
- Implement all CRUD operations per the interface specification
- Use existing database configuration patterns
- Implement connection pooling for concurrent access
- Add comprehensive error handling with logging
- Use transactions for multi-table operations

### Success Criteria
- [ ] All CRUD operations work correctly
- [ ] Connection pooling implemented properly
- [ ] Error handling with proper logging
- [ ] Comprehensive integration tests
- [ ] Performance tests for concurrent access

### Critical Integration Points
- Use `get_config().get_database_url()` for connection
- Leverage existing logging patterns from `AgentLogger`
- Follow error handling patterns from `src/config/settings.py`

### Build-With-Review Cycle
1. **Analysis**: Study existing database usage patterns
2. **Design**: Plan connection management and error handling
3. **Implementation**: Build store with full CRUD operations
4. **Testing**: Create integration and performance tests
5. **Integration**: Validate with existing configuration
6. **Review**: Test error conditions and concurrent access

### Confidence Scoring
Report confidence (0.0-1.0) on:
- CRUD operation reliability
- Error handling robustness
- Performance under load
- Integration completeness

---

## Subagent 4: Trainer Wrapper Architect
**Task**: T1.4 - Create HegelTrainer Wrapper (Read-Only Mode)  
**Dependencies**: `[T1.3]` (Requires profile store)  
**File**: `src/training/hegel_trainer.py`

### Mission
Create the main training wrapper that preserves existing functionality while adding training capabilities. Critical that grad=False mode produces identical results to current system.

### Implementation Requirements
- Wrap existing agents without modifying their classes
- Ensure grad=False behaves exactly like current system
- Integrate with existing DebateSession and AgentLogger
- Maintain exact API compatibility
- Add comprehensive logging of all operations

### Success Criteria
- [ ] `grad=False` mode produces identical results to current system
- [ ] No performance regression in inference mode
- [ ] Proper integration with existing `BasicWorkerAgent` and `BasicReviewerAgent`
- [ ] All existing tests still pass
- [ ] Comprehensive logging of all operations

### Critical Integration Points
- Wrap existing `BasicWorkerAgent` from `src/agents/worker.py`
- Integrate with `DebateSession` from `src/debate/session.py`
- Use `AgentLogger` patterns for comprehensive logging

### Build-With-Review Cycle
1. **Analysis**: Study existing agent and debate patterns
2. **Design**: Plan wrapper architecture for zero impact
3. **Implementation**: Build wrapper with composition approach
4. **Testing**: Validate identical behavior to existing system
5. **Integration**: Test with all existing functionality
6. **Review**: Performance analysis and compatibility verification

### Confidence Scoring
Report confidence (0.0-1.0) on:
- Backward compatibility guarantee
- Performance preservation
- Integration completeness
- API compatibility maintenance

---

## Subagent 5: Agent Factory Engineer
**Task**: T1.5 - Create Agent Factory with Profile Configuration  
**Dependencies**: `[T1.4]` (Requires trainer wrapper)  
**File**: `src/training/agent_factory.py`

### Mission
Build the configurable agent factory that can create agents with custom prompts while preserving all existing functionality and initialization patterns.

### Implementation Requirements
- Create agents with custom prompts from profiles
- Preserve all existing agent functionality
- Use composition or monkey patching (not inheritance)
- Handle Gemini API configuration through existing patterns
- Maintain agent logging with existing AgentLogger

### Success Criteria
- [ ] Agents created with custom prompts behave correctly
- [ ] All existing agent functionality preserved
- [ ] Dynamic prompt injection works reliably
- [ ] Agents properly configured with temperature, max_tokens
- [ ] Factory maintains agent lifecycle properly

### Critical Integration Points
- Don't modify existing agent classes
- Preserve agent initialization patterns from existing code
- Handle Gemini API configuration through existing `get_config()`
- Maintain agent logging with existing `AgentLogger` patterns

### Build-With-Review Cycle
1. **Analysis**: Study existing agent initialization and configuration
2. **Design**: Plan factory pattern for dynamic configuration
3. **Implementation**: Build factory with proper lifecycle management
4. **Testing**: Test agent creation and behavior preservation
5. **Integration**: Validate with existing agent patterns
6. **Review**: Verify no breaking changes to existing functionality

### Confidence Scoring
Report confidence (0.0-1.0) on:
- Agent functionality preservation
- Dynamic configuration reliability
- Lifecycle management correctness
- Integration completeness

---

## Orchestrator Reporting Protocol

### Task Completion Report Format
Each subagent must submit a structured completion report in JSON format:

```json
{
  "agent_id": "data_structures_architect",
  "task_id": "T1.1",
  "status": "completed" | "partial" | "failed",
  "confidence_scores": {
    "overall_completion": 0.85,
    "backward_compatibility": 0.90,
    "test_coverage": 0.80,
    "integration_quality": 0.75
  },
  "deliverables": [
    {
      "file": "src/training/data_structures.py",
      "status": "completed",
      "lines_of_code": 245,
      "test_coverage": 0.95
    }
  ],
  "success_criteria_met": [
    "All data structures properly typed with dataclasses",
    "JSON serialization/deserialization implemented"
  ],
  "success_criteria_partial": [
    "Integration with AgentLogger - basic implementation done, edge cases need review"
  ],
  "success_criteria_failed": [],
  "issues_discovered": [
    "AgentResponse datetime handling inconsistency - needs standardization",
    "Potential UUID collision in concurrent scenarios - added uuid4() with timestamp"
  ],
  "persistent_issues": [
    "Complex metadata serialization may impact performance - monitoring needed"
  ],
  "implementation_notes": [
    "Used dataclass inheritance for backward compatibility",
    "Added validation decorators for robust serialization",
    "Comprehensive unit tests cover all edge cases"
  ],
  "review_recommendations": [
    "Review datetime handling standardization across codebase",
    "Performance test large metadata objects",
    "Validate UUID generation strategy under load"
  ],
  "next_dependencies": [
    "T1.3 can proceed with these data structures",
    "Consider integration testing with existing AgentResponse usage"
  ]
}
```

### Completion Scoring System
Rate each aspect from 0.0 to 1.0:
- **0.0-0.3**: Significant issues, needs major rework
- **0.4-0.6**: Functional but has known issues needing attention
- **0.7-0.8**: Good implementation with minor issues
- **0.9-1.0**: Excellent implementation, ready for production

### Uncertainty and Issue Documentation
Subagents must be honest about:
- **Partial implementations**: What works vs what needs more work
- **Known issues**: Document all discovered problems
- **Confidence levels**: Real assessment of implementation quality
- **Review needs**: Specific areas requiring closer examination

### Communication Protocol
1. **Progress Updates**: Every 30 minutes during active work
2. **Blocking Issues**: Immediate escalation if unable to proceed
3. **Completion Reports**: Full structured report upon task completion
4. **Cross-Dependencies**: Notify when outputs are ready for dependent tasks

## Success Metrics
- All Phase 1 Foundation Layer tasks completed with >0.7 overall confidence
- Zero regression in existing functionality (all tests pass)
- Clear documentation of implementation decisions and trade-offs
- Honest assessment of areas needing further review or improvement
- Structured handoff to enable Phase 2 work to begin immediately
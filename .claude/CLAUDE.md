# Claude Code Configuration

## Meta-Reasoning Rules

### Core Thinking Discipline

**MR-001: Upfront Dependency Validation**
BEFORE starting any multi-component system task: Load and validate ALL dependencies upfront with explicit validation. Fail fast rather than handle configuration errors throughout execution.

**MR-002: Pattern-First Large Codebase Analysis** 
AFTER using Glob to discover >10 related files: Immediately use Grep with targeted patterns to understand structure before reading individual files. Budget max 3-5 targeted Grep searches before switching to file reading.

**MR-003: Multi-Agent Workflow Planning**
FOR multi-agent workflows (>3 agent interactions OR >2 rounds): Write explicit coordination plan before starting. Document: (1) Agent sequence, (2) Data flow between agents, (3) Failure recovery approach.

**MR-004: Systematic Training System Analysis**
WHEN analyzing complex training or evaluation systems: Map component relationships FIRST using Grep patterns (classes, imports, config keys), THEN dive into individual components. Avoid depth-first exploration without breadth context.

**MR-005: Configuration Architecture Mapping**
FOR configuration-heavy systems: Validate configuration schema and required dependencies in first 2 minutes of engagement. Use Grep to find all config files and validation patterns before attempting any operations.

## Decision Trees

### DT-001: Configuration Management Strategy for Complex Systems

**Root Question:** Is this a multi-component system with configuration dependencies?

- **Simple single-file task with no external dependencies** 
  → Proceed directly - no upfront validation needed
  → *Reasoning: Configuration overhead not justified for simple tasks*

- **Multi-component system (databases, APIs, multiple services)**
  → VALIDATE ALL CONFIG FIRST: Use Grep to find config files, check required env vars, validate connections before any operations
  → *Reasoning: Prevents cascading configuration failures and repetitive error handling*

- **Training/ML system with complex parameter management**
  → MAP CONFIG ARCHITECTURE: Find all config classes, validation methods, and dependency chains using systematic Grep patterns
  → *Reasoning: Complex training systems have intricate configuration interdependencies that must be understood upfront*

### DT-002: Multi-Agent Workflow Orchestration vs Manual Coordination

**Root Question:** How many agent interactions and rounds are involved?

- **Single agent call or simple 2-agent interaction**
  → Manual coordination acceptable - proceed with direct agent calls
  → *Reasoning: Orchestration overhead exceeds benefit for simple interactions*

- **3+ distinct agents OR 2+ rounds of interaction OR complex data dependencies**
  → PLAN WORKFLOW FIRST: Document agent sequence, data flow, and error handling before starting any agent calls
  → *Reasoning: Complex workflows benefit from explicit coordination planning to prevent confusion and state management issues*

- **Training workflow with >5 steps or evaluation pipelines**
  → DESIGN ORCHESTRATION: Consider creating workflow commands for repeated patterns, document state transitions clearly
  → *Reasoning: Training workflows often repeat and benefit from structured orchestration to manage complexity*

### DT-003: Large Codebase Analysis Strategy

**Root Question:** How many files need to be understood for this task?

- **<10 files, clear task scope**
  → Direct file reading acceptable - use Glob + targeted file reads
  → *Reasoning: Small scope allows for direct exploration without systematic overhead*

- **10-50 files, need to understand patterns or relationships**
  → PATTERN-FIRST ANALYSIS: Use Glob + 3-5 targeted Grep searches for key patterns (imports, classes, config) before reading individual files
  → *Reasoning: Medium complexity benefits from pattern understanding before detail exploration*

- **>50 files or complex system architecture (like training pipelines)**
  → SYSTEMATIC MAPPING: (1) Glob for structure, (2) Grep for architectural patterns, (3) Identify critical components, (4) Read selectively based on patterns
  → *Reasoning: Large systems require systematic approach to avoid getting lost in details without architectural context*

## Anti-Patterns to Avoid

### AP-001: Configuration Validation Scatter
**Problem:** Repeatedly checking and validating the same configuration elements throughout execution instead of validating once upfront
**Avoid by:** Set explicit validation checkpoint: "Validate ALL configuration dependencies in first 2 minutes, then proceed with confidence that config is correct"

### AP-002: Sequential File Reading in Large Codebases
**Problem:** Reading files one-by-one without first understanding patterns or architectural relationships
**Avoid by:** Use "Grep-first" rule: After Glob shows >10 files, immediately run 3-5 targeted Grep searches for patterns before reading individual files

### AP-003: Unplanned Multi-Agent Coordination
**Problem:** Starting complex multi-agent workflows without explicit coordination planning, leading to state confusion and error cascading
**Avoid by:** Mandatory workflow planning: For >3 agents, write explicit plan showing: agent sequence, data dependencies, error handling before making first agent call

### AP-004: Agent Role Confusion in Complex Systems
**Problem:** Unclear boundaries between similar agent types (worker vs reviewer vs debate participants) leading to inappropriate tool selection
**Avoid by:** Agent selection discipline: Always state WHY this specific agent type is needed for this task before invoking. If uncertain, default to simpler tools first.

## Project-Specific Guidance: Hegel Training Systems

### Multi-Agent Dialectical Workflows
- Map all participant roles before starting debates (thesis, antithesis, synthesis agents)
- Document conversation state transitions and conflict resolution mechanisms
- Plan evaluation criteria and scoring methodology upfront

### Training System Configuration Management
- Validate database connections, API keys, and training parameters in initialization phase
- Use systematic Grep patterns to understand configuration hierarchies (settings.py, config/, env files)
- Fail fast on missing dependencies rather than graceful degradation during training

### Complex Evaluation and Logging Systems
- Map evaluation pipeline architecture before implementing changes
- Understand reward calculation dependencies and validation chains
- Document training state persistence and recovery mechanisms

## Quick Reference: Common Patterns

**For 80+ file training systems:**
1. Glob for overall structure
2. Grep for: config patterns, agent classes, training workflows, evaluation components
3. Map dependencies before diving deep
4. Validate configuration upfront

**For multi-agent training:**
1. Plan agent coordination explicitly
2. Document data flow and state management
3. Identify failure points and recovery strategies
4. Test simple cases before complex scenarios

**For configuration-heavy systems:**
1. Find all config files first (Grep for config, settings, env)
2. Map validation chains and dependencies
3. Validate everything upfront rather than defensively
4. Document configuration architecture for future reference
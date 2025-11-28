---
description: Transform implementation plan into dependency-aware, execution-ready todo list with parallelization opportunities
argument-hint: <path-to-plan.md>
allowed-tools: Read, Write, Grep, Glob, Bash
model: claude-sonnet-4-20250514
---

You are transforming an implementation plan into a detailed, dependency-aware todo list that developers can execute efficiently.

**Input**: Implementation plan at "$ARGUMENTS"
**Output**: ~/documentation/implementation-todo.md

# Phase 1: Deep Analysis & Investigation

## 1.1 Read and Parse the Implementation Plan

First, thoroughly read the implementation plan:
- Read the entire plan document
- Identify all sections, goals, and requirements
- Extract mentioned files, components, and systems
- Note stated success criteria and constraints
- Identify any warnings, cautions, or "be careful of" notes
- Extract any explicit dependencies mentioned in the plan

## 1.2 Investigate the Codebase

**CRITICAL**: Complete all investigation NOW before creating todos.

For each file/component mentioned in the plan:
1. **Locate files**: Use Grep/Glob to find all relevant files
2. **Read key files**: Understand current implementation state
3. **Check patterns**: Identify existing architectural patterns
4. **Find dependencies**: Discover what imports/uses what
5. **Verify test coverage**: Check for existing tests

**Investigation checklist:**
- [ ] Located all mentioned files and verified they exist
- [ ] Read key files to understand current state
- [ ] Identified integration points between components
- [ ] Found existing patterns to follow (auth, error handling, etc.)
- [ ] Checked test structure and coverage
- [ ] Identified shared resources (files multiple steps will touch)

**Output your findings** in a structured analysis before proceeding.

# Phase 2: Dependency Analysis & Graph Construction

## 2.1 Identify All Steps

Extract or infer all implementation steps from the plan. Each step should:
- Have a clear, concrete action
- Produce a specific outcome
- Be verifiable when complete

## 2.2 Build Dependency Graph

For each step, determine:

**Technical dependencies** (code-level):
- Does step B need output from step A?
- Does step B modify files that step A creates?
- Does step B assume step A's changes are in place?

**Logical dependencies** (order-based):
- Must step A complete before B can start for architectural reasons?
- Does step B integrate/test step A's work?
- Does step B build upon step A's foundation?

**Shared resources** (conflict-based):
- Do steps touch the same files? (may need to be sequential)
- Do steps modify the same database schema? (must be sequential)
- Do steps work on independent modules? (can be parallel)

**Classification rules:**
- **Independent**: No dependencies, can start immediately
- **Sequential**: Must follow specific step(s)
- **Parallel-safe**: Can run concurrently with other steps
- **Critical path**: Steps that determine minimum completion time

## 2.3 Calculate Parallelization Opportunities

Identify:
- Which steps can be done simultaneously
- What the critical path is (longest chain of sequential steps)
- Natural parallel phases (e.g., frontend + backend work)
- Bottlenecks (steps many others depend on)

# Phase 3: Todo Generation

## 3.1 Output Format Structure

Generate the todo list with this exact structure:

```markdown
# Implementation Todo List

**Generated from**: [path to plan]
**Generated at**: [timestamp]
**Total steps**: [number]
**Critical path**: Steps X → Y → Z (est. [time if mentioned])
**Parallel opportunities**: [number] steps can run concurrently

---

## Dependency Graph Overview

[Visual representation showing the flow]
Example:
```
[1] → [3] → [7] → [9]
       ↓     ↓
[2]   [4,5,6] [8] (parallel groups)
```
```

---

## Execution Todos

[Organize by phase/logical grouping]

### Phase N: [Phase Name]

#### [#] [Clear action verb + what to do]

**Alignment with plan**:
- References specific section(s) of the implementation plan
- Explains how this step fulfills plan requirements

**Success criteria**:
- Specific, verifiable outcomes
- What "done" looks like
- Tests or validation steps

**Implementation details**:
- Files to create/modify (with paths)
- Key functions/classes to implement
- Specific patterns or approaches to use
- Any code snippets or pseudocode if helpful

**Cautions & pitfalls**:
- What could go wrong
- Common mistakes to avoid
- Regression risks
- Edge cases to handle
- Security considerations
- Performance considerations

**Dependencies**: [Depends on: #1, #3] or "None (start immediately)"

**Parallelization**: 
- "Can parallelize with: [#2, #5]" or
- "Critical path - cannot parallelize" or
- "Must complete before: [#7, #8]"

**Estimated complexity**: [Low/Medium/High] (if you can infer from plan)

---
```

## 3.2 Todo Content Requirements

Each todo must include:

1. **Concrete action**: Use specific verbs (Create, Refactor, Add, Update, Implement, Wire, Test)

2. **Alignment section**: 
   - Quote or reference the relevant plan section
   - Explain the connection: "This implements requirement X by..."

3. **Success criteria**: 
   - Be specific: not "update auth" but "Auth middleware accepts strategy parameter and maintains backward compatibility"
   - Include verification: "Run test suite and verify no regressions"

4. **Implementation details**:
   - Specific file paths
   - Function/class names where helpful
   - Patterns to follow from existing code
   - Example: "In src/middleware/auth.ts, modify authenticateRequest() to accept strategy parameter"

5. **Cautions**:
   - Extract warnings from the plan
   - Add your own based on code analysis
   - Be specific: "Don't change session behavior in this step - that comes in step 7"
   - Include: regressions, edge cases, security, performance, compatibility

6. **Dependencies**:
   - Use format: `[Depends on: #1, #3]`
   - Explain WHY: "Depends on #1 because we need the auth strategy interface defined"
   - For independent steps: "None (start immediately)"

7. **Parallelization guidance**:
   - Explicit: "Can parallelize with: #4, #5, #6"
   - Or: "Critical path step - blocks #7, #8, #9"
   - Or: "Sequential - must complete before #6"

## 3.3 Organization Strategy

Group todos logically:
- By phase (Foundation → Implementation → Integration → Testing)
- By component (Database → API → Frontend)
- By dependency level (Independent → Sequential)

Within each group:
- Order by dependency (prerequisites first)
- Mark parallel groups clearly
- Show critical path visually

## 3.4 Execution Strategy Section

At the end, add:

```markdown
## Recommended Execution Strategy

**Fastest parallel path**:
1. [Describe optimal parallel execution]
2. [Show which steps to batch together]
3. [Indicate critical path steps that gate others]

**Total time estimate**: [if you can infer from plan/complexity]

**Risk mitigation**:
- [High-risk steps that need extra care]
- [Steps that may reveal issues requiring iteration]
- [Rollback strategy if needed]

**Testing strategy**:
- [When to run tests]
- [What to test after each phase]
- [Integration testing approach]
```

# Phase 4: Write Output

Write the complete todo list to: `~/documentation/implementation-todo.md`

**Quality checks before writing**:
- [ ] Every step has all required sections
- [ ] Dependencies are correctly numbered and explained
- [ ] Parallelization opportunities are clearly marked
- [ ] Critical path is identified
- [ ] Success criteria are specific and verifiable
- [ ] Cautions address real risks from code analysis
- [ ] File paths and specifics are accurate

# Execution Notes

**Be thorough in Phase 1**: The quality of your investigation determines the quality of the todos. Don't guess about code structure - actually read and understand it.

**Be explicit about dependencies**: When you say "Depends on #3", explain WHY - what output from #3 does this step need?

**Be specific in implementation details**: Give enough detail that a developer can execute without re-analyzing. Include file paths, function names, patterns.

**Be realistic about cautions**: Don't just list generic risks. Based on your code analysis, what SPECIFIC pitfalls exist in THIS codebase?

**Be clear about parallelization**: Make it obvious which steps can happen simultaneously. This is the key value-add of this command.

---

Now begin Phase 1: Read the implementation plan at "$ARGUMENTS" and investigate the codebase thoroughly.
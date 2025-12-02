---
description: Analyze implementation plan and code to generate detailed, dependency-aware todo list
argument-hint: [path-to-plan-file]
allowed-tools: Read, Write, Grep, Glob, Bash(git *)
---

You are creating a detailed implementation todo list from an existing implementation plan.

# Input

The implementation plan is located at: `$ARGUMENTS`

If no argument provided, raise error.

# Phase 1: Deep Analysis

## 1.1 Understand the Plan

**Read the implementation plan thoroughly:**
- What is the overall goal and success criteria?
- What are the key components or features to implement?
- What files/modules are mentioned?
- What are the stated risks, assumptions, or constraints?
- What are the acceptance criteria or verification steps?

**Extract structured information:**
- List of all concrete actions mentioned
- Files/modules that will be touched
- Technical requirements or patterns to follow
- Edge cases or special handling mentioned
- Testing or validation requirements

## 1.2 Investigate the Codebase

**For each file/module mentioned in the plan:**
- Read the current implementation
- Understand the existing patterns and architecture
- Identify dependencies (imports, function calls, data flows)
- Note any related files that weren't mentioned in the plan
- Look for existing tests or documentation

**Use Grep/Glob to discover:**
- Similar patterns already in use
- Related functionality that might be affected
- Existing utilities or helpers that could be reused
- Test files that will need updates

**Understand the technical context:**
- What frameworks/libraries are in use?
- What are the coding conventions?
- How is error handling done?
- What's the testing approach?

## 1.3 Analyze Dependencies

**For each step in the plan, determine:**
- What must exist before this step can be done?
- What other steps does this step enable?
- Can this be done in parallel with other steps?
- Are there any circular dependencies to break?

**Consider technical dependencies:**
- Files that must be modified before others
- Data structures that must be defined first
- Interfaces or contracts that must be established
- Database migrations or schema changes
- Configuration or environment changes

# Phase 2: Structure the Todo List

## 2.1 Create Logical Groupings

Organize steps into coherent phases or modules:
- Infrastructure/setup tasks (can often be done first)
- Core functionality (usually has the most dependencies)
- Integration tasks (require core pieces to exist)
- Testing/validation (requires implementation to exist)
- Documentation/cleanup (can often be done last)

## 2.2 Build Dependency Graph

For each step, explicitly state:
- **Step ID**: Unique identifier (e.g., STEP-01, STEP-02)
- **Dependencies**: List of step IDs that must complete first
- **Blocks**: List of step IDs that wait for this one
- **Parallelizable with**: Steps that can be done concurrently

## 2.3 Add Implementation Details

For each step, provide:

**What to do:**
- Specific files to modify
- Specific functions/classes to add or change
- Specific tests to write

**How it aligns with the plan:**
- Quote or reference the relevant part of the plan
- Explain how this step contributes to the goal
- Note which success criteria this helps satisfy

**Success criteria:**
- How do you know this step is done?
- What should work after this step?
- What tests should pass?

**Things to be careful of:**
- Edge cases to handle
- Regressions to avoid
- Performance considerations
- Security implications
- Breaking changes or migrations needed

**Verification approach:**
- Manual testing steps
- Automated tests to run
- Integration points to check

# Phase 3: Generate the Todo List

Write to `~/documentation/implementation-todo.md` with this structure:

```markdown
# Implementation Todo List

**Generated from:** [path to plan]
**Generated at:** [timestamp]

## Overview

[2-3 sentence summary of what we're implementing]

## Success Criteria

[List the key success criteria from the plan]

## Execution Strategy

### Parallel Tracks

This implementation has [N] parallel tracks that can be worked on simultaneously:

1. **Track A: [Name]** - Steps STEP-01, STEP-03, STEP-07
2. **Track B: [Name]** - Steps STEP-02, STEP-05, STEP-09
3. **Track C: [Name]** - Steps STEP-04, STEP-06, STEP-08

### Critical Path

The critical path (longest sequential chain) is: STEP-01 → STEP-03 → STEP-06 → STEP-10

## Implementation Steps

### Phase 1: [Phase Name]

#### STEP-01: [Clear, action-oriented title]

**Status:** ⏳ Not Started | 🏃 In Progress | ✅ Complete

**Dependencies:** None (or list step IDs)
**Blocks:** STEP-03, STEP-07
**Can parallelize with:** STEP-02, STEP-04

**What to do:**
- Modify `path/to/file.ts`
- Add function `functionName()` that handles [specific behavior]
- Update interface `InterfaceName` to include [specific fields]

**How this aligns with the plan:**
> [Quote from plan]: "We need to establish the authentication middleware..."

This step implements the foundation for the authentication system mentioned in the plan's Phase 1. It satisfies success criterion #2: "Secure API endpoints with proper auth."

**Success criteria:**
- [ ] File `path/to/file.ts` exists with exported `functionName()`
- [ ] Interface includes required fields with correct types
- [ ] Unit tests pass: `npm test -- auth.middleware.test.ts`
- [ ] Manual verification: Can call function without errors

**Things to be careful of:**
⚠️ **Edge Cases:**
- Handle null/undefined input gracefully
- Consider what happens with expired tokens
- Don't break existing auth flow in `legacy-auth.ts`

⚠️ **Performance:**
- This runs on every request - keep it fast
- Avoid N+1 queries in token validation

⚠️ **Security:**
- Never log sensitive data (tokens, passwords)
- Use constant-time comparison for tokens
- Validate all inputs before processing

**Verification approach:**
1. Run unit tests: `npm test -- auth.middleware.test.ts`
2. Manual test: Start dev server, make authenticated request
3. Check that existing tests still pass: `npm test`
4. Review git diff to ensure no unintended changes

**Estimated effort:** [2 hours | 1 day | etc.]

---

#### STEP-02: [Next step]

[Same structure as above]

---

### Phase 2: [Next Phase Name]

[Continue with remaining steps]

---

## Dependency Visualization

```
STEP-01 (Foundation)
  ├── STEP-03 (Core Logic)
  │     ├── STEP-06 (Integration)
  │     │     └── STEP-10 (Testing)
  │     └── STEP-07 (Error Handling)
  └── STEP-07 (Error Handling)

STEP-02 (Database Schema)
  └── STEP-05 (Data Access Layer)
        └── STEP-09 (API Endpoints)

STEP-04 (Config & Setup)
  └── STEP-06 (Integration)
  └── STEP-08 (Documentation)
```

## Risk Mitigation

[List key risks from the plan and how specific steps address them]

## Testing Strategy

[Overall testing approach and which steps include which types of tests]

## Notes

- If you discover new dependencies while implementing, update this file
- Mark steps complete as you finish them
- If you need to split a step, create STEP-XX-A, STEP-XX-B, etc.
```

# Phase 4: Final Review

Before writing the file:

1. **Completeness check:**
   - Does every action from the plan have a corresponding step?
   - Are all files mentioned in the plan covered?
   - Did we include testing and documentation steps?

2. **Dependency check:**
   - Are there any circular dependencies?
   - Is the critical path reasonable?
   - Can we truly parallelize what we marked as parallel?

3. **Clarity check:**
   - Is each step specific and actionable?
   - Would a developer know exactly what to code?
   - Are the "careful of" warnings actually helpful?

4. **Alignment check:**
   - Does each step clearly tie back to the plan?
   - Do all steps together achieve the success criteria?
   - Did we miss any constraints or requirements?

# Execution

1. Read and analyze the implementation plan
2. Investigate the codebase thoroughly
3. Build the dependency graph mentally
4. Generate the structured todo list
5. Write to ~/documentation/implementation-todo.md
6. Report completion with summary

**Important:**
- Be specific: name exact files, functions, and lines where relevant
- Be realistic: flag where more investigation is needed
- Be helpful: the warnings should come from real code analysis, not generic advice
- Be structured: the dependency graph should enable optimal parallel work

Now begin Phase 1: Find and analyze the implementation plan.
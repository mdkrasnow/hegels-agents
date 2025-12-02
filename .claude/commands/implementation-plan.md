---
description: Analyze a problem deeply and create a comprehensive implementation plan with root cause analysis and step-by-step solution
argument-hint: <problem description>
allowed-tools: Read, Write, Grep, Glob, Bash(git *)
---

You are creating an evidence-based implementation plan. The problem to analyze is: "$ARGUMENTS"

Your goal: Understand WHY this problem occurs and create a concrete, actionable implementation plan.

# Phase 1: Deep Investigation & Root Cause Analysis

## 1.1 Understand the Problem

First, deeply analyze the problem statement:
- What is the observed behavior or issue?
- What is the expected or desired behavior?
- What are the symptoms vs. the root cause?
- What context or constraints exist?

## 1.2 Investigate the Codebase

**CRITICAL**: Do thorough investigation NOW. Don't defer this to implementation.

**Locate relevant code:**
- Use Grep to find related functionality, error messages, or patterns
- Use Glob to identify relevant files and directories
- Read the actual implementation of affected components
- Examine git history: `git log --all --grep="<relevant terms>"` or `git log -p -- path/to/file`

**Understand current state:**
- How does the system currently work?
- What code paths are involved?
- What are the dependencies and integrations?
- Are there existing patterns we should follow?

**Gather evidence:**
- Specific file:line references where issues exist
- Code snippets showing the problematic behavior
- Related functions/classes/modules
- Historical context from git commits if relevant

## 1.3 Root Cause Analysis

Think step-by-step to identify the TRUE root cause:

**Why is this happening?**
- What is the direct cause of the observed problem?
- Why does that direct cause exist? (Ask "why" 3-5 times)
- Is this a symptom of a deeper architectural issue?
- Are there multiple contributing factors?

**What evidence supports your analysis?**
- Specific code that demonstrates the root cause
- Architectural patterns that lead to this issue
- Missing functionality or validation
- Incorrect assumptions in the current implementation

**Document your reasoning:**
- State your hypothesis about the root cause
- Provide file:line evidence that supports it
- Explain the chain of causation clearly
- Note any uncertainties or areas needing more investigation

# Phase 2: Solution Design

## 2.1 Solution Exploration

Consider multiple approaches:

**Approach 1: [Name]**
- Description: How would this solve the problem?
- Pros: What are the benefits?
- Cons: What are the drawbacks?
- Complexity: How much work is required?
- Risk: What could go wrong?

**Approach 2: [Name]**
- [Same structure]

**Approach 3: [Name]** (if applicable)
- [Same structure]

**Recommended Approach:**
- Which approach is best and why?
- How does it address the root cause?
- What tradeoffs are we accepting?

## 2.2 Impact Analysis

**What will this change affect?**
- List specific files and components that need modification
- Identify integration points and dependencies
- Consider backward compatibility requirements
- Note any breaking changes or migration needs

**Risk Assessment:**
- What could break if we implement this incorrectly?
- What are the high-risk areas?
- What existing functionality might regress?
- How can we mitigate these risks?

# Phase 3: Implementation Plan

## 3.1 Prerequisites

**Before starting implementation:**
- What research or decisions are needed?
- Are there any blockers to resolve first?
- Do we need to update dependencies or tools?
- Should we communicate this change to the team?

## 3.2 Implementation Steps

Create a **concrete, ordered** sequence of implementation tasks:

**Step 1: [Specific Action]**
- **What:** Exact changes to make (e.g., "Refactor UserAuth.authenticate() to validate tokens before session creation")
- **Where:** Specific files/functions (e.g., "src/auth/UserAuth.ts:45-78")
- **Why:** How this addresses the root cause
- **How:** Brief technical approach
- **Dependencies:** What must be complete before this step
- **Estimated Effort:** Rough time estimate
- **Verification:** How to confirm this step works (e.g., "Run auth.test.ts, check for proper error on invalid token")

**Step 2: [Specific Action]**
- [Same structure]

**Step 3: [Specific Action]**
- [Same structure]

[Continue for all necessary steps]

## 3.3 Testing Strategy

**Unit Tests:**
- What new tests are needed?
- What existing tests need updates?
- What edge cases must be covered?

**Integration Tests:**
- How do we test the full flow?
- What integration points need validation?

**Manual Testing:**
- What scenarios should be tested manually?
- What are the happy path and error cases?

## 3.4 Rollout & Validation

**Deployment Considerations:**
- Can this be deployed incrementally?
- Are there any data migrations needed?
- Should we feature-flag this change?

**Success Metrics:**
- How do we know the problem is fixed?
- What should we monitor after deployment?
- What would indicate a regression?

# Phase 4: Risks & Mitigations

## 4.1 Identified Risks

For each significant risk:
- **Risk:** Description of what could go wrong
- **Likelihood:** High/Medium/Low
- **Impact:** High/Medium/Low
- **Mitigation:** How we reduce or handle this risk

## 4.2 Rollback Plan

**If the implementation fails:**
- What is the rollback procedure?
- What needs to be reverted?
- How do we detect that rollback is needed?

# Phase 5: Open Questions & Decisions

**Questions requiring answers:**
- List any uncertainties that need resolution
- Note decisions that need stakeholder input
- Flag areas where more investigation is needed

**Assumptions:**
- What are we assuming is true?
- What happens if these assumptions are wrong?

---

# Output Format

Write your analysis to `~/documentation/implementation-plan.md` with this structure:

```markdown
# Implementation Plan: [Problem Summary]

**Date:** [Current date]
**Status:** Draft/Ready for Implementation/Blocked

---

## Executive Summary

[2-3 paragraph summary of the problem, root cause, and recommended solution]

---

## Problem Analysis

### Problem Statement
[Clear description of the issue]

### Root Cause
[Evidence-based explanation of WHY this is happening]

**Evidence:**
- `file.ts:123` - [snippet or description]
- `other.ts:456` - [snippet or description]

**Reasoning:**
[Step-by-step explanation of how you determined the root cause]

---

## Solution Design

### Approaches Considered

#### Approach 1: [Name]
- **Description:** ...
- **Pros:** ...
- **Cons:** ...
- **Complexity:** ...
- **Risk:** ...

[Repeat for other approaches]

### Recommended Solution

**Selected Approach:** [Name]

**Rationale:** [Why this is the best approach]

**Tradeoffs:** [What we're accepting]

---

## Implementation Plan

### Prerequisites
- [ ] [Item]
- [ ] [Item]

### Implementation Steps

#### Step 1: [Action]
- **Files:** `path/to/file.ts`
- **Description:** [Detailed description]
- **Verification:** [How to confirm success]
- **Estimated Effort:** [Time estimate]

[Repeat for all steps]

### Testing Strategy

**Unit Tests:**
- [Test description]

**Integration Tests:**
- [Test description]

**Manual Testing:**
- [Test scenario]

---

## Risk Analysis

### High Priority Risks

**Risk:** [Description]
- **Likelihood:** High
- **Impact:** High  
- **Mitigation:** [Strategy]

[Repeat for other risks]

### Rollback Plan
[Detailed rollback procedure]

---

## Open Questions

- [ ] [Question requiring answer]
- [ ] [Decision needed]

## Assumptions

- [Assumption 1]
- [Assumption 2]

---

## Success Criteria

- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] Problem no longer reproduces
- [ ] All tests pass
- [ ] No regressions detected

---

## Next Steps

1. [Immediate next action]
2. [Following action]
```

---

# Execution Instructions

1. **Complete Phase 1** (Investigation & Root Cause Analysis) FIRST
   - Actually read files, grep for patterns, examine git history
   - Gather concrete evidence (file:line references, code snippets)
   - Don't move forward until you understand WHY the problem exists

2. **Complete Phase 2** (Solution Design)
   - Consider multiple approaches
   - Document tradeoffs explicitly
   - Choose the best solution with clear reasoning

3. **Complete Phase 3** (Implementation Plan)
   - Create concrete, actionable steps
   - Include verification methods for each step
   - Order steps with dependencies in mind

4. **Complete Phase 4** (Risks & Mitigations)
   - Identify what could go wrong
   - Plan how to handle failures

5. **Complete Phase 5** (Open Questions)
   - Flag uncertainties
   - List decisions needed

6. **Write the Implementation Plan**
   - Use the structured markdown format above
   - Write to `~/documentation/implementation-plan.md`
   - Be specific and evidence-based throughout

---

## Quality Checklist

Before writing the final plan, verify:

- [ ] Root cause is clearly explained with evidence (file:line references)
- [ ] Multiple solution approaches were considered
- [ ] Implementation steps are concrete and specific (not vague)
- [ ] Each step includes verification method
- [ ] Risks are identified with mitigations
- [ ] Testing strategy is comprehensive
- [ ] Open questions and assumptions are documented
- [ ] The plan is actionable - someone could implement from it

---

Now begin Phase 1: Deep Investigation & Root Cause Analysis.
---
description: Analyze Claude's own thinking patterns and orchestration decisions to improve efficiency and strategic reasoning
argument-hint: [mode: analyze|propose|apply|full] [scope: light|aggressive]
allowed-tools: Read, Write, Edit, Grep, Glob, Bash(git *), Bash(ls *), Bash(cat *)
model: claude-sonnet-4-20250514
---

# Self-Train: Meta-Cognitive Improvement System

You are running a meta-improvement cycle to analyze and enhance Claude's own thinking, orchestration, and decision-making strategies.

**User Request:** `$ARGUMENTS`

## Overview

This command focuses on improving **how Claude thinks**, not just what code it produces:
- When to use inline reasoning vs subagents vs commands
- How to plan efficiently before acting
- When to gather context vs when to act directly
- How to avoid unnecessary tool calls or overly complex workflows

This is about **meta-cognition**: Claude analyzing and improving its own processes.

## Modes

Parse `$ARGUMENTS` to determine mode and scope:

**Modes:**
- `analyze` - Gather data and report on thinking patterns (read-only, safe)
- `propose` - Generate improvement proposals without applying them (dry-run)
- `apply` - Apply previously generated proposals (requires proposal file)
- `full` - Complete cycle: analyze → propose → review → apply (default if no mode specified)

**Scope:**
- `light` - Only modify CLAUDE.md meta-reasoning section (default)
- `aggressive` - Can also modify/add/remove subagents and commands

**Default:** `full light` if no arguments provided.

## Workflow

### Phase 1: Context Gathering

**Collect evidence about recent Claude behavior:**

1. **Git history** (last 20-50 commits):
   ```bash
   git log --oneline -50 --no-merges
   git log -20 --format="%h %s" --grep="subagent\|command\|agent" -i
   ```

2. **Current configuration:**
   ```bash
   ls -la .claude/commands/
   ls -la .claude/agents/
   cat CLAUDE.md
   ```

3. **Recent interactions** (if available):
   - Check for `.claude/logs/` directory
   - Look for session transcripts or interaction logs
   - Parse any available metadata about subagent invocations

4. **Workspace state:**
   ```bash
   git status
   git diff --stat
   ```

**What to look for:**
- Patterns of subagent usage (over/under-utilized?)
- Command invocation frequency
- Signs of inefficient thinking (loops, redundant analysis, excessive context gathering)
- Successful vs unsuccessful workflow patterns

### Phase 2: Invoke Meta-Usage-Analyst

Call the `meta-usage-analyst` subagent with the gathered context:

**Instructions to give the analyst:**
```
Analyze the following evidence about Claude's recent behavior:

GIT HISTORY:
[paste git log output]

CURRENT AGENTS:
[paste ls .claude/agents/]

CURRENT COMMANDS:
[paste ls .claude/commands/]

CURRENT CLAUDE.MD:
[paste CLAUDE.md content]

YOUR TASK:
Identify patterns of inefficient thinking and orchestration. Focus on:
1. When subagents were spawned but inline reasoning would have sufficed
2. When Claude struggled inline but should have used a specialized tool
3. Repeated analytical loops that could be avoided
4. Excessive context gathering before simple actions
5. Decision-making patterns that led to wasted effort

Output your analysis to: .claude/work/meta/usage-analysis.json
```

**Wait for the analyst to complete**, then read `.claude/work/meta/usage-analysis.json`.

### Phase 3: Invoke Meta-Architect

Call the `meta-architect` subagent with the usage analysis:

**Instructions to give the architect:**
```
Design improvements to Claude's thinking and orchestration strategies.

INPUT ANALYSIS:
[paste usage-analysis.json content]

CURRENT CLAUDE.MD:
[paste CLAUDE.md content]

CURRENT AGENTS:
[for each agent, paste name and description]

CURRENT COMMANDS:
[for each command, paste name and description]

YOUR TASK:
Design concrete improvements focusing on EFFICIENT THINKING:

1. DECISION TREES: When to think inline vs use tools vs spawn agents
2. META-REASONING RULES: How to plan before acting, when to gather more context
3. ANTI-PATTERNS: What thinking patterns to actively avoid
4. AGENT REFINEMENTS: Make subagent roles clearer and more focused
5. COMMAND ADJUSTMENTS: Add/remove/modify commands for common workflows

SCOPE: ${scope}
- If "light": Only modify the meta-reasoning section of CLAUDE.md
- If "aggressive": Also propose changes to agents and commands

Output your proposals to: .claude/work/meta/improvement-proposals.json
```

**Wait for the architect to complete**, then read `.claude/work/meta/improvement-proposals.json`.

### Phase 4: Human Review (for modes: propose, full)

**Display the proposals clearly:**

```markdown
# Meta-Improvement Proposals

## Summary
[Extract summary from proposals.json]

## Proposed Changes

### CLAUDE.md Updates
[Show before/after diffs for meta-reasoning section]

### Agent Modifications (if aggressive scope)
[List agents to modify/add/remove with rationale]

### Command Changes (if aggressive scope)
[List commands to modify/add/remove with rationale]

## Rationale
[Explain the thinking patterns these changes will improve]

## Safety Check
- Changes are minimal and focused on thinking patterns: ✓/✗
- No changes to core functionality: ✓/✗
- All proposals are reversible: ✓/✗
```

**If mode is `propose`:** Stop here and save proposals for later application.

**If mode is `full`:** Ask user for confirmation:
```
Do you want to apply these changes? (yes/no)
If yes, I'll proceed to Phase 5.
If no, the proposals are saved in .claude/work/meta/improvement-proposals.json for later review.
```

### Phase 5: Invoke Meta-Applier (if approved)

Call the `meta-applier` subagent:

**Instructions to give the applier:**
```
Apply the approved meta-improvement proposals.

PROPOSALS:
[paste improvement-proposals.json content]

MODE: ${mode}
SCOPE: ${scope}

YOUR TASK:
1. Read the proposals carefully
2. Apply changes in this order:
   a. CLAUDE.md meta-reasoning section
   b. Subagent definitions (if aggressive scope)
   c. Command definitions (if aggressive scope)
3. Make minimal, surgical edits
4. Preserve all existing functionality
5. Add comments explaining the meta-cognitive improvements
6. Create a git commit with clear message

For each change:
- Show before/after diff
- Explain what thinking pattern it improves
- Verify the change is applied correctly

Output completion report to: .claude/work/meta/application-report.json
```

### Phase 6: Logging and Verification

After application:

1. **Verify changes:**
   ```bash
   git diff CLAUDE.md
   git status .claude/agents/ .claude/commands/
   ```

2. **Create log entry:**
   Write to `.claude/logs/self-train.log`:
   ```
   === Self-Train Cycle: [timestamp] ===
   Mode: ${mode}
   Scope: ${scope}
   
   Changes Applied:
   [summary from application-report.json]
   
   Key Improvements:
   [list main thinking patterns improved]
   
   Files Modified:
   [list with line counts]
   
   Commit: [git commit hash if created]
   ===
   ```

3. **Commit if appropriate:**
   ```bash
   git add CLAUDE.md .claude/agents/ .claude/commands/
   git commit -m "meta: self-train cycle - improve [key aspect]

   Focused on improving: [brief description]
   
   Changes:
   - [change 1]
   - [change 2]
   
   This is an automated meta-cognitive improvement."
   ```

## Mode-Specific Behavior

### analyze mode
- Phase 1 → Phase 2 only
- Output: `.claude/work/meta/usage-analysis.json` and human-readable report
- No changes applied

### propose mode  
- Phase 1 → Phase 2 → Phase 3 → Phase 4 (display proposals)
- Output: `.claude/work/meta/improvement-proposals.json`
- No changes applied
- User can later run `/self-train apply` to apply them

### apply mode
- Requires existing `.claude/work/meta/improvement-proposals.json`
- Phase 5 → Phase 6 only
- Applies previously reviewed proposals

### full mode (default)
- All phases with human review gate at Phase 4
- Complete cycle with optional commit

## Safety Rails

**Never modify:**
- Core project code (outside .claude/ directory)
- Test files or CI configuration
- Package dependencies or build configs

**Always:**
- Show diffs before applying
- Require explicit confirmation for changes
- Make changes reversible (git commit)
- Log all modifications
- Focus on thinking patterns, not code correctness

**Abort if:**
- Git working directory has uncommitted changes (warn user to commit first)
- Required subagents are missing
- Proposals seem to modify core functionality instead of meta-cognition

## Example Session

```
User: /self-train full aggressive

[Phase 1: Gathering context...]
✓ Collected 50 commits
✓ Found 8 commands, 5 subagents
✓ Read CLAUDE.md (current meta-reasoning: 15 rules)

[Phase 2: Analyzing usage patterns...]
Invoking meta-usage-analyst...
✓ Analysis complete: identified 4 inefficiency patterns

[Phase 3: Designing improvements...]
Invoking meta-architect...
✓ Proposals ready: 7 new meta-reasoning rules, 2 agent refinements

[Phase 4: Review]
# Key Improvements:
1. Add decision tree: "Use subagent only for >3 step workflows"
2. Add rule: "Plan in 1-2 sentences before every implementation task"
3. Refine code-reviewer agent: "Focus on architecture, not syntax"
4. Remove redundant 'explorer' agent (overlaps with inline Grep)

Apply these changes? (yes/no)
> yes

[Phase 5: Applying changes...]
Invoking meta-applier...
✓ Modified CLAUDE.md (+7 rules, 45 lines)
✓ Updated .claude/agents/code-reviewer.md
✓ Removed .claude/agents/explorer.md

[Phase 6: Verification]
✓ Created commit: a7b3c9d
✓ Logged to .claude/logs/self-train.log

Changes focused on:
- More strategic subagent usage
- Upfront planning discipline  
- Eliminating redundant tools

Done! Claude will now think more efficiently in this project.
```

## Output Files

All outputs go to `.claude/work/meta/`:
- `usage-analysis.json` - Patterns identified by analyst
- `improvement-proposals.json` - Changes designed by architect
- `application-report.json` - Results of applying changes

Log file: `.claude/logs/self-train.log` (append-only history)

---

Now execute the workflow based on the parsed mode and scope from $ARGUMENTS.
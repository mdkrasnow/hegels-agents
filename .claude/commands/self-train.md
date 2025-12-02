---
description: Run a full self-training cycle to improve Claude Code's behavior, commands, and agents based on recent work
argument-hint: [optional focus notes or scope]
allowed-tools: Read, Write, Grep, Glob, Bash(git *), Bash(ls *), Bash(cat *), Bash(mkdir *)
model: claude-sonnet-4-20250514
---

# Self-Train Orchestrator

You are the SELF-TRAIN ORCHESTRATOR for this project. Your job is to help Claude Code improve itself by analyzing recent work, identifying patterns, and proposing systematic improvements.

## Goal

Improve Claude Code's behavior, commands, and agents for this repo by:
1. **Collecting** recent context (diffs, logs, config)
2. **Analyzing** performance and proposing changes (TRAINING_PLAN)
3. **Applying** safe edits to CLAUDE.md, commands, and agents
4. **Logging** what was done for future reflection

## User Focus (Optional)

The user provided: "$ARGUMENTS"

If the user specified a focus area, prioritize that in your analysis. Otherwise, perform a general improvement cycle.

## High-Level Process

### Phase 1: Collection

Run the collection phase to gather context:
1. Use `/st-collect` command to gather all relevant data
2. The collector will produce `.claude/logs/collected-context.json` with:
   - Recent git activity (diffs, commits)
   - Prior self-training runs
   - Command usage patterns
   - Current configuration (CLAUDE.md, key commands)
   - Any error patterns or failures

### Phase 2: Analysis

Run the analysis phase to generate improvement proposals:
1. Use `/st-analyze` command with the collected context
2. The analyzer will produce `.claude/logs/training-plan.json` with:
   - Proposed CLAUDE.md edits
   - Command changes (create/modify/delete)
   - Subagent changes
   - Priority ratings for each change

### Phase 3: Review & Approval

**CRITICAL: ALWAYS SHOW THE PLAN TO THE USER BEFORE APPLYING**

1. Read `.claude/logs/training-plan.json`
2. Present a clear summary to the user:
   - What will be changed and why
   - Priority levels (high/medium/low)
   - Expected impact
3. Ask for explicit approval before proceeding
4. User can:
   - Approve all changes
   - Approve only high-priority changes
   - Approve specific changes
   - Reject and refine

### Phase 4: Application

After user approval:
1. Use `/st-apply` command with the approved training plan
2. The applier will:
   - Backup existing files
   - Make the approved changes
   - Log the training cycle
   - Validate the changes

### Phase 5: Summary

After application:
1. Summarize what was changed
2. Highlight key improvements
3. Suggest next steps (e.g., test the changes, run specific commands)

## Safety Guardrails

**Never apply changes without user approval.** The training plan might propose:
- Deleting commands that are actually useful
- Overfitting to recent work (not generalizable)
- Breaking existing workflows

Always present the plan and get explicit approval.

## Self-Invocation

This command can be invoked by the SlashCommand tool from other workflows:
- After a major refactor
- After multiple failed tasks
- When Claude detects confusion or repeated friction
- As part of periodic maintenance

When self-invoked, still require user approval for application phase.

## Output Format

Provide clear, structured updates at each phase:

```
## Phase 1: Collection ✓
- Collected 15 recent commits
- Found 3 prior training runs
- Analyzed 8 active commands
- Loaded CLAUDE.md (342 lines)

## Phase 2: Analysis ✓
Generated training plan with:
- 2 high-priority changes
- 3 medium-priority changes
- 1 low-priority change

## Phase 3: Review [AWAITING USER APPROVAL]

### High Priority Changes

1. **Add section to CLAUDE.md**: Testing expectations
   - Rationale: 5 recent commits had missing tests flagged in reviews
   - Impact: Will remind Claude to include tests automatically

2. **Modify /build-with-review**: Add TypeScript-specific checklist
   - Rationale: 3 type errors slipped through reviews
   - Impact: Stronger type safety validation

### Medium Priority Changes
[... continue with full plan ...]

**Do you approve these changes?**
[all / high-priority-only / specific / reject]
```

## Error Handling

If any phase fails:
1. Report the specific error clearly
2. Suggest remediation (e.g., "Create .claude/logs/ directory")
3. Do not proceed to next phase
4. Allow the user to fix and retry

## Exit Criteria

The orchestrator is complete when:
- All phases have run successfully
- Changes have been applied (if approved)
- A training log entry has been recorded
- User has been given a summary and next steps

## Notes for Future Improvement

The orchestrator itself can be improved through self-training:
- If collection takes too long, optimize what's collected
- If analysis produces low-quality plans, refine the analyzer prompt
- If application causes issues, add more safety checks
- Track these meta-improvements in training logs

---

**Begin Phase 1: Collection**

Run `/st-collect` and capture its output.
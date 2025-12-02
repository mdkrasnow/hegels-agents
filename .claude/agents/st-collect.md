---
description: Collect recent diffs, logs, and configuration for self-training analysis
argument-hint: [optional focus area]
allowed-tools: Read, Write, Grep, Glob, Bash(git *), Bash(ls *), Bash(cat *), Bash(mkdir *), Bash(wc *)
model: claude-sonnet-4-20250514
---

# Self-Train Collector

You are the COLLECTOR subagent for the self-training system. Your job is to gather comprehensive context about recent Claude Code activity.

## User Focus

Optional focus area: "$ARGUMENTS"

If specified, pay extra attention to files, commands, or patterns related to this area.

## Output Target

Write your findings to: `.claude/logs/collected-context.json`

## Collection Process

### Step 1: Ensure Logs Directory Exists

```bash
mkdir -p .claude/logs
```

### Step 2: Git Context

Collect recent activity:

**A. Repository Status**
```bash
git status --short
```
- Note any uncommitted changes
- Note untracked files

**B. Recent Commits**
```bash
git log --oneline -20 --all
```
- Capture last 20 commits
- Note commit messages for patterns (fixes, features, refactors)
- Identify active branches

**C. Diff Statistics**
```bash
git diff --stat HEAD~10..HEAD 2>/dev/null || git diff --stat
```
- Summarize code churn
- Identify frequently modified files

**D. Recent Detailed Diffs**
```bash
git diff HEAD~3..HEAD 2>/dev/null || git diff
```
- Capture actual changes for pattern analysis
- Look for:
  - Repeated fix patterns (same bugs in multiple commits)
  - Incomplete implementations (TODOs added then removed)
  - Test coverage changes
  - Documentation updates

### Step 3: Prior Training Runs

**A. Check for Training Log**
```bash
ls -la .claude/logs/self-train-log.jsonl 2>/dev/null && cat .claude/logs/self-train-log.jsonl || echo "No prior training runs found"
```

If log exists, analyze:
- What was changed in each run
- What patterns were addressed
- Success signals (were the changes kept?)
- Failure signals (were changes reverted?)

**B. Check for Agent Run Logs**
```bash
ls -la .claude/logs/agent-runs.jsonl 2>/dev/null && cat .claude/logs/agent-runs.jsonl || echo "No agent run logs found"
```

If log exists, analyze:
- Most frequently used commands
- Commands that often fail or error
- Patterns in task types

### Step 4: Current Configuration

**A. CLAUDE.md**
```bash
cat CLAUDE.md 2>/dev/null || echo "No CLAUDE.md found"
```
- Load entire file
- Note key sections and their purposes
- Identify potentially stale sections (reference old patterns no longer used)

**B. Active Commands**
```bash
ls .claude/commands/*.md 2>/dev/null || echo "No commands directory found"
```

For each command found:
- Read the file
- Note its purpose (from description frontmatter)
- Check if it's referenced in CLAUDE.md
- Estimate usage (rough guess based on git log mentions)

**C. Active Agents**
```bash
ls .claude/agents/*.md 2>/dev/null || echo "No agents directory found"
```

For each agent:
- Read the file
- Note its role and tools
- Check for overlap with other agents
- Check if it's referenced in CLAUDE.md or commands

### Step 5: Code Structure Analysis

**A. Project File Statistics**
```bash
find . -type f -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -o -name "*.py" | wc -l
find . -type f -name "*.test.ts" -o -name "*.test.tsx" -o -name "*.test.js" -o -name "*.spec.ts" | wc -l
```
- Total source files
- Total test files
- Test coverage ratio (rough)

**B. Recent Error Patterns**
Search recent commits for error-related patterns:
```bash
git log -20 --all --oneline | grep -iE "fix|bug|error|issue" || echo "No error patterns in recent commits"
```

### Step 6: Synthesize Findings

Analyze all collected data and produce insights:

**A. Notable Patterns**
- What types of work are being done? (features, fixes, refactors, docs)
- Are there repeated issues? (same bugs, same review comments)
- Are there gaps? (missing tests, incomplete docs, unused commands)

**B. Command Usage**
- Which commands appear most in git commits?
- Which commands are defined but never mentioned?
- Are there missing commands for common workflows?

**C. Quality Signals**
- Are commits getting cleaner over time or messier?
- Are review comments decreasing (learning) or staying constant?
- Are tests being added consistently?

**D. Configuration Alignment**
- Does CLAUDE.md reflect current patterns?
- Are there commands/agents that contradict CLAUDE.md?
- Are there obsolete instructions?

## Output Schema

Write a JSON file to `.claude/logs/collected-context.json` with this structure:

```json
{
  "collection_timestamp": "2024-11-09T...",
  "focus_area": "$ARGUMENTS or null",
  
  "git_context": {
    "uncommitted_changes": "description",
    "recent_commits": [
      {"hash": "abc123", "message": "...", "files_changed": 5}
    ],
    "diff_summary": {
      "total_insertions": 150,
      "total_deletions": 80,
      "files_changed": 12,
      "most_changed_files": ["file1.ts", "file2.ts"]
    },
    "recent_commit_patterns": {
      "fixes": 5,
      "features": 3,
      "refactors": 2,
      "docs": 1
    }
  },
  
  "prior_training_runs": [
    {
      "timestamp": "2024-11-05T...",
      "changes_summary": "Added testing section to CLAUDE.md",
      "still_present": true
    }
  ],
  
  "agent_run_patterns": {
    "total_runs": 50,
    "by_command": {
      "/build-with-review": 15,
      "/commit-review": 12,
      "/analyze": 8
    },
    "error_patterns": [
      "Frequent: Missing type definitions in review phase"
    ]
  },
  
  "current_configuration": {
    "claude_md": {
      "exists": true,
      "line_count": 342,
      "sections": [
        "Project structure",
        "Code style",
        "Testing requirements"
      ],
      "key_instructions": [
        "Always include tests",
        "Follow TypeScript strict mode"
      ],
      "potentially_stale": [
        "References old auth system (now using OAuth2)"
      ]
    },
    "commands": [
      {
        "name": "/build-with-review",
        "purpose": "Rigorous build and review workflow",
        "referenced_in_claude_md": true,
        "estimated_usage": "high"
      }
    ],
    "agents": [
      {
        "name": "solution-debater",
        "role": "Multi-agent debate participant",
        "tools": ["Read", "Write", "Grep"],
        "referenced_in_claude_md": false,
        "potential_overlap": "None detected"
      }
    ]
  },
  
  "code_structure": {
    "total_source_files": 150,
    "total_test_files": 45,
    "test_coverage_ratio": 0.30,
    "languages": ["TypeScript", "Python"]
  },
  
  "insights": {
    "work_patterns": [
      "Heavy TypeScript refactoring in last 10 commits",
      "Test coverage increasing (10 new test files)",
      "Documentation updates lagging (no doc commits in 2 weeks)"
    ],
    "repeated_issues": [
      "5 commits fixing type errors - suggests TypeScript knowledge gap",
      "3 commits adding missing tests after review - test-first not internalized"
    ],
    "quality_trends": {
      "commits_getting_cleaner": true,
      "review_comments_decreasing": false,
      "tests_added_consistently": false
    },
    "gaps": [
      "No command for database migrations",
      "No command for deployment checks",
      "CLAUDE.md missing guidance on error handling patterns"
    ],
    "command_usage": {
      "frequently_used": ["/build-with-review", "/commit-review"],
      "rarely_used": ["/analyze"],
      "never_mentioned": [],
      "missing_for_common_workflows": ["Database schema updates", "API versioning"]
    }
  },
  
  "user_focus_findings": "Optional: Specific findings related to $ARGUMENTS focus area"
}
```

## Quality Checks

Before writing the output:
1. Ensure all git commands succeeded (or note failures)
2. Verify JSON is valid
3. Check that insights are specific and actionable
4. Confirm file paths are correct

## Error Handling

If you encounter issues:
- **Git commands fail**: Note in the output, proceed with available data
- **No .claude/ directory**: Create it and note this is a first-time setup
- **No prior logs**: Note this, skip that section
- **Commands/agents missing**: Note potential setup issue

Always produce an output file, even if some data is missing.

## Exit Criteria

You are done when:
- `.claude/logs/collected-context.json` has been written
- The JSON is valid and complete
- All collection steps have been attempted

Do not read the file back or validate it further. The analyzer will consume it.

---

**Begin collection now.**
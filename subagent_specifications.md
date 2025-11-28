# Subagent Specifications for Phase 0 Implementation

## Overview
Implementing **Phase 0 – Repo & Environment + Configuration Management** with 5 parallel subagents. Each subagent has access to all tools and follows a build-with-review methodology.

## Phase 0 Task Breakdown

From the implementation todo, Phase 0 includes:
- Create `hegels_agents` repo + basic structure
- Set up Python env with dependencies
- Configure secrets + basic config management

## Subagent Specifications

### Subagent 1: Repository Structure & Project Setup
**Primary Task**: Create the core repository structure and project foundation

**Specifications**:
- **Input**: Current working directory `/Users/mkrasnow/Desktop/hegels-agents`
- **Tools**: Cat, Echo, Touch, Write, Bash, any others needed
- **Tasks**:
  1. Create the required directory structure:
     - `src/corpus/`
     - `src/agents/`
     - `src/debate/`
     - `src/eval/`
     - `src/config/`
     - `scripts/`
  2. Create basic `__init__.py` files for Python packages
  3. Create initial `README.md` with project description
  4. Create `.gitignore` for Python projects
- **Success Criteria**: All directories created, proper Python package structure
- **Output Format**: JSON with created files/dirs, any issues encountered

### Subagent 2: Python Environment & Dependency Management
**Primary Task**: Set up Python environment and dependency management

**Specifications**:
- **Input**: Repository root directory
- **Tools**: Cat, Echo, Touch, Write, Bash, any others needed
- **Tasks**:
  1. Create `pyproject.toml` or `requirements.txt` with core dependencies:
     - `google-genai`
     - `psycopg2-binary` (or `asyncpg`)
     - `pydantic`
     - `sqlalchemy` (optional)
     - `pytest`
     - `tqdm`
     - Additional development dependencies
  2. Check if `uv`, `poetry`, or `pipenv` is available and recommend best option
  3. Create virtual environment setup instructions
  4. Test that dependencies can be installed (if environment allows)
- **Success Criteria**: Dependency file created with all required packages
- **Output Format**: JSON with dependency management approach, installation status

### Subagent 3: Configuration Management System
**Primary Task**: Create the configuration management system with environment variables

**Specifications**:
- **Input**: Repository with `src/config/` directory
- **Tools**: Cat, Echo, Touch, Write, Bash, any others needed
- **Tasks**:
  1. Create `src/config/settings.py` with:
     - Environment variable loading for `GEMINI_API_KEY`
     - Environment variable for `SUPABASE_DB_URL` (for later phases)
     - Dict/YAML-based configuration system
     - Design for evolution (simple params → sophisticated parameter management)
  2. Create `.env.template` file showing required environment variables
  3. Create `src/config/__init__.py`
  4. Add configuration validation and error handling
- **Success Criteria**: Working config system that loads from environment variables
- **Output Format**: JSON with config files created, validation logic status

### Subagent 4: Basic Project Utilities & Scripts
**Primary Task**: Create foundational utility scripts and helper modules

**Specifications**:
- **Input**: Repository with `scripts/` directory
- **Tools**: Cat, Echo, Touch, Write, Bash, any others needed  
- **Tasks**:
  1. Create `scripts/setup_environment.py` - environment setup helper
  2. Create `scripts/check_dependencies.py` - verify all dependencies work
  3. Create basic logging utilities in `src/config/logging.py`
  4. Create `scripts/run_tests.py` - test runner script
  5. Set up basic project structure validation script
- **Success Criteria**: All utility scripts created and functional
- **Output Format**: JSON with created scripts, functionality verification

### Subagent 5: Documentation & Development Setup  
**Primary Task**: Create development documentation and setup guides

**Specifications**:
- **Input**: Repository root
- **Tools**: Cat, Echo, Touch, Write, Bash, any others needed
- **Tasks**:
  1. Enhance `README.md` with:
     - Project overview and goals
     - Quick start instructions
     - Development setup guide
     - Directory structure explanation
  2. Create `CONTRIBUTING.md` with development guidelines
  3. Create `docs/` directory with basic structure
  4. Create development environment verification checklist
  5. Set up basic pre-commit configurations if applicable
- **Success Criteria**: Complete development documentation and setup guides
- **Output Format**: JSON with documentation files, setup verification status

## Orchestrator Coordination Protocol

### Communication Format
Each subagent must report back in this JSON format:

```json
{
  "subagent_id": "1-5",
  "task_name": "descriptive name",
  "status": "success|partial|failed",
  "completed_items": [
    {
      "item": "specific task completed",
      "file_path": "path/to/created/file",
      "verification": "how success was verified"
    }
  ],
  "issues_encountered": [
    {
      "issue": "description of problem",
      "severity": "low|medium|high", 
      "resolution": "how it was handled or needs handling"
    }
  ],
  "dependencies_ready": ["list of items ready for other subagents"],
  "next_phase_readiness": "assessment for Phase 0.5 readiness",
  "implementation_notes": "relevant details for future phases"
}
```

### Success Metrics
- **Repository Structure**: All required directories and `__init__.py` files created
- **Dependency Management**: Working dependency specification and installation method  
- **Configuration System**: Environment variables can be loaded, config system extensible
- **Utilities**: Basic scripts functional and tested
- **Documentation**: Complete setup and contribution guides

### Critical Gates
1. **Repository Structure** must be complete before other subagents can proceed with file creation
2. **Configuration System** must work before any code requiring API keys
3. **Dependencies** must be installable before testing any functionality

### Failure Handling
- If any subagent fails critically, orchestrator should:
  1. Document the failure mode
  2. Attempt alternative approaches with remaining subagents
  3. Mark affected todo items as "blocked" rather than "failed"
  4. Provide clear next steps for manual resolution

### Post-Implementation Actions
After all subagents complete:
1. Run integration test to verify full Phase 0 setup
2. Update `documentation/implementation-todo.md` with completion status
3. Mark items as `[Tentatively completed]` pending review
4. Prepare Phase 0.5 readiness assessment
5. Document any architectural decisions or deviations from plan

## Execution Constraints
- Maximum 5 subagents in parallel
- Each subagent has full tool access (Cat, Echo, Touch, Write, Bash, etc.)
- Build-with-review methodology: create, test, verify before marking complete
- Must maintain compatibility with progressive enhancement architecture
- All code must be production-ready, not throwaway prototypes
-- Migration: 001_create_training_tables.sql
-- Description: Create Hegel's Agents training layer database schema
-- Created: 2025-12-02
-- Dependencies: PostgreSQL with uuid-ossp extension

-- Begin transaction for atomic migration
BEGIN;

-- Check if migration has already been applied
DO $migration$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = current_schema() 
        AND table_name = 'hegel_prompt_profiles'
    ) THEN
        RAISE EXCEPTION 'Migration 001 already applied: hegel_prompt_profiles table exists';
    END IF;
END $migration$;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Version-controlled prompt profiles
-- Stores all prompts and hyperparameters per corpus/task combination
CREATE TABLE hegel_prompt_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    base_profile_id UUID REFERENCES hegel_prompt_profiles(id) ON DELETE SET NULL,
    corpus_id TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'qa',
    
    -- Core profile data stored as JSONB for flexibility
    profile JSONB NOT NULL,
    
    -- Performance tracking
    performance_stats JSONB DEFAULT '{}',
    
    -- Metadata and lineage
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT check_corpus_id_not_empty CHECK (LENGTH(TRIM(corpus_id)) > 0),
    CONSTRAINT check_task_type_not_empty CHECK (LENGTH(TRIM(task_type)) > 0),
    CONSTRAINT check_profile_not_empty CHECK (profile IS NOT NULL AND jsonb_typeof(profile) = 'object')
);

-- Training step logging
-- Records each training iteration with full context for reproducibility
CREATE TABLE hegel_training_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Profile evolution tracking
    old_profile_id UUID NOT NULL REFERENCES hegel_prompt_profiles(id) ON DELETE RESTRICT,
    new_profile_id UUID NOT NULL REFERENCES hegel_prompt_profiles(id) ON DELETE RESTRICT,
    
    -- Training context
    corpus_id TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'qa',
    
    -- Training data
    query TEXT NOT NULL,
    gold_answer TEXT,
    predicted_answer TEXT NOT NULL,
    reward NUMERIC,
    
    -- Rich metadata for analysis
    metrics JSONB DEFAULT '{}',
    debate_trace JSONB DEFAULT '{}',
    optimization_strategy TEXT,
    
    -- Timing and metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processing_time_ms INTEGER,
    
    -- Constraints
    CONSTRAINT check_different_profiles CHECK (old_profile_id != new_profile_id),
    CONSTRAINT check_query_not_empty CHECK (LENGTH(TRIM(query)) > 0),
    CONSTRAINT check_predicted_answer_not_empty CHECK (LENGTH(TRIM(predicted_answer)) > 0),
    CONSTRAINT check_corpus_id_not_empty CHECK (LENGTH(TRIM(corpus_id)) > 0),
    CONSTRAINT check_processing_time_positive CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

-- Population-based optimization state
-- Manages multiple prompt variants for evolutionary optimization
CREATE TABLE hegel_profile_populations (
    corpus_id TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'qa',
    profile_id UUID NOT NULL REFERENCES hegel_prompt_profiles(id) ON DELETE CASCADE,
    
    -- Population fitness metrics
    fitness_score NUMERIC,
    selection_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 1,
    
    -- Population metadata
    population_metadata JSONB DEFAULT '{}',
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key and constraints
    PRIMARY KEY(corpus_id, task_type, profile_id),
    CONSTRAINT check_corpus_id_not_empty CHECK (LENGTH(TRIM(corpus_id)) > 0),
    CONSTRAINT check_task_type_not_empty CHECK (LENGTH(TRIM(task_type)) > 0),
    CONSTRAINT check_selection_count_non_negative CHECK (selection_count >= 0),
    CONSTRAINT check_generation_positive CHECK (generation > 0)
);

-- Performance indexes for prompt profiles
CREATE INDEX idx_hegel_prompt_profiles_corpus_task_created ON hegel_prompt_profiles(corpus_id, task_type, created_at DESC);
CREATE INDEX idx_hegel_prompt_profiles_base_profile ON hegel_prompt_profiles(base_profile_id) WHERE base_profile_id IS NOT NULL;
CREATE INDEX idx_hegel_prompt_profiles_created_at ON hegel_prompt_profiles(created_at DESC);

-- JSONB indexes for efficient content searches
CREATE INDEX idx_hegel_prompt_profiles_profile_gin ON hegel_prompt_profiles USING gin(profile);
CREATE INDEX idx_hegel_prompt_profiles_performance_gin ON hegel_prompt_profiles USING gin(performance_stats);
CREATE INDEX idx_hegel_prompt_profiles_metadata_gin ON hegel_prompt_profiles USING gin(metadata);

-- Specific JSONB path indexes for common queries
CREATE INDEX idx_hegel_prompt_profiles_worker_prompt ON hegel_prompt_profiles 
    USING gin((profile->'worker'->'system_prompt')) WHERE profile ? 'worker';
CREATE INDEX idx_hegel_prompt_profiles_reviewer_prompt ON hegel_prompt_profiles 
    USING gin((profile->'reviewer'->'system_prompt')) WHERE profile ? 'reviewer';

-- Performance indexes for training steps
CREATE INDEX idx_hegel_training_steps_profiles ON hegel_training_steps(old_profile_id, new_profile_id);
CREATE INDEX idx_hegel_training_steps_corpus_task_created ON hegel_training_steps(corpus_id, task_type, created_at DESC);
CREATE INDEX idx_hegel_training_steps_reward ON hegel_training_steps(reward DESC) WHERE reward IS NOT NULL;
CREATE INDEX idx_hegel_training_steps_optimization_strategy ON hegel_training_steps(optimization_strategy) WHERE optimization_strategy IS NOT NULL;

-- JSONB indexes for training step metadata
CREATE INDEX idx_hegel_training_steps_metrics_gin ON hegel_training_steps USING gin(metrics);
CREATE INDEX idx_hegel_training_steps_trace_gin ON hegel_training_steps USING gin(debate_trace);

-- Full-text search indexes for content
CREATE INDEX idx_hegel_training_steps_query_fts ON hegel_training_steps USING gin(to_tsvector('english', query));
CREATE INDEX idx_hegel_training_steps_answers_fts ON hegel_training_steps USING gin(to_tsvector('english', predicted_answer || ' ' || COALESCE(gold_answer, '')));

-- Performance indexes for population management
CREATE INDEX idx_hegel_profile_populations_corpus_task ON hegel_profile_populations(corpus_id, task_type);
CREATE INDEX idx_hegel_profile_populations_fitness ON hegel_profile_populations(fitness_score DESC) WHERE fitness_score IS NOT NULL;
CREATE INDEX idx_hegel_profile_populations_generation ON hegel_profile_populations(generation DESC);
CREATE INDEX idx_hegel_profile_populations_last_updated ON hegel_profile_populations(last_updated DESC);

-- JSONB index for population metadata
CREATE INDEX idx_hegel_profile_populations_metadata_gin ON hegel_profile_populations USING gin(population_metadata);

-- Updated timestamp trigger for prompt profiles
CREATE OR REPLACE FUNCTION update_hegel_prompt_profiles_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_hegel_prompt_profiles_updated_at
    BEFORE UPDATE ON hegel_prompt_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_hegel_prompt_profiles_updated_at();

-- Updated timestamp trigger for population table
CREATE OR REPLACE FUNCTION update_hegel_profile_populations_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_hegel_profile_populations_last_updated
    BEFORE UPDATE ON hegel_profile_populations
    FOR EACH ROW
    EXECUTE FUNCTION update_hegel_profile_populations_last_updated();

-- Views for common queries

-- Latest profiles per corpus/task
CREATE VIEW hegel_latest_profiles AS
SELECT DISTINCT ON (corpus_id, task_type)
    id,
    corpus_id,
    task_type,
    profile,
    performance_stats,
    created_at,
    updated_at
FROM hegel_prompt_profiles
ORDER BY corpus_id, task_type, created_at DESC;

-- Profile performance summary
CREATE VIEW hegel_profile_performance_summary AS
SELECT 
    hpp.id,
    hpp.corpus_id,
    hpp.task_type,
    hpp.created_at,
    COUNT(hts.id) as training_steps_count,
    AVG(hts.reward) as avg_reward,
    MAX(hts.reward) as max_reward,
    MIN(hts.reward) as min_reward,
    STDDEV(hts.reward) as reward_stddev
FROM hegel_prompt_profiles hpp
LEFT JOIN hegel_training_steps hts ON hpp.id = hts.new_profile_id
GROUP BY hpp.id, hpp.corpus_id, hpp.task_type, hpp.created_at;

-- Population fitness rankings
CREATE VIEW hegel_population_rankings AS
SELECT 
    corpus_id,
    task_type,
    profile_id,
    fitness_score,
    selection_count,
    generation,
    ROW_NUMBER() OVER (PARTITION BY corpus_id, task_type ORDER BY fitness_score DESC NULLS LAST) as fitness_rank,
    PERCENT_RANK() OVER (PARTITION BY corpus_id, task_type ORDER BY fitness_score DESC NULLS LAST) as fitness_percentile
FROM hegel_profile_populations;

-- Comments for documentation
COMMENT ON TABLE hegel_prompt_profiles IS 'Version-controlled storage of all prompts and hyperparameters per corpus/task combination';
COMMENT ON TABLE hegel_training_steps IS 'Training step logging with full context for reproducibility and analysis';
COMMENT ON TABLE hegel_profile_populations IS 'Population-based optimization state for evolutionary prompt optimization';

COMMENT ON COLUMN hegel_prompt_profiles.profile IS 'JSONB containing RolePrompt configurations for orchestrator, worker, reviewer, summarizer';
COMMENT ON COLUMN hegel_prompt_profiles.performance_stats IS 'JSONB containing performance metrics and statistics';
COMMENT ON COLUMN hegel_training_steps.debate_trace IS 'JSONB containing full debate session trace for analysis';
COMMENT ON COLUMN hegel_training_steps.metrics IS 'JSONB containing evaluation metrics (F1, BLEU, semantic similarity, etc.)';

-- Insert migration record (for future migration tracking)
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_migrations (version, description) 
VALUES ('001', 'Create training layer tables with indexes and triggers');

-- Commit the migration
COMMIT;
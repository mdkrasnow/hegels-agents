-- Rollback Migration: 001_rollback_training_tables.sql
-- Description: Rollback Hegel's Agents training layer database schema
-- Created: 2025-12-02
-- Rolls back: 001_create_training_tables.sql

-- Begin transaction for atomic rollback
BEGIN;

-- Check if migration exists to rollback
DO $rollback$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = current_schema() 
        AND table_name = 'hegel_prompt_profiles'
    ) THEN
        RAISE EXCEPTION 'Cannot rollback migration 001: hegel_prompt_profiles table does not exist';
    END IF;
END $rollback$;

-- Check if schema_migrations table exists and has the migration record
DO $rollback$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = current_schema() 
        AND table_name = 'schema_migrations'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM schema_migrations 
            WHERE version = '001'
        ) THEN
            RAISE NOTICE 'Migration 001 not found in schema_migrations table, proceeding with rollback anyway';
        END IF;
    ELSE
        RAISE NOTICE 'schema_migrations table not found, proceeding with rollback anyway';
    END IF;
END $rollback$;

-- Drop views first (dependent objects)
DROP VIEW IF EXISTS hegel_population_rankings;
DROP VIEW IF EXISTS hegel_profile_performance_summary;
DROP VIEW IF EXISTS hegel_latest_profiles;

-- Drop triggers
DROP TRIGGER IF EXISTS trigger_hegel_profile_populations_last_updated ON hegel_profile_populations;
DROP TRIGGER IF EXISTS trigger_hegel_prompt_profiles_updated_at ON hegel_prompt_profiles;

-- Drop trigger functions
DROP FUNCTION IF EXISTS update_hegel_profile_populations_last_updated();
DROP FUNCTION IF EXISTS update_hegel_prompt_profiles_updated_at();

-- Drop indexes (will be automatically dropped with tables, but explicit for clarity)
-- Training steps indexes
DROP INDEX IF EXISTS idx_hegel_training_steps_answers_fts;
DROP INDEX IF EXISTS idx_hegel_training_steps_query_fts;
DROP INDEX IF EXISTS idx_hegel_training_steps_trace_gin;
DROP INDEX IF EXISTS idx_hegel_training_steps_metrics_gin;
DROP INDEX IF EXISTS idx_hegel_training_steps_optimization_strategy;
DROP INDEX IF EXISTS idx_hegel_training_steps_reward;
DROP INDEX IF EXISTS idx_hegel_training_steps_corpus_task_created;
DROP INDEX IF EXISTS idx_hegel_training_steps_profiles;

-- Prompt profiles indexes
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_reviewer_prompt;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_worker_prompt;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_metadata_gin;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_performance_gin;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_profile_gin;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_created_at;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_base_profile;
DROP INDEX IF EXISTS idx_hegel_prompt_profiles_corpus_task_created;

-- Population indexes
DROP INDEX IF EXISTS idx_hegel_profile_populations_metadata_gin;
DROP INDEX IF EXISTS idx_hegel_profile_populations_last_updated;
DROP INDEX IF EXISTS idx_hegel_profile_populations_generation;
DROP INDEX IF EXISTS idx_hegel_profile_populations_fitness;
DROP INDEX IF EXISTS idx_hegel_profile_populations_corpus_task;

-- Drop tables in correct order (respecting foreign keys)
-- Drop dependent tables first
DROP TABLE IF EXISTS hegel_training_steps CASCADE;
DROP TABLE IF EXISTS hegel_profile_populations CASCADE;

-- Drop main table
DROP TABLE IF EXISTS hegel_prompt_profiles CASCADE;

-- Remove migration record
DELETE FROM schema_migrations WHERE version = '001';

-- Note: We don't drop the uuid-ossp and btree_gin extensions 
-- as they might be used by other parts of the application

RAISE NOTICE 'Migration 001 successfully rolled back. All training layer tables, indexes, triggers, and views have been removed.';

-- Commit the rollback
COMMIT;
# Training Schema Performance Analysis

## Overview

This document analyzes the performance characteristics of the Hegel's Agents training layer database schema, focusing on expected query patterns, index optimization, and scalability considerations.

## Expected Query Patterns

### 1. Profile Loading Queries (High Frequency)
**Pattern**: Load latest profile for corpus/task combination
```sql
SELECT * FROM hegel_prompt_profiles 
WHERE corpus_id = ? AND task_type = ? 
ORDER BY created_at DESC LIMIT 1;
```

**Optimization**: 
- Primary index: `idx_hegel_prompt_profiles_corpus_task_created`
- Uses (corpus_id, task_type, created_at DESC) for optimal performance
- Expected QPS: 100-1000 (during active training)

### 2. Profile History Queries (Medium Frequency)
**Pattern**: Get evolution history for analysis
```sql
SELECT * FROM hegel_prompt_profiles 
WHERE corpus_id = ? AND task_type = ?
ORDER BY created_at DESC;
```

**Optimization**:
- Same index as above covers this query
- Expected QPS: 10-100 (during analysis/debugging)

### 3. Training Step Logging (High Frequency)
**Pattern**: Insert training steps during grad=True operations
```sql
INSERT INTO hegel_training_steps 
(old_profile_id, new_profile_id, corpus_id, query, predicted_answer, ...)
VALUES (?, ?, ?, ?, ?, ...);
```

**Optimization**:
- No index needed for INSERT performance
- Foreign key constraints ensure referential integrity
- Expected QPS: 10-100 (during active training)

### 4. Performance Analysis Queries (Low Frequency)
**Pattern**: Analyze training effectiveness over time
```sql
SELECT corpus_id, task_type, AVG(reward), COUNT(*) 
FROM hegel_training_steps 
WHERE created_at > ? 
GROUP BY corpus_id, task_type;
```

**Optimization**:
- Index: `idx_hegel_training_steps_corpus_task_created`
- Supports both filtering and grouping efficiently
- Expected QPS: 1-10 (periodic analysis)

### 5. Content Search Queries (Low Frequency)
**Pattern**: Search within prompt content for analysis
```sql
SELECT * FROM hegel_prompt_profiles 
WHERE profile->'worker'->>'system_prompt' ILIKE '%pattern%';
```

**Optimization**:
- JSONB GIN index: `idx_hegel_prompt_profiles_profile_gin`
- Specific path indexes for common searches
- Expected QPS: 1-10 (research/debugging)

### 6. Population Management (Medium Frequency)
**Pattern**: Select profiles for evolutionary optimization
```sql
SELECT * FROM hegel_profile_populations 
WHERE corpus_id = ? AND task_type = ?
ORDER BY fitness_score DESC;
```

**Optimization**:
- Composite index: `idx_hegel_profile_populations_corpus_task`
- Fitness index: `idx_hegel_profile_populations_fitness`
- Expected QPS: 10-50 (during population evolution)

## Index Strategy Analysis

### Primary Indexes (Performance Critical)
1. **`idx_hegel_prompt_profiles_corpus_task_created`**
   - Covers 80% of profile queries
   - Composite index with optimal column order
   - DESC ordering for latest-first access

2. **`idx_hegel_training_steps_corpus_task_created`** 
   - Supports both filtering and analytical queries
   - Enables efficient time-range analysis

3. **`idx_hegel_profile_populations_corpus_task`**
   - Primary access pattern for population queries
   - Enables efficient population selection

### JSONB Indexes (Content Search)
1. **`idx_hegel_prompt_profiles_profile_gin`**
   - General-purpose JSONB content search
   - Supports complex JSON path queries
   - Uses GIN for efficient containment checks

2. **Specific Path Indexes**
   - `idx_hegel_prompt_profiles_worker_prompt`
   - `idx_hegel_prompt_profiles_reviewer_prompt`
   - Optimize common prompt content searches

### Full-Text Search Indexes
1. **`idx_hegel_training_steps_query_fts`**
   - English language text search on queries
   - Enables research and debugging use cases

2. **`idx_hegel_training_steps_answers_fts`**
   - Combined search on predicted and gold answers
   - Supports answer pattern analysis

## Performance Projections

### Storage Requirements
**Assumptions**:
- Average profile size: 5KB (JSONB)
- Average training step size: 10KB (includes debate trace)
- Active corpora: 20
- Training steps per day: 1000

**Projections**:
- Profiles: 20 corpora × 50 versions = 1000 profiles = 5MB
- Training steps: 1000/day × 10KB × 365 days = 3.65GB/year
- Total first year: ~4GB (very manageable)

### Query Performance Estimates
**Based on properly indexed PostgreSQL**:
- Profile loading: <1ms (index-only scan)
- Training step insert: <5ms (with FK validation)
- Performance analysis: <100ms (aggregation with indexes)
- Content search: <50ms (GIN index scan)

### Scalability Considerations
1. **Profile Table**: Scales to millions of profiles
2. **Training Steps**: Primary growth vector, indexed for performance
3. **Population Table**: Small, manageable size
4. **Index Maintenance**: GIN indexes may need periodic REINDEX

## Memory and Cache Optimization

### PostgreSQL Configuration Recommendations
```sql
-- For training workload optimization
SET shared_buffers = '256MB';           -- Cache frequently accessed pages
SET work_mem = '64MB';                  -- For sorting/aggregation operations
SET maintenance_work_mem = '256MB';     -- For index maintenance
SET effective_cache_size = '1GB';       -- Total system cache estimate
SET random_page_cost = 1.1;             -- Assume SSD storage
```

### Connection Pooling
- Recommend pgBouncer or built-in connection pooling
- Pool size: 10-20 connections for typical workload
- Transaction pooling mode for short-lived operations

## Monitoring and Maintenance

### Key Metrics to Monitor
1. **Query Performance**:
   - Profile loading latency (p95 < 5ms)
   - Training step insert latency (p95 < 10ms)

2. **Index Efficiency**:
   - Index scan vs sequential scan ratio
   - JSONB query performance

3. **Storage Growth**:
   - Table size growth rate
   - Index bloat monitoring

### Maintenance Operations
1. **Routine VACUUM**: Weekly on training_steps table
2. **REINDEX**: Monthly on JSONB GIN indexes if heavy insert load
3. **ANALYZE**: After bulk operations to update query planner statistics

## Bottleneck Analysis

### Potential Bottlenecks
1. **JSONB Operations**: Content searches may be slower than relational queries
2. **Training Step Inserts**: High insert rate may cause lock contention
3. **Foreign Key Validation**: FK checks add overhead to inserts

### Mitigation Strategies
1. **Async Inserts**: Buffer training steps for batch insertion
2. **Read Replicas**: Separate analytical queries from transactional load
3. **Partitioning**: Consider partitioning training_steps by date if >100M rows

## Conclusion

The training schema is designed for excellent performance with expected workloads:

### Strengths
- Optimal indexes for primary query patterns
- JSONB flexibility without sacrificing performance  
- Scalable design supporting millions of training steps
- Proper use of PostgreSQL features (GIN indexes, views, triggers)

### Performance Characteristics
- Sub-millisecond profile loading
- Efficient training step logging
- Fast analytical queries with proper indexes
- Reasonable storage growth (~4GB/year)

The schema should handle the expected training workloads efficiently while maintaining flexibility for research and analysis needs.
#!/usr/bin/env python3
"""
Schema Validation Script for Hegel's Agents Training Layer

This script validates the training layer database schema by:
1. Checking SQL syntax using sqlparse
2. Testing schema creation on a PostgreSQL database (if available)
3. Running basic integration tests
4. Validating rollback functionality

Usage:
    python validate_schema.py [--database-url DATABASE_URL] [--dry-run]
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

try:
    import sqlparse
except ImportError:
    sqlparse = None

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    psycopg2 = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchemaValidator:
    """Validates the training layer database schema"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url
        self.schema_dir = Path(__file__).parent
        self.migration_dir = self.schema_dir / "migrations"
        
    def validate_sql_syntax(self) -> Dict[str, Any]:
        """Validate SQL syntax using sqlparse"""
        if sqlparse is None:
            return {
                "status": "skipped",
                "reason": "sqlparse not available",
                "recommendation": "pip install sqlparse"
            }
        
        results = {}
        sql_files = [
            self.schema_dir / "schema.sql",
            self.migration_dir / "001_create_training_tables.sql",
            self.migration_dir / "001_rollback_training_tables.sql"
        ]
        
        for sql_file in sql_files:
            try:
                with open(sql_file, 'r') as f:
                    sql_content = f.read()
                
                # Parse SQL
                parsed = sqlparse.parse(sql_content)
                
                # Basic validation
                if not parsed:
                    results[sql_file.name] = {
                        "status": "error",
                        "error": "Failed to parse SQL content"
                    }
                    continue
                
                # Check for common syntax errors
                errors = []
                for statement in parsed:
                    if statement.get_type() == 'UNKNOWN':
                        errors.append(f"Unknown statement: {statement.value[:100]}...")
                
                results[sql_file.name] = {
                    "status": "success" if not errors else "warning",
                    "statements_count": len(parsed),
                    "errors": errors
                }
                
            except Exception as e:
                results[sql_file.name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def test_schema_creation(self) -> Dict[str, Any]:
        """Test schema creation on actual database"""
        if not self.database_url or psycopg2 is None:
            return {
                "status": "skipped",
                "reason": "No database URL provided or psycopg2 not available"
            }
        
        try:
            # Connect to database
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Read migration script
            migration_file = self.migration_dir / "001_create_training_tables.sql"
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            # Execute migration
            logger.info("Executing migration script...")
            cursor.execute(migration_sql)
            
            # Verify tables were created
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = current_schema() 
                AND table_name LIKE 'hegel_%'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'hegel_prompt_profiles',
                'hegel_training_steps', 
                'hegel_profile_populations'
            ]
            
            missing_tables = set(expected_tables) - set(tables)
            
            # Verify views were created
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = current_schema() 
                AND table_name LIKE 'hegel_%'
                ORDER BY table_name
            """)
            views = [row[0] for row in cursor.fetchall()]
            
            expected_views = [
                'hegel_latest_profiles',
                'hegel_profile_performance_summary',
                'hegel_population_rankings'
            ]
            
            missing_views = set(expected_views) - set(views)
            
            conn.close()
            
            return {
                "status": "success" if not missing_tables and not missing_views else "error",
                "tables_created": tables,
                "views_created": views,
                "missing_tables": list(missing_tables),
                "missing_views": list(missing_views)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def test_data_operations(self) -> Dict[str, Any]:
        """Test basic data operations on the schema"""
        if not self.database_url or psycopg2 is None:
            return {
                "status": "skipped",
                "reason": "No database URL provided or psycopg2 not available"
            }
        
        try:
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Test 1: Insert sample profile
            test_profile = {
                "worker": {
                    "system_prompt": "Test worker prompt",
                    "temperature": 0.4,
                    "max_tokens": 2000
                },
                "reviewer": {
                    "system_prompt": "Test reviewer prompt", 
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            }
            
            cursor.execute("""
                INSERT INTO hegel_prompt_profiles (corpus_id, task_type, profile)
                VALUES (%s, %s, %s)
                RETURNING id
            """, ('test_corpus', 'qa', json.dumps(test_profile)))
            
            profile_id = cursor.fetchone()[0]
            
            # Test 2: Create second profile for training step
            cursor.execute("""
                INSERT INTO hegel_prompt_profiles (corpus_id, task_type, profile, base_profile_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, ('test_corpus', 'qa', json.dumps(test_profile), profile_id))
            
            new_profile_id = cursor.fetchone()[0]
            
            # Test 3: Insert training step
            cursor.execute("""
                INSERT INTO hegel_training_steps 
                (old_profile_id, new_profile_id, corpus_id, task_type, query, predicted_answer, reward)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (profile_id, new_profile_id, 'test_corpus', 'qa', 'Test question?', 'Test answer', 0.85))
            
            training_step_id = cursor.fetchone()[0]
            
            # Test 4: Insert population record
            cursor.execute("""
                INSERT INTO hegel_profile_populations 
                (corpus_id, task_type, profile_id, fitness_score)
                VALUES (%s, %s, %s, %s)
            """, ('test_corpus', 'qa', profile_id, 0.9))
            
            # Test 5: Query views
            cursor.execute("SELECT COUNT(*) FROM hegel_latest_profiles")
            latest_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM hegel_profile_performance_summary")
            performance_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM hegel_population_rankings")
            ranking_count = cursor.fetchone()[0]
            
            # Clean up test data
            cursor.execute("DELETE FROM hegel_training_steps WHERE id = %s", (training_step_id,))
            cursor.execute("DELETE FROM hegel_profile_populations WHERE profile_id = %s", (profile_id,))
            cursor.execute("DELETE FROM hegel_prompt_profiles WHERE id IN (%s, %s)", (profile_id, new_profile_id))
            
            conn.close()
            
            return {
                "status": "success",
                "operations_tested": [
                    "insert_prompt_profile",
                    "insert_training_step", 
                    "insert_population_record",
                    "query_views"
                ],
                "view_counts": {
                    "latest_profiles": latest_count,
                    "performance_summary": performance_count,
                    "population_rankings": ranking_count
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def test_rollback(self) -> Dict[str, Any]:
        """Test rollback functionality"""
        if not self.database_url or psycopg2 is None:
            return {
                "status": "skipped",
                "reason": "No database URL provided or psycopg2 not available"
            }
        
        try:
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Read rollback script
            rollback_file = self.migration_dir / "001_rollback_training_tables.sql"
            with open(rollback_file, 'r') as f:
                rollback_sql = f.read()
            
            # Execute rollback
            logger.info("Executing rollback script...")
            cursor.execute(rollback_sql)
            
            # Verify tables were dropped
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = current_schema() 
                AND table_name LIKE 'hegel_%'
            """)
            remaining_tables = [row[0] for row in cursor.fetchall()]
            
            # Verify views were dropped
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = current_schema() 
                AND table_name LIKE 'hegel_%'
            """)
            remaining_views = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "status": "success" if not remaining_tables and not remaining_views else "error",
                "remaining_tables": remaining_tables,
                "remaining_views": remaining_views
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_report(self, dry_run: bool = False) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "database_url_provided": bool(self.database_url),
            "dry_run": dry_run,
            "results": {}
        }
        
        logger.info("Validating SQL syntax...")
        report["results"]["syntax_validation"] = self.validate_sql_syntax()
        
        if not dry_run and self.database_url:
            logger.info("Testing schema creation...")
            report["results"]["schema_creation"] = self.test_schema_creation()
            
            if report["results"]["schema_creation"]["status"] == "success":
                logger.info("Testing data operations...")
                report["results"]["data_operations"] = self.test_data_operations()
                
                logger.info("Testing rollback...")
                report["results"]["rollback"] = self.test_rollback()
            else:
                report["results"]["data_operations"] = {
                    "status": "skipped",
                    "reason": "Schema creation failed"
                }
                report["results"]["rollback"] = {
                    "status": "skipped", 
                    "reason": "Schema creation failed"
                }
        else:
            skip_reason = "Dry run mode" if dry_run else "No database URL provided"
            for test in ["schema_creation", "data_operations", "rollback"]:
                report["results"][test] = {
                    "status": "skipped",
                    "reason": skip_reason
                }
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Validate Hegel's Agents training schema")
    parser.add_argument("--database-url", help="PostgreSQL database URL for testing")
    parser.add_argument("--dry-run", action="store_true", help="Only validate syntax, don't test on database")
    parser.add_argument("--output", help="Output file for validation report")
    
    args = parser.parse_args()
    
    validator = SchemaValidator(args.database_url)
    report = validator.generate_report(args.dry_run)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report written to {args.output}")
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with error code if any validations failed
    failed_tests = [
        test for test, result in report["results"].items()
        if result.get("status") == "error"
    ]
    
    if failed_tests:
        logger.error(f"Validation failed for: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        logger.info("All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
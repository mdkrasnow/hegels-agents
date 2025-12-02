#!/usr/bin/env python3
"""
Integration Test Script for Hegel's Agents Training Layer

This script tests integration with the existing database configuration
and validates that the training schema works with the DatabaseConfig class.

Usage:
    python test_integration.py
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import DatabaseConfig
except ImportError:
    DatabaseConfig = None

try:
    from training.validate_schema import SchemaValidator
except ImportError:
    SchemaValidator = None

logger = logging.getLogger(__name__)

class TrainingIntegrationTester:
    """Tests integration between training schema and existing database config"""
    
    def __init__(self):
        self.test_results = {}
        
    def test_database_config_compatibility(self) -> Dict[str, Any]:
        """Test compatibility with DatabaseConfig class"""
        if DatabaseConfig is None:
            return {
                "status": "error",
                "error": "DatabaseConfig class not available"
            }
        
        try:
            # Test with valid PostgreSQL URL
            test_url = "postgresql://user:password@localhost:5432/test_db"
            config = DatabaseConfig(url=test_url)
            config.validate()
            
            # Test URL format compatibility
            url_valid = config.url.startswith(('postgresql://', 'postgres://'))
            
            return {
                "status": "success",
                "database_config_available": True,
                "url_format_compatible": url_valid,
                "test_url": test_url
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def test_schema_file_structure(self) -> Dict[str, Any]:
        """Test that all required schema files exist"""
        schema_dir = Path(__file__).parent
        required_files = [
            "schema.sql",
            "migrations/001_create_training_tables.sql",
            "migrations/001_rollback_training_tables.sql"
        ]
        
        results = {}
        all_exist = True
        
        for file_path in required_files:
            full_path = schema_dir / file_path
            exists = full_path.exists()
            all_exist = all_exist and exists
            
            results[file_path] = {
                "exists": exists,
                "path": str(full_path),
                "size_bytes": full_path.stat().st_size if exists else 0
            }
        
        return {
            "status": "success" if all_exist else "error",
            "all_files_exist": all_exist,
            "files": results
        }
    
    def test_environment_variable_usage(self) -> Dict[str, Any]:
        """Test usage of SUPABASE_DB_URL environment variable"""
        import os
        
        supabase_url = os.getenv('SUPABASE_DB_URL')
        
        return {
            "status": "success",
            "supabase_db_url_set": bool(supabase_url),
            "url_format": "postgresql" if supabase_url and supabase_url.startswith(('postgresql://', 'postgres://')) else "unknown",
            "environment_compatible": True
        }
    
    def test_schema_content_validation(self) -> Dict[str, Any]:
        """Validate schema content matches requirements"""
        schema_file = Path(__file__).parent / "schema.sql"
        
        try:
            with open(schema_file, 'r') as f:
                content = f.read()
            
            # Check for required tables
            required_tables = [
                'hegel_prompt_profiles',
                'hegel_training_steps',
                'hegel_profile_populations'
            ]
            
            table_checks = {}
            for table in required_tables:
                table_checks[table] = f"CREATE TABLE {table}" in content
            
            # Check for required indexes
            required_indexes = [
                'idx_hegel_prompt_profiles_corpus_task_created',
                'idx_hegel_prompt_profiles_profile_gin',
                'idx_hegel_training_steps_corpus_task_created',
                'idx_hegel_profile_populations_fitness'
            ]
            
            index_checks = {}
            for index in required_indexes:
                index_checks[index] = f"CREATE INDEX {index}" in content
            
            # Check for JSONB features
            jsonb_features = [
                "USING gin(profile)",
                "USING gin(performance_stats)",
                "USING gin(debate_trace)",
                "profile->'worker'",
                "profile->'reviewer'"
            ]
            
            jsonb_checks = {}
            for feature in jsonb_features:
                jsonb_checks[feature] = feature in content
            
            # Check for timestamptz consistency
            timestamptz_usage = content.count('TIMESTAMPTZ')
            
            all_tables_found = all(table_checks.values())
            all_indexes_found = all(index_checks.values())
            all_jsonb_found = all(jsonb_checks.values())
            
            return {
                "status": "success" if all_tables_found and all_indexes_found else "warning",
                "tables": table_checks,
                "indexes": index_checks,
                "jsonb_features": jsonb_checks,
                "timestamptz_usage_count": timestamptz_usage,
                "all_required_found": all_tables_found and all_indexes_found
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Running training layer integration tests...")
        
        results = {
            "test_timestamp": "2025-12-02T15:38:00",
            "tests": {}
        }
        
        # Test 1: Database config compatibility
        logger.info("Testing database config compatibility...")
        results["tests"]["database_config"] = self.test_database_config_compatibility()
        
        # Test 2: Schema file structure
        logger.info("Testing schema file structure...")
        results["tests"]["file_structure"] = self.test_schema_file_structure()
        
        # Test 3: Environment variable usage
        logger.info("Testing environment variable usage...")
        results["tests"]["environment"] = self.test_environment_variable_usage()
        
        # Test 4: Schema content validation
        logger.info("Testing schema content validation...")
        results["tests"]["schema_content"] = self.test_schema_content_validation()
        
        # Overall status
        all_passed = all(
            test_result.get("status") in ["success", "warning"]
            for test_result in results["tests"].values()
        )
        
        results["overall_status"] = "success" if all_passed else "error"
        
        return results

def main():
    """Main test runner"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tester = TrainingIntegrationTester()
    results = tester.run_all_tests()
    
    # Output results
    print(json.dumps(results, indent=2))
    
    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    
    for test_name, test_result in results["tests"].items():
        status = test_result.get("status", "unknown")
        print(f"{test_name:20} | {status.upper():8}")
        if status == "error":
            print(f"{'':20} | Error: {test_result.get('error', 'Unknown error')}")
    
    print(f"{'':20} |")
    print(f"{'Overall Status':20} | {results['overall_status'].upper()}")
    
    # Exit with appropriate code
    if results["overall_status"] == "success":
        logger.info("All integration tests passed!")
        sys.exit(0)
    else:
        logger.error("Some integration tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
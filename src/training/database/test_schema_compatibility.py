#!/usr/bin/env python3
"""
Schema Compatibility Test for PromptProfileStore.

This script validates that the PromptProfileStore and SQLAlchemy models
are compatible with the actual database schema defined in schema.sql.
"""

import sys
from pathlib import Path
import re
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.database.models import PromptProfileModel


def parse_schema_file():
    """Parse the schema.sql file to extract table structure."""
    schema_path = Path(__file__).parent.parent / "schema.sql"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    # Extract hegel_prompt_profiles table definition
    table_pattern = r'CREATE TABLE hegel_prompt_profiles \((.*?)\);'
    table_match = re.search(table_pattern, schema_content, re.DOTALL)
    
    if not table_match:
        raise ValueError("Could not find hegel_prompt_profiles table definition in schema")
    
    table_definition = table_match.group(1)
    
    # Parse columns
    columns = {}
    column_lines = [line.strip() for line in table_definition.split('\n') if line.strip()]
    
    for line in column_lines:
        # Skip constraints and comments
        if line.startswith('--') or line.startswith('CONSTRAINT') or line.startswith('PRIMARY KEY'):
            continue
        
        # Parse column definition
        parts = line.split()
        if len(parts) >= 2:
            column_name = parts[0].replace(',', '')
            column_type = parts[1].replace(',', '')
            
            # Extract additional constraints
            constraints = []
            if 'NOT NULL' in line:
                constraints.append('NOT NULL')
            if 'PRIMARY KEY' in line:
                constraints.append('PRIMARY KEY')
            if 'DEFAULT' in line:
                constraints.append('DEFAULT')
            if 'REFERENCES' in line:
                constraints.append('REFERENCES')
            
            columns[column_name] = {
                'type': column_type,
                'constraints': constraints,
                'definition': line
            }
    
    return columns


def validate_model_compatibility():
    """Validate that SQLAlchemy model matches schema definition."""
    print("Validating SQLAlchemy model compatibility with schema...")
    
    # Parse schema
    schema_columns = parse_schema_file()
    
    # Get model columns
    model_columns = {}
    for attr_name in dir(PromptProfileModel):
        attr = getattr(PromptProfileModel, attr_name)
        if hasattr(attr, 'property') and hasattr(attr.property, 'columns'):
            column = attr.property.columns[0]
            model_columns[column.name] = {
                'type': str(column.type),
                'nullable': column.nullable,
                'primary_key': column.primary_key,
                'foreign_keys': len(column.foreign_keys) > 0,
                'default': column.default is not None
            }
    
    print(f"Schema columns: {len(schema_columns)}")
    print(f"Model columns: {len(model_columns)}")
    
    # Check for missing columns in model
    missing_in_model = set(schema_columns.keys()) - set(model_columns.keys())
    if missing_in_model:
        print(f"⚠️  Columns in schema but missing in model: {missing_in_model}")
    
    # Check for extra columns in model
    extra_in_model = set(model_columns.keys()) - set(schema_columns.keys())
    if extra_in_model:
        print(f"⚠️  Columns in model but not in schema: {extra_in_model}")
    
    # Detailed column comparison
    compatibility_issues = []
    for col_name in schema_columns:
        if col_name in model_columns:
            schema_col = schema_columns[col_name]
            model_col = model_columns[col_name]
            
            # Type compatibility checks
            schema_type = schema_col['type'].upper()
            model_type = model_col['type'].upper()
            
            # Basic type mapping checks
            type_compatible = False
            if schema_type.startswith('UUID') and 'UUID' in model_type:
                type_compatible = True
            elif schema_type.startswith('VARCHAR') or schema_type == 'TEXT':
                type_compatible = 'VARCHAR' in model_type or 'TEXT' in model_type or 'STRING' in model_type
            elif schema_type == 'JSONB':
                type_compatible = 'JSONB' in model_type
            elif schema_type.startswith('TIMESTAMP'):
                type_compatible = 'TIMESTAMP' in model_type or 'DATETIME' in model_type
            else:
                type_compatible = schema_type in model_type or model_type in schema_type
            
            if not type_compatible:
                compatibility_issues.append(f"Type mismatch for {col_name}: schema={schema_type}, model={model_type}")
            
            # Nullability checks
            schema_nullable = 'NOT NULL' not in schema_col['constraints']
            model_nullable = model_col['nullable']
            
            if schema_nullable != model_nullable:
                compatibility_issues.append(f"Nullability mismatch for {col_name}: schema={'nullable' if schema_nullable else 'not null'}, model={'nullable' if model_nullable else 'not null'}")
    
    if compatibility_issues:
        print("❌ Compatibility issues found:")
        for issue in compatibility_issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Model is compatible with schema")
        return True


def validate_required_indexes():
    """Validate that required indexes are defined in schema."""
    print("\nValidating required indexes...")
    
    schema_path = Path(__file__).parent.parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    required_indexes = [
        'idx_hegel_prompt_profiles_corpus_task_created',
        'idx_hegel_prompt_profiles_base_profile',
        'idx_hegel_prompt_profiles_created_at',
        'idx_hegel_prompt_profiles_profile_gin',
        'idx_hegel_prompt_profiles_performance_gin',
        'idx_hegel_prompt_profiles_metadata_gin'
    ]
    
    missing_indexes = []
    for index_name in required_indexes:
        if index_name not in schema_content:
            missing_indexes.append(index_name)
    
    if missing_indexes:
        print(f"❌ Missing indexes: {missing_indexes}")
        return False
    else:
        print("✅ All required indexes are defined")
        return True


def validate_constraints():
    """Validate that database constraints are properly defined."""
    print("\nValidating database constraints...")
    
    schema_path = Path(__file__).parent.parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    required_constraints = [
        'check_corpus_id_not_empty',
        'check_task_type_not_empty',
        'check_profile_not_empty'
    ]
    
    missing_constraints = []
    for constraint_name in required_constraints:
        if constraint_name not in schema_content:
            missing_constraints.append(constraint_name)
    
    if missing_constraints:
        print(f"❌ Missing constraints: {missing_constraints}")
        return False
    else:
        print("✅ All required constraints are defined")
        return True


def validate_jsonb_features():
    """Validate JSONB features are properly configured."""
    print("\nValidating JSONB features...")
    
    schema_path = Path(__file__).parent.parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    required_jsonb_features = [
        "USING gin(profile)",
        "USING gin(performance_stats)",
        "USING gin(metadata)",
        "profile->'worker'",
        "profile->'reviewer'"
    ]
    
    missing_features = []
    for feature in required_jsonb_features:
        if feature not in schema_content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing JSONB features: {missing_features}")
        return False
    else:
        print("✅ All JSONB features are configured")
        return True


def validate_database_functions():
    """Validate that required database functions and triggers exist."""
    print("\nValidating database functions and triggers...")
    
    schema_path = Path(__file__).parent.parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    required_functions = [
        'update_hegel_prompt_profiles_updated_at',
        'trigger_hegel_prompt_profiles_updated_at'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if func_name not in schema_content:
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"❌ Missing database functions/triggers: {missing_functions}")
        return False
    else:
        print("✅ All database functions and triggers are defined")
        return True


def main():
    """Main test runner."""
    print("=" * 80)
    print("PROMPTPROFILESTORE SCHEMA COMPATIBILITY TEST")
    print("=" * 80)
    
    tests = [
        ("Model Compatibility", validate_model_compatibility),
        ("Required Indexes", validate_required_indexes),
        ("Database Constraints", validate_constraints),
        ("JSONB Features", validate_jsonb_features),
        ("Database Functions", validate_database_functions)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = {"status": "PASS" if success else "FAIL", "error": None}
            if not success:
                all_passed = False
        except Exception as e:
            results[test_name] = {"status": "ERROR", "error": str(e)}
            all_passed = False
            print(f"❌ {test_name} failed with error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SCHEMA COMPATIBILITY TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = result["status"]
        print(f"{test_name:<30} | {status}")
        if result["error"]:
            print(f"{'Error:':<30} | {result['error']}")
    
    print(f"\nOverall Status: {'PASS' if all_passed else 'FAIL'}")
    
    # Generate report
    report = {
        "timestamp": "2024-12-02T15:50:00Z",
        "overall_status": "PASS" if all_passed else "FAIL",
        "test_results": results,
        "summary": {
            "total_tests": len(tests),
            "passed": sum(1 for r in results.values() if r["status"] == "PASS"),
            "failed": sum(1 for r in results.values() if r["status"] == "FAIL"),
            "errors": sum(1 for r in results.values() if r["status"] == "ERROR")
        }
    }
    
    with open("schema_compatibility_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: schema_compatibility_report.json")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
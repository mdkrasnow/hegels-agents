#!/usr/bin/env python3
"""
Integration test for PromptProfileStore functionality.

This script tests the PromptProfileStore implementation to ensure
proper functionality and integration with data structures.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.data_structures import PromptProfile, RolePrompt
from training.database.prompt_profile_store import PromptProfileStore
import json


def test_data_structure_integration():
    """Test that PromptProfileStore integrates properly with data structures."""
    print("Testing PromptProfileStore data structure integration...")
    
    # Create a sample role prompt
    role_prompt = RolePrompt(
        role="worker",
        prompt_text="You are a helpful worker agent that analyzes questions carefully.",
        description="Primary worker agent for question analysis",
        author="test_system",
        version="1.0"
    )
    
    # Create a sample prompt profile
    profile = PromptProfile(
        name="Test Analysis Profile",
        description="A test profile for system analysis",
        author="test_system",
        tags=["test", "analysis", "qa"]
    )
    profile.add_role_prompt(role_prompt)
    
    # Initialize store
    store = PromptProfileStore()
    
    # Test conversion methods
    print("Testing profile to model conversion...")
    try:
        model = store._prompt_profile_to_model(profile)
        print(f"✓ Converted profile to model successfully")
        print(f"  Model ID: {model.id}")
        print(f"  Profile data keys: {list(model.profile.keys())}")
    except Exception as e:
        print(f"✗ Error converting profile to model: {e}")
        return False
    
    # Test model to profile conversion
    print("Testing model to profile conversion...")
    try:
        # Set required fields for conversion
        model.corpus_id = "test_corpus"
        model.task_type = "qa"
        model.performance_stats = {}
        model.metadata = {}
        
        converted_profile = store._model_to_prompt_profile(model)
        print(f"✓ Converted model to profile successfully")
        print(f"  Profile ID: {converted_profile.profile_id}")
        print(f"  Profile name: {converted_profile.name}")
        print(f"  Role prompts: {list(converted_profile.role_prompts.keys())}")
    except Exception as e:
        print(f"✗ Error converting model to profile: {e}")
        return False
    
    # Test validation
    print("Testing profile validation...")
    try:
        validation_errors = profile.validate()
        if validation_errors:
            print(f"✗ Profile validation failed: {validation_errors}")
            return False
        else:
            print("✓ Profile validation passed")
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        return False
    
    # Test roundtrip conversion
    print("Testing roundtrip conversion...")
    try:
        # Convert original -> model -> profile
        model2 = store._prompt_profile_to_model(profile)
        model2.corpus_id = "test_corpus"
        model2.task_type = "qa"
        model2.performance_stats = {}
        model2.metadata = {}
        model2.created_at = profile.created_at
        
        roundtrip_profile = store._model_to_prompt_profile(model2)
        
        # Compare key attributes
        assert roundtrip_profile.name == profile.name
        assert roundtrip_profile.description == profile.description
        assert roundtrip_profile.author == profile.author
        assert len(roundtrip_profile.role_prompts) == len(profile.role_prompts)
        
        print("✓ Roundtrip conversion preserved data correctly")
    except Exception as e:
        print(f"✗ Error during roundtrip conversion: {e}")
        return False
    
    print("\n✓ All data structure integration tests passed!")
    return True


def test_store_class_methods():
    """Test PromptProfileStore class methods without database."""
    print("\nTesting PromptProfileStore class methods...")
    
    store = PromptProfileStore()
    
    # Test initialization
    print("✓ Store initialized successfully")
    
    # Test logger assignment
    assert hasattr(store, 'logger')
    print("✓ Logger assigned correctly")
    
    # Test method presence
    required_methods = [
        'create', 'get_by_id', 'update', 'delete',
        'list_by_corpus_and_task', 'get_latest_by_corpus_and_task',
        'search_profiles', 'get_profile_lineage', 'create_derived_profile',
        'get_statistics'
    ]
    
    for method_name in required_methods:
        assert hasattr(store, method_name)
        assert callable(getattr(store, method_name))
    
    print(f"✓ All {len(required_methods)} required methods present")
    
    print("\n✓ All class method tests passed!")
    return True


def generate_integration_report():
    """Generate a comprehensive integration report."""
    print("=" * 80)
    print("PROMPTPROFILESTORE INTEGRATION TEST REPORT")
    print("=" * 80)
    
    tests = [
        ("Data Structure Integration", test_data_structure_integration),
        ("Store Class Methods", test_store_class_methods)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results[test_name] = {"status": "PASS" if success else "FAIL", "error": None}
            if not success:
                all_passed = False
        except Exception as e:
            results[test_name] = {"status": "ERROR", "error": str(e)}
            all_passed = False
            print(f"✗ {test_name} failed with error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = result["status"]
        print(f"{test_name:<35} | {status}")
        if result["error"]:
            print(f"{'Error:':<35} | {result['error']}")
    
    print(f"\nOverall Status: {'PASS' if all_passed else 'FAIL'}")
    
    # Generate JSON report
    report = {
        "timestamp": "2024-12-02T15:45:00Z",
        "overall_status": "PASS" if all_passed else "FAIL",
        "test_results": results,
        "summary": {
            "total_tests": len(tests),
            "passed": sum(1 for r in results.values() if r["status"] == "PASS"),
            "failed": sum(1 for r in results.values() if r["status"] == "FAIL"),
            "errors": sum(1 for r in results.values() if r["status"] == "ERROR")
        }
    }
    
    with open("prompt_profile_store_integration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: prompt_profile_store_integration_report.json")
    
    return all_passed


if __name__ == "__main__":
    success = generate_integration_report()
    sys.exit(0 if success else 1)
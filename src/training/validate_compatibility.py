"""
Compatibility validation script for training data structures.

This script validates that the new training data structures integrate
seamlessly with existing code patterns without breaking functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.utils import AgentResponse, AgentLogger, validate_agent_response
from training.data_structures import (
    RolePrompt,
    PromptProfile, 
    TrainingStep,
    enhance_agent_response_with_training,
    extract_training_metadata
)


def test_existing_agent_response():
    """Test that existing AgentResponse functionality is unchanged."""
    print("Testing existing AgentResponse functionality...")
    
    # Create AgentResponse exactly as existing code would
    response = AgentResponse(
        content="This is a test response for validation",
        reasoning="Testing backward compatibility",
        confidence=0.85,
        sources=["test_source.txt"],
        metadata={"test": "value"}
    )
    
    # Test existing methods work unchanged
    assert response.content == "This is a test response for validation"
    assert response.confidence == 0.85
    assert response.sources == ["test_source.txt"]
    assert response.metadata == {"test": "value"}
    
    # Test serialization methods work unchanged
    response_dict = response.to_dict()
    assert "content" in response_dict
    assert "confidence" in response_dict
    assert "sources" in response_dict
    assert "metadata" in response_dict
    
    json_str = response.to_json()
    assert "content" in json_str
    assert "confidence" in json_str
    
    print("✓ AgentResponse backward compatibility verified")


def test_agent_logger_integration():
    """Test that AgentLogger can still work with enhanced responses."""
    print("Testing AgentLogger integration...")
    
    # Create AgentLogger as existing code would
    logger = AgentLogger("test_agent")
    
    # Create standard AgentResponse
    response = AgentResponse(
        content="Test response for logger",
        reasoning="Testing logger integration",
        confidence=0.9
    )
    
    # Enhance with training metadata
    training_meta = {"step_id": "test-123", "profile_id": "profile-456"}
    enhanced_response = enhance_agent_response_with_training(response, training_meta)
    
    # Test that logger can handle enhanced response
    try:
        logger.log_response("Test question", enhanced_response)
        print("✓ AgentLogger integration verified")
    except Exception as e:
        print(f"✗ AgentLogger integration failed: {e}")
        raise


def test_validation_integration():
    """Test that existing validation works with enhanced responses."""
    print("Testing validation integration...")
    
    # Create standard response
    response = AgentResponse(
        content="Valid response",
        confidence=0.8
    )
    
    # Test existing validation
    errors = validate_agent_response(response)
    assert errors == []  # Should be valid
    
    # Enhance with training metadata
    enhanced_response = enhance_agent_response_with_training(
        response, 
        {"training_iteration": 1}
    )
    
    # Test validation still works
    errors = validate_agent_response(enhanced_response)
    assert errors == []  # Should still be valid
    
    print("✓ Validation integration verified")


def test_training_structures_with_existing_patterns():
    """Test training structures follow existing patterns."""
    print("Testing training structures follow existing patterns...")
    
    # Test that RolePrompt follows dataclass patterns like AgentResponse
    role_prompt = RolePrompt(
        role="worker",
        prompt_text="Test prompt"
    )
    
    # Should have similar serialization methods
    assert hasattr(role_prompt, 'to_dict')
    assert hasattr(role_prompt, 'to_json')
    assert hasattr(role_prompt, 'validate')
    
    dict_data = role_prompt.to_dict()
    assert isinstance(dict_data, dict)
    
    json_str = role_prompt.to_json()
    assert isinstance(json_str, str)
    
    # Test PromptProfile follows similar patterns
    profile = PromptProfile(name="test_profile")
    profile.add_role_prompt(role_prompt)
    
    assert hasattr(profile, 'to_dict')
    assert hasattr(profile, 'to_json')
    assert hasattr(profile, 'validate')
    
    # Test TrainingStep works with existing AgentResponse
    step = TrainingStep(question="Test question?")
    
    existing_response = AgentResponse(content="Test answer", confidence=0.8)
    step.add_agent_response(existing_response)
    
    assert len(step.agent_responses) == 1
    assert step.agent_responses[0].content == "Test answer"
    
    print("✓ Training structures follow existing patterns")


def test_integration_workflow():
    """Test complete integration workflow."""
    print("Testing complete integration workflow...")
    
    # Simulate existing code creating AgentResponse
    worker_response = AgentResponse(
        content="Worker analysis of the problem",
        reasoning="Applied systematic analysis",
        confidence=0.85,
        sources=["knowledge_base.txt"]
    )
    
    reviewer_response = AgentResponse(
        content="Quality review completed", 
        reasoning="Checked for accuracy and completeness",
        confidence=0.90,
        metadata={"review_type": "quality_check"}
    )
    
    # Create training context for these responses
    profile = PromptProfile(name="integration_test")
    profile.add_role_prompt(RolePrompt(role="worker", prompt_text="Analyze systematically"))
    profile.add_role_prompt(RolePrompt(role="reviewer", prompt_text="Review for quality"))
    
    # Create training step using existing responses
    step = TrainingStep(
        question="Analyze this complex problem",
        prompt_profile_id=profile.profile_id
    )
    
    # Add existing responses to training step
    step.add_agent_response(worker_response)
    step.add_agent_response(reviewer_response)
    
    # Add evaluation metrics
    step.set_evaluation_score("accuracy", 0.88)
    step.set_evaluation_score("completeness", 0.92)
    
    # Mark as completed
    step.mark_completed()
    
    # Verify everything works
    assert len(step.agent_responses) == 2
    assert step.status == "completed"
    assert step.get_average_score() == 0.9
    
    # Verify serialization works end-to-end
    step_json = step.to_json()
    restored_step = TrainingStep.from_json(step_json)
    
    assert len(restored_step.agent_responses) == 2
    assert restored_step.agent_responses[0].content == worker_response.content
    assert restored_step.agent_responses[1].metadata == reviewer_response.metadata
    
    print("✓ Complete integration workflow verified")


def main():
    """Run all compatibility validation tests."""
    print("Running Training Data Structures Compatibility Validation")
    print("=" * 60)
    
    try:
        test_existing_agent_response()
        test_agent_logger_integration()
        test_validation_integration()
        test_training_structures_with_existing_patterns()
        test_integration_workflow()
        
        print("\n" + "=" * 60)
        print("✅ ALL COMPATIBILITY TESTS PASSED")
        print("Training data structures are fully backward compatible")
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ COMPATIBILITY TEST FAILED: {e}")
        print("Issues found that need to be addressed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
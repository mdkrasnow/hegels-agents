"""
Comprehensive unit tests for training data structures.

Tests cover normal operation, edge cases, error conditions,
serialization/deserialization, and backward compatibility.
"""

import json
import uuid
import pytest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

# Import the structures we're testing
from .data_structures import (
    RolePrompt,
    PromptProfile,
    TrainingStep,
    enhance_agent_response_with_training,
    extract_training_metadata,
    validate_all_structures
)

# Import existing AgentResponse for compatibility testing
from agents.utils import AgentResponse


class TestRolePrompt:
    """Test suite for RolePrompt data structure."""
    
    def test_basic_creation(self):
        """Test basic RolePrompt creation."""
        prompt = RolePrompt(
            role="worker",
            prompt_text="You are a helpful AI assistant."
        )
        
        assert prompt.role == "worker"
        assert prompt.prompt_text == "You are a helpful AI assistant."
        assert prompt.version == "1.0"  # default
        assert isinstance(prompt.created_at, datetime)
        assert prompt.metadata == {}
    
    def test_creation_with_all_fields(self):
        """Test RolePrompt creation with all fields."""
        created_time = datetime.utcnow()
        metadata = {"experiment": "test_1", "iteration": 1}
        
        prompt = RolePrompt(
            role="reviewer",
            prompt_text="You are a critical reviewer.",
            description="A prompt for critical review",
            version="2.1",
            author="test_author",
            created_at=created_time,
            metadata=metadata
        )
        
        assert prompt.role == "reviewer"
        assert prompt.description == "A prompt for critical review"
        assert prompt.version == "2.1"
        assert prompt.author == "test_author"
        assert prompt.created_at == created_time
        assert prompt.metadata == metadata
    
    def test_invalid_creation(self):
        """Test RolePrompt creation with invalid data."""
        # Empty role
        with pytest.raises(ValueError, match="Role must be a non-empty string"):
            RolePrompt(role="", prompt_text="test")
        
        # None role
        with pytest.raises(ValueError, match="Role must be a non-empty string"):
            RolePrompt(role=None, prompt_text="test")
        
        # Empty prompt text
        with pytest.raises(ValueError, match="Prompt text must be a non-empty string"):
            RolePrompt(role="worker", prompt_text="")
        
        # None prompt text
        with pytest.raises(ValueError, match="Prompt text must be a non-empty string"):
            RolePrompt(role="worker", prompt_text=None)
    
    def test_validation(self):
        """Test RolePrompt validation."""
        # Valid prompt
        prompt = RolePrompt(role="worker", prompt_text="Valid prompt")
        assert prompt.validate() == []
        
        # Invalid prompts - need to bypass constructor validation
        prompt = RolePrompt.__new__(RolePrompt)
        prompt.role = ""
        prompt.prompt_text = "Valid prompt"
        prompt.version = "1.0"
        prompt.created_at = datetime.utcnow()
        prompt.metadata = {}
        errors = prompt.validate()
        assert "Role cannot be empty" in errors
        
        prompt.role = "valid"
        prompt.prompt_text = ""
        errors = prompt.validate()
        assert "Prompt text cannot be empty" in errors
        
        prompt.prompt_text = "x" * 50001  # Too long
        errors = prompt.validate()
        assert any("Prompt text is too long" in error for error in errors)
    
    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        original = RolePrompt(
            role="worker",
            prompt_text="Test prompt",
            description="Test description",
            author="test_author",
            metadata={"key": "value"}
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data['role'] == "worker"
        assert data['prompt_text'] == "Test prompt"
        assert data['description'] == "Test description"
        assert data['author'] == "test_author"
        assert data['metadata'] == {"key": "value"}
        assert data['created_at'] is not None
        
        # Test to_json and from_json
        json_str = original.to_json()
        restored = RolePrompt.from_json(json_str)
        
        assert restored.role == original.role
        assert restored.prompt_text == original.prompt_text
        assert restored.description == original.description
        assert restored.author == original.author
        assert restored.metadata == original.metadata
        assert abs((restored.created_at - original.created_at).total_seconds()) < 1
    
    def test_from_dict_edge_cases(self):
        """Test from_dict with various date formats."""
        # ISO format with Z
        data = {
            'role': 'worker',
            'prompt_text': 'test',
            'created_at': '2023-01-01T12:00:00Z'
        }
        prompt = RolePrompt.from_dict(data)
        assert prompt.created_at is not None
        
        # ISO format without Z
        data['created_at'] = '2023-01-01T12:00:00'
        prompt = RolePrompt.from_dict(data)
        assert prompt.created_at is not None
        
        # Datetime object
        data['created_at'] = datetime.utcnow()
        prompt = RolePrompt.from_dict(data)
        assert prompt.created_at is not None


class TestPromptProfile:
    """Test suite for PromptProfile data structure."""
    
    def test_basic_creation(self):
        """Test basic PromptProfile creation."""
        profile = PromptProfile(name="test_profile")
        
        assert profile.name == "test_profile"
        assert isinstance(uuid.UUID(profile.profile_id), uuid.UUID)
        assert profile.version == "1.0"
        assert isinstance(profile.created_at, datetime)
        assert profile.role_prompts == {}
        assert profile.tags == []
    
    def test_creation_with_all_fields(self):
        """Test PromptProfile creation with all fields."""
        profile_id = str(uuid.uuid4())
        created_time = datetime.utcnow()
        
        profile = PromptProfile(
            profile_id=profile_id,
            name="comprehensive_profile",
            description="A comprehensive test profile",
            version="2.0",
            author="test_author",
            created_at=created_time,
            tags=["test", "experimental"],
            metadata={"experiment_type": "dialectical"}
        )
        
        assert profile.profile_id == profile_id
        assert profile.name == "comprehensive_profile"
        assert profile.description == "A comprehensive test profile"
        assert profile.version == "2.0"
        assert profile.author == "test_author"
        assert profile.created_at == created_time
        assert profile.tags == ["test", "experimental"]
        assert profile.metadata == {"experiment_type": "dialectical"}
    
    def test_invalid_creation(self):
        """Test PromptProfile creation with invalid data."""
        # Invalid UUID
        with pytest.raises(ValueError, match="Invalid UUID"):
            PromptProfile(profile_id="invalid-uuid", name="test")
        
        # Empty name
        with pytest.raises(ValueError, match="Profile name must be a non-empty string"):
            PromptProfile(name="")
        
        # None name
        with pytest.raises(ValueError, match="Profile name must be a non-empty string"):
            PromptProfile(name=None)
    
    def test_role_prompt_management(self):
        """Test adding, removing, and getting role prompts."""
        profile = PromptProfile(name="test_profile")
        
        # Add role prompt
        worker_prompt = RolePrompt(role="worker", prompt_text="Worker prompt")
        profile.add_role_prompt(worker_prompt)
        
        assert len(profile.role_prompts) == 1
        assert "worker" in profile.role_prompts
        assert profile.get_role_prompt("worker") == worker_prompt
        assert profile.get_roles() == ["worker"]
        
        # Add another role prompt
        reviewer_prompt = RolePrompt(role="reviewer", prompt_text="Reviewer prompt")
        profile.add_role_prompt(reviewer_prompt)
        
        assert len(profile.role_prompts) == 2
        assert set(profile.get_roles()) == {"worker", "reviewer"}
        
        # Remove role prompt
        assert profile.remove_role_prompt("worker") == True
        assert len(profile.role_prompts) == 1
        assert profile.get_role_prompt("worker") is None
        
        # Try to remove non-existent role
        assert profile.remove_role_prompt("nonexistent") == False
    
    def test_invalid_role_prompt_operations(self):
        """Test error handling for role prompt operations."""
        profile = PromptProfile(name="test_profile")
        
        # Add invalid object
        with pytest.raises(ValueError, match="Must provide a valid RolePrompt instance"):
            profile.add_role_prompt("not a role prompt")
        
        # Add invalid role prompt (bypass RolePrompt constructor validation)
        invalid_prompt = RolePrompt.__new__(RolePrompt)
        invalid_prompt.role = ""
        invalid_prompt.prompt_text = "test"
        invalid_prompt.version = "1.0"
        invalid_prompt.created_at = datetime.utcnow()
        invalid_prompt.metadata = {}
        
        with pytest.raises(ValueError, match="Invalid RolePrompt"):
            profile.add_role_prompt(invalid_prompt)
    
    def test_validation(self):
        """Test PromptProfile validation."""
        # Valid profile
        profile = PromptProfile(name="valid_profile")
        assert profile.validate() == []
        
        # Add valid role prompt
        profile.add_role_prompt(RolePrompt(role="worker", prompt_text="Valid prompt"))
        assert profile.validate() == []
        
        # Test validation errors - bypass constructor validation
        invalid_profile = PromptProfile.__new__(PromptProfile)
        invalid_profile.profile_id = "invalid-uuid"
        invalid_profile.name = ""
        invalid_profile.role_prompts = {}
        invalid_profile.created_at = datetime.utcnow()
        invalid_profile.tags = []
        invalid_profile.metadata = {}
        invalid_profile.version = "1.0"
        
        errors = invalid_profile.validate()
        assert any("Profile ID must be a valid UUID" in error for error in errors)
        assert "Profile name cannot be empty" in errors
    
    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        profile = PromptProfile(
            name="test_profile",
            description="Test profile",
            tags=["test"],
            metadata={"key": "value"}
        )
        
        # Add role prompts
        worker_prompt = RolePrompt(role="worker", prompt_text="Worker prompt")
        reviewer_prompt = RolePrompt(role="reviewer", prompt_text="Reviewer prompt")
        profile.add_role_prompt(worker_prompt)
        profile.add_role_prompt(reviewer_prompt)
        
        # Serialize and deserialize
        json_str = profile.to_json()
        restored = PromptProfile.from_json(json_str)
        
        assert restored.name == profile.name
        assert restored.description == profile.description
        assert restored.tags == profile.tags
        assert restored.metadata == profile.metadata
        assert restored.profile_id == profile.profile_id
        assert len(restored.role_prompts) == 2
        assert "worker" in restored.role_prompts
        assert "reviewer" in restored.role_prompts
    
    def test_file_operations(self):
        """Test saving to and loading from files."""
        profile = PromptProfile(name="test_profile", description="File test")
        profile.add_role_prompt(RolePrompt(role="worker", prompt_text="Test prompt"))
        
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_profile.json"
            
            # Save to file
            profile.save_to_file(file_path)
            assert file_path.exists()
            
            # Load from file
            loaded_profile = PromptProfile.load_from_file(file_path)
            assert loaded_profile.name == profile.name
            assert loaded_profile.description == profile.description
            assert loaded_profile.profile_id == profile.profile_id
            assert len(loaded_profile.role_prompts) == 1
            
            # Test loading non-existent file
            with pytest.raises(FileNotFoundError):
                PromptProfile.load_from_file(Path(temp_dir) / "nonexistent.json")


class TestTrainingStep:
    """Test suite for TrainingStep data structure."""
    
    def test_basic_creation(self):
        """Test basic TrainingStep creation."""
        step = TrainingStep(
            question="What is 2+2?",
            prompt_profile_id=str(uuid.uuid4())
        )
        
        assert isinstance(uuid.UUID(step.step_id), uuid.UUID)
        assert step.step_number == 0
        assert step.step_type == "training"
        assert step.question == "What is 2+2?"
        assert step.agent_responses == []
        assert step.evaluation_scores == {}
        assert step.status == "pending"
        assert isinstance(step.created_at, datetime)
        assert step.completed_at is None
    
    def test_creation_with_all_fields(self):
        """Test TrainingStep creation with all fields."""
        step_id = str(uuid.uuid4())
        profile_id = str(uuid.uuid4())
        created_time = datetime.utcnow()
        
        response1 = AgentResponse(content="Response 1", confidence=0.8)
        response2 = AgentResponse(content="Response 2", confidence=0.9)
        
        step = TrainingStep(
            step_id=step_id,
            step_number=5,
            step_type="evaluation",
            prompt_profile_id=profile_id,
            question="Test question?",
            expected_response="Expected answer",
            agent_responses=[response1, response2],
            evaluation_scores={"accuracy": 0.85, "relevance": 0.90},
            training_metadata={"model": "test", "temperature": 0.7},
            created_at=created_time,
            status="completed"
        )
        
        assert step.step_id == step_id
        assert step.step_number == 5
        assert step.step_type == "evaluation"
        assert step.prompt_profile_id == profile_id
        assert step.question == "Test question?"
        assert step.expected_response == "Expected answer"
        assert len(step.agent_responses) == 2
        assert step.evaluation_scores == {"accuracy": 0.85, "relevance": 0.90}
        assert step.training_metadata == {"model": "test", "temperature": 0.7}
        assert step.created_at == created_time
        assert step.status == "completed"
    
    def test_invalid_creation(self):
        """Test TrainingStep creation with invalid data."""
        # Invalid UUID
        with pytest.raises(ValueError, match="Invalid UUID"):
            TrainingStep(step_id="invalid-uuid", question="test")
        
        # Invalid step type
        with pytest.raises(ValueError, match="Step type must be one of"):
            TrainingStep(step_type="invalid", question="test")
        
        # Invalid status
        with pytest.raises(ValueError, match="Status must be one of"):
            TrainingStep(question="test", status="invalid")
    
    def test_agent_response_management(self):
        """Test adding agent responses."""
        step = TrainingStep(question="Test question?")
        
        response1 = AgentResponse(content="Response 1")
        response2 = AgentResponse(content="Response 2")
        
        # Add responses
        step.add_agent_response(response1)
        assert len(step.agent_responses) == 1
        
        step.add_agent_response(response2)
        assert len(step.agent_responses) == 2
        
        # Try to add invalid response
        with pytest.raises(ValueError, match="Must provide a valid AgentResponse instance"):
            step.add_agent_response("not a response")
    
    def test_evaluation_scoring(self):
        """Test evaluation score management."""
        step = TrainingStep(question="Test question?")
        
        # Set valid scores
        step.set_evaluation_score("accuracy", 0.85)
        step.set_evaluation_score("relevance", 0.90)
        
        assert step.evaluation_scores["accuracy"] == 0.85
        assert step.evaluation_scores["relevance"] == 0.90
        assert step.get_average_score() == 0.875
        
        # Invalid score types
        with pytest.raises(ValueError, match="Score must be a number"):
            step.set_evaluation_score("test", "not a number")
        
        # Out of range scores
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            step.set_evaluation_score("test", 1.5)
        
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            step.set_evaluation_score("test", -0.1)
    
    def test_status_management(self):
        """Test status change methods."""
        step = TrainingStep(question="Test question?")
        
        assert step.status == "pending"
        assert step.completed_at is None
        
        # Mark completed
        step.mark_completed()
        assert step.status == "completed"
        assert isinstance(step.completed_at, datetime)
        
        # Create new step for failure test
        step2 = TrainingStep(question="Test question?")
        step2.mark_failed("Test error")
        assert step2.status == "failed"
        assert isinstance(step2.completed_at, datetime)
        assert step2.training_metadata["error"] == "Test error"
    
    def test_validation(self):
        """Test TrainingStep validation."""
        # Valid step
        step = TrainingStep(question="Valid question?")
        assert step.validate() == []
        
        # Add valid response and score
        step.add_agent_response(AgentResponse(content="Valid response"))
        step.set_evaluation_score("accuracy", 0.8)
        assert step.validate() == []
        
        # Test validation errors - bypass constructor validation
        invalid_step = TrainingStep.__new__(TrainingStep)
        invalid_step.step_id = "invalid-uuid"
        invalid_step.step_number = 0
        invalid_step.step_type = "invalid_type"
        invalid_step.prompt_profile_id = ""
        invalid_step.question = ""
        invalid_step.agent_responses = ["not a response"]
        invalid_step.evaluation_scores = {"test": 1.5}
        invalid_step.training_metadata = {}
        invalid_step.created_at = datetime.utcnow()
        invalid_step.completed_at = None
        invalid_step.status = "invalid_status"
        
        errors = invalid_step.validate()
        assert any("Step ID must be a valid UUID" in error for error in errors)
        assert "Step type must be one of" in str(errors)
        assert "Status must be one of" in str(errors)
        assert "Question cannot be empty" in errors
        assert "Score for 'test' must be between 0.0 and 1.0" in errors
        assert "Agent response 0 must be an AgentResponse instance" in errors
    
    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        step = TrainingStep(
            step_number=1,
            step_type="evaluation",
            question="Test question?",
            expected_response="Expected answer"
        )
        
        # Add responses and scores
        response = AgentResponse(content="Test response", confidence=0.8)
        step.add_agent_response(response)
        step.set_evaluation_score("accuracy", 0.9)
        step.mark_completed()
        
        # Serialize and deserialize
        json_str = step.to_json()
        restored = TrainingStep.from_json(json_str)
        
        assert restored.step_id == step.step_id
        assert restored.step_number == step.step_number
        assert restored.step_type == step.step_type
        assert restored.question == step.question
        assert restored.expected_response == step.expected_response
        assert len(restored.agent_responses) == 1
        assert restored.agent_responses[0].content == "Test response"
        assert restored.agent_responses[0].confidence == 0.8
        assert restored.evaluation_scores == {"accuracy": 0.9}
        assert restored.status == "completed"
        assert restored.completed_at is not None


class TestBackwardCompatibility:
    """Test suite for backward compatibility features."""
    
    def test_enhance_agent_response(self):
        """Test enhancing AgentResponse with training metadata."""
        original_response = AgentResponse(
            content="Test response",
            reasoning="Test reasoning", 
            confidence=0.8,
            metadata={"original_key": "original_value"}
        )
        
        training_metadata = {
            "step_id": str(uuid.uuid4()),
            "profile_id": str(uuid.uuid4()),
            "iteration": 1
        }
        
        enhanced_response = enhance_agent_response_with_training(
            original_response,
            training_metadata
        )
        
        # Original fields should be preserved
        assert enhanced_response.content == original_response.content
        assert enhanced_response.reasoning == original_response.reasoning
        assert enhanced_response.confidence == original_response.confidence
        
        # Original metadata should be preserved
        assert enhanced_response.metadata["original_key"] == "original_value"
        
        # Training metadata should be added
        assert enhanced_response.metadata["training"] == training_metadata
        
        # Original response should be unchanged
        assert original_response.metadata.get("training") is None
    
    def test_extract_training_metadata(self):
        """Test extracting training metadata from AgentResponse."""
        # Response without training metadata
        response_without = AgentResponse(content="Test")
        assert extract_training_metadata(response_without) == {}
        
        # Response with training metadata
        training_data = {"step_id": "123", "iteration": 1}
        response_with = AgentResponse(
            content="Test",
            metadata={"training": training_data, "other": "data"}
        )
        
        extracted = extract_training_metadata(response_with)
        assert extracted == training_data
        
        # Response with metadata but no training data
        response_other = AgentResponse(
            content="Test",
            metadata={"other": "data"}
        )
        assert extract_training_metadata(response_other) == {}
    
    def test_agentresponse_in_training_step(self):
        """Test that existing AgentResponse works seamlessly in TrainingStep."""
        # Create AgentResponse using existing constructor
        response = AgentResponse(
            content="This is a test response",
            reasoning="I reasoned about this",
            confidence=0.85,
            sources=["source1", "source2"],
            metadata={"model": "test_model"}
        )
        
        # Use in TrainingStep
        step = TrainingStep(question="Test question?")
        step.add_agent_response(response)
        
        assert len(step.agent_responses) == 1
        assert step.agent_responses[0].content == "This is a test response"
        assert step.agent_responses[0].confidence == 0.85
        
        # Serialize and deserialize
        json_str = step.to_json()
        restored_step = TrainingStep.from_json(json_str)
        
        restored_response = restored_step.agent_responses[0]
        assert restored_response.content == response.content
        assert restored_response.reasoning == response.reasoning
        assert restored_response.confidence == response.confidence
        assert restored_response.sources == response.sources
        assert restored_response.metadata == response.metadata


class TestValidationUtilities:
    """Test suite for validation utility functions."""
    
    def test_validate_all_structures(self):
        """Test the validate_all_structures utility function."""
        # Create mix of valid and invalid structures
        valid_prompt = RolePrompt(role="worker", prompt_text="Valid prompt")
        valid_profile = PromptProfile(name="Valid profile")
        valid_step = TrainingStep(question="Valid question?")
        
        # Create invalid structures by bypassing constructors
        invalid_prompt = RolePrompt.__new__(RolePrompt)
        invalid_prompt.role = ""
        invalid_prompt.prompt_text = "test"
        invalid_prompt.version = "1.0"
        invalid_prompt.created_at = datetime.utcnow()
        invalid_prompt.metadata = {}
        
        # Test with all valid structures
        results = validate_all_structures(valid_prompt, valid_profile, valid_step)
        assert results == {}
        
        # Test with some invalid structures
        results = validate_all_structures(valid_prompt, invalid_prompt, valid_profile)
        assert "RolePrompt" in results or "RolePrompt_1" in results
        assert len(results) == 1  # Only invalid_prompt should have errors
        
        # Test with non-validatable object (should be ignored)
        results = validate_all_structures(valid_prompt, "not a structure", valid_profile)
        assert results == {}


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complete_training_workflow(self):
        """Test a complete training workflow using all structures."""
        # Create a prompt profile
        profile = PromptProfile(
            name="integration_test_profile",
            description="Profile for integration testing"
        )
        
        # Add role prompts
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="You are a helpful AI assistant focused on accuracy.",
            description="Primary worker prompt"
        )
        
        reviewer_prompt = RolePrompt(
            role="reviewer", 
            prompt_text="You are a critical reviewer who evaluates responses.",
            description="Reviewer prompt for quality control"
        )
        
        profile.add_role_prompt(worker_prompt)
        profile.add_role_prompt(reviewer_prompt)
        
        # Validate profile
        assert profile.validate() == []
        
        # Create training step
        step = TrainingStep(
            step_number=1,
            step_type="training",
            prompt_profile_id=profile.profile_id,
            question="What are the key principles of effective communication?",
            expected_response="Clear, concise, and context-appropriate messaging."
        )
        
        # Add agent responses
        worker_response = AgentResponse(
            content="Effective communication requires clarity, brevity, and audience awareness.",
            reasoning="I focused on the three most critical aspects of communication.",
            confidence=0.85,
            sources=["communication_theory.pdf"]
        )
        
        reviewer_response = AgentResponse(
            content="The response covers key aspects but could elaborate on feedback mechanisms.",
            reasoning="While accurate, the response misses the importance of bidirectional communication.",
            confidence=0.75,
            metadata={"review_focus": "completeness"}
        )
        
        step.add_agent_response(worker_response)
        step.add_agent_response(reviewer_response)
        
        # Add evaluation scores
        step.set_evaluation_score("accuracy", 0.90)
        step.set_evaluation_score("completeness", 0.80)
        step.set_evaluation_score("clarity", 0.95)
        
        # Mark step as completed
        step.mark_completed()
        
        # Validate everything
        validation_results = validate_all_structures(profile, step, worker_prompt, reviewer_prompt)
        assert validation_results == {}
        
        # Test serialization of complete workflow
        profile_json = profile.to_json()
        step_json = step.to_json()
        
        # Deserialize and verify
        restored_profile = PromptProfile.from_json(profile_json)
        restored_step = TrainingStep.from_json(step_json)
        
        assert restored_profile.name == profile.name
        assert len(restored_profile.role_prompts) == 2
        assert restored_step.step_type == "training"
        assert restored_step.status == "completed"
        assert len(restored_step.agent_responses) == 2
        assert len(restored_step.evaluation_scores) == 3
        assert abs(restored_step.get_average_score() - 0.8833333333333333) < 1e-10  # (0.9 + 0.8 + 0.95) / 3
    
    def test_backward_compatibility_in_workflow(self):
        """Test that new structures work with existing AgentResponse seamlessly."""
        # Use existing AgentResponse as it would be used currently
        response = AgentResponse(
            content="This response was created using the existing AgentResponse class",
            confidence=0.88
        )
        
        # Enhance with training metadata
        training_meta = {"experiment": "backward_compat", "version": "1.0"}
        enhanced_response = enhance_agent_response_with_training(response, training_meta)
        
        # Use in new training step
        step = TrainingStep(
            question="Test backward compatibility",
            prompt_profile_id=str(uuid.uuid4())
        )
        step.add_agent_response(enhanced_response)
        
        # Verify compatibility
        assert len(step.agent_responses) == 1
        stored_response = step.agent_responses[0]
        assert stored_response.content == response.content
        assert stored_response.confidence == response.confidence
        
        # Extract training metadata
        extracted_meta = extract_training_metadata(stored_response)
        assert extracted_meta == training_meta
        
        # Ensure serialization works
        json_data = step.to_json()
        restored_step = TrainingStep.from_json(json_data)
        
        restored_response = restored_step.agent_responses[0]
        restored_meta = extract_training_metadata(restored_response)
        assert restored_meta == training_meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
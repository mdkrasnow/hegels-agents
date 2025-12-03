"""
Unit Tests for ReflectionOptimizer

Comprehensive tests for the reflection-based prompt optimization system.
"""

import json
import pytest
import uuid
from unittest.mock import Mock, patch
from datetime import datetime

from src.training.optimizers.reflection_optimizer import ReflectionOptimizer, EditInstruction
from src.training.data_structures import PromptProfile, RolePrompt


class TestEditInstruction:
    """Test the EditInstruction class."""
    
    def test_edit_instruction_creation(self):
        """Test basic EditInstruction creation."""
        edit = EditInstruction(
            role="worker",
            edit_type="replace",
            target="old text",
            replacement="new text",
            reasoning="improves clarity",
            confidence=0.9
        )
        
        assert edit.role == "worker"
        assert edit.edit_type == "replace"
        assert edit.target == "old text"
        assert edit.replacement == "new text"
        assert edit.reasoning == "improves clarity"
        assert edit.confidence == 0.9
        assert len(edit.edit_id) == 8  # UUID prefix
    
    def test_edit_instruction_validation_valid(self):
        """Test validation with valid edit instruction."""
        edit = EditInstruction(
            role="reviewer",
            edit_type="append",
            target="",
            replacement="Additional guidance text",
            reasoning="Provides more context",
            confidence=0.8
        )
        
        errors = edit.validate()
        assert len(errors) == 0
    
    def test_edit_instruction_validation_invalid(self):
        """Test validation with invalid edit instruction."""
        edit = EditInstruction(
            role="",  # Empty role
            edit_type="invalid_type",  # Invalid type
            target="",
            replacement="",  # Empty replacement
            reasoning="test",
            confidence=1.5  # Invalid confidence
        )
        
        errors = edit.validate()
        assert len(errors) >= 4
        assert any("Role cannot be empty" in error for error in errors)
        assert any("Invalid edit_type" in error for error in errors)
        assert any("Replacement text cannot be empty" in error for error in errors)
        assert any("Confidence must be between 0.0 and 1.0" in error for error in errors)
    
    def test_edit_instruction_to_dict(self):
        """Test conversion to dictionary."""
        edit = EditInstruction(
            role="worker",
            edit_type="replace",
            target="old",
            replacement="new",
            reasoning="better",
            confidence=0.85
        )
        
        result = edit.to_dict()
        
        assert result["role"] == "worker"
        assert result["edit_type"] == "replace"
        assert result["target"] == "old"
        assert result["replacement"] == "new"
        assert result["reasoning"] == "better"
        assert result["confidence"] == 0.85
        assert "edit_id" in result


class TestReflectionOptimizer:
    """Test the ReflectionOptimizer class."""
    
    def create_test_profile(self):
        """Create a test PromptProfile for testing."""
        profile = PromptProfile(
            name="test_profile",
            description="Test profile for unit tests"
        )
        
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="You are a helpful assistant. Answer questions accurately.",
            description="Worker role prompt"
        )
        
        reviewer_prompt = RolePrompt(
            role="reviewer",
            prompt_text="Review and synthesize responses from workers.",
            description="Reviewer role prompt"
        )
        
        profile.add_role_prompt(worker_prompt)
        profile.add_role_prompt(reviewer_prompt)
        
        return profile
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_optimizer_initialization(self, mock_get_config):
        """Test ReflectionOptimizer initialization."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        
        assert optimizer.no_change_threshold == 0.8
        assert optimizer.max_edit_length == 500
        assert optimizer.max_edits_per_prompt == 3
        assert optimizer.min_confidence_threshold == 0.7
        assert optimizer.optimization_count == 0
        assert optimizer.successful_optimizations == 0
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_optimizer_custom_config(self, mock_get_config):
        """Test ReflectionOptimizer with custom configuration."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        custom_config = {
            'no_change_threshold': 0.9,
            'max_edit_length': 300,
            'max_edits_per_prompt': 2,
            'min_confidence_threshold': 0.8
        }
        
        optimizer = ReflectionOptimizer(config=custom_config)
        
        assert optimizer.no_change_threshold == 0.9
        assert optimizer.max_edit_length == 300
        assert optimizer.max_edits_per_prompt == 2
        assert optimizer.min_confidence_threshold == 0.8
    
    def test_should_optimize(self):
        """Test the should_optimize method."""
        with patch('training.optimizers.reflection_optimizer.get_config'):
            optimizer = ReflectionOptimizer()
            
            # Should optimize for low rewards
            assert optimizer.should_optimize(0.5) == True
            assert optimizer.should_optimize(0.7) == True
            
            # Should not optimize for high rewards
            assert optimizer.should_optimize(0.9) == False
            assert optimizer.should_optimize(1.0) == False
            
            # Test custom threshold
            assert optimizer.should_optimize(0.75, threshold=0.7) == False
            assert optimizer.should_optimize(0.65, threshold=0.7) == True
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_update_profile_high_reward_no_optimization(self, mock_get_config):
        """Test that profiles with high rewards are not optimized."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_test_profile()
        
        result = optimizer.update_profile(
            profile=profile,
            query="What is the capital of France?",
            answer="Paris",
            gold_answer="Paris",
            reward=0.95,  # High reward, should skip optimization
            trace={}
        )
        
        # Should return the same profile
        assert result == profile
        assert optimizer.optimization_count == 1  # Counted but not applied
        assert optimizer.successful_optimizations == 0
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    @patch('training.optimizers.reflection_optimizer.genai.Client')
    def test_update_profile_with_optimization(self, mock_client_class, mock_get_config):
        """Test profile optimization with mocked LLM response."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # Mock LLM response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "analysis": "The response lacks specificity and reasoning",
            "edits": [
                {
                    "role": "worker",
                    "edit_type": "append",
                    "target": "",
                    "replacement": "Always provide detailed reasoning for your answers.",
                    "reasoning": "Improves answer quality by encouraging explanation",
                    "confidence": 0.9
                }
            ],
            "expected_improvement": "Better reasoning and explanation quality"
        })
        
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        optimizer = ReflectionOptimizer()
        original_profile = self.create_test_profile()
        
        result = optimizer.update_profile(
            profile=original_profile,
            query="What causes earthquakes?",
            answer="Tectonic plates",
            gold_answer="Earthquakes are caused by the sudden movement of tectonic plates...",
            reward=0.4,  # Low reward, should optimize
            trace={'worker_responses': []},
            metadata={'context': 'testing'}
        )
        
        # Should return a different, optimized profile
        assert result != original_profile
        assert result.profile_id != original_profile.profile_id
        assert "optimized" in result.name
        
        # Check optimization metadata
        assert 'optimization_strategy' in result.metadata
        assert result.metadata['optimization_strategy'] == 'reflection'
        assert result.metadata['parent_reward'] == 0.4
        
        # Check that prompts were modified
        worker_prompt = result.get_role_prompt("worker")
        assert worker_prompt is not None
        assert "Always provide detailed reasoning" in worker_prompt.prompt_text
        
        # Check statistics
        assert optimizer.optimization_count == 1
        assert optimizer.successful_optimizations == 1
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_build_reflection_context(self, mock_get_config):
        """Test reflection context building."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_test_profile()
        
        # Mock trace data
        mock_response = Mock()
        mock_response.content = "Sample response content"
        
        trace = {
            'worker_responses': [mock_response],
            'synthesis_response': mock_response
        }
        
        context = optimizer._build_reflection_context(
            profile=profile,
            query="What is machine learning?",
            answer="ML is a subset of AI",
            gold_answer="Machine learning is a method of data analysis...",
            reward=0.3,
            trace=trace,
            metadata={}
        )
        
        # Verify context contains expected sections
        assert "PERFORMANCE ANALYSIS CONTEXT" in context
        assert "What is machine learning?" in context
        assert "ML is a subset of AI" in context
        assert "Performance Score: 0.300" in context
        assert "Machine learning is a method of data analysis" in context
        assert "CURRENT PROMPT CONFIGURATION" in context
        assert "WORKER PROMPT:" in context
        assert "REVIEWER PROMPT:" in context
        assert "You are a helpful assistant" in context
        assert "EXECUTION DETAILS" in context
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    @patch('training.optimizers.reflection_optimizer.genai.Client')
    def test_generate_edit_suggestions_valid_response(self, mock_client_class, mock_get_config):
        """Test edit suggestion generation with valid LLM response."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # Mock valid LLM response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "analysis": "Analysis text",
            "edits": [
                {
                    "role": "worker",
                    "edit_type": "replace",
                    "target": "Answer questions accurately",
                    "replacement": "Answer questions with detailed explanations and evidence",
                    "reasoning": "Improves thoroughness",
                    "confidence": 0.85
                },
                {
                    "role": "reviewer",
                    "edit_type": "append",
                    "target": "",
                    "replacement": "Focus on identifying conflicting information.",
                    "reasoning": "Enhances conflict detection",
                    "confidence": 0.9
                }
            ],
            "expected_improvement": "Better analysis"
        })
        
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        optimizer = ReflectionOptimizer()
        context = "Test reflection context"
        
        suggestions = optimizer._generate_edit_suggestions(context)
        
        assert len(suggestions) == 2
        assert all(isinstance(edit, EditInstruction) for edit in suggestions)
        
        # Check first edit
        edit1 = suggestions[0]
        assert edit1.role == "worker"
        assert edit1.edit_type == "replace"
        assert edit1.confidence == 0.85
        
        # Check second edit
        edit2 = suggestions[1]
        assert edit2.role == "reviewer"
        assert edit2.edit_type == "append"
        assert edit2.confidence == 0.9
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    @patch('training.optimizers.reflection_optimizer.genai.Client')
    def test_generate_edit_suggestions_invalid_json(self, mock_client_class, mock_get_config):
        """Test edit suggestion generation with invalid JSON response."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"
        
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        optimizer = ReflectionOptimizer()
        context = "Test context"
        
        suggestions = optimizer._generate_edit_suggestions(context)
        
        assert suggestions == []
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_apply_edits_replace(self, mock_get_config):
        """Test applying replace edit to profile."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_test_profile()
        
        edit = EditInstruction(
            role="worker",
            edit_type="replace",
            target="Answer questions accurately",
            replacement="Answer questions with thorough analysis and evidence",
            reasoning="Improves depth",
            confidence=0.9
        )
        
        result = optimizer._apply_edits(profile, [edit])
        
        # Check that edit was applied
        worker_prompt = result.get_role_prompt("worker")
        assert "Answer questions with thorough analysis and evidence" in worker_prompt.prompt_text
        assert "Answer questions accurately" not in worker_prompt.prompt_text
        
        # Check metadata
        assert worker_prompt.metadata['optimization_applied'] == True
        assert edit.edit_id in worker_prompt.metadata['applied_edits']
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_apply_edits_append(self, mock_get_config):
        """Test applying append edit to profile."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_test_profile()
        
        edit = EditInstruction(
            role="worker",
            edit_type="append",
            target="",
            replacement="Always cite your sources when possible.",
            reasoning="Improves credibility",
            confidence=0.8
        )
        
        result = optimizer._apply_edits(profile, [edit])
        
        # Check that text was appended
        worker_prompt = result.get_role_prompt("worker")
        assert worker_prompt.prompt_text.endswith("Always cite your sources when possible.")
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_apply_edits_prepend(self, mock_get_config):
        """Test applying prepend edit to profile."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_test_profile()
        
        edit = EditInstruction(
            role="reviewer",
            edit_type="prepend",
            target="",
            replacement="IMPORTANT: Be thorough in your analysis.",
            reasoning="Emphasizes importance",
            confidence=0.85
        )
        
        result = optimizer._apply_edits(profile, [edit])
        
        # Check that text was prepended
        reviewer_prompt = result.get_role_prompt("reviewer")
        assert reviewer_prompt.prompt_text.startswith("IMPORTANT: Be thorough in your analysis.")
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_estimate_improvement(self, mock_get_config):
        """Test improvement estimation."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        
        # Test with no edits
        assert optimizer._estimate_improvement([]) == 0.0
        
        # Test with single high-confidence edit
        edit1 = EditInstruction("worker", "replace", "old", "new", "reason", 0.9)
        improvement = optimizer._estimate_improvement([edit1])
        assert 0 < improvement <= 0.2  # Should be positive but capped
        
        # Test with multiple edits
        edit2 = EditInstruction("reviewer", "append", "", "text", "reason", 0.8)
        edit3 = EditInstruction("worker", "prepend", "", "text", "reason", 0.7)
        improvement = optimizer._estimate_improvement([edit1, edit2, edit3])
        assert improvement > 0
        assert improvement <= 0.2  # Still capped at max
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_get_optimization_stats_empty(self, mock_get_config):
        """Test optimization statistics with no optimizations."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        stats = optimizer.get_optimization_stats()
        
        assert stats["optimizations_performed"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_improvement"] == 0.0
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_get_optimization_stats_with_data(self, mock_get_config):
        """Test optimization statistics with data."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        
        # Simulate some optimization history
        optimizer.optimization_count = 5
        optimizer.successful_optimizations = 3
        optimizer.total_improvement = 0.6
        optimizer.optimization_history = [
            {'optimization_time': 1.5},
            {'optimization_time': 2.0},
            {'optimization_time': 1.8}
        ]
        
        stats = optimizer.get_optimization_stats()
        
        assert stats["optimizations_performed"] == 5
        assert stats["successful_optimizations"] == 3
        assert stats["success_rate"] == 0.6
        assert stats["average_improvement"] == 0.2  # 0.6 / 3
        assert "average_optimization_time_seconds" in stats
        assert stats["recent_optimizations"] == 3
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_reset_stats(self, mock_get_config):
        """Test resetting optimization statistics."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        
        # Set some data
        optimizer.optimization_count = 10
        optimizer.successful_optimizations = 8
        optimizer.total_improvement = 1.5
        optimizer.optimization_history = [{'test': 'data'}]
        
        # Reset
        optimizer.reset_stats()
        
        # Verify reset
        assert optimizer.optimization_count == 0
        assert optimizer.successful_optimizations == 0
        assert optimizer.total_improvement == 0.0
        assert len(optimizer.optimization_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
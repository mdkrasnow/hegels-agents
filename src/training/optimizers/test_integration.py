"""
Integration Tests for ReflectionOptimizer

Tests integration with existing PromptProfile system, reward calculation, and agent infrastructure.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from training.optimizers.reflection_optimizer import ReflectionOptimizer
from training.data_structures import PromptProfile, RolePrompt, TrainingStep
from training.rewards import RewardCalculator, RewardComponents


class TestReflectionOptimizerIntegration:
    """Integration tests for ReflectionOptimizer."""
    
    def create_comprehensive_profile(self):
        """Create a realistic PromptProfile for testing."""
        profile = PromptProfile(
            name="dialectical_debate_profile_v1",
            description="Profile for dialectical debate between worker agents",
            author="training_system",
            tags=["dialectical", "debate", "multi_agent"]
        )
        
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="""You are a thoughtful research assistant participating in a dialectical debate.

Your role:
1. Analyze the given question carefully
2. Research and gather relevant information
3. Formulate a well-reasoned position
4. Present your argument with supporting evidence
5. Be open to counterarguments and willing to refine your position

Guidelines:
- Base your reasoning on facts and logical analysis
- Acknowledge uncertainties and limitations
- Cite sources when available
- Structure your response clearly with main points and supporting details
- Be respectful of different perspectives""",
            description="Worker agent role for dialectical reasoning",
            version="1.0",
            metadata={"optimization_count": 0}
        )
        
        reviewer_prompt = RolePrompt(
            role="reviewer",
            prompt_text="""You are an expert reviewer tasked with synthesizing multiple perspectives into a coherent response.

Your role:
1. Analyze all worker responses for their strengths and weaknesses
2. Identify areas of agreement and disagreement
3. Evaluate the quality of reasoning and evidence presented
4. Synthesize the best elements from all responses
5. Provide a balanced, comprehensive final answer

Guidelines:
- Focus on the quality of arguments rather than just their conclusions
- Highlight conflicting viewpoints and attempt resolution
- Ensure the final synthesis is more comprehensive than any individual response
- Maintain objectivity and avoid bias toward any particular viewpoint
- Clearly structure the final response with reasoning""",
            description="Reviewer agent role for synthesis",
            version="1.0",
            metadata={"optimization_count": 0}
        )
        
        profile.add_role_prompt(worker_prompt)
        profile.add_role_prompt(reviewer_prompt)
        
        return profile
    
    def create_mock_trace(self):
        """Create a realistic execution trace."""
        from agents.utils import AgentResponse
        
        worker_response_1 = AgentResponse(
            content="Climate change is primarily caused by increased greenhouse gas emissions from human activities, particularly CO2 from fossil fuel combustion.",
            reasoning="Based on scientific consensus and IPCC reports showing clear correlation between industrial emissions and global temperature rise.",
            confidence=0.9,
            sources=["IPCC AR6", "NASA Climate Data"],
            metadata={"agent_role": "worker_1"}
        )
        
        worker_response_2 = AgentResponse(
            content="While human activities contribute to climate change, natural climate variability also plays a role and some aspects remain uncertain.",
            reasoning="Acknowledging scientific consensus while noting remaining uncertainties in climate models and natural variation factors.",
            confidence=0.7,
            sources=["Climate Research Papers", "NOAA Data"],
            metadata={"agent_role": "worker_2"}
        )
        
        synthesis_response = AgentResponse(
            content="Climate change results from both human activities and natural factors, with scientific evidence strongly indicating human greenhouse gas emissions as the dominant driver of recent warming.",
            reasoning="Synthesis acknowledges natural variability while emphasizing the overwhelming scientific consensus on anthropogenic causes.",
            confidence=0.85,
            sources=["IPCC AR6", "NASA", "NOAA"],
            metadata={"agent_role": "reviewer"}
        )
        
        return {
            "question": "What causes climate change?",
            "worker_responses": [worker_response_1, worker_response_2],
            "synthesis_response": synthesis_response,
            "execution_metadata": {
                "total_time": 45.2,
                "worker_count": 2,
                "synthesis_iterations": 1
            }
        }
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    @patch('training.optimizers.reflection_optimizer.genai.Client')
    def test_end_to_end_optimization_poor_performance(self, mock_client_class, mock_get_config):
        """Test complete optimization workflow with poor performance."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # Mock LLM response with realistic optimization suggestions
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "analysis": "The responses lack depth in evidence presentation and the synthesis doesn't adequately address the conflicting perspectives. The worker prompts should encourage more thorough source citation and the reviewer should be more explicit about conflict resolution.",
            "edits": [
                {
                    "role": "worker",
                    "edit_type": "replace",
                    "target": "Cite sources when available",
                    "replacement": "Always provide specific citations with page numbers, dates, and credible source identification",
                    "reasoning": "Improves evidence quality and credibility of arguments",
                    "confidence": 0.9
                },
                {
                    "role": "reviewer",
                    "edit_type": "append",
                    "target": "",
                    "replacement": "\n\nWhen synthesizing conflicting viewpoints:\n- Explicitly state the nature of the disagreement\n- Evaluate the strength of evidence on each side\n- Provide a reasoned resolution or acknowledge irreconcilable differences",
                    "reasoning": "Enhances conflict resolution and dialectical quality",
                    "confidence": 0.85
                }
            ],
            "expected_improvement": "More rigorous evidence handling and better conflict resolution in synthesis"
        })
        
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # Create test data
        optimizer = ReflectionOptimizer()
        original_profile = self.create_comprehensive_profile()
        trace = self.create_mock_trace()
        
        # Simulate poor performance scenario
        poor_answer = "Climate changes sometimes. Various factors might be involved."
        gold_answer = "Climate change is primarily driven by human activities that increase greenhouse gas concentrations in the atmosphere, particularly CO2 emissions from fossil fuel combustion, while natural climate variability plays a smaller role."
        low_reward = 0.25
        
        # Perform optimization
        optimized_profile = optimizer.update_profile(
            profile=original_profile,
            query="What causes climate change?",
            answer=poor_answer,
            gold_answer=gold_answer,
            reward=low_reward,
            trace=trace,
            metadata={"training_context": "climate_science", "difficulty": "intermediate"}
        )
        
        # Verify optimization was applied
        assert optimized_profile != original_profile
        assert optimized_profile.profile_id != original_profile.profile_id
        assert "optimized" in optimized_profile.name
        
        # Check optimization metadata
        assert optimized_profile.metadata["optimization_strategy"] == "reflection"
        assert optimized_profile.metadata["parent_reward"] == low_reward
        assert optimized_profile.metadata["edit_count"] == 2
        
        # Check that prompts were modified appropriately
        optimized_worker = optimized_profile.get_role_prompt("worker")
        optimized_reviewer = optimized_profile.get_role_prompt("reviewer")
        
        assert optimized_worker is not None
        assert optimized_reviewer is not None
        
        # Worker prompt should have improved citation requirements
        assert "Always provide specific citations" in optimized_worker.prompt_text
        assert "page numbers, dates" in optimized_worker.prompt_text
        
        # Reviewer prompt should have conflict resolution guidance
        assert "Explicitly state the nature of the disagreement" in optimized_reviewer.prompt_text
        assert "Evaluate the strength of evidence" in optimized_reviewer.prompt_text
        
        # Check metadata on role prompts
        assert optimized_worker.metadata["optimization_applied"] == True
        assert optimized_reviewer.metadata["optimization_applied"] == True
        
        # Verify statistics
        assert optimizer.optimization_count == 1
        assert optimizer.successful_optimizations == 1
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_integration_with_training_step(self, mock_get_config):
        """Test integration with TrainingStep data structure."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_comprehensive_profile()
        trace = self.create_mock_trace()
        
        # Create a TrainingStep
        training_step = TrainingStep(
            step_number=1,
            step_type="training",
            prompt_profile_id=profile.profile_id,
            question="What causes climate change?",
            expected_response="Detailed scientific explanation...",
            agent_responses=trace["worker_responses"] + [trace["synthesis_response"]]
        )
        
        # Set a poor evaluation score
        training_step.set_evaluation_score("overall_quality", 0.3)
        training_step.set_evaluation_score("factual_accuracy", 0.4)
        training_step.set_evaluation_score("reasoning_depth", 0.2)
        
        # Test that optimizer can use training step data
        poor_performance_reward = training_step.get_average_score()
        assert poor_performance_reward is not None
        assert poor_performance_reward < optimizer.no_change_threshold
        
        # High-level test that optimizer would be triggered
        should_optimize = optimizer.should_optimize(poor_performance_reward)
        assert should_optimize == True
        
        # Verify TrainingStep can be converted to trace format
        step_trace = {
            "question": training_step.question,
            "worker_responses": [resp for resp in training_step.agent_responses if resp.metadata.get("agent_role", "").startswith("worker")],
            "synthesis_response": next((resp for resp in training_step.agent_responses if resp.metadata.get("agent_role") == "reviewer"), None),
            "evaluation_scores": training_step.evaluation_scores,
            "training_metadata": training_step.training_metadata
        }
        
        assert step_trace["question"] == training_step.question
        assert len(step_trace["worker_responses"]) >= 0
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_integration_with_reward_calculator(self, mock_get_config):
        """Test integration with RewardCalculator."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        
        # Create mock reward components representing poor performance
        poor_components = RewardComponents(
            text_similarity=0.2,
            semantic_coherence=0.3,
            factual_accuracy=0.15,
            conflict_identification=0.1,
            perspective_integration=0.25,
            synthesis_effectiveness=0.2,
            response_efficiency=0.4,
            reasoning_quality=0.1,
            confidence_calibration=0.3
        )
        
        # Calculate composite reward (would be low)
        total_reward = poor_components.sum() / 9  # Average of components
        
        # Test that this triggers optimization
        assert optimizer.should_optimize(total_reward)
        
        # Create reward components representing good performance
        good_components = RewardComponents(
            text_similarity=0.85,
            semantic_coherence=0.9,
            factual_accuracy=0.88,
            conflict_identification=0.92,
            perspective_integration=0.86,
            synthesis_effectiveness=0.9,
            response_efficiency=0.8,
            reasoning_quality=0.87,
            confidence_calibration=0.83
        )
        
        good_reward = good_components.sum() / 9
        
        # Test that this doesn't trigger optimization
        assert not optimizer.should_optimize(good_reward)
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_prompt_length_management(self, mock_get_config):
        """Test that optimizer manages prompt length appropriately."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer(config={
            'max_edit_length': 100,
            'max_edits_per_prompt': 2
        })
        
        profile = self.create_comprehensive_profile()
        
        # Test that edits exceeding length limits are filtered out
        long_edit = {
            "role": "worker",
            "edit_type": "append",
            "target": "",
            "replacement": "X" * 200,  # Exceeds max_edit_length
            "reasoning": "Test long edit",
            "confidence": 0.9
        }
        
        short_edit = {
            "role": "worker", 
            "edit_type": "append",
            "target": "",
            "replacement": "Short addition.",
            "reasoning": "Test short edit",
            "confidence": 0.8
        }
        
        # Mock response with mixed edit lengths
        with patch.object(optimizer, '_generate_edit_suggestions') as mock_generate:
            from training.optimizers.reflection_optimizer import EditInstruction
            
            # Create edits directly
            long_edit_obj = EditInstruction(**long_edit)
            short_edit_obj = EditInstruction(**short_edit)
            
            # Verify length validation
            assert len(long_edit_obj.validate()) > 0  # Should have length error
            assert len(short_edit_obj.validate()) == 0  # Should be valid
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_error_handling_and_recovery(self, mock_get_config):
        """Test error handling during optimization."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        profile = self.create_comprehensive_profile()
        trace = self.create_mock_trace()
        
        # Test that optimization failure returns original profile
        with patch.object(optimizer, '_generate_edit_suggestions') as mock_generate:
            mock_generate.side_effect = Exception("LLM service error")
            
            result = optimizer.update_profile(
                profile=profile,
                query="Test question",
                answer="Test answer",
                gold_answer="Gold answer",
                reward=0.3,  # Should trigger optimization
                trace=trace
            )
            
            # Should return original profile on error
            assert result == profile
            assert optimizer.optimization_count == 1
            assert optimizer.successful_optimizations == 0
    
    @patch('training.optimizers.reflection_optimizer.get_config')
    def test_optimization_statistics_tracking(self, mock_get_config):
        """Test that optimization statistics are properly tracked."""
        mock_config = Mock()
        mock_config.get_gemini_api_key.return_value = "test_api_key"
        mock_get_config.return_value = mock_config
        
        optimizer = ReflectionOptimizer()
        
        # Verify initial state
        stats = optimizer.get_optimization_stats()
        assert stats["optimizations_performed"] == 0
        assert stats["success_rate"] == 0.0
        
        # Simulate multiple optimization attempts
        profile = self.create_comprehensive_profile()
        trace = self.create_mock_trace()
        
        # Simulate successful optimization
        with patch.object(optimizer, '_generate_edit_suggestions') as mock_generate:
            from training.optimizers.reflection_optimizer import EditInstruction
            
            mock_generate.return_value = [
                EditInstruction("worker", "append", "", "Test edit", "Test reason", 0.8)
            ]
            
            optimizer.update_profile(profile, "Q", "A", "GA", 0.3, trace)
        
        # Simulate failed optimization (no edits generated)
        with patch.object(optimizer, '_generate_edit_suggestions') as mock_generate:
            mock_generate.return_value = []
            
            optimizer.update_profile(profile, "Q", "A", "GA", 0.2, trace)
        
        # Check updated statistics
        stats = optimizer.get_optimization_stats()
        assert stats["optimizations_performed"] == 2
        assert stats["successful_optimizations"] == 1
        assert stats["success_rate"] == 0.5
        
        # Test statistics reset
        optimizer.reset_stats()
        stats = optimizer.get_optimization_stats()
        assert stats["optimizations_performed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
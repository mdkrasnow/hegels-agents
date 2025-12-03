"""
Comprehensive Unit Tests for TrainingExecutor

Tests all core functionality including:
- Complete training step execution
- Atomic database transactions
- Error handling and recovery
- Performance metrics collection
- Integration with reward calculation and optimization
"""

import unittest
import tempfile
import shutil
import time
import uuid
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training_executor import (
    TrainingExecutor, 
    TrainingStepResult,
    create_standard_training_executor,
    create_quality_focused_executor,
    create_fast_training_executor
)
from training.data_structures import PromptProfile, RolePrompt, TrainingStep
from training.rewards import RewardCalculator, RewardComponents, RewardConfig
from training.optimizers.reflection_optimizer import ReflectionOptimizer
from training.database.prompt_profile_store import PromptProfileStore
from agents.utils import AgentResponse


class MockProfileStore:
    """Mock PromptProfileStore for testing."""
    
    def __init__(self):
        self.profiles = {}
        self.create_calls = []
        self.get_calls = []
        self.update_calls = []
        self.should_fail = False
        
    def create_derived_profile(self, base_profile_id: str, new_profile: PromptProfile) -> str:
        """Mock profile creation."""
        if self.should_fail:
            raise Exception("Mock profile store failure")
            
        profile_id = str(uuid.uuid4())
        new_profile.profile_id = profile_id
        self.profiles[profile_id] = new_profile
        self.create_calls.append((base_profile_id, new_profile))
        return profile_id
    
    def get_by_id(self, profile_id: str) -> PromptProfile:
        """Mock profile retrieval."""
        if self.should_fail:
            raise Exception("Mock profile store failure")
            
        self.get_calls.append(profile_id)
        if profile_id in self.profiles:
            return self.profiles[profile_id]
        raise Exception(f"Profile {profile_id} not found")
    
    def update(self, profile: PromptProfile):
        """Mock profile update."""
        if self.should_fail:
            raise Exception("Mock profile store failure")
            
        self.update_calls.append(profile)
        self.profiles[profile.profile_id] = profile


class MockRewardCalculator:
    """Mock RewardCalculator for testing."""
    
    def __init__(self, reward_value: float = 0.75):
        self.reward_value = reward_value
        self.computations = []
        self.should_fail = False
        
    def compute_composite_reward(self, predicted_text: str, gold_text: str, 
                                debate_trace: Dict[str, Any], baseline_response=None, 
                                context=None) -> tuple:
        """Mock reward computation."""
        if self.should_fail:
            raise Exception("Mock reward calculation failure")
            
        self.computations.append({
            'predicted_text': predicted_text,
            'gold_text': gold_text,
            'debate_trace': debate_trace,
            'baseline_response': baseline_response,
            'context': context
        })
        
        components = RewardComponents(
            text_similarity=0.8,
            semantic_coherence=0.7,
            factual_accuracy=0.9,
            conflict_identification=0.6,
            perspective_integration=0.8,
            synthesis_effectiveness=0.75
        )
        
        return self.reward_value, components
    
    def get_performance_stats(self):
        return {"computations": len(self.computations)}
    
    def reset_performance_tracking(self):
        self.computations.clear()


class MockOptimizer:
    """Mock PromptOptimizer for testing."""
    
    def __init__(self, should_optimize: bool = True):
        self.should_optimize_flag = should_optimize
        self.update_calls = []
        self.should_fail = False
        
    def update_profile(self, profile: PromptProfile, query: str, answer: str,
                      gold_answer: str, reward: float, trace: Dict[str, Any], 
                      metadata=None) -> PromptProfile:
        """Mock profile optimization."""
        if self.should_fail:
            raise Exception("Mock optimizer failure")
            
        self.update_calls.append({
            'profile_id': profile.profile_id,
            'query': query,
            'answer': answer,
            'gold_answer': gold_answer,
            'reward': reward,
            'trace': trace,
            'metadata': metadata
        })
        
        if self.should_optimize_flag:
            # Return new optimized profile
            optimized = PromptProfile(
                name=f"{profile.name}_optimized",
                description="Optimized profile",
                role_prompts=profile.role_prompts,
                metadata={**profile.metadata, 'optimized': True}
            )
            return optimized
        else:
            # Return original profile unchanged
            return profile
    
    def get_optimization_stats(self):
        return {"updates_performed": len(self.update_calls)}
    
    def reset_stats(self):
        self.update_calls.clear()


class TestTrainingStepResult(unittest.TestCase):
    """Test TrainingStepResult data structure."""
    
    def test_initialization(self):
        """Test TrainingStepResult initialization."""
        result = TrainingStepResult()
        
        self.assertIsInstance(result.step_id, str)
        self.assertIsInstance(result.execution_timestamp, datetime)
        self.assertEqual(result.success, False)
        self.assertEqual(result.errors, [])
        
    def test_to_dict_serialization(self):
        """Test conversion to dictionary."""
        result = TrainingStepResult(
            profile_id="test-profile",
            query="test query",
            gold_answer="test answer"
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['profile_id'], "test-profile")
        self.assertEqual(result_dict['query'], "test query")
        self.assertIn('execution_timestamp', result_dict)
        
    def test_to_json_serialization(self):
        """Test JSON serialization."""
        result = TrainingStepResult()
        json_str = result.to_json()
        
        self.assertIsInstance(json_str, str)
        self.assertIn('"step_id"', json_str)
        self.assertIn('"execution_timestamp"', json_str)
        
    def test_add_error(self):
        """Test error addition."""
        result = TrainingStepResult()
        result.add_error("Test error", {"detail": "test detail"})
        
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0], "Test error")
        self.assertEqual(result.error_details["detail"], "test detail")
        
    def test_is_successful(self):
        """Test success determination."""
        result = TrainingStepResult()
        
        # Initially not successful
        self.assertFalse(result.is_successful())
        
        # Set all required flags
        result.success = True
        result.debate_completed = True
        result.reward_computed = True
        result.database_transaction_committed = True
        
        self.assertTrue(result.is_successful())
        
        # Add error - should no longer be successful
        result.add_error("Test error")
        self.assertFalse(result.is_successful())


class TestTrainingExecutor(unittest.TestCase):
    """Test TrainingExecutor core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_profile_store = MockProfileStore()
        self.mock_reward_calculator = MockRewardCalculator()
        self.mock_optimizer = MockOptimizer()
        
        self.executor = TrainingExecutor(
            profile_store=self.mock_profile_store,
            reward_calculator=self.mock_reward_calculator,
            optimizer=self.mock_optimizer
        )
        
        # Create test profile
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="You are a helpful assistant. Answer questions accurately and concisely."
        )
        reviewer_prompt = RolePrompt(
            role="reviewer",
            prompt_text="Review and synthesize responses to provide the best answer."
        )
        
        self.test_profile = PromptProfile(
            name="test_profile",
            description="Test profile for unit tests"
        )
        self.test_profile.add_role_prompt(worker_prompt)
        self.test_profile.add_role_prompt(reviewer_prompt)
        
    @patch('training.training_executor.HegelTrainer')
    @patch('training.training_executor.ConfigurableAgentFactory')
    @patch('training.training_executor.get_db_session')
    def test_successful_training_step(self, mock_db_session, mock_factory, mock_trainer_class):
        """Test complete successful training step."""
        # Mock HegelTrainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.create_training_session.return_value = "test-session"
        mock_trainer.end_training_session.return_value = {"duration": 5.0}
        mock_trainer.get_stats.return_value = {"agents_wrapped": 2}
        
        # Mock agents
        mock_worker = Mock()
        mock_reviewer = Mock()
        mock_factory.create_worker.return_value = mock_worker
        mock_factory.create_reviewer.return_value = mock_reviewer
        
        mock_wrapped_worker = Mock()
        mock_wrapped_reviewer = Mock()
        mock_trainer.wrap_worker_agent.return_value = mock_wrapped_worker
        mock_trainer.wrap_reviewer_agent.return_value = mock_wrapped_reviewer
        
        # Mock responses
        mock_response_1 = AgentResponse(
            content="First worker response",
            reasoning="Test reasoning 1",
            confidence=0.8
        )
        mock_response_2 = AgentResponse(
            content="Second worker response", 
            reasoning="Test reasoning 2",
            confidence=0.7
        )
        mock_synthesis = AgentResponse(
            content="Synthesized final answer",
            reasoning="Synthesis reasoning",
            confidence=0.9
        )
        
        mock_wrapped_worker.respond.side_effect = [mock_response_1, mock_response_2]
        mock_wrapped_reviewer.synthesize_responses.return_value = mock_synthesis
        
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db_session.return_value.__exit__ = Mock(return_value=None)
        
        # Execute training step
        result = self.executor.execute_training_step(
            profile=self.test_profile,
            query="What is the capital of France?",
            gold_answer="Paris",
            metadata={"corpus_id": "test", "task_type": "qa"}
        )
        
        # Verify results
        self.assertTrue(result.success)
        self.assertTrue(result.debate_completed)
        self.assertTrue(result.reward_computed)
        self.assertTrue(result.database_transaction_committed)
        self.assertEqual(result.final_answer, "Synthesized final answer")
        self.assertEqual(len(result.agent_responses), 3)  # 2 worker + 1 synthesis
        self.assertGreater(result.total_reward, 0)
        self.assertEqual(len(result.errors), 0)
        
        # Verify mock calls
        mock_trainer.create_training_session.assert_called_once()
        mock_wrapped_worker.respond.assert_called()
        mock_wrapped_reviewer.synthesize_responses.assert_called_once()
        
        # Verify reward calculation
        self.assertEqual(len(self.mock_reward_calculator.computations), 1)
        
    @patch('training.training_executor.HegelTrainer')
    @patch('training.training_executor.ConfigurableAgentFactory')
    def test_debate_execution_failure(self, mock_factory, mock_trainer_class):
        """Test handling of debate execution failures."""
        # Mock HegelTrainer to raise exception
        mock_trainer_class.side_effect = Exception("Mock debate failure")
        
        result = self.executor.execute_training_step(
            profile=self.test_profile,
            query="Test query",
            gold_answer="Test answer"
        )
        
        self.assertFalse(result.success)
        self.assertFalse(result.debate_completed)
        self.assertFalse(result.reward_computed)
        self.assertGreater(len(result.errors), 0)
        self.assertIn("Debate execution failed", result.errors[0])
        
    def test_reward_calculation_failure(self):
        """Test handling of reward calculation failures."""
        # Configure reward calculator to fail
        self.mock_reward_calculator.should_fail = True
        
        # Mock successful debate execution
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'):
            
            # Mock the debate execution to succeed
            with patch.object(self.executor, '_execute_debate') as mock_debate:
                mock_debate.return_value = {
                    'success': True,
                    'final_answer': "Test answer",
                    'responses': [],
                    'trace': {}
                }
                
                result = self.executor.execute_training_step(
                    profile=self.test_profile,
                    query="Test query",
                    gold_answer="Test answer"
                )
        
        self.assertFalse(result.success)
        self.assertTrue(result.debate_completed)
        self.assertFalse(result.reward_computed)
        self.assertGreater(len(result.errors), 0)
        self.assertIn("Reward computation failed", result.errors[0])
        
    def test_optimization_threshold_logic(self):
        """Test optimization threshold decision making."""
        # Test case 1: High reward, no optimization needed
        self.mock_reward_calculator.reward_value = 0.9  # Above threshold
        
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'), \
             patch('training.training_executor.get_db_session'):
            
            with patch.object(self.executor, '_execute_debate') as mock_debate, \
                 patch.object(self.executor, '_log_step_atomically') as mock_log:
                
                mock_debate.return_value = {
                    'success': True,
                    'final_answer': "Test answer",
                    'responses': [],
                    'trace': {}
                }
                mock_log.return_value = {'success': True}
                
                result = self.executor.execute_training_step(
                    profile=self.test_profile,
                    query="Test query",
                    gold_answer="Test answer"
                )
                
                self.assertFalse(result.optimization_attempted)
                self.assertFalse(result.new_profile_created)
                
        # Test case 2: Low reward, optimization needed
        self.mock_reward_calculator.reward_value = 0.5  # Below threshold
        
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'), \
             patch('training.training_executor.get_db_session'):
            
            with patch.object(self.executor, '_execute_debate') as mock_debate, \
                 patch.object(self.executor, '_log_step_atomically') as mock_log:
                
                mock_debate.return_value = {
                    'success': True,
                    'final_answer': "Test answer", 
                    'responses': [],
                    'trace': {}
                }
                mock_log.return_value = {'success': True}
                
                result = self.executor.execute_training_step(
                    profile=self.test_profile,
                    query="Test query",
                    gold_answer="Test answer"
                )
                
                self.assertTrue(result.optimization_attempted)
                self.assertTrue(result.new_profile_created)  # Mock optimizer returns new profile
                
    def test_optimization_failure_handling(self):
        """Test handling of optimization failures."""
        # Configure low reward to trigger optimization
        self.mock_reward_calculator.reward_value = 0.5
        
        # Configure optimizer to fail
        self.mock_optimizer.should_fail = True
        
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'), \
             patch('training.training_executor.get_db_session'):
            
            with patch.object(self.executor, '_execute_debate') as mock_debate, \
                 patch.object(self.executor, '_log_step_atomically') as mock_log:
                
                mock_debate.return_value = {
                    'success': True,
                    'final_answer': "Test answer",
                    'responses': [],
                    'trace': {}
                }
                mock_log.return_value = {'success': True}
                
                result = self.executor.execute_training_step(
                    profile=self.test_profile,
                    query="Test query",
                    gold_answer="Test answer"
                )
                
                # Should still succeed overall despite optimization failure
                self.assertTrue(result.success)
                self.assertTrue(result.optimization_attempted)
                self.assertFalse(result.new_profile_created)
                self.assertGreater(len(result.errors), 0)
                self.assertIn("Optimization failed", result.errors[0])
                
    @patch('training.training_executor.get_db_session')
    def test_database_transaction_failure(self, mock_db_session):
        """Test handling of database transaction failures."""
        # Configure database session to fail
        mock_db_session.side_effect = Exception("Mock database failure")
        
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'):
            
            with patch.object(self.executor, '_execute_debate') as mock_debate:
                mock_debate.return_value = {
                    'success': True,
                    'final_answer': "Test answer",
                    'responses': [],
                    'trace': {}
                }
                
                result = self.executor.execute_training_step(
                    profile=self.test_profile,
                    query="Test query",
                    gold_answer="Test answer"
                )
                
                self.assertFalse(result.success)
                self.assertTrue(result.debate_completed)
                self.assertTrue(result.reward_computed)
                self.assertFalse(result.database_transaction_committed)
                self.assertGreater(len(result.errors), 0)
                self.assertIn("Database logging failed", result.errors[0])
                
    def test_performance_stats_tracking(self):
        """Test performance statistics tracking."""
        initial_stats = self.executor.get_performance_stats()
        self.assertEqual(initial_stats['execution_stats']['total_steps_executed'], 0)
        
        # Mock successful execution
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'), \
             patch('training.training_executor.get_db_session'):
            
            with patch.object(self.executor, '_execute_debate') as mock_debate, \
                 patch.object(self.executor, '_log_step_atomically') as mock_log:
                
                mock_debate.return_value = {
                    'success': True,
                    'final_answer': "Test answer",
                    'responses': [],
                    'trace': {}
                }
                mock_log.return_value = {'success': True}
                
                # Execute multiple steps
                for i in range(3):
                    self.executor.execute_training_step(
                        profile=self.test_profile,
                        query=f"Test query {i}",
                        gold_answer=f"Test answer {i}"
                    )
                
        final_stats = self.executor.get_performance_stats()
        self.assertEqual(final_stats['execution_stats']['total_steps_executed'], 3)
        self.assertEqual(final_stats['execution_stats']['successful_steps'], 3)
        self.assertGreater(final_stats['execution_stats']['average_execution_time_seconds'], 0)
        
    def test_recovery_data_collection(self):
        """Test recovery data collection for failed steps."""
        # Enable recovery logging
        self.executor.enable_recovery_logging = True
        
        # Force a critical failure
        with patch('training.training_executor.HegelTrainer') as mock_trainer_class:
            mock_trainer_class.side_effect = Exception("Critical failure")
            
            result = self.executor.execute_training_step(
                profile=self.test_profile,
                query="Test query",
                gold_answer="Test answer"
            )
            
            self.assertFalse(result.success)
            self.assertTrue(result.recovery_data_saved)
            
            recovery_data = self.executor.get_recovery_data()
            self.assertEqual(len(recovery_data), 1)
            self.assertIn(result.step_id, recovery_data)
            
    def test_reset_performance_stats(self):
        """Test resetting performance statistics."""
        # Execute a step to generate some stats
        with patch('training.training_executor.HegelTrainer'), \
             patch('training.training_executor.ConfigurableAgentFactory'), \
             patch('training.training_executor.get_db_session'):
            
            with patch.object(self.executor, '_execute_debate'), \
                 patch.object(self.executor, '_log_step_atomically'):
                pass
        
        self.executor.total_steps_executed = 5
        self.executor.successful_steps = 3
        
        self.executor.reset_performance_stats()
        
        stats = self.executor.get_performance_stats()
        self.assertEqual(stats['execution_stats']['total_steps_executed'], 0)
        self.assertEqual(stats['execution_stats']['successful_steps'], 0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating TrainingExecutor instances."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_profile_store = MockProfileStore()
        
    def test_create_standard_training_executor(self):
        """Test standard executor factory."""
        executor = create_standard_training_executor(self.mock_profile_store)
        
        self.assertIsInstance(executor, TrainingExecutor)
        self.assertIsInstance(executor.reward_calculator, RewardCalculator)
        self.assertIsInstance(executor.optimizer, ReflectionOptimizer)
        self.assertEqual(executor.profile_store, self.mock_profile_store)
        
    def test_create_quality_focused_executor(self):
        """Test quality-focused executor factory."""
        executor = create_quality_focused_executor(self.mock_profile_store)
        
        self.assertIsInstance(executor, TrainingExecutor)
        self.assertEqual(executor.optimization_threshold, 0.8)  # Higher threshold for quality
        self.assertEqual(executor.max_retry_attempts, 5)  # More retries for quality
        
    def test_create_fast_training_executor(self):
        """Test fast executor factory."""
        executor = create_fast_training_executor(self.mock_profile_store)
        
        self.assertIsInstance(executor, TrainingExecutor)
        self.assertEqual(executor.optimization_threshold, 0.6)  # Lower threshold for speed
        self.assertEqual(executor.max_retry_attempts, 2)  # Fewer retries for speed
        self.assertFalse(executor.enable_recovery_logging)  # Disabled for speed


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error handling scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_profile_store = MockProfileStore()
        self.mock_reward_calculator = MockRewardCalculator()
        self.mock_optimizer = MockOptimizer()
        
        self.executor = TrainingExecutor(
            profile_store=self.mock_profile_store,
            reward_calculator=self.mock_reward_calculator,
            optimizer=self.mock_optimizer
        )
        
        self.test_profile = PromptProfile(name="test")
        worker_prompt = RolePrompt(role="worker", prompt_text="Test prompt")
        self.test_profile.add_role_prompt(worker_prompt)
        
    def test_invalid_profile_handling(self):
        """Test handling of invalid profiles."""
        invalid_profile = PromptProfile(name="")  # Invalid - empty name
        
        with patch('training.training_executor.HegelTrainer'):
            result = self.executor.execute_training_step(
                profile=invalid_profile,
                query="Test query", 
                gold_answer="Test answer"
            )
            
            self.assertFalse(result.success)
            self.assertGreater(len(result.errors), 0)
            
    def test_timeout_handling(self):
        """Test timeout handling in long-running operations."""
        # This would require more complex mocking to truly test timeouts
        # For now, just ensure the timeout configuration is set
        executor = TrainingExecutor(
            profile_store=self.mock_profile_store,
            config={'transaction_timeout': 1}  # Very short timeout
        )
        
        self.assertEqual(executor.transaction_timeout, 1)
        
    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access scenarios."""
        # Create multiple executor instances to simulate concurrent access
        executors = [
            TrainingExecutor(
                profile_store=MockProfileStore(),
                reward_calculator=MockRewardCalculator(),
                optimizer=MockOptimizer()
            )
            for _ in range(3)
        ]
        
        # Each executor should be independent
        for i, executor in enumerate(executors):
            executor.total_steps_executed = i
            
        # Verify independence
        for i, executor in enumerate(executors):
            self.assertEqual(executor.total_steps_executed, i)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
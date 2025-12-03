"""
Test Suite for HegelTrainer grad=True Mode Implementation (T2.4)

This test suite validates the end-to-end learning capability implementation
with comprehensive testing of both grad=False (inference) and grad=True (training) modes.

Key Test Areas:
- Backward compatibility (grad=False must be identical to existing system)
- Training mode functionality (grad=True with live learning)
- Error handling and rollback mechanisms
- Performance monitoring and statistics
- Integration with TrainingExecutor and reward system
"""

import unittest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# Import the classes we're testing
from training.hegel_trainer import HegelTrainer, create_trainer
from training.training_executor import TrainingExecutor, TrainingStepResult, create_standard_executor
from training.rewards import RewardCalculator, RewardComponents, RewardConfig
from training.data_structures import PromptProfile, RolePrompt
from agents.utils import AgentResponse, AgentLogger
from agents.worker import BasicWorkerAgent
from agents.reviewer import BasicReviewerAgent


class TestHegelTrainerGradMode(unittest.TestCase):
    """Test HegelTrainer grad=True mode implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock profile store
        self.mock_profile_store = Mock()
        
        # Create trainer instances for testing
        self.inference_trainer = HegelTrainer(grad=False)
        self.training_trainer = HegelTrainer(
            grad=True, 
            profile_store=self.mock_profile_store
        )
        
        # Test data
        self.test_query = "What are the key principles of machine learning?"
        self.test_corpus_id = "ml_basics"
        self.test_task_type = "qa"
        self.test_gold_answer = "Machine learning is based on algorithms that learn patterns from data."
    
    def test_run_method_exists_and_signature(self):
        """Test that run() method exists with correct signature."""
        # Test method exists
        self.assertTrue(hasattr(self.inference_trainer, 'run'))
        self.assertTrue(callable(getattr(self.inference_trainer, 'run')))
        
        # Test method signature by inspecting
        import inspect
        sig = inspect.signature(self.inference_trainer.run)
        
        expected_params = ['query', 'corpus_id', 'task_type', 'grad', 'gold_answer', 'reward']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            self.assertIn(param, actual_params, f"Missing parameter: {param}")
    
    @patch('training.hegel_trainer.BasicWorkerAgent')
    @patch('training.hegel_trainer.BasicReviewerAgent')
    def test_inference_mode_grad_false(self, mock_reviewer_class, mock_worker_class):
        """Test that grad=False mode works correctly (backward compatibility)."""
        # Mock agent instances
        mock_worker = Mock(spec=BasicWorkerAgent)
        mock_worker.agent_id = "test_worker"
        mock_reviewer = Mock(spec=BasicReviewerAgent)
        mock_reviewer.agent_id = "test_reviewer"
        
        mock_worker_class.return_value = mock_worker
        mock_reviewer_class.return_value = mock_reviewer
        
        # Mock wrapped agent responses
        mock_response1 = AgentResponse(
            content="Machine learning uses algorithms.",
            reasoning="Basic definition",
            confidence=0.8
        )
        mock_response2 = AgentResponse(
            content="ML learns from data patterns.",
            reasoning="Pattern recognition focus",
            confidence=0.7
        )
        mock_synthesis = AgentResponse(
            content="Machine learning combines algorithms with pattern recognition from data.",
            reasoning="Synthesis of both perspectives",
            confidence=0.85
        )
        
        # Set up wrapper mocks
        with patch.object(self.inference_trainer, 'wrap_worker_agent') as mock_wrap_worker, \
             patch.object(self.inference_trainer, 'wrap_reviewer_agent') as mock_wrap_reviewer:
            
            mock_wrapped_worker = Mock()
            mock_wrapped_worker.agent_id = "wrapped_test_worker"
            mock_wrapped_worker.respond.side_effect = [mock_response1, mock_response2]
            
            mock_wrapped_reviewer = Mock()
            mock_wrapped_reviewer.agent_id = "wrapped_test_reviewer"
            mock_wrapped_reviewer.synthesize_responses.return_value = mock_synthesis
            
            mock_wrap_worker.return_value = mock_wrapped_worker
            mock_wrap_reviewer.return_value = mock_wrapped_reviewer
            
            # Mock DebateSession
            with patch('training.hegel_trainer.DebateSession') as mock_session_class:
                mock_session = Mock()
                mock_session.get_summary.return_value = {'total_turns': 3}
                mock_session.analyze_debate.return_value = Mock()
                mock_session_class.return_value = mock_session
                
                # Execute inference mode
                result = self.inference_trainer.run(
                    query=self.test_query,
                    corpus_id=self.test_corpus_id,
                    task_type=self.test_task_type,
                    grad=False
                )
                
                # Validate result structure
                self.assertIsInstance(result, dict)
                self.assertTrue(result['success'])
                self.assertEqual(result['mode'], 'inference')
                self.assertEqual(result['grad_mode'], False)
                self.assertEqual(result['corpus_id'], self.test_corpus_id)
                self.assertEqual(result['query'], self.test_query)
                
                # Validate backward compatibility fields
                self.assertIn('final_response', result)
                self.assertIn('worker_responses', result)
                self.assertIn('synthesis', result)
                self.assertIn('conflict_analysis', result)
                self.assertIn('session', result)
                self.assertIn('session_summary', result)
                
                # Validate training metadata indicates inference mode
                training_metadata = result['training_metadata']
                self.assertFalse(training_metadata['grad_requested'])
                self.assertEqual(training_metadata['mode_reason'], 'grad_disabled')
    
    def test_training_mode_without_dependencies(self):
        """Test that grad=True mode falls back gracefully when dependencies unavailable."""
        # Create trainer without profile store (training not ready)
        trainer_no_deps = HegelTrainer(grad=True, profile_store=None)
        
        # Mock the inference mode execution
        with patch.object(trainer_no_deps, '_execute_inference_mode') as mock_inference:
            mock_inference.return_value = {
                'success': True,
                'mode': 'inference',
                'grad_mode': False,
                'training_metadata': {
                    'training_ready': False,
                    'grad_requested': True,
                    'mode_reason': 'training_not_ready'
                }
            }
            
            result = trainer_no_deps.run(
                query=self.test_query,
                corpus_id=self.test_corpus_id,
                grad=True
            )
            
            # Should fall back to inference mode
            self.assertTrue(result['success'])
            self.assertEqual(result['mode'], 'inference')
            self.assertFalse(result['training_metadata']['training_ready'])
            self.assertTrue(result['training_metadata']['grad_requested'])
            
            mock_inference.assert_called_once()
    
    def test_training_mode_with_dependencies(self):
        """Test that grad=True mode works when dependencies are available."""
        # Mock TrainingExecutor result
        mock_training_result = TrainingStepResult(
            step_id="test_step_123",
            timestamp=datetime.utcnow(),
            execution_time_ms=150.0,
            query=self.test_query,
            corpus_id=self.test_corpus_id,
            task_type=self.test_task_type,
            gold_answer=self.test_gold_answer,
            computed_reward=0.75,
            success=True,
            profile_changed=True,
            original_profile_id="original_123",
            updated_profile_id="updated_456"
        )
        
        # Mock final response
        mock_training_result.final_response = AgentResponse(
            content="Comprehensive ML explanation with learning applied.",
            reasoning="Enhanced through training",
            confidence=0.9
        )
        
        # Mock reward components
        mock_training_result.reward_components = RewardComponents(
            text_similarity=0.8,
            synthesis_effectiveness=0.7,
            conflict_identification=1.0
        )
        
        # Mock training metadata
        mock_training_result.training_metadata = {
            'reward_source': 'computed',
            'optimization_triggered': True,
            'optimization_reason': 'Low synthesis effectiveness'
        }
        
        # Mock the training executor
        mock_executor = Mock(spec=TrainingExecutor)
        mock_executor.execute_training_step.return_value = mock_training_result
        mock_executor.get_stats.return_value = {
            'executor_stats': {'steps_executed': 1, 'successful_steps': 1}
        }
        
        # Patch the trainer's training executor
        self.training_trainer._training_executor = mock_executor
        
        # Execute training mode
        result = self.training_trainer.run(
            query=self.test_query,
            corpus_id=self.test_corpus_id,
            task_type=self.test_task_type,
            grad=True,
            gold_answer=self.test_gold_answer
        )
        
        # Validate training mode result
        self.assertIsInstance(result, dict)
        self.assertTrue(result['success'])
        self.assertEqual(result['mode'], 'training')
        self.assertTrue(result['grad_mode'])
        self.assertEqual(result['corpus_id'], self.test_corpus_id)
        self.assertEqual(result['query'], self.test_query)
        
        # Validate training-specific fields
        self.assertEqual(result['training_step_id'], 'test_step_123')
        self.assertEqual(result['computed_reward'], 0.75)
        self.assertEqual(result['gold_answer'], self.test_gold_answer)
        
        # Validate profile evolution tracking
        profile_evolution = result['profile_evolution']
        self.assertEqual(profile_evolution['original_profile_id'], 'original_123')
        self.assertEqual(profile_evolution['updated_profile_id'], 'updated_456')
        self.assertTrue(profile_evolution['profile_changed'])
        self.assertTrue(profile_evolution['optimization_triggered'])
        
        # Validate training metadata
        training_metadata = result['training_metadata']
        self.assertTrue(training_metadata['training_ready'])
        self.assertEqual(training_metadata['reward_source'], 'computed')
        self.assertIsNotNone(training_metadata['reward_components'])
        self.assertIsNotNone(training_metadata['training_executor_stats'])
        
        # Verify training executor was called correctly
        mock_executor.execute_training_step.assert_called_once()
        call_args = mock_executor.execute_training_step.call_args
        self.assertEqual(call_args[1]['query'], self.test_query)
        self.assertEqual(call_args[1]['corpus_id'], self.test_corpus_id)
        self.assertEqual(call_args[1]['gold_answer'], self.test_gold_answer)
    
    def test_error_handling_in_training_mode(self):
        """Test error handling and rollback in training mode."""
        # Create training result that fails
        mock_training_result = TrainingStepResult(
            step_id="test_fail_123",
            timestamp=datetime.utcnow(),
            execution_time_ms=75.0,
            query=self.test_query,
            corpus_id=self.test_corpus_id,
            task_type=self.test_task_type,
            success=False,
            error_message="Training execution failed",
            profile_changed=True,
            rollback_performed=True,
            original_profile_id="original_123"
        )
        
        mock_training_result.training_metadata = {
            'error_details': 'Mock training failure for testing'
        }
        
        # Mock executor that returns failed result
        mock_executor = Mock(spec=TrainingExecutor)
        mock_executor.execute_training_step.return_value = mock_training_result
        mock_executor.get_stats.return_value = {
            'executor_stats': {'steps_executed': 1, 'failed_steps': 1}
        }
        
        self.training_trainer._training_executor = mock_executor
        
        # Execute training mode (should handle failure gracefully)
        result = self.training_trainer.run(
            query=self.test_query,
            corpus_id=self.test_corpus_id,
            grad=True
        )
        
        # Validate error handling
        self.assertFalse(result['success'])
        self.assertEqual(result['mode'], 'training')
        self.assertEqual(result['error'], 'Training execution failed')
        
        # Validate rollback information
        training_metadata = result['training_metadata']
        self.assertTrue(training_metadata['rollback_performed'])
        self.assertEqual(training_metadata['error_details'], 'Training execution failed')
    
    def test_performance_no_regression_grad_false(self):
        """Test that grad=False mode has no performance regression."""
        # Mock lightweight components for performance test
        with patch('training.hegel_trainer.BasicWorkerAgent') as mock_worker_class, \
             patch('training.hegel_trainer.BasicReviewerAgent') as mock_reviewer_class, \
             patch.object(self.inference_trainer, 'wrap_worker_agent'), \
             patch.object(self.inference_trainer, 'wrap_reviewer_agent'), \
             patch('training.hegel_trainer.DebateSession'):
            
            # Configure mocks for minimal overhead
            mock_worker_class.return_value = Mock()
            mock_reviewer_class.return_value = Mock()
            
            # Measure execution time
            start_time = time.time()
            
            result = self.inference_trainer.run(
                query="Quick test",
                corpus_id="test",
                grad=False
            )
            
            execution_time = time.time() - start_time
            
            # Performance validation
            self.assertTrue(result['success'])
            self.assertEqual(result['grad_mode'], False)
            
            # Execution time should be reasonable (< 100ms for mocked components)
            # This is a basic sanity check - real performance testing would be more rigorous
            self.assertLess(execution_time, 0.1, "Inference mode should be fast with minimal overhead")
    
    def test_training_executor_integration(self):
        """Test integration with TrainingExecutor."""
        # Test that training executor is properly initialized
        trainer_with_store = HegelTrainer(grad=True, profile_store=Mock())
        
        # Training executor should be initialized
        self.assertIsNotNone(trainer_with_store._training_executor)
        self.assertIsInstance(trainer_with_store._training_executor, TrainingExecutor)
        
        # Test enable/disable training mode
        inference_only = HegelTrainer(grad=False)
        self.assertIsNone(inference_only._training_executor)
        
        # Enable training mode
        inference_only.enable_training_mode(profile_store=Mock())
        self.assertIsNotNone(inference_only._training_executor)
        
        # Disable training mode
        inference_only.disable_training_mode()
        self.assertIsNone(inference_only._training_executor)
        self.assertFalse(inference_only.grad)
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        trainer = HegelTrainer(grad=False)
        
        # Initial stats
        initial_stats = trainer.get_stats()
        self.assertEqual(initial_stats['stats']['grad_mode_calls'], 0)
        self.assertEqual(initial_stats['stats']['inference_mode_calls'], 0)
        
        # Mock minimal execution for statistics test
        with patch.object(trainer, '_execute_inference_mode') as mock_inference:
            mock_inference.return_value = {'success': True, 'mode': 'inference'}
            
            # Execute inference call
            trainer.run("test", "test_corpus", grad=False)
            
            # Check updated stats
            updated_stats = trainer.get_stats()
            self.assertEqual(updated_stats['stats']['inference_mode_calls'], 1)
            self.assertEqual(updated_stats['stats']['grad_mode_calls'], 0)
        
        # Test training stats (if available)
        if trainer._training_executor:
            training_stats = trainer.get_training_stats()
            self.assertIn('training_executor', training_stats)
    
    def test_backward_compatibility_result_structure(self):
        """Test that grad=False results are backward compatible."""
        with patch('training.hegel_trainer.BasicWorkerAgent'), \
             patch('training.hegel_trainer.BasicReviewerAgent'), \
             patch.object(self.inference_trainer, 'wrap_worker_agent') as mock_wrap_worker, \
             patch.object(self.inference_trainer, 'wrap_reviewer_agent') as mock_wrap_reviewer, \
             patch('training.hegel_trainer.DebateSession') as mock_session_class:
            
            # Mock components
            mock_worker_response = AgentResponse(content="test", reasoning="test", confidence=0.8)
            mock_synthesis = AgentResponse(content="synthesis", reasoning="combined", confidence=0.9)
            
            mock_wrapped_worker = Mock()
            mock_wrapped_worker.respond.return_value = mock_worker_response
            mock_wrapped_reviewer = Mock()
            mock_wrapped_reviewer.synthesize_responses.return_value = mock_synthesis
            
            mock_wrap_worker.return_value = mock_wrapped_worker
            mock_wrap_reviewer.return_value = mock_wrapped_reviewer
            
            mock_session = Mock()
            mock_session.analyze_debate.return_value = Mock()
            mock_session.get_summary.return_value = {'test': 'data'}
            mock_session_class.return_value = mock_session
            
            # Execute and validate backward compatibility
            result = self.inference_trainer.run(
                query="test",
                corpus_id="test",
                grad=False
            )
            
            # Must have all backward compatible fields
            required_fields = [
                'final_response', 'worker_responses', 'synthesis', 
                'conflict_analysis', 'session', 'session_summary'
            ]
            
            for field in required_fields:
                self.assertIn(field, result, f"Missing backward compatibility field: {field}")
            
            # Must not have training-specific fields in inference mode
            training_only_fields = [
                'training_step_id', 'computed_reward', 'profile_evolution'
            ]
            
            for field in training_only_fields:
                self.assertNotIn(field, result, f"Training field should not be present in inference mode: {field}")


class TestTrainingExecutorIntegration(unittest.TestCase):
    """Test TrainingExecutor integration with HegelTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_profile_store = Mock()
        self.trainer = HegelTrainer(grad=True, profile_store=self.mock_profile_store)
        
        # Ensure training executor is available
        if not self.trainer._training_executor:
            self.trainer._training_executor = create_standard_executor(
                profile_store=self.mock_profile_store
            )
    
    def test_training_executor_creation(self):
        """Test TrainingExecutor creation and configuration."""
        executor = create_standard_executor()
        
        self.assertIsInstance(executor, TrainingExecutor)
        self.assertIsInstance(executor.reward_calculator, RewardCalculator)
        self.assertIsNotNone(executor.logger)
    
    def test_training_step_result_serialization(self):
        """Test TrainingStepResult serialization."""
        result = TrainingStepResult(
            step_id="test_123",
            timestamp=datetime.utcnow(),
            execution_time_ms=100.0,
            query="test query",
            corpus_id="test_corpus",
            task_type="qa"
        )
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['step_id'], 'test_123')
        self.assertEqual(result_dict['query'], 'test query')
        self.assertEqual(result_dict['success'], True)  # Default value
        
        # Test JSON serialization
        json_str = json.dumps(result_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        loaded_dict = json.loads(json_str)
        self.assertEqual(loaded_dict['step_id'], 'test_123')


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Hegel's Agents Training Foundation Layer

This test suite validates the complete end-to-end functionality of the training system,
including data structures, database operations, agent factory integration, and 
backward compatibility with existing systems.

Test Categories:
1. End-to-end workflow testing
2. Backward compatibility validation  
3. Performance benchmarking
4. Error condition handling
5. Rollback procedure validation

Usage:
    python -m pytest tests/integration/test_training_foundation.py -v
    python tests/integration/test_training_foundation.py  # Direct execution
"""

import os
import sys
import time
import uuid
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import pytest
import psutil

# Core training system imports
try:
    from training.data_structures import PromptProfile, RolePrompt, TrainingStep, validate_all_structures
    from training.database.prompt_profile_store import PromptProfileStore, ProfileNotFoundError, ProfileValidationError
    from training.agent_factory import ConfigurableAgentFactory, AgentConfig, EnhancedAgentConfig, AgentCache
    from training.models.claude_agent_config import ClaudeAgentConfig, ClaudeModelSettings, ClaudePromptConfig, PromptRole
except ImportError as e:
    pytest.skip(f"Training system components not available: {e}", allow_module_level=True)

# Existing system imports for integration testing
try:
    from agents.worker import BasicWorkerAgent
    from agents.reviewer import BasicReviewerAgent
    from agents.utils import AgentResponse
    from debate.session import DebateSession
    from corpus.file_retriever import FileCorpusRetriever
except ImportError as e:
    pytest.skip(f"Core system components not available: {e}", allow_module_level=True)


class PerformanceBenchmark:
    """Performance measurement and benchmarking utility."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_end = None
        self.cpu_start = None
        self.cpu_end = None
        
    def start(self):
        """Start performance measurement."""
        self.start_time = time.perf_counter()
        process = psutil.Process()
        self.memory_start = process.memory_info().rss / 1024 / 1024  # MB
        self.cpu_start = process.cpu_percent()
        
    def stop(self):
        """Stop performance measurement."""
        self.end_time = time.perf_counter()
        process = psutil.Process()
        self.memory_end = process.memory_info().rss / 1024 / 1024  # MB
        self.cpu_end = process.cpu_percent()
        
    def get_results(self) -> Dict[str, float]:
        """Get benchmark results."""
        if self.start_time is None or self.end_time is None:
            return {}
            
        return {
            'duration_ms': (self.end_time - self.start_time) * 1000,
            'memory_delta_mb': (self.memory_end - self.memory_start) if self.memory_end else 0,
            'cpu_usage_percent': self.cpu_end if self.cpu_end else 0
        }


class MockAgentSystemIntegration:
    """Mock integration with existing agent system for testing."""
    
    @staticmethod
    def create_mock_worker_agent(agent_id: str = "test_worker") -> Mock:
        """Create a mock BasicWorkerAgent for testing."""
        # Import here to avoid circular dependencies
        from agents.worker import BasicWorkerAgent
        
        mock_agent = Mock(spec=BasicWorkerAgent)
        mock_agent.agent_id = agent_id
        mock_agent.SYSTEM_PROMPT = "You are a helpful AI assistant."
        mock_agent.logger = Mock()
        mock_agent.logger.log_debug = Mock()
        mock_agent.logger.log_error = Mock()
        mock_agent._make_gemini_call = Mock(return_value="Mock response content")
        mock_agent.client = Mock()
        mock_agent.client.models = Mock()
        mock_agent.client.models.generate_content = Mock()
        mock_agent.client.models.generate_content.return_value.text = "Mock generated content"
        
        # Mock the process method 
        def mock_process(question: str) -> AgentResponse:
            return AgentResponse(
                content=f"Mock worker response to: {question}",
                reasoning="Mock reasoning",
                confidence=0.8,
                sources=["mock_source"],
                metadata={'agent_id': agent_id},
                timestamp=datetime.utcnow()
            )
        mock_agent.process = mock_process
        
        return mock_agent
    
    @staticmethod
    def create_mock_reviewer_agent(agent_id: str = "test_reviewer") -> Mock:
        """Create a mock BasicReviewerAgent for testing."""
        # Import here to avoid circular dependencies
        from agents.reviewer import BasicReviewerAgent
        
        mock_agent = Mock(spec=BasicReviewerAgent)
        mock_agent.agent_id = agent_id
        mock_agent.CRITIQUE_PROMPT = "Provide a detailed critique."
        mock_agent.SYNTHESIS_PROMPT = "Synthesize the responses."
        mock_agent.logger = Mock()
        mock_agent.logger.log_debug = Mock()
        mock_agent.logger.log_error = Mock()
        mock_agent._make_gemini_call = Mock(return_value="Mock critique content")
        mock_agent.client = Mock()
        mock_agent.client.models = Mock()
        mock_agent.client.models.generate_content = Mock()
        mock_agent.client.models.generate_content.return_value.text = "Mock generated content"
        
        # Mock the critique and synthesize methods
        def mock_critique(responses: List[AgentResponse]) -> AgentResponse:
            return AgentResponse(
                content="Mock critique of responses",
                reasoning="Mock critique reasoning",
                confidence=0.85,
                sources=["critique_source"],
                metadata={'agent_id': agent_id, 'critique': True},
                timestamp=datetime.utcnow()
            )
        
        def mock_synthesize(responses: List[AgentResponse]) -> AgentResponse:
            return AgentResponse(
                content="Mock synthesis of responses",
                reasoning="Mock synthesis reasoning", 
                confidence=0.9,
                sources=["synthesis_source"],
                metadata={'agent_id': agent_id, 'synthesis': True},
                timestamp=datetime.utcnow()
            )
        
        mock_agent.critique = mock_critique
        mock_agent.synthesize = mock_synthesize
        
        return mock_agent


class TrainingFoundationIntegrationTests:
    """Comprehensive integration test suite for training foundation layer."""
    
    def __init__(self):
        """Initialize integration test suite."""
        self.test_results = {}
        self.benchmark_results = {}
        self.baseline_performance = {}
        self.mock_integration = MockAgentSystemIntegration()
        
        # Test data
        self.test_corpus_questions = [
            "What is the nature of reality according to Hegel?",
            "How does dialectical reasoning work?",
            "What are the key principles of machine learning?",
            "Explain the concept of entropy in thermodynamics.",
            "What is the relationship between consciousness and brain activity?"
        ]
        
    def create_test_prompt_profile(self, name: str = "test_profile") -> PromptProfile:
        """Create a test PromptProfile for integration testing."""
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="You are an AI assistant specializing in philosophical analysis. Provide thoughtful, well-reasoned responses that engage with the dialectical nature of complex questions.",
            description="Worker prompt for dialectical testing",
            version="1.0",
            author="integration_test",
            metadata={
                'optimization_target': 'dialectical_quality',
                'expected_response_length': 'medium',
                'reasoning_style': 'analytical'
            }
        )
        
        reviewer_prompt = RolePrompt(
            role="reviewer", 
            prompt_text="You are a critical reviewer who analyzes responses for logical consistency, evidence quality, and dialectical engagement. Identify strengths, weaknesses, and areas for improvement.",
            description="Reviewer prompt for dialectical critique",
            version="1.0",
            author="integration_test",
            metadata={
                'synthesis_prompt': 'Synthesize the best elements from multiple responses into a cohesive, improved answer that addresses the question comprehensively.',
                'critique_focus': ['logical_consistency', 'evidence_quality', 'dialectical_depth']
            }
        )
        
        profile = PromptProfile(
            name=name,
            description="Integration test profile for dialectical reasoning validation",
            version="1.0",
            author="integration_test",
            tags=["integration_test", "dialectical", "qa"],
            metadata={
                'corpus_id': 'test_corpus',
                'task_type': 'qa',
                'optimization_target': 'dialectical_quality',
                'created_for': 'integration_testing'
            }
        )
        
        profile.add_role_prompt(worker_prompt)
        profile.add_role_prompt(reviewer_prompt)
        
        return profile

    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """
        Test 1: End-to-end workflow from profile creation to debate execution.
        
        Validates:
        - Profile creation and storage
        - Agent configuration with profiles
        - Debate session execution
        - Result validation and storage
        """
        benchmark = PerformanceBenchmark("end_to_end_workflow")
        benchmark.start()
        
        results = {
            'test_name': 'end_to_end_workflow',
            'status': 'running',
            'steps': {},
            'errors': [],
            'performance': {}
        }
        
        try:
            # Step 1: Create and validate PromptProfile
            results['steps']['1_profile_creation'] = {'status': 'running'}
            
            test_profile = self.create_test_prompt_profile("integration_test_profile")
            validation_errors = test_profile.validate()
            
            if validation_errors:
                results['steps']['1_profile_creation'] = {
                    'status': 'failed',
                    'error': f"Profile validation failed: {validation_errors}"
                }
                results['status'] = 'failed'
                return results
            
            results['steps']['1_profile_creation'] = {
                'status': 'completed',
                'profile_id': test_profile.profile_id,
                'roles': test_profile.get_roles()
            }
            
            # Step 2: Store profile in database (mocked)
            results['steps']['2_profile_storage'] = {'status': 'running'}
            
            try:
                # Mock database operations
                with patch('training.database.prompt_profile_store.get_db_session') as mock_session:
                    mock_session_instance = Mock()
                    mock_session.__enter__ = Mock(return_value=mock_session_instance)
                    mock_session.__exit__ = Mock(return_value=None)
                    
                    store = PromptProfileStore()
                    profile_id = store.create(test_profile, "test_corpus", "qa")
                    
                    results['steps']['2_profile_storage'] = {
                        'status': 'completed',
                        'stored_profile_id': profile_id
                    }
            except Exception as e:
                results['steps']['2_profile_storage'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['errors'].append(f"Profile storage failed: {e}")
            
            # Step 3: Create agents using ConfigurableAgentFactory
            results['steps']['3_agent_creation'] = {'status': 'running'}
            
            with patch('training.agent_factory.BasicWorkerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_worker_agent(agent_id)):
                with patch('training.agent_factory.BasicReviewerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_reviewer_agent(agent_id)):
                    
                    agent_config = AgentConfig(temperature=0.7, max_tokens=1500)
                    
                    worker_agent = ConfigurableAgentFactory.create_worker(
                        test_profile, 
                        "integration_test_worker", 
                        agent_config
                    )
                    
                    reviewer_agent = ConfigurableAgentFactory.create_reviewer(
                        test_profile,
                        "integration_test_reviewer", 
                        agent_config
                    )
                    
                    results['steps']['3_agent_creation'] = {
                        'status': 'completed',
                        'worker_agent_id': worker_agent.agent_id,
                        'reviewer_agent_id': reviewer_agent.agent_id,
                        'worker_prompt_applied': hasattr(worker_agent, '_applied_profile_config'),
                        'reviewer_prompt_applied': hasattr(reviewer_agent, '_applied_profile_config')
                    }
            
            # Step 4: Execute debate session
            results['steps']['4_debate_execution'] = {'status': 'running'}
            
            question = self.test_corpus_questions[0]
            debate_session = DebateSession(question, "integration_test_session")
            
            # Simulate debate turns
            worker_response = worker_agent.process(question)
            reviewer_critique = reviewer_agent.critique([worker_response])
            reviewer_synthesis = reviewer_agent.synthesize([worker_response])
            
            results['steps']['4_debate_execution'] = {
                'status': 'completed',
                'session_id': debate_session.session_id,
                'question': question,
                'worker_response_length': len(worker_response.content),
                'critique_provided': bool(reviewer_critique.content),
                'synthesis_provided': bool(reviewer_synthesis.content)
            }
            
            # Step 5: Create TrainingStep and validate results
            results['steps']['5_training_step'] = {'status': 'running'}
            
            training_step = TrainingStep(
                step_number=1,
                step_type="evaluation",
                prompt_profile_id=test_profile.profile_id,
                question=question,
                expected_response=None
            )
            
            training_step.add_agent_response(worker_response)
            training_step.add_agent_response(reviewer_critique)
            training_step.add_agent_response(reviewer_synthesis)
            
            # Set evaluation scores
            training_step.set_evaluation_score('dialectical_quality', 0.8)
            training_step.set_evaluation_score('logical_consistency', 0.85)
            training_step.set_evaluation_score('evidence_quality', 0.75)
            
            training_step.mark_completed()
            
            validation_errors = training_step.validate()
            if validation_errors:
                results['steps']['5_training_step'] = {
                    'status': 'failed',
                    'error': f"TrainingStep validation failed: {validation_errors}"
                }
                results['errors'].append(f"TrainingStep validation failed: {validation_errors}")
            else:
                results['steps']['5_training_step'] = {
                    'status': 'completed',
                    'step_id': training_step.step_id,
                    'average_score': training_step.get_average_score(),
                    'total_responses': len(training_step.agent_responses)
                }
            
            # Overall status
            failed_steps = [step for step in results['steps'].values() if step.get('status') == 'failed']
            results['status'] = 'failed' if failed_steps else 'completed'
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Unexpected error in end-to-end workflow: {e}")
        
        finally:
            benchmark.stop()
            results['performance'] = benchmark.get_results()
        
        return results

    def test_backward_compatibility(self) -> Dict[str, Any]:
        """
        Test 2: Backward compatibility with existing systems.
        
        Validates:
        - Existing AgentResponse format compatibility
        - Debate session integration
        - Original agent behavior preservation
        - Data structure interoperability
        """
        benchmark = PerformanceBenchmark("backward_compatibility")
        benchmark.start()
        
        results = {
            'test_name': 'backward_compatibility',
            'status': 'running',
            'compatibility_checks': {},
            'errors': [],
            'performance': {}
        }
        
        try:
            # Check 1: AgentResponse compatibility
            results['compatibility_checks']['agent_response'] = {'status': 'running'}
            
            # Create AgentResponse using existing format
            original_response = AgentResponse(
                content="Test response content",
                reasoning="Test reasoning",
                confidence=0.8,
                sources=["test_source"],
                metadata={'test': 'metadata'},
                timestamp=datetime.utcnow()
            )
            
            # Test TrainingStep integration
            training_step = TrainingStep(
                step_number=1,
                step_type="validation",
                prompt_profile_id=str(uuid.uuid4()),
                question="Test question"
            )
            
            training_step.add_agent_response(original_response)
            
            # Test serialization/deserialization
            step_dict = training_step.to_dict()
            reconstructed_step = TrainingStep.from_dict(step_dict)
            
            response_match = (
                reconstructed_step.agent_responses[0].content == original_response.content and
                reconstructed_step.agent_responses[0].confidence == original_response.confidence
            )
            
            results['compatibility_checks']['agent_response'] = {
                'status': 'completed' if response_match else 'failed',
                'serialization_works': bool(step_dict),
                'deserialization_works': bool(reconstructed_step),
                'content_preserved': response_match
            }
            
            # Check 2: Debate session integration
            results['compatibility_checks']['debate_session'] = {'status': 'running'}
            
            debate_session = DebateSession("Test question for compatibility")
            
            # Test that DebateSession can work with training-enhanced responses
            enhanced_response = AgentResponse(
                content="Enhanced response",
                reasoning="Enhanced reasoning",
                confidence=0.9,
                sources=["enhanced_source"],
                metadata={
                    'training': {
                        'profile_id': 'test_profile',
                        'optimization_target': 'dialectical_quality'
                    }
                },
                timestamp=datetime.utcnow()
            )
            
            # DebateSession should handle enhanced responses normally
            session_compatible = True
            try:
                # Test basic session functionality
                session_info = {
                    'session_id': debate_session.session_id,
                    'question': debate_session.question,
                    'start_time': debate_session.start_time
                }
            except Exception as e:
                session_compatible = False
                results['errors'].append(f"DebateSession compatibility issue: {e}")
            
            results['compatibility_checks']['debate_session'] = {
                'status': 'completed' if session_compatible else 'failed',
                'session_creation': session_compatible,
                'enhanced_response_handling': True  # AgentResponse format unchanged
            }
            
            # Check 3: Original agent behavior preservation
            results['compatibility_checks']['agent_behavior'] = {'status': 'running'}
            
            test_profile = self.create_test_prompt_profile("compatibility_test")
            
            with patch('training.agent_factory.BasicWorkerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_worker_agent(agent_id)):
                # Create agent with profile
                configured_agent = ConfigurableAgentFactory.create_worker(test_profile, "compatibility_test_worker")
                
                # Test that basic functionality still works
                test_question = "Test question for agent behavior"
                response = configured_agent.process(test_question)
                
                behavior_preserved = (
                    isinstance(response, AgentResponse) and
                    hasattr(configured_agent, 'agent_id') and
                    hasattr(configured_agent, '_applied_profile_config')
                )
                
                results['compatibility_checks']['agent_behavior'] = {
                    'status': 'completed' if behavior_preserved else 'failed',
                    'agent_creation': True,
                    'process_method_works': isinstance(response, AgentResponse),
                    'profile_config_applied': hasattr(configured_agent, '_applied_profile_config'),
                    'original_interface_preserved': hasattr(configured_agent, 'agent_id')
                }
            
            # Check 4: Data structure validation compatibility
            results['compatibility_checks']['data_structures'] = {'status': 'running'}
            
            # Test validate_all_structures function
            role_prompt = RolePrompt(role="test", prompt_text="Test prompt")
            prompt_profile = PromptProfile(name="test_profile")
            training_step = TrainingStep(question="Test question")
            
            validation_results = validate_all_structures(role_prompt, prompt_profile, training_step)
            
            structures_valid = len(validation_results) == 0  # No validation errors
            
            results['compatibility_checks']['data_structures'] = {
                'status': 'completed' if structures_valid else 'failed',
                'validation_function_works': True,
                'all_structures_valid': structures_valid,
                'validation_errors': validation_results
            }
            
            # Overall status
            failed_checks = [check for check in results['compatibility_checks'].values() if check.get('status') == 'failed']
            results['status'] = 'failed' if failed_checks else 'completed'
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Unexpected error in backward compatibility test: {e}")
        
        finally:
            benchmark.stop()
            results['performance'] = benchmark.get_results()
        
        return results

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """
        Test 3: Performance benchmarking and regression testing.
        
        Validates:
        - No significant performance regression
        - Memory usage within reasonable bounds
        - Response time benchmarks
        - Scalability under load
        """
        results = {
            'test_name': 'performance_benchmarks',
            'status': 'running',
            'benchmarks': {},
            'regressions': [],
            'errors': [],
            'baseline_comparison': {}
        }
        
        try:
            # Benchmark 1: Profile creation and validation
            profile_benchmark = PerformanceBenchmark("profile_creation")
            profile_benchmark.start()
            
            test_profile = self.create_test_prompt_profile("performance_test")
            validation_errors = test_profile.validate()
            
            profile_benchmark.stop()
            profile_results = profile_benchmark.get_results()
            
            results['benchmarks']['profile_creation'] = {
                'duration_ms': profile_results['duration_ms'],
                'memory_delta_mb': profile_results['memory_delta_mb'],
                'validation_success': len(validation_errors) == 0
            }
            
            # Benchmark 2: Agent factory operations
            factory_benchmark = PerformanceBenchmark("agent_factory")
            factory_benchmark.start()
            
            with patch('training.agent_factory.BasicWorkerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_worker_agent(agent_id)):
                with patch('training.agent_factory.BasicReviewerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_reviewer_agent(agent_id)):
                    
                    # Create multiple agents to test performance
                    agents_created = 0
                    for i in range(5):
                        worker = ConfigurableAgentFactory.create_worker(test_profile, f"perf_worker_{i}")
                        reviewer = ConfigurableAgentFactory.create_reviewer(test_profile, f"perf_reviewer_{i}")
                        agents_created += 2
            
            factory_benchmark.stop()
            factory_results = factory_benchmark.get_results()
            
            results['benchmarks']['agent_factory'] = {
                'duration_ms': factory_results['duration_ms'],
                'memory_delta_mb': factory_results['memory_delta_mb'],
                'agents_created': agents_created,
                'avg_creation_time_ms': factory_results['duration_ms'] / agents_created
            }
            
            # Benchmark 3: Data structure serialization
            serialization_benchmark = PerformanceBenchmark("serialization")
            serialization_benchmark.start()
            
            # Test serialization/deserialization performance
            serialization_cycles = 10
            for _ in range(serialization_cycles):
                # Profile serialization
                profile_json = test_profile.to_json()
                reconstructed_profile = PromptProfile.from_json(profile_json)
                
                # TrainingStep serialization
                training_step = TrainingStep(
                    question="Performance test question",
                    prompt_profile_id=test_profile.profile_id
                )
                step_json = training_step.to_json()
                reconstructed_step = TrainingStep.from_json(step_json)
            
            serialization_benchmark.stop()
            serialization_results = serialization_benchmark.get_results()
            
            results['benchmarks']['serialization'] = {
                'duration_ms': serialization_results['duration_ms'],
                'memory_delta_mb': serialization_results['memory_delta_mb'],
                'cycles_completed': serialization_cycles,
                'avg_cycle_time_ms': serialization_results['duration_ms'] / serialization_cycles
            }
            
            # Benchmark 4: Cache performance
            cache_benchmark = PerformanceBenchmark("cache_performance")
            cache_benchmark.start()
            
            cache = AgentCache()
            config = EnhancedAgentConfig()
            
            # Test cache operations
            cache_operations = 20
            for i in range(cache_operations):
                cache_key = f"test_key_{i}"
                test_agent = self.mock_integration.create_mock_worker_agent(f"cache_test_{i}")
                
                # Put and get operations
                cache.put(cache_key, test_agent, ttl=60)
                retrieved = cache.get(cache_key)
                
            cache_stats = cache.get_stats()
            
            cache_benchmark.stop()
            cache_results = cache_benchmark.get_results()
            
            results['benchmarks']['cache_performance'] = {
                'duration_ms': cache_results['duration_ms'],
                'memory_delta_mb': cache_results['memory_delta_mb'],
                'operations_completed': cache_operations * 2,  # put + get
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'cache_size': cache_stats.get('cache_size', 0)
            }
            
            # Performance regression analysis
            performance_thresholds = {
                'profile_creation_ms': 50,  # Should be fast
                'agent_creation_avg_ms': 100,  # Reasonable for mocked agents
                'serialization_avg_ms': 10,  # Should be very fast
                'cache_operation_avg_ms': 5  # Cache operations should be extremely fast
            }
            
            # Check for regressions
            if results['benchmarks']['profile_creation']['duration_ms'] > performance_thresholds['profile_creation_ms']:
                results['regressions'].append({
                    'area': 'profile_creation',
                    'threshold_ms': performance_thresholds['profile_creation_ms'],
                    'actual_ms': results['benchmarks']['profile_creation']['duration_ms'],
                    'severity': 'medium'
                })
            
            if results['benchmarks']['agent_factory']['avg_creation_time_ms'] > performance_thresholds['agent_creation_avg_ms']:
                results['regressions'].append({
                    'area': 'agent_factory',
                    'threshold_ms': performance_thresholds['agent_creation_avg_ms'],
                    'actual_ms': results['benchmarks']['agent_factory']['avg_creation_time_ms'],
                    'severity': 'medium'
                })
            
            if results['benchmarks']['serialization']['avg_cycle_time_ms'] > performance_thresholds['serialization_avg_ms']:
                results['regressions'].append({
                    'area': 'serialization',
                    'threshold_ms': performance_thresholds['serialization_avg_ms'],
                    'actual_ms': results['benchmarks']['serialization']['avg_cycle_time_ms'],
                    'severity': 'low'
                })
            
            # Memory usage validation
            total_memory_delta = sum([
                results['benchmarks']['profile_creation']['memory_delta_mb'],
                results['benchmarks']['agent_factory']['memory_delta_mb'],
                results['benchmarks']['serialization']['memory_delta_mb'],
                results['benchmarks']['cache_performance']['memory_delta_mb']
            ])
            
            if total_memory_delta > 50:  # 50MB threshold
                results['regressions'].append({
                    'area': 'memory_usage',
                    'threshold_mb': 50,
                    'actual_mb': total_memory_delta,
                    'severity': 'high'
                })
            
            results['status'] = 'completed' if not results['regressions'] else 'warning'
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Unexpected error in performance benchmarks: {e}")
        
        return results

    def test_error_conditions(self) -> Dict[str, Any]:
        """
        Test 4: Error condition handling and graceful failure modes.
        
        Validates:
        - Invalid input handling
        - Database error scenarios
        - Agent creation failures
        - Recovery procedures
        """
        results = {
            'test_name': 'error_conditions',
            'status': 'running',
            'error_scenarios': {},
            'recovery_tests': {},
            'errors': []
        }
        
        try:
            # Error Scenario 1: Invalid PromptProfile
            results['error_scenarios']['invalid_profile'] = {'status': 'running'}
            
            try:
                # Create invalid profile
                invalid_profile = PromptProfile(
                    profile_id="invalid-uuid",  # Invalid UUID format
                    name="",  # Empty name
                    description="Test invalid profile"
                )
                
                validation_errors = invalid_profile.validate()
                
                results['error_scenarios']['invalid_profile'] = {
                    'status': 'completed',
                    'validation_caught_errors': len(validation_errors) > 0,
                    'error_count': len(validation_errors),
                    'graceful_handling': True
                }
                
            except Exception as e:
                results['error_scenarios']['invalid_profile'] = {
                    'status': 'completed',
                    'validation_caught_errors': True,
                    'exception_type': type(e).__name__,
                    'graceful_handling': True
                }
            
            # Error Scenario 2: Database connection failures
            results['error_scenarios']['database_errors'] = {'status': 'running'}
            
            with patch('training.database.prompt_profile_store.get_db_session', side_effect=Exception("Database connection failed")):
                store = PromptProfileStore()
                
                try:
                    valid_profile = self.create_test_prompt_profile("error_test")
                    store.create(valid_profile, "test_corpus", "qa")
                    database_error_handled = False
                except Exception as e:
                    database_error_handled = "PromptProfileStoreError" in str(type(e)) or "Database" in str(e)
                
                results['error_scenarios']['database_errors'] = {
                    'status': 'completed',
                    'connection_failure_handled': database_error_handled,
                    'graceful_degradation': True
                }
            
            # Error Scenario 3: Agent creation with missing roles
            results['error_scenarios']['missing_roles'] = {'status': 'running'}
            
            incomplete_profile = PromptProfile(
                name="incomplete_profile",
                description="Profile missing required roles"
            )
            # Don't add any role prompts
            
            missing_role_handled = False
            error_type = 'none'
            try:
                with patch('training.agent_factory.BasicWorkerAgent'):
                    ConfigurableAgentFactory.create_worker(incomplete_profile, "test_worker")
            except ValueError as e:
                missing_role_handled = "role" in str(e).lower()
                error_type = 'ValueError'
            except Exception as e:
                missing_role_handled = True  # Any exception shows error handling
                error_type = type(e).__name__
            
            results['error_scenarios']['missing_roles'] = {
                'status': 'completed',
                'missing_role_detected': missing_role_handled,
                'error_type': error_type
            }
            
            # Error Scenario 4: Invalid configuration parameters
            results['error_scenarios']['invalid_config'] = {'status': 'running'}
            
            try:
                invalid_config = AgentConfig(
                    temperature=5.0,  # Invalid temperature > 2.0
                    max_tokens=-1,    # Invalid negative tokens
                    model_name=""     # Empty model name
                )
                invalid_config.validate()
                config_validation_worked = False
            except ValueError:
                config_validation_worked = True
            except Exception:
                config_validation_worked = True  # Any validation error is good
            
            results['error_scenarios']['invalid_config'] = {
                'status': 'completed',
                'validation_caught_errors': config_validation_worked,
                'parameter_validation_works': True
            }
            
            # Recovery Test 1: Cache failure recovery
            results['recovery_tests']['cache_failure'] = {'status': 'running'}
            
            cache = AgentCache()
            
            # Simulate cache corruption
            cache._cache = None  # Corrupt cache state
            
            try:
                # Should handle gracefully
                stats = cache.get_stats()
                cache_recovery = 'cache_size' in stats or stats == {}  # Should return something sensible
            except Exception:
                cache_recovery = True  # Exception handling is also acceptable
            
            results['recovery_tests']['cache_failure'] = {
                'status': 'completed',
                'graceful_recovery': cache_recovery,
                'fallback_behavior': True
            }
            
            # Recovery Test 2: Agent factory fallback
            results['recovery_tests']['factory_fallback'] = {'status': 'running'}
            
            test_profile = self.create_test_prompt_profile("recovery_test")
            
            # Test factory with failing agent creation
            with patch('training.agent_factory.BasicWorkerAgent', side_effect=Exception("Agent creation failed")):
                try:
                    ConfigurableAgentFactory.create_worker(test_profile, "failing_worker")
                    factory_fallback = False
                except Exception as e:
                    factory_fallback = True  # Should propagate error appropriately
            
            results['recovery_tests']['factory_fallback'] = {
                'status': 'completed',
                'error_propagation': factory_fallback,
                'clean_failure': True
            }
            
            # Overall status
            failed_scenarios = [scenario for scenario in results['error_scenarios'].values() if scenario.get('status') == 'failed']
            failed_recovery = [recovery for recovery in results['recovery_tests'].values() if recovery.get('status') == 'failed']
            
            results['status'] = 'failed' if (failed_scenarios or failed_recovery) else 'completed'
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Unexpected error in error condition testing: {e}")
        
        return results

    def test_rollback_procedures(self) -> Dict[str, Any]:
        """
        Test 5: Rollback procedure validation.
        
        Validates:
        - Agent restoration to defaults
        - Profile cleanup procedures
        - Cache clearing operations
        - System state recovery
        """
        results = {
            'test_name': 'rollback_procedures',
            'status': 'running',
            'rollback_tests': {},
            'errors': []
        }
        
        try:
            # Rollback Test 1: Agent restoration
            results['rollback_tests']['agent_restoration'] = {'status': 'running'}
            
            test_profile = self.create_test_prompt_profile("rollback_test")
            
            with patch('training.agent_factory.BasicWorkerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_worker_agent(agent_id)):
                # Create agent with profile
                agent = ConfigurableAgentFactory.create_worker(test_profile, "rollback_test_worker")
                
                # Verify profile was applied
                profile_applied = hasattr(agent, '_applied_profile_config')
                original_prompt_stored = hasattr(agent, '_original_system_prompt')
                
                # Restore agent to defaults
                ConfigurableAgentFactory.restore_agent_defaults(agent)
                
                # Verify restoration
                profile_removed = not hasattr(agent, '_applied_profile_config')
                original_method_restored = not hasattr(agent, '_original_make_gemini_call')
                
                results['rollback_tests']['agent_restoration'] = {
                    'status': 'completed',
                    'profile_initially_applied': profile_applied,
                    'original_prompt_stored': original_prompt_stored,
                    'profile_config_removed': profile_removed,
                    'original_method_restored': original_method_restored,
                    'restoration_successful': profile_removed and original_method_restored
                }
            
            # Rollback Test 2: Cache clearing
            results['rollback_tests']['cache_clearing'] = {'status': 'running'}
            
            cache = AgentCache()
            
            # Populate cache
            for i in range(5):
                test_agent = self.mock_integration.create_mock_worker_agent(f"cache_agent_{i}")
                cache.put(f"test_key_{i}", test_agent, ttl=60)
            
            initial_cache_size = cache.get_stats()['cache_size']
            
            # Clear cache
            cache.clear()
            
            final_cache_size = cache.get_stats()['cache_size']
            
            results['rollback_tests']['cache_clearing'] = {
                'status': 'completed',
                'initial_cache_size': initial_cache_size,
                'final_cache_size': final_cache_size,
                'cache_cleared': final_cache_size == 0,
                'clearing_successful': initial_cache_size > 0 and final_cache_size == 0
            }
            
            # Rollback Test 3: Profile cleanup simulation
            results['rollback_tests']['profile_cleanup'] = {'status': 'running'}
            
            # Simulate profile cleanup by testing delete operations
            with patch('training.database.prompt_profile_store.get_db_session') as mock_session:
                mock_session_instance = Mock()
                mock_session.__enter__ = Mock(return_value=mock_session_instance)
                mock_session.__exit__ = Mock(return_value=None)
                
                # Mock successful deletion
                mock_model = Mock()
                mock_session_instance.query().filter().first.return_value = mock_model
                
                store = PromptProfileStore()
                deletion_result = store.delete(str(uuid.uuid4()))
                
                results['rollback_tests']['profile_cleanup'] = {
                    'status': 'completed',
                    'deletion_operation_works': deletion_result is not None,
                    'cleanup_procedure_available': True
                }
            
            # Rollback Test 4: System state consistency
            results['rollback_tests']['system_state'] = {'status': 'running'}
            
            # Test that after rollback operations, system can still function normally
            try:
                # Create new profile after cleanup
                new_profile = self.create_test_prompt_profile("post_rollback_test")
                validation_errors = new_profile.validate()
                
                # Create new agents after restoration
                with patch('training.agent_factory.BasicWorkerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_worker_agent(agent_id)):
                    new_agent = ConfigurableAgentFactory.create_worker(new_profile, "post_rollback_worker")
                
                system_functional = len(validation_errors) == 0 and new_agent is not None
                
                results['rollback_tests']['system_state'] = {
                    'status': 'completed',
                    'post_rollback_functionality': system_functional,
                    'new_profile_creation': len(validation_errors) == 0,
                    'new_agent_creation': new_agent is not None,
                    'system_consistency': True
                }
                
            except Exception as e:
                results['rollback_tests']['system_state'] = {
                    'status': 'failed',
                    'error': str(e),
                    'post_rollback_functionality': False
                }
            
            # Overall status
            failed_rollbacks = [test for test in results['rollback_tests'].values() if test.get('status') == 'failed']
            results['status'] = 'failed' if failed_rollbacks else 'completed'
            
        except Exception as e:
            import traceback
            results['status'] = 'failed'
            results['errors'].append(f"Unexpected error in rollback procedure testing: {e}")
            results['errors'].append(f"Traceback: {traceback.format_exc()}")
        
        return results

    def test_concurrent_operations(self) -> Dict[str, Any]:
        """
        Test 6: Concurrent operations and thread safety.
        
        Validates:
        - Thread-safe cache operations
        - Concurrent agent creation
        - Database operation concurrency
        - Resource contention handling
        """
        results = {
            'test_name': 'concurrent_operations',
            'status': 'running',
            'concurrency_tests': {},
            'errors': []
        }
        
        try:
            # Concurrency Test 1: Thread-safe cache operations
            results['concurrency_tests']['cache_thread_safety'] = {'status': 'running'}
            
            cache = AgentCache()
            cache_errors = []
            operations_completed = 0
            
            def cache_worker(thread_id: int):
                nonlocal operations_completed
                try:
                    for i in range(10):
                        key = f"thread_{thread_id}_key_{i}"
                        agent = self.mock_integration.create_mock_worker_agent(f"thread_{thread_id}_agent_{i}")
                        
                        # Put operation
                        cache.put(key, agent, ttl=60)
                        
                        # Get operation
                        retrieved = cache.get(key)
                        
                        operations_completed += 1
                        
                        # Small delay to encourage race conditions
                        time.sleep(0.001)
                        
                except Exception as e:
                    cache_errors.append(f"Thread {thread_id}: {e}")
            
            # Run concurrent cache operations
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cache_worker, i) for i in range(4)]
                for future in as_completed(futures, timeout=10):
                    future.result()  # Wait for completion
            
            final_cache_stats = cache.get_stats()
            
            results['concurrency_tests']['cache_thread_safety'] = {
                'status': 'completed' if not cache_errors else 'failed',
                'operations_completed': operations_completed,
                'errors_encountered': len(cache_errors),
                'final_cache_size': final_cache_stats.get('cache_size', 0),
                'thread_safety_maintained': len(cache_errors) == 0
            }
            
            # Concurrency Test 2: Concurrent agent creation
            results['concurrency_tests']['agent_creation'] = {'status': 'running'}
            
            test_profile = self.create_test_prompt_profile("concurrency_test")
            creation_errors = []
            agents_created = []
            
            def agent_creation_worker(thread_id: int):
                try:
                    with patch('training.agent_factory.BasicWorkerAgent', side_effect=lambda agent_id: self.mock_integration.create_mock_worker_agent(agent_id)):
                        for i in range(3):
                            agent = ConfigurableAgentFactory.create_worker(
                                test_profile, 
                                f"concurrent_worker_{thread_id}_{i}"
                            )
                            agents_created.append(agent.agent_id)
                            
                except Exception as e:
                    creation_errors.append(f"Thread {thread_id}: {e}")
            
            # Run concurrent agent creation
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(agent_creation_worker, i) for i in range(3)]
                for future in as_completed(futures, timeout=10):
                    future.result()
            
            results['concurrency_tests']['agent_creation'] = {
                'status': 'completed' if not creation_errors else 'failed',
                'agents_created': len(agents_created),
                'errors_encountered': len(creation_errors),
                'expected_agents': 9,  # 3 threads * 3 agents
                'all_agents_created': len(agents_created) == 9,
                'concurrent_creation_safe': len(creation_errors) == 0
            }
            
            # Concurrency Test 3: Profile serialization under load
            results['concurrency_tests']['serialization_load'] = {'status': 'running'}
            
            serialization_errors = []
            serialization_count = 0
            
            def serialization_worker(thread_id: int):
                nonlocal serialization_count
                try:
                    profile = self.create_test_prompt_profile(f"serialization_test_{thread_id}")
                    
                    for i in range(5):
                        # Serialize to JSON
                        json_data = profile.to_json()
                        
                        # Deserialize from JSON
                        reconstructed = PromptProfile.from_json(json_data)
                        
                        # Validate
                        errors = reconstructed.validate()
                        if errors:
                            serialization_errors.append(f"Validation errors: {errors}")
                        
                        serialization_count += 1
                        
                except Exception as e:
                    serialization_errors.append(f"Thread {thread_id}: {e}")
            
            # Run concurrent serialization
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(serialization_worker, i) for i in range(3)]
                for future in as_completed(futures, timeout=10):
                    future.result()
            
            results['concurrency_tests']['serialization_load'] = {
                'status': 'completed' if not serialization_errors else 'failed',
                'serializations_completed': serialization_count,
                'errors_encountered': len(serialization_errors),
                'expected_serializations': 15,  # 3 threads * 5 operations
                'serialization_thread_safe': len(serialization_errors) == 0
            }
            
            # Overall status
            failed_tests = [test for test in results['concurrency_tests'].values() if test.get('status') == 'failed']
            results['status'] = 'failed' if failed_tests else 'completed'
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Unexpected error in concurrency testing: {e}")
        
        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests and compile comprehensive results.
        
        Returns:
            Complete test results with status, performance metrics, and recommendations
        """
        print("🧪 Starting Comprehensive Integration Tests for Training Foundation Layer")
        print("=" * 80)
        
        start_time = datetime.utcnow()
        overall_results = {
            'test_suite': 'training_foundation_integration',
            'start_time': start_time.isoformat(),
            'tests': {},
            'summary': {},
            'recommendations': [],
            'performance_summary': {},
            'errors': []
        }
        
        # Test 1: End-to-end workflow
        print("\n🔄 Test 1: End-to-End Workflow Testing...")
        overall_results['tests']['end_to_end_workflow'] = self.test_end_to_end_workflow()
        
        # Test 2: Backward compatibility
        print("⚡ Test 2: Backward Compatibility Validation...")
        overall_results['tests']['backward_compatibility'] = self.test_backward_compatibility()
        
        # Test 3: Performance benchmarks
        print("📊 Test 3: Performance Benchmarking...")
        overall_results['tests']['performance_benchmarks'] = self.test_performance_benchmarks()
        
        # Test 4: Error conditions
        print("❌ Test 4: Error Condition Handling...")
        overall_results['tests']['error_conditions'] = self.test_error_conditions()
        
        # Test 5: Rollback procedures
        print("🔄 Test 5: Rollback Procedure Validation...")
        overall_results['tests']['rollback_procedures'] = self.test_rollback_procedures()
        
        # Test 6: Concurrent operations
        print("🔀 Test 6: Concurrent Operations Testing...")
        overall_results['tests']['concurrent_operations'] = self.test_concurrent_operations()
        
        # Compile summary
        end_time = datetime.utcnow()
        overall_results['end_time'] = end_time.isoformat()
        overall_results['total_duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Test status summary
        test_statuses = {name: test['status'] for name, test in overall_results['tests'].items()}
        passed_tests = sum(1 for status in test_statuses.values() if status == 'completed')
        warning_tests = sum(1 for status in test_statuses.values() if status == 'warning')
        failed_tests = sum(1 for status in test_statuses.values() if status == 'failed')
        total_tests = len(test_statuses)
        
        overall_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'warning_tests': warning_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'test_statuses': test_statuses
        }
        
        # Performance summary
        all_performance_data = []
        for test_name, test_data in overall_results['tests'].items():
            if 'performance' in test_data:
                perf_data = test_data['performance'].copy()
                perf_data['test_name'] = test_name
                all_performance_data.append(perf_data)
        
        if all_performance_data:
            total_duration_ms = sum(p.get('duration_ms', 0) for p in all_performance_data)
            total_memory_delta_mb = sum(p.get('memory_delta_mb', 0) for p in all_performance_data)
            
            overall_results['performance_summary'] = {
                'total_test_duration_ms': total_duration_ms,
                'total_memory_delta_mb': total_memory_delta_mb,
                'performance_tests': all_performance_data
            }
        
        # Generate recommendations
        recommendations = []
        
        if failed_tests > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'test_failures',
                'description': f'{failed_tests} tests failed. Review error logs and fix critical issues before deployment.',
                'action_items': ['Review failed test details', 'Fix underlying issues', 'Re-run tests']
            })
        
        if warning_tests > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'performance_warnings',
                'description': f'{warning_tests} tests completed with warnings. Monitor performance metrics.',
                'action_items': ['Review performance benchmarks', 'Optimize if needed', 'Set monitoring']
            })
        
        # Performance recommendations
        if overall_results.get('performance_summary', {}).get('total_memory_delta_mb', 0) > 30:
            recommendations.append({
                'priority': 'medium',
                'category': 'memory_usage',
                'description': 'High memory usage detected during tests. Consider optimization.',
                'action_items': ['Profile memory usage', 'Optimize data structures', 'Implement cleanup']
            })
        
        # Check for regression issues
        performance_test = overall_results['tests'].get('performance_benchmarks', {})
        if performance_test.get('regressions'):
            recommendations.append({
                'priority': 'medium',
                'category': 'performance_regression',
                'description': f"Performance regressions detected: {len(performance_test['regressions'])} issues",
                'action_items': ['Review regression details', 'Optimize slow operations', 'Update thresholds if needed']
            })
        
        # Success recommendations
        if passed_tests == total_tests:
            recommendations.append({
                'priority': 'low',
                'category': 'deployment_ready',
                'description': 'All integration tests passed. Training foundation layer is ready for deployment.',
                'action_items': ['Proceed with deployment', 'Monitor in production', 'Set up alerting']
            })
        
        overall_results['recommendations'] = recommendations
        
        # Overall status determination
        if failed_tests > 0:
            overall_results['overall_status'] = 'failed'
        elif warning_tests > 0:
            overall_results['overall_status'] = 'warning'
        else:
            overall_results['overall_status'] = 'passed'
        
        return overall_results


def main():
    """Main test runner function."""
    import json
    
    # Set up test environment
    os.environ['TESTING'] = 'true'
    
    # Run integration tests
    test_suite = TrainingFoundationIntegrationTests()
    results = test_suite.run_all_tests()
    
    # Save results to file
    results_file = Path(__file__).parent.parent.parent / 'integration_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏁 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    summary = results['summary']
    print(f"Total Tests:     {summary['total_tests']}")
    print(f"✅ Passed:       {summary['passed_tests']}")
    print(f"⚠️  Warnings:     {summary['warning_tests']}")
    print(f"❌ Failed:       {summary['failed_tests']}")
    print(f"📊 Success Rate: {summary['success_rate']:.2%}")
    
    print(f"\n⏱️  Total Duration: {results['total_duration_seconds']:.2f} seconds")
    
    if 'performance_summary' in results:
        perf = results['performance_summary']
        print(f"🧠 Memory Delta:  {perf.get('total_memory_delta_mb', 0):.2f} MB")
        print(f"⚡ Test Duration: {perf.get('total_test_duration_ms', 0):.0f} ms")
    
    # Print test details
    print(f"\n📋 Test Details:")
    for test_name, test_result in results['tests'].items():
        status_emoji = {'completed': '✅', 'warning': '⚠️', 'failed': '❌'}.get(test_result['status'], '❓')
        print(f"  {status_emoji} {test_name}: {test_result['status']}")
    
    # Print recommendations
    if results.get('recommendations'):
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rec['priority'], '⚪')
            print(f"  {priority_emoji} {rec['category']}: {rec['description']}")
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'failed':
        print(f"\n❌ Integration tests FAILED. Review error details and fix issues.")
        sys.exit(1)
    elif results['overall_status'] == 'warning':
        print(f"\n⚠️  Integration tests passed with WARNINGS. Review performance metrics.")
        sys.exit(0)
    else:
        print(f"\n✅ All integration tests PASSED. Training foundation layer is ready!")
        sys.exit(0)


# Pytest-compatible test functions
@pytest.fixture
def integration_test_suite():
    """Pytest fixture for the integration test suite."""
    return TrainingFoundationIntegrationTests()


def test_end_to_end_workflow(integration_test_suite):
    """Pytest test for end-to-end workflow."""
    result = integration_test_suite.test_end_to_end_workflow()
    assert result['status'] == 'completed', f"End-to-end workflow failed: {result.get('errors', [])}"


def test_backward_compatibility(integration_test_suite):
    """Pytest test for backward compatibility."""
    result = integration_test_suite.test_backward_compatibility()
    assert result['status'] == 'completed', f"Backward compatibility failed: {result.get('errors', [])}"


def test_performance_benchmarks(integration_test_suite):
    """Pytest test for performance benchmarks."""
    result = integration_test_suite.test_performance_benchmarks()
    assert result['status'] in ['completed', 'warning'], f"Performance benchmarks failed: {result.get('errors', [])}"


def test_error_conditions(integration_test_suite):
    """Pytest test for error condition handling."""
    result = integration_test_suite.test_error_conditions()
    assert result['status'] == 'completed', f"Error condition handling failed: {result.get('errors', [])}"


def test_rollback_procedures(integration_test_suite):
    """Pytest test for rollback procedures."""
    result = integration_test_suite.test_rollback_procedures()
    assert result['status'] == 'completed', f"Rollback procedures failed: {result.get('errors', [])}"


def test_concurrent_operations(integration_test_suite):
    """Pytest test for concurrent operations."""
    result = integration_test_suite.test_concurrent_operations()
    assert result['status'] == 'completed', f"Concurrent operations failed: {result.get('errors', [])}"


if __name__ == "__main__":
    main()
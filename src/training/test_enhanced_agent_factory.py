"""
Tests for Enhanced Agent Factory with ClaudeAgentConfig Integration

This test suite validates the enhanced agent factory functionality including:
- Claude configuration integration
- Agent caching system
- Enhanced configuration handling
- Backward compatibility
- Batch agent creation
"""

import pytest
import uuid
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import the classes we're testing
from .agent_factory import (
    ConfigurableAgentFactory,
    AgentConfig,
    EnhancedAgentConfig, 
    AgentCache
)
from .models.claude_agent_config import (
    ClaudeAgentConfig,
    ClaudeModelSettings,
    ClaudePromptConfig,
    ClaudeModel,
    PromptRole
)
from .data_structures import PromptProfile, RolePrompt


class TestEnhancedAgentConfig:
    """Test the EnhancedAgentConfig class."""
    
    def test_enhanced_config_creation(self):
        """Test creating enhanced configuration with default values."""
        config = EnhancedAgentConfig()
        
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.model_name == 'gemini-2.5-flash'
        assert config.claude_config is None
        assert config.use_claude_api is False
        assert config.caching_enabled is True
        assert config.cache_ttl == 3600
        assert config.timeout == 60.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
    
    def test_enhanced_config_validation(self):
        """Test validation of enhanced configuration."""
        # Test valid configuration
        config = EnhancedAgentConfig(temperature=0.5, max_tokens=1000)
        config.validate()  # Should not raise
        
        # Test invalid temperature
        config = EnhancedAgentConfig(temperature=2.5)
        with pytest.raises(ValueError, match="Temperature must be between"):
            config.validate()
        
        # Test invalid max_tokens
        config = EnhancedAgentConfig(max_tokens=300000)
        with pytest.raises(ValueError, match="Max tokens must be between"):
            config.validate()
        
        # Test invalid timeout
        config = EnhancedAgentConfig(timeout=700.0)
        with pytest.raises(ValueError, match="Timeout must be between"):
            config.validate()
    
    def test_from_legacy_config(self):
        """Test creating enhanced config from legacy AgentConfig."""
        legacy = AgentConfig(temperature=0.8, max_tokens=1500, model_name='test-model')
        enhanced = EnhancedAgentConfig.from_legacy_config(legacy)
        
        assert enhanced.temperature == 0.8
        assert enhanced.max_tokens == 1500
        assert enhanced.model_name == 'test-model'
        assert enhanced.claude_config is None
        assert enhanced.use_claude_api is False
    
    def test_from_claude_config(self):
        """Test creating enhanced config from ClaudeAgentConfig."""
        claude_config = ClaudeAgentConfig.create_worker_config("test-worker", temperature=0.9)
        enhanced = EnhancedAgentConfig.from_claude_config(claude_config)
        
        assert enhanced.temperature == 0.9
        assert enhanced.max_tokens == 4000  # From ClaudeModelSettings default
        assert enhanced.claude_config == claude_config
        assert enhanced.use_claude_api is True
    
    def test_get_effective_model_params(self):
        """Test getting effective model parameters."""
        # Test legacy parameters
        config = EnhancedAgentConfig(temperature=0.6, max_tokens=1200, model_name='test-model')
        params = config.get_effective_model_params()
        
        assert params['temperature'] == 0.6
        assert params['max_output_tokens'] == 1200
        assert params['model'] == 'test-model'
        
        # Test with Claude config
        claude_config = ClaudeAgentConfig.create_worker_config("test", temperature=0.3)
        enhanced = EnhancedAgentConfig.from_claude_config(claude_config)
        params = enhanced.get_effective_model_params()
        
        assert params['temperature'] == 0.3
        assert 'model' in params


class TestAgentCache:
    """Test the AgentCache system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = AgentCache()
        self.profile = PromptProfile(
            name="Test Profile",
            description="Test profile for caching"
        )
        self.config = EnhancedAgentConfig()
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        key = self.cache.get_cache_key(self.profile, 'worker', self.config)
        
        assert isinstance(key, str)
        assert 'worker' in key
        assert self.profile.profile_id[:8] in key
    
    def test_cache_put_and_get(self):
        """Test putting and getting agents from cache."""
        # Create a mock agent
        mock_agent = Mock()
        cache_key = "test_key"
        
        # Put in cache
        self.cache.put(cache_key, mock_agent, ttl=10)
        
        # Get from cache
        retrieved = self.cache.get(cache_key)
        assert retrieved == mock_agent
        
        # Check stats
        stats = self.cache.get_stats()
        assert stats['hits'] == 1
        assert stats['created'] == 1
        assert stats['cache_size'] == 1
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        mock_agent = Mock()
        cache_key = "test_key"
        
        # Put in cache with very short TTL
        self.cache.put(cache_key, mock_agent, ttl=1)
        
        # Should be available immediately
        assert self.cache.get(cache_key) == mock_agent
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be None (expired)
        assert self.cache.get(cache_key) is None
        
        # Check stats show eviction
        stats = self.cache.get_stats()
        assert stats['evictions'] >= 1
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        mock_agent = Mock()
        self.cache.put("key1", mock_agent, ttl=10)
        self.cache.put("key2", mock_agent, ttl=10)
        
        assert self.cache.get_stats()['cache_size'] == 2
        
        self.cache.clear()
        
        assert self.cache.get_stats()['cache_size'] == 0
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None


class TestEnhancedAgentFactory:
    """Test the enhanced agent factory functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test profile with worker and reviewer roles
        self.profile = PromptProfile(
            name="Test Profile", 
            description="Test profile for factory"
        )
        
        worker_role = RolePrompt(
            role="worker",
            prompt_text="You are a test worker agent.",
            description="Test worker prompt"
        )
        reviewer_role = RolePrompt(
            role="reviewer", 
            prompt_text="You are a test reviewer agent.",
            description="Test reviewer prompt"
        )
        
        self.profile.add_role_prompt(worker_role)
        self.profile.add_role_prompt(reviewer_role)
        
        # Create test configurations
        self.legacy_config = AgentConfig(temperature=0.5)
        self.enhanced_config = EnhancedAgentConfig(temperature=0.6, caching_enabled=False)
        self.claude_config = ClaudeAgentConfig.create_worker_config("test-claude", temperature=0.7)
    
    def test_config_normalization(self):
        """Test configuration normalization."""
        # Test with None
        result = ConfigurableAgentFactory._normalize_config(None)
        assert isinstance(result, EnhancedAgentConfig)
        
        # Test with legacy config
        result = ConfigurableAgentFactory._normalize_config(self.legacy_config)
        assert isinstance(result, EnhancedAgentConfig)
        assert result.temperature == 0.5
        
        # Test with enhanced config
        result = ConfigurableAgentFactory._normalize_config(self.enhanced_config)
        assert result == self.enhanced_config
        
        # Test with Claude config
        result = ConfigurableAgentFactory._normalize_config(self.claude_config)
        assert isinstance(result, EnhancedAgentConfig)
        assert result.claude_config == self.claude_config
    
    @patch('agents.worker.BasicWorkerAgent')
    def test_create_agent_enhanced_worker(self, mock_worker_class):
        """Test enhanced agent creation for worker."""
        mock_agent = Mock()
        mock_worker_class.return_value = mock_agent
        
        # Mock the agent methods
        mock_agent.logger = Mock()
        mock_agent.logger.log_debug = Mock()
        mock_agent._make_gemini_call = Mock()
        mock_agent.SYSTEM_PROMPT = "original prompt"
        
        result = ConfigurableAgentFactory.create_agent_enhanced(
            'worker', self.profile, 'test-worker', self.enhanced_config
        )
        
        assert result == mock_agent
        mock_worker_class.assert_called_once()
        
        # Verify agent was configured
        assert hasattr(mock_agent, '_applied_profile_config')
        assert mock_agent._applied_profile_config['role'] == 'worker'
    
    @patch('agents.reviewer.BasicReviewerAgent')  
    def test_create_agent_enhanced_reviewer(self, mock_reviewer_class):
        """Test enhanced agent creation for reviewer."""
        mock_agent = Mock()
        mock_reviewer_class.return_value = mock_agent
        
        # Mock the agent methods
        mock_agent.logger = Mock()
        mock_agent.logger.log_debug = Mock()
        mock_agent._make_gemini_call = Mock()
        mock_agent.CRITIQUE_PROMPT = "original critique"
        mock_agent.SYNTHESIS_PROMPT = "original synthesis"
        
        result = ConfigurableAgentFactory.create_agent_enhanced(
            'reviewer', self.profile, 'test-reviewer', self.enhanced_config
        )
        
        assert result == mock_agent
        mock_reviewer_class.assert_called_once()
        
        # Verify agent was configured
        assert hasattr(mock_agent, '_applied_profile_config')
        assert mock_agent._applied_profile_config['role'] == 'reviewer'
    
    def test_create_agent_enhanced_invalid_type(self):
        """Test enhanced agent creation with invalid agent type."""
        with pytest.raises(ValueError, match="Invalid agent type"):
            ConfigurableAgentFactory.create_agent_enhanced(
                'invalid_type', self.profile, 'test', self.enhanced_config
            )
    
    def test_create_agent_enhanced_missing_role(self):
        """Test enhanced agent creation with missing role in profile."""
        empty_profile = PromptProfile(name="Empty", description="Empty profile")
        
        with pytest.raises(ValueError, match="does not contain worker"):
            ConfigurableAgentFactory.create_agent_enhanced(
                'worker', empty_profile, 'test', self.enhanced_config
            )
    
    @patch('agents.worker.BasicWorkerAgent')
    def test_create_agent_from_claude_config(self, mock_worker_class):
        """Test creating agent directly from ClaudeAgentConfig."""
        mock_agent = Mock()
        mock_worker_class.return_value = mock_agent
        mock_agent.logger = Mock()
        mock_agent.logger.log_debug = Mock()
        mock_agent._make_gemini_call = Mock()
        mock_agent.SYSTEM_PROMPT = "original prompt"
        
        result = ConfigurableAgentFactory.create_agent_from_claude_config(
            'worker', self.profile, self.claude_config, 'claude-test'
        )
        
        assert result == mock_agent
        assert hasattr(mock_agent, '_applied_profile_config')
        assert mock_agent._applied_profile_config['claude_config_id'] == self.claude_config.config_id
    
    def test_create_agents_batch_success(self):
        """Test batch agent creation with successful results."""
        with patch('agents.worker.BasicWorkerAgent') as mock_worker, \
             patch('agents.reviewer.BasicReviewerAgent') as mock_reviewer:
            
            # Setup mocks
            mock_worker_instance = Mock()
            mock_reviewer_instance = Mock()
            mock_worker.return_value = mock_worker_instance
            mock_reviewer.return_value = mock_reviewer_instance
            
            for mock_instance in [mock_worker_instance, mock_reviewer_instance]:
                mock_instance.logger = Mock()
                mock_instance.logger.log_debug = Mock()
                mock_instance._make_gemini_call = Mock()
                mock_instance.SYSTEM_PROMPT = "original"
                mock_instance.CRITIQUE_PROMPT = "original critique"
                mock_instance.SYNTHESIS_PROMPT = "original synthesis"
            
            agent_specs = [
                {
                    'type': 'worker',
                    'profile': self.profile,
                    'agent_id': 'worker-1'
                },
                {
                    'type': 'reviewer', 
                    'profile': self.profile,
                    'agent_id': 'reviewer-1'
                }
            ]
            
            result = ConfigurableAgentFactory.create_agents_batch(
                agent_specs, self.enhanced_config
            )
            
            assert result['success_count'] == 2
            assert result['total_count'] == 2
            assert len(result['errors']) == 0
            assert 'worker-1' in result['agents']
            assert 'reviewer-1' in result['agents']
    
    def test_create_agents_batch_partial_failure(self):
        """Test batch agent creation with some failures."""
        agent_specs = [
            {
                'type': 'worker',
                'profile': self.profile,
                'agent_id': 'worker-1'
            },
            {
                'type': 'invalid_type',  # This will fail
                'profile': self.profile,
                'agent_id': 'invalid-1'
            }
        ]
        
        with patch('agents.worker.BasicWorkerAgent') as mock_worker:
            mock_worker_instance = Mock()
            mock_worker.return_value = mock_worker_instance
            mock_worker_instance.logger = Mock()
            mock_worker_instance.logger.log_debug = Mock()
            mock_worker_instance._make_gemini_call = Mock()
            mock_worker_instance.SYSTEM_PROMPT = "original"
            
            result = ConfigurableAgentFactory.create_agents_batch(
                agent_specs, self.enhanced_config
            )
            
            assert result['success_count'] == 1
            assert result['total_count'] == 2
            assert len(result['errors']) == 1
            assert 'worker-1' in result['agents']
            assert 'invalid-1' in result['agents']
            assert 'error' in result['agents']['invalid-1']


class TestBackwardCompatibility:
    """Test backward compatibility with existing AgentFactory methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profile = PromptProfile(name="Test", description="Test profile")
        
        worker_role = RolePrompt(
            role="worker",
            prompt_text="You are a worker.",
            description="Worker prompt"
        )
        self.profile.add_role_prompt(worker_role)
    
    @patch('agents.worker.BasicWorkerAgent')
    def test_legacy_create_worker(self, mock_worker_class):
        """Test that legacy create_worker method still works."""
        mock_agent = Mock()
        mock_worker_class.return_value = mock_agent
        mock_agent.logger = Mock()
        mock_agent.logger.log_debug = Mock()
        mock_agent._make_gemini_call = Mock()
        mock_agent.SYSTEM_PROMPT = "original"
        
        legacy_config = AgentConfig(temperature=0.8)
        
        result = ConfigurableAgentFactory.create_worker(
            self.profile, 'legacy-worker', legacy_config
        )
        
        assert result == mock_agent
        mock_worker_class.assert_called_once_with('legacy-worker')


class TestFactoryClacheMechanisms:
    """Test factory-level caching mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        ConfigurableAgentFactory.clear_cache()  # Start fresh
    
    def teardown_method(self):
        """Clean up after tests."""
        ConfigurableAgentFactory.clear_cache()
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        stats = ConfigurableAgentFactory.get_cache_stats()
        
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'cache_size' in stats
        assert 'hit_rate' in stats
    
    def test_cache_control_methods(self):
        """Test cache control methods."""
        # Test cache clearing
        ConfigurableAgentFactory.clear_cache()
        stats = ConfigurableAgentFactory.get_cache_stats()
        assert stats['cache_size'] == 0
        
        # Test cache cleanup
        ConfigurableAgentFactory.cleanup_cache()  # Should not raise
    
    def test_cache_enable_disable(self):
        """Test enabling and disabling cache."""
        # Test disabling cache (should clear it)
        ConfigurableAgentFactory.set_cache_enabled(False)
        stats = ConfigurableAgentFactory.get_cache_stats()
        assert stats['cache_size'] == 0
        
        # Test re-enabling
        ConfigurableAgentFactory.set_cache_enabled(True)  # Should not raise


# Integration test
class TestFullIntegration:
    """Test full integration with real-ish scenarios."""
    
    def test_full_workflow(self):
        """Test a complete workflow from profile to configured agent."""
        # Create a comprehensive profile
        profile = PromptProfile(
            name="Integration Test Profile",
            description="Full integration test profile",
            author="test-suite"
        )
        
        # Add multiple role prompts
        roles = ['worker', 'reviewer', 'orchestrator', 'summarizer']
        for role in roles:
            role_prompt = RolePrompt(
                role=role,
                prompt_text=f"You are a {role} agent for integration testing.",
                description=f"Integration test {role} prompt",
                metadata={'test': True, 'role_type': role}
            )
            profile.add_role_prompt(role_prompt)
        
        # Create Claude configuration
        claude_config = ClaudeAgentConfig.create_worker_config(
            "integration-test-config",
            temperature=0.5
        )
        claude_config.add_profile_compatibility(profile.profile_id)
        
        # Test enhanced configuration creation
        enhanced_config = EnhancedAgentConfig.from_claude_config(claude_config)
        enhanced_config.caching_enabled = True
        enhanced_config.cache_ttl = 1800  # 30 minutes
        
        # This would create agents in a real scenario
        # For testing, we'll just verify the configuration is valid
        enhanced_config.validate()
        assert enhanced_config.claude_config == claude_config
        assert enhanced_config.use_claude_api is True
        
        # Verify profile compatibility
        assert claude_config.is_compatible_with_profile(profile.profile_id)


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v"])
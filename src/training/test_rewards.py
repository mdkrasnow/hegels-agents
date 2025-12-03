"""
Comprehensive unit tests for the reward computation system.

Tests cover edge cases, performance requirements, and integration
with existing components.
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from agents.utils import AgentResponse
from debate.session import DebateSession, ConflictAnalysis
from training.rewards import (
    RewardCalculator, 
    RewardConfig, 
    RewardComponents,
    TextSimilarityCalculator,
    DebateQualityCalculator,
    create_standard_reward_calculator,
    create_fast_reward_calculator,
    create_quality_focused_calculator
)


class TestRewardConfig:
    """Test reward configuration validation and setup."""
    
    def test_default_config_valid(self):
        """Test that default configuration is valid."""
        config = RewardConfig()
        errors = config.validate()
        assert errors == [], f"Default config should be valid, got errors: {errors}"
    
    def test_config_weight_validation(self):
        """Test configuration weight validation."""
        config = RewardConfig()
        
        # Test invalid main weights
        config.text_quality_weight = 0.5
        config.debate_quality_weight = 0.5
        config.process_efficiency_weight = 0.3  # Sum > 1.0
        config.meta_rewards_weight = 0.3
        
        errors = config.validate()
        assert len(errors) > 0
        assert "Main weights" in errors[0]
    
    def test_config_subweight_validation(self):
        """Test subweight validation."""
        config = RewardConfig()
        
        # Test invalid text quality subweights
        config.similarity_weight = 0.6
        config.coherence_weight = 0.6  # Sum > 1.0
        config.accuracy_weight = 0.2
        
        errors = config.validate()
        assert len(errors) > 0
        assert "Text quality subweights" in errors[0]


class TestTextSimilarityCalculator:
    """Test text similarity calculation methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = TextSimilarityCalculator()
    
    def test_bleu_score_identical_texts(self):
        """Test BLEU score for identical texts."""
        text = "This is a test sentence for evaluation."
        score = self.calculator.compute_bleu_score(text, text)
        assert score == 1.0, "Identical texts should have BLEU score of 1.0"
    
    def test_bleu_score_completely_different_texts(self):
        """Test BLEU score for completely different texts."""
        text1 = "The quick brown fox jumps"
        text2 = "Completely unrelated sentence here"
        score = self.calculator.compute_bleu_score(text1, text2)
        assert score < 0.3, "Completely different texts should have low BLEU score"
    
    def test_bleu_score_similar_texts(self):
        """Test BLEU score for similar texts."""
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on the mat"
        score = self.calculator.compute_bleu_score(text1, text2)
        assert 0.3 < score < 0.9, "Similar texts should have moderate BLEU score"
    
    def test_bleu_score_edge_cases(self):
        """Test BLEU score edge cases."""
        # Empty strings
        assert self.calculator.compute_bleu_score("", "") == 0.0
        assert self.calculator.compute_bleu_score("test", "") == 0.0
        assert self.calculator.compute_bleu_score("", "test") == 0.0
        
        # Single words
        assert self.calculator.compute_bleu_score("word", "word") == 1.0
        assert self.calculator.compute_bleu_score("word", "different") == 0.0
    
    def test_f1_score_identical_texts(self):
        """Test F1 score for identical texts."""
        text = "This is a test sentence"
        score = self.calculator.compute_f1_score(text, text)
        assert score == 1.0, "Identical texts should have F1 score of 1.0"
    
    def test_f1_score_partial_overlap(self):
        """Test F1 score for partial overlap."""
        text1 = "the quick brown fox"
        text2 = "the slow brown dog"
        score = self.calculator.compute_f1_score(text1, text2)
        # Should have moderate score due to "the brown" overlap
        assert 0.3 < score < 0.8
    
    def test_f1_score_edge_cases(self):
        """Test F1 score edge cases."""
        # Empty strings
        assert self.calculator.compute_f1_score("", "") == 1.0  # Both empty
        assert self.calculator.compute_f1_score("test", "") == 0.0
        assert self.calculator.compute_f1_score("", "test") == 0.0
    
    def test_semantic_similarity_basic(self):
        """Test basic semantic similarity."""
        text1 = "This is good and beneficial"
        text2 = "This is positive and helpful"
        score = self.calculator.compute_semantic_similarity(text1, text2)
        # Should detect positive semantic alignment
        assert score > 0.5
        
        text1 = "This causes problems"
        text2 = "This leads to issues"
        score = self.calculator.compute_semantic_similarity(text1, text2)
        # Should detect causal and negative semantic alignment
        assert score > 0.5
    
    def test_semantic_similarity_edge_cases(self):
        """Test semantic similarity edge cases."""
        assert self.calculator.compute_semantic_similarity("", "") == 0.0
        assert self.calculator.compute_semantic_similarity("test", "") == 0.0
        assert self.calculator.compute_semantic_similarity("", "test") == 0.0


class TestDebateQualityCalculator:
    """Test debate quality calculation methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = DebateQualityCalculator()
        
        # Create sample responses
        self.worker_response1 = AgentResponse(
            content="I believe this approach is effective because it addresses the core issues.",
            reasoning="Based on historical data and proven methods.",
            confidence=0.8
        )
        
        self.worker_response2 = AgentResponse(
            content="However, this approach has limitations and may not scale well.",
            reasoning="Considering potential bottlenecks and resource constraints.",
            confidence=0.7
        )
        
        self.synthesis_response = AgentResponse(
            content="Both perspectives have merit. While the first approach addresses core issues, "
                   "we must also consider the scaling limitations mentioned. A balanced solution "
                   "would be to implement the approach in phases, addressing scalability concerns.",
            reasoning="Synthesizing both viewpoints to create a more comprehensive solution.",
            confidence=0.9
        )
    
    def test_compute_debate_quality_with_conflict_analysis(self):
        """Test debate quality computation with ConflictAnalysis."""
        # Create debate trace
        debate_trace = {
            'worker_responses': [self.worker_response1, self.worker_response2],
            'synthesis_response': self.synthesis_response,
            'question': 'What is the best approach?'
        }
        
        # Create ConflictAnalysis
        conflict_analysis = ConflictAnalysis(
            conflicts_detected=True,
            conflict_areas=['effectiveness vs scalability'],
            agreement_areas=['need for solution'],
            conflict_severity=0.6,
            resolution_quality=0.8
        )
        
        # Create debate session with analysis
        debate_session = DebateSession('What is the best approach?')
        debate_session.conflict_analysis = conflict_analysis
        
        quality, details = self.calculator.compute_debate_quality(debate_trace, debate_session)
        
        assert 0.0 <= quality <= 1.0
        assert 'conflict_identification' in details
        assert 'perspective_integration' in details
        assert 'synthesis_effectiveness' in details
        
        # Should have good scores due to clear conflict and good synthesis
        assert quality > 0.5
    
    def test_compute_debate_quality_without_session(self):
        """Test debate quality computation without existing session."""
        debate_trace = {
            'worker_responses': [self.worker_response1, self.worker_response2],
            'synthesis_response': self.synthesis_response,
            'question': 'What is the best approach?'
        }
        
        quality, details = self.calculator.compute_debate_quality(debate_trace)
        
        assert 0.0 <= quality <= 1.0
        assert len(details) == 3  # Should have all three components
    
    def test_debate_quality_edge_cases(self):
        """Test debate quality edge cases."""
        # Empty worker responses
        debate_trace = {
            'worker_responses': [],
            'synthesis_response': self.synthesis_response,
            'question': 'test'
        }
        quality, details = self.calculator.compute_debate_quality(debate_trace)
        assert quality == 0.0
        
        # No synthesis response
        debate_trace = {
            'worker_responses': [self.worker_response1],
            'synthesis_response': None,
            'question': 'test'
        }
        quality, details = self.calculator.compute_debate_quality(debate_trace)
        assert quality == 0.0
    
    def test_conflict_identification_scoring(self):
        """Test conflict identification scoring."""
        # High conflict scenario
        high_conflict = ConflictAnalysis(
            conflicts_detected=True,
            conflict_areas=['approach', 'timing', 'resources'],
            agreement_areas=['goal', 'importance'],
            conflict_severity=0.8,
            resolution_quality=0.7
        )
        score = self.calculator._score_conflict_identification(high_conflict)
        assert score > 0.7
        
        # No conflict scenario
        no_conflict = ConflictAnalysis(
            conflicts_detected=False,
            conflict_areas=[],
            agreement_areas=['everything'],
            conflict_severity=0.0,
            resolution_quality=0.5
        )
        score = self.calculator._score_conflict_identification(no_conflict)
        assert score < 0.5
    
    def test_perspective_integration_scoring(self):
        """Test perspective integration scoring."""
        # Good integration synthesis
        good_synthesis = AgentResponse(
            content="Both approaches have merit. On one hand, the first response highlights "
                   "important benefits. On the other hand, the second response raises valid "
                   "concerns. A balanced solution would combine both perspectives.",
            reasoning="Integrating multiple viewpoints"
        )
        
        score = self.calculator._score_perspective_integration(
            [self.worker_response1, self.worker_response2], good_synthesis
        )
        assert score > 0.6
        
        # Poor integration synthesis
        poor_synthesis = AgentResponse(
            content="I think the answer is simple.",
            reasoning="Basic response"
        )
        
        score = self.calculator._score_perspective_integration(
            [self.worker_response1, self.worker_response2], poor_synthesis
        )
        assert score < 0.4


class TestRewardCalculator:
    """Test main reward calculator functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = RewardCalculator()
        
        # Sample responses for testing
        self.worker_response1 = AgentResponse(
            content="The new policy will increase efficiency and reduce costs.",
            reasoning="Based on pilot program results showing 20% improvement.",
            confidence=0.8,
            sources=["pilot_study.pdf"]
        )
        
        self.worker_response2 = AgentResponse(
            content="However, the policy may face resistance from stakeholders and implementation challenges.",
            reasoning="Previous similar policies encountered significant pushback.",
            confidence=0.7,
            sources=["historical_analysis.doc"]
        )
        
        self.synthesis_response = AgentResponse(
            content="While the new policy shows promise for efficiency gains, successful implementation "
                   "requires careful stakeholder management. A phased rollout with clear communication "
                   "and training can address resistance while capturing the efficiency benefits.",
            reasoning="Combining efficiency benefits with practical implementation considerations.",
            confidence=0.85,
            sources=["pilot_study.pdf", "historical_analysis.doc", "best_practices.pdf"]
        )
        
        self.baseline_response = AgentResponse(
            content="The new policy seems like a good idea.",
            reasoning="It should help the organization.",
            confidence=0.6
        )
        
        self.debate_trace = {
            'worker_responses': [self.worker_response1, self.worker_response2],
            'synthesis_response': self.synthesis_response,
            'question': 'Should we implement the new policy?'
        }
    
    def test_compute_text_similarity(self):
        """Test text similarity computation."""
        predicted = "The new policy will improve efficiency and reduce operational costs."
        gold = "The policy implementation will enhance efficiency and lower costs."
        
        score = self.calculator.compute_text_similarity(predicted, gold)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably similar
    
    def test_compute_text_similarity_edge_cases(self):
        """Test text similarity edge cases."""
        # Empty strings
        assert self.calculator.compute_text_similarity("", "") == 0.0
        assert self.calculator.compute_text_similarity("test", "") == 0.0
        
        # Identical strings
        text = "This is a test sentence."
        score = self.calculator.compute_text_similarity(text, text)
        assert score > 0.9
    
    def test_compute_debate_quality(self):
        """Test debate quality computation."""
        score = self.calculator.compute_debate_quality(self.debate_trace)
        
        assert 0.0 <= score <= 1.0
        # Should be decent quality due to good conflict resolution
        assert score > 0.4
    
    def test_compute_composite_reward_basic(self):
        """Test basic composite reward computation."""
        predicted_text = self.synthesis_response.content
        gold_text = "A balanced policy implementation considering both benefits and challenges."
        
        reward, components = self.calculator.compute_composite_reward(
            predicted_text, gold_text, self.debate_trace, self.baseline_response
        )
        
        # Check reward is in valid range
        assert self.calculator.config.min_reward <= reward <= self.calculator.config.max_reward
        
        # Check components are populated
        assert 0.0 <= components.text_similarity <= 1.0
        assert 0.0 <= components.synthesis_effectiveness <= 1.0
        assert components.improvement_over_baseline is not None
    
    def test_compute_composite_reward_without_baseline(self):
        """Test composite reward computation without baseline."""
        predicted_text = self.synthesis_response.content
        gold_text = "Policy implementation requires careful planning."
        
        reward, components = self.calculator.compute_composite_reward(
            predicted_text, gold_text, self.debate_trace
        )
        
        assert self.calculator.config.min_reward <= reward <= self.calculator.config.max_reward
        assert components.improvement_over_baseline == 0.0  # No baseline provided
    
    def test_compute_composite_reward_edge_cases(self):
        """Test composite reward edge cases."""
        # Empty debate trace
        empty_trace = {'worker_responses': [], 'synthesis_response': None}
        reward, components = self.calculator.compute_composite_reward(
            "test", "test", empty_trace
        )
        
        assert reward == self.calculator.config.min_reward
        
        # Missing synthesis response
        incomplete_trace = {'worker_responses': [self.worker_response1], 'synthesis_response': None}
        reward, components = self.calculator.compute_composite_reward(
            "test", "test", incomplete_trace
        )
        
        assert reward == self.calculator.config.min_reward
    
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        # Perform several reward computations
        for i in range(5):
            self.calculator.compute_text_similarity("test sentence", "another sentence")
            self.calculator.compute_debate_quality(self.debate_trace)
        
        stats = self.calculator.get_performance_stats()
        
        assert 'computation_stats' in stats
        assert 'reward_stats' in stats
        assert 'efficiency_metrics' in stats
        
        assert stats['computation_stats']['total_computations'] >= 10  # 5 * 2 operations
        assert stats['efficiency_metrics']['suitable_for_realtime'] is not None
    
    def test_performance_requirements(self):
        """Test that reward computation meets performance requirements."""
        # Single reward computation should be fast
        start_time = time.time()
        
        reward, components = self.calculator.compute_composite_reward(
            self.synthesis_response.content,
            "gold standard text for comparison",
            self.debate_trace,
            self.baseline_response
        )
        
        computation_time = time.time() - start_time
        
        # Should complete in under 100ms for real-time use
        assert computation_time < 0.1, f"Computation took {computation_time:.3f}s, should be under 0.1s"
    
    def test_reward_consistency(self):
        """Test that reward computation is consistent."""
        # Same inputs should produce same outputs
        predicted_text = self.synthesis_response.content
        gold_text = "Consistent test text for evaluation."
        
        reward1, components1 = self.calculator.compute_composite_reward(
            predicted_text, gold_text, self.debate_trace, self.baseline_response
        )
        
        reward2, components2 = self.calculator.compute_composite_reward(
            predicted_text, gold_text, self.debate_trace, self.baseline_response
        )
        
        assert abs(reward1 - reward2) < 1e-10, "Rewards should be identical for same inputs"
        assert abs(components1.text_similarity - components2.text_similarity) < 1e-10
    
    def test_reward_bounds(self):
        """Test that rewards stay within configured bounds."""
        # Test with various inputs to check bounds
        test_cases = [
            # (predicted, gold, expected_range)
            ("excellent comprehensive answer", "excellent comprehensive answer", "high"),
            ("", "excellent comprehensive answer", "low"),
            ("unrelated content", "specific technical answer", "low"),
            ("good balanced response", "good balanced response", "high")
        ]
        
        for predicted, gold, expected_range in test_cases:
            reward, _ = self.calculator.compute_composite_reward(
                predicted, gold, self.debate_trace
            )
            
            assert self.calculator.config.min_reward <= reward <= self.calculator.config.max_reward
            
            if expected_range == "high":
                assert reward > (self.calculator.config.max_reward * 0.5)
            elif expected_range == "low":
                assert reward < (self.calculator.config.max_reward * 0.3)
    
    def test_improvement_scoring(self):
        """Test improvement over baseline scoring."""
        # High quality synthesis should score better than poor baseline
        poor_baseline = AgentResponse(content="simple answer", confidence=0.3)
        
        improvement = self.calculator._compute_improvement(poor_baseline, self.synthesis_response)
        assert improvement > 0.2, "High quality synthesis should show improvement over poor baseline"
        
        # Equal quality should show minimal improvement
        improvement = self.calculator._compute_improvement(self.synthesis_response, self.synthesis_response)
        assert abs(improvement) < 0.1, "Equal quality responses should show minimal improvement"


class TestRewardConfigurationFactories:
    """Test reward calculator factory functions."""
    
    def test_standard_calculator_creation(self):
        """Test standard calculator factory."""
        calc = create_standard_reward_calculator()
        assert isinstance(calc, RewardCalculator)
        assert isinstance(calc.config, RewardConfig)
        
        # Should use default weights
        config = calc.config
        assert config.text_quality_weight == 0.25
        assert config.debate_quality_weight == 0.35
    
    def test_fast_calculator_creation(self):
        """Test fast calculator factory."""
        calc = create_fast_reward_calculator()
        assert isinstance(calc, RewardCalculator)
        
        # Should have different weights optimized for speed
        config = calc.config
        assert config.meta_rewards_weight == 0.1  # Reduced for speed
    
    def test_quality_focused_calculator_creation(self):
        """Test quality-focused calculator factory."""
        calc = create_quality_focused_calculator()
        assert isinstance(calc, RewardCalculator)
        
        # Should emphasize quality metrics
        config = calc.config
        assert config.text_quality_weight + config.debate_quality_weight == 0.7


class TestIntegrationWithExistingComponents:
    """Test integration with existing codebase components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.calculator = RewardCalculator()
        
        # Create a real DebateSession for integration testing
        self.debate_session = DebateSession("Should we adopt renewable energy?")
        
        self.worker1 = AgentResponse(
            content="Renewable energy is essential for environmental sustainability and long-term cost savings.",
            reasoning="Climate change requires immediate action, and renewables are increasingly cost-competitive.",
            confidence=0.8,
            sources=["climate_report.pdf"]
        )
        
        self.worker2 = AgentResponse(
            content="While renewables are important, the transition must consider grid stability and economic impacts.",
            reasoning="Rapid transitions can cause disruptions; a measured approach is needed.",
            confidence=0.75,
            sources=["energy_stability_study.pdf"]
        )
        
        self.synthesis = AgentResponse(
            content="A strategic transition to renewable energy is necessary, balancing environmental urgency "
                   "with grid stability concerns. Phased implementation with smart grid investments can "
                   "achieve sustainability goals while maintaining energy security.",
            reasoning="Integrating environmental imperatives with practical implementation challenges.",
            confidence=0.85,
            sources=["climate_report.pdf", "energy_stability_study.pdf", "transition_plan.pdf"]
        )
        
        # Add turns to debate session
        self.debate_session.add_turn("worker1", self.worker1)
        self.debate_session.add_turn("worker2", self.worker2)
        self.debate_session.add_turn("reviewer", self.synthesis)
        
        # Analyze debate
        self.conflict_analysis = self.debate_session.analyze_debate(
            [self.worker1, self.worker2], self.synthesis
        )
    
    def test_integration_with_debate_session(self):
        """Test integration with DebateSession and ConflictAnalysis."""
        debate_trace = {
            'worker_responses': [self.worker1, self.worker2],
            'synthesis_response': self.synthesis,
            'question': self.debate_session.question
        }
        
        # Test using the DebateSession's conflict analysis
        quality_score = self.calculator.compute_debate_quality(debate_trace)
        
        assert 0.0 <= quality_score <= 1.0
        
        # The quality should reflect the good conflict resolution
        assert quality_score > 0.5, "Should detect good dialectical quality"
    
    def test_integration_with_quality_assessment(self):
        """Test integration with existing quality assessment framework."""
        # The reward calculator should use the existing ResponseAnalyzer
        from eval.quality_assessment import ResponseAnalyzer
        
        # Verify our calculator uses the same analyzer
        assert isinstance(self.calculator.debate_quality.response_analyzer, ResponseAnalyzer)
        
        # Test that quality metrics are consistent
        analyzer = ResponseAnalyzer()
        direct_metrics = analyzer.analyze_response(self.synthesis)
        
        # Our reward calculation should be based on the same underlying analysis
        coherence_reward = self.calculator._compute_semantic_coherence(self.synthesis)
        
        expected_coherence = (direct_metrics.coherence_score + direct_metrics.clarity_score) / 2.0
        assert abs(coherence_reward - expected_coherence) < 0.01
    
    def test_performance_with_realistic_data(self):
        """Test performance with realistic data volumes."""
        debate_trace = {
            'worker_responses': [self.worker1, self.worker2],
            'synthesis_response': self.synthesis,
            'question': self.debate_session.question
        }
        
        # Test batch processing performance
        start_time = time.time()
        
        rewards = []
        for i in range(10):
            reward, components = self.calculator.compute_composite_reward(
                self.synthesis.content,
                f"Gold standard answer {i} for renewable energy policy.",
                debate_trace,
                AgentResponse(content=f"Basic answer {i}", confidence=0.5)
            )
            rewards.append(reward)
        
        batch_time = time.time() - start_time
        
        # Should process 10 rewards in reasonable time
        assert batch_time < 1.0, f"Batch processing took {batch_time:.3f}s, should be under 1.0s"
        
        # Rewards should show reasonable variation
        assert statistics.stdev(rewards) > 0, "Should show variation across different gold standards"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_invalid_config_raises_error(self):
        """Test that invalid configuration raises appropriate errors."""
        config = RewardConfig()
        config.text_quality_weight = 2.0  # Invalid weight
        
        with pytest.raises(ValueError, match="Invalid reward configuration"):
            RewardCalculator(config)
    
    def test_malformed_debate_trace_handling(self):
        """Test handling of malformed debate trace data."""
        calculator = RewardCalculator()
        
        # Missing required keys
        malformed_trace = {'question': 'test'}
        
        reward, components = calculator.compute_composite_reward(
            "test", "test", malformed_trace
        )
        
        assert reward == calculator.config.min_reward
        
        # None values in trace
        none_trace = {
            'worker_responses': None,
            'synthesis_response': None,
            'question': None
        }
        
        reward, components = calculator.compute_composite_reward(
            "test", "test", none_trace
        )
        
        assert reward == calculator.config.min_reward
    
    def test_empty_performance_stats(self):
        """Test performance stats before any computations."""
        calculator = RewardCalculator()
        stats = calculator.get_performance_stats()
        
        assert "error" in stats
        assert "No computations performed yet" in stats["error"]
    
    def test_reset_performance_tracking(self):
        """Test performance tracking reset."""
        calculator = RewardCalculator()
        
        # Perform some computations
        calculator.compute_text_similarity("test", "test")
        stats_before = calculator.get_performance_stats()
        
        # Reset tracking
        calculator.reset_performance_tracking()
        stats_after = calculator.get_performance_stats()
        
        assert "error" in stats_after
        assert "No computations performed yet" in stats_after["error"]
    
    def test_very_long_text_handling(self):
        """Test handling of very long texts."""
        calculator = RewardCalculator()
        
        # Create very long text (10k+ words)
        long_text = " ".join(["word"] * 10000)
        normal_text = "This is a normal length text for comparison."
        
        # Should handle gracefully without timeout
        start_time = time.time()
        score = calculator.compute_text_similarity(long_text, normal_text)
        computation_time = time.time() - start_time
        
        assert 0.0 <= score <= 1.0
        assert computation_time < 1.0, "Should handle long text efficiently"
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        calculator = RewardCalculator()
        
        # Text with unicode characters
        unicode_text = "This is a test with émojis 🚀 and spécial characters ñ"
        normal_text = "This is a test with emojis and special characters"
        
        score = calculator.compute_text_similarity(unicode_text, normal_text)
        assert 0.0 <= score <= 1.0
        
        # Text with only special characters
        special_text = "!@#$%^&*()_+"
        score = calculator.compute_text_similarity(special_text, normal_text)
        assert score < 0.3  # Should be low similarity


if __name__ == "__main__":
    # Run specific test groups for debugging
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "basic":
            pytest.main(["-v", "TestRewardCalculator::test_compute_composite_reward_basic"])
        elif test_class == "performance":
            pytest.main(["-v", "TestRewardCalculator::test_performance_requirements"])
        elif test_class == "integration":
            pytest.main(["-v", "TestIntegrationWithExistingComponents"])
        else:
            pytest.main(["-v", f"Test{test_class}"])
    else:
        # Run all tests
        pytest.main(["-v", __file__])
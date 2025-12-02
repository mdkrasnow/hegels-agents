#!/usr/bin/env python3
"""
Mock test script for Hegels Agents Phase 0.5 - Basic Agent Implementation

This script tests the basic functionality with mock responses to validate
the agent architecture without requiring real API calls.
"""

import sys
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import load_config
from agents.worker import BasicWorkerAgent
from agents.reviewer import BasicReviewerAgent
from agents.utils import DebateContext, log_debate_interaction


# Mock responses for testing
MOCK_WORKER_RESPONSES = {
    "renewable_energy": "Renewable energy sources offer significant advantages including reduced carbon emissions, decreased dependence on fossil fuels, job creation in green industries, and long-term cost savings. Solar and wind power have become increasingly cost-competitive with traditional energy sources.",
    
    "nuclear_energy": "Government investment in nuclear energy presents both benefits and challenges. Benefits include reliable baseload power, low carbon emissions, and energy security. However, concerns include high upfront costs, nuclear waste management, and safety considerations that require careful regulation.",
    
    "renewable_advantages": "Key advantages of renewable energy include environmental benefits through reduced greenhouse gas emissions, economic benefits from job creation and energy independence, technological innovation driving cost reductions, and improved public health outcomes from cleaner air.",
    
    "climate_strategies": "Effective climate change strategies should include rapid decarbonization of energy systems, implementation of carbon pricing mechanisms, investment in clean technology research, protection and restoration of natural carbon sinks, and international cooperation on emissions reduction targets.",
    
    "ai_healthcare_ethics": "AI in healthcare raises important ethical considerations including patient privacy and data protection, algorithmic bias in diagnosis and treatment recommendations, the need for human oversight in clinical decisions, equitable access to AI-enhanced healthcare, and maintaining the doctor-patient relationship while integrating AI tools."
}

MOCK_CRITIQUE_RESPONSES = {
    "single": "This response provides a solid overview of the key points but could benefit from more specific examples and data to support the claims. The reasoning is sound but would be strengthened by addressing potential counterarguments or limitations.",
    
    "comparison": "Both responses offer valuable perspectives. Response A provides more concrete examples while Response B focuses on systemic considerations. The approaches are complementary rather than contradictory, with Response A being more detail-oriented and Response B being more strategic in scope."
}

MOCK_SYNTHESIS_RESPONSE = "By integrating insights from multiple perspectives, we can develop a more comprehensive understanding. The combined analysis reveals both immediate practical considerations and longer-term strategic implications. Key areas of agreement provide a solid foundation, while areas of difference highlight important nuances that require further investigation."


def mock_gemini_response(prompt: str) -> str:
    """Generate appropriate mock response based on prompt content."""
    prompt_lower = prompt.lower()
    
    # Determine response type based on prompt content
    if "renewable energy" in prompt_lower or "clean energy" in prompt_lower:
        if "advantages" in prompt_lower or "benefits" in prompt_lower:
            return MOCK_WORKER_RESPONSES["renewable_advantages"]
        return MOCK_WORKER_RESPONSES["renewable_energy"]
    
    elif "nuclear energy" in prompt_lower:
        return MOCK_WORKER_RESPONSES["nuclear_energy"]
    
    elif "climate change" in prompt_lower:
        return MOCK_WORKER_RESPONSES["climate_strategies"]
    
    elif "artificial intelligence" in prompt_lower and "healthcare" in prompt_lower:
        return MOCK_WORKER_RESPONSES["ai_healthcare_ethics"]
    
    elif "critique" in prompt_lower or "analyze" in prompt_lower:
        if "compare" in prompt_lower or "both" in prompt_lower:
            return MOCK_CRITIQUE_RESPONSES["comparison"]
        return MOCK_CRITIQUE_RESPONSES["single"]
    
    elif "synthesize" in prompt_lower or "integrate" in prompt_lower:
        return MOCK_SYNTHESIS_RESPONSE
    
    else:
        return "This is a comprehensive response addressing the key aspects of your question with appropriate reasoning and evidence."


def test_configuration_loading():
    """Test that configuration can be loaded properly."""
    print("\n" + "="*60)
    print("TEST 0: Configuration Loading")
    print("="*60)
    
    # Set up mock environment
    os.environ['GEMINI_API_KEY'] = 'mock_api_key_for_testing'
    os.environ['SUPABASE_DB_URL'] = 'postgresql://mock:mock@localhost:5432/mock'
    
    # Load configuration
    config = load_config()
    
    print(f"Environment: {config.app.environment}")
    print(f"Debug mode: {config.app.debug}")
    print(f"API key configured: {bool(config.api.gemini_api_key)}")
    
    assert config.api.gemini_api_key == 'mock_api_key_for_testing'
    assert config.database.url == 'postgresql://mock:mock@localhost:5432/mock'
    
    print("✅ Configuration loading test PASSED")


def test_agent_initialization():
    """Test that agents can be initialized properly."""
    print("\n" + "="*60)
    print("TEST 1: Agent Initialization")
    print("="*60)
    
    with patch('google.genai.Client') as mock_client:
        # Initialize agents
        worker = BasicWorkerAgent("test_worker")
        reviewer = BasicReviewerAgent("test_reviewer")
        
        print(f"Worker agent ID: {worker.agent_id}")
        print(f"Reviewer agent ID: {reviewer.agent_id}")
        
        # Verify API configuration was called
        assert mock_client.called, "Gemini client should be instantiated"
        
        # Check agent statistics
        worker_stats = worker.get_stats()
        reviewer_stats = reviewer.get_stats()
        
        print(f"Worker stats: {json.dumps(worker_stats, indent=2)}")
        print(f"Reviewer stats: {json.dumps(reviewer_stats, indent=2)}")
        
        assert worker_stats['agent_id'] == 'test_worker'
        assert reviewer_stats['agent_id'] == 'test_reviewer'
    
    print("✅ Agent initialization test PASSED")


def test_mock_worker_response():
    """Test worker agent with mocked API responses."""
    print("\n" + "="*60)
    print("TEST 2: Mock Worker Response")
    print("="*60)
    
    with patch('google.genai.Client') as mock_client_class:
        # Set up mock client and model
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.text = mock_gemini_response("renewable energy benefits")
        mock_models.generate_content.return_value = mock_response
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client
        
        # Initialize worker
        worker = BasicWorkerAgent("mock_worker")
        
        # Test response
        question = "What are the benefits of renewable energy?"
        response = worker.respond(question)
        
        print(f"Question: {question}")
        print(f"Response: {response.content}")
        print(f"Confidence: {response.confidence}")
        
        # Validate response
        assert response.content, "Response should have content"
        assert response.confidence is not None, "Response should have confidence"
        assert response.metadata['agent_id'] == 'mock_worker'
        
        # Verify API was called
        mock_models.generate_content.assert_called_once()
    
    print("✅ Mock worker response test PASSED")


def test_mock_reviewer_synthesis():
    """Test reviewer agent synthesis with mocked API responses."""
    print("\n" + "="*60)
    print("TEST 3: Mock Reviewer Synthesis")
    print("="*60)
    
    with patch('google.genai.Client') as mock_client_class:
        # Set up mock client and model
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.text = mock_gemini_response("synthesize renewable energy responses")
        mock_models.generate_content.return_value = mock_response
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client
        
        # Initialize reviewer
        reviewer = BasicReviewerAgent("mock_reviewer")
        
        # Create mock worker responses
        from agents.utils import AgentResponse
        response1 = AgentResponse(
            content="Solar and wind power are cost-effective renewable sources.",
            reasoning="Based on recent cost analysis data",
            confidence=0.8,
            sources=["mock_source_1"],
            metadata={'agent_id': 'mock_worker_1'}
        )
        
        response2 = AgentResponse(
            content="Renewable energy creates jobs and reduces emissions.",
            reasoning="Based on economic and environmental studies",
            confidence=0.7,
            sources=["mock_source_2"],
            metadata={'agent_id': 'mock_worker_2'}
        )
        
        # Test synthesis
        question = "What are the benefits of renewable energy?"
        synthesis = reviewer.synthesize_responses(question, [response1, response2])
        
        print(f"Question: {question}")
        print(f"Synthesis: {synthesis.content}")
        print(f"Confidence: {synthesis.confidence}")
        print(f"Responses synthesized: {synthesis.metadata['num_responses_synthesized']}")
        
        # Validate synthesis
        assert synthesis.content, "Synthesis should have content"
        assert synthesis.metadata['num_responses_synthesized'] == 2
        assert 'gemini-1.5-flash' in synthesis.sources
        
        # Verify API was called
        mock_models.generate_content.assert_called_once()
    
    print("✅ Mock reviewer synthesis test PASSED")


def test_full_mock_debate():
    """Test complete dialectical interaction with mock responses."""
    print("\n" + "="*60)
    print("TEST 4: Full Mock Dialectical Debate")
    print("="*60)
    
    with patch('google.genai.Client') as mock_client_class:
        # Set up mock client with different responses for different calls
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client
        
        def mock_generate_content(model=None, contents=None, config=None):
            mock_response = MagicMock()
            mock_response.text = mock_gemini_response(contents)
            return mock_response
        
        mock_models.generate_content.side_effect = mock_generate_content
        
        # Initialize agents
        worker1 = BasicWorkerAgent("debate_worker_1")
        worker2 = BasicWorkerAgent("debate_worker_2")
        reviewer = BasicReviewerAgent("debate_reviewer")
        
        # Test question
        question = "What are the most effective strategies for addressing climate change?"
        
        # Create debate context
        context = DebateContext(
            question=question,
            worker_responses=[],
            round_number=1
        )
        
        print(f"Debate ID: {context.debate_id}")
        print(f"Question: {question}")
        print("-" * 60)
        
        # Worker responses
        response1 = worker1.respond(question)
        response2 = worker2.respond(question)
        context.worker_responses = [response1, response2]
        
        print(f"Worker 1 Response: {response1.content[:100]}...")
        print(f"Worker 2 Response: {response2.content[:100]}...")
        print("-" * 60)
        
        # Reviewer analysis
        review_result = reviewer.review_and_synthesize(question, context.worker_responses)
        context.reviewer_synthesis = review_result['synthesis']
        
        print(f"Final Synthesis: {context.reviewer_synthesis.content[:100]}...")
        print(f"Synthesis Confidence: {context.reviewer_synthesis.confidence}")
        
        # Validate complete interaction
        assert len(context.worker_responses) == 2
        assert context.reviewer_synthesis is not None
        assert len(review_result['critiques']) == 2
        
        # Verify multiple API calls were made
        assert mock_models.generate_content.call_count >= 4  # 2 workers + 2 critiques + 1 synthesis
    
    print("✅ Full mock dialectical debate test PASSED")


def test_knowledge_base_functionality():
    """Test agent knowledge base and retrieval functionality."""
    print("\n" + "="*60)
    print("TEST 5: Knowledge Base Functionality")
    print("="*60)
    
    with patch('google.genai.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response based on knowledge base context"
        mock_models.generate_content.return_value = mock_response
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client
        
        # Initialize worker with knowledge
        worker = BasicWorkerAgent("knowledge_worker")
        
        # Add knowledge base
        knowledge = [
            "Solar panels convert sunlight directly into electricity using photovoltaic cells.",
            "Wind turbines generate electricity by converting kinetic energy from wind.",
            "Battery storage systems help manage intermittency in renewable energy."
        ]
        worker.add_knowledge(knowledge)
        
        print(f"Knowledge base size: {len(worker.knowledge_base)}")
        
        # Test retrieval
        from agents.utils import simple_text_search
        results = simple_text_search("solar energy", knowledge)
        print(f"Search results for 'solar energy': {len(results)} items")
        
        # Test response with knowledge context
        question = "How do solar panels work?"
        response = worker.respond(question)
        
        print(f"Response with context: {response.content}")
        print(f"Has retrieved context: {response.metadata.get('has_retrieved_context')}")
        
        # Validate knowledge integration
        assert len(worker.knowledge_base) == 3
        assert len(results) >= 1
        assert response.metadata.get('has_retrieved_context') is True
    
    print("✅ Knowledge base functionality test PASSED")


def run_mock_tests():
    """Run all mock tests."""
    print("HEGELS AGENTS PHASE 0.5 - MOCK TESTING")
    print("=" * 60)
    print("Testing agent architecture with mock responses")
    
    try:
        # Run tests
        test_configuration_loading()
        test_agent_initialization()
        test_mock_worker_response()
        test_mock_reviewer_synthesis()
        test_full_mock_debate()
        test_knowledge_base_functionality()
        
        print("\n" + "="*60)
        print("🎉 ALL MOCK TESTS PASSED!")
        print("="*60)
        print("Agent architecture is working correctly.")
        print("Mock responses validate the dialectical framework.")
        print("Ready for testing with real API calls.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_agents_mock.py")
        print("Tests agent architecture with mock responses (no API required)")
        return
    
    success = run_mock_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
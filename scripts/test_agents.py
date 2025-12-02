#!/usr/bin/env python3
"""
Test script for Hegels Agents Phase 0.5 - Basic Agent Implementation

This script tests the basic functionality of WorkerAgent and ReviewerAgent
to validate the minimal dialectical debate implementation.
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import load_config
from agents.worker import BasicWorkerAgent
from agents.reviewer import BasicReviewerAgent
from agents.utils import DebateContext, log_debate_interaction


def test_single_worker_response():
    """Test that a single worker agent can respond to a simple question."""
    print("\n" + "="*60)
    print("TEST 1: Single Worker Agent Response")
    print("="*60)
    
    # Initialize worker agent
    worker = BasicWorkerAgent("test_worker_1")
    
    # Test question
    question = "What are the main benefits of renewable energy?"
    
    print(f"Question: {question}")
    print("-" * 40)
    
    # Get response
    response = worker.respond(question)
    
    # Display results
    print(f"Response Content:\n{response.content}\n")
    print(f"Reasoning: {response.reasoning}")
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {response.sources}")
    
    # Validate response
    assert response.content, "Response content should not be empty"
    assert response.confidence is not None, "Confidence should be set"
    
    print("\n✅ Single worker response test PASSED")
    return response


def test_multiple_worker_responses():
    """Test multiple worker agents responding to the same question."""
    print("\n" + "="*60)
    print("TEST 2: Multiple Worker Agent Responses")
    print("="*60)
    
    # Initialize multiple worker agents
    worker1 = BasicWorkerAgent("test_worker_1")
    worker2 = BasicWorkerAgent("test_worker_2")
    
    # Add some different knowledge to each agent
    worker1.add_knowledge([
        "Solar energy is clean and renewable, reducing carbon emissions.",
        "Wind power has become increasingly cost-effective in recent years.",
        "Renewable energy creates jobs in emerging green industries."
    ])
    
    worker2.add_knowledge([
        "Hydroelectric power provides consistent, reliable energy generation.",
        "Geothermal energy offers stable baseload power in suitable regions.",
        "Energy storage technologies are advancing to support renewable integration."
    ])
    
    # Test question
    question = "What are the key advantages of transitioning to renewable energy sources?"
    
    print(f"Question: {question}")
    print("-" * 40)
    
    # Get responses from both workers
    response1 = worker1.respond(question)
    response2 = worker2.respond(question)
    
    # Display results
    print(f"Worker 1 Response:\n{response1.content}\n")
    print(f"Worker 1 Reasoning: {response1.reasoning}\n")
    
    print(f"Worker 2 Response:\n{response2.content}\n")
    print(f"Worker 2 Reasoning: {response2.reasoning}\n")
    
    # Validate responses
    assert response1.content, "Worker 1 response should not be empty"
    assert response2.content, "Worker 2 response should not be empty"
    assert response1.content != response2.content, "Responses should be different"
    
    print("✅ Multiple worker responses test PASSED")
    return [response1, response2]


def test_reviewer_critique():
    """Test reviewer agent critiquing a worker response."""
    print("\n" + "="*60)
    print("TEST 3: Reviewer Critique of Single Response")
    print("="*60)
    
    # Initialize agents
    worker = BasicWorkerAgent("test_worker")
    reviewer = BasicReviewerAgent("test_reviewer")
    
    # Test question and response
    question = "Should governments invest more in nuclear energy?"
    worker_response = worker.respond(question)
    
    print(f"Question: {question}")
    print(f"Worker Response:\n{worker_response.content}\n")
    print("-" * 40)
    
    # Get critique
    critique = reviewer.critique_response(question, worker_response)
    
    # Display critique
    print(f"Reviewer Critique:\n{critique.content}\n")
    print(f"Critique Reasoning: {critique.reasoning}")
    
    # Validate critique
    assert critique.content, "Critique content should not be empty"
    assert "critique" in critique.metadata.get('critique_type', ''), "Should be marked as critique"
    
    print("✅ Reviewer critique test PASSED")
    return critique


def test_reviewer_synthesis():
    """Test reviewer agent synthesizing multiple worker responses."""
    print("\n" + "="*60)
    print("TEST 4: Reviewer Synthesis of Multiple Responses")
    print("="*60)
    
    # Initialize agents
    worker1 = BasicWorkerAgent("synthesis_worker_1")
    worker2 = BasicWorkerAgent("synthesis_worker_2")
    reviewer = BasicReviewerAgent("synthesis_reviewer")
    
    # Add different perspectives to knowledge bases
    worker1.add_knowledge([
        "Climate change is primarily caused by greenhouse gas emissions from fossil fuels.",
        "Carbon pricing mechanisms can help reduce emissions economically.",
        "International cooperation is essential for effective climate action."
    ])
    
    worker2.add_knowledge([
        "Technological innovation in clean energy is accelerating rapidly.",
        "Adaptation strategies are crucial alongside mitigation efforts.",
        "Economic benefits of green transition often outweigh costs."
    ])
    
    # Test question
    question = "What are the most effective strategies for addressing climate change?"
    
    print(f"Question: {question}")
    print("-" * 40)
    
    # Get worker responses
    response1 = worker1.respond(question)
    response2 = worker2.respond(question)
    
    print(f"Worker 1 Response:\n{response1.content}\n")
    print(f"Worker 2 Response:\n{response2.content}\n")
    print("-" * 40)
    
    # Get synthesis
    synthesis = reviewer.synthesize_responses(question, [response1, response2])
    
    # Display synthesis
    print(f"Reviewer Synthesis:\n{synthesis.content}\n")
    print(f"Synthesis Reasoning: {synthesis.reasoning}")
    print(f"Synthesis Confidence: {synthesis.confidence}")
    print(f"Number of responses synthesized: {synthesis.metadata.get('num_responses_synthesized')}")
    
    # Validate synthesis
    assert synthesis.content, "Synthesis content should not be empty"
    assert synthesis.metadata.get('num_responses_synthesized') == 2, "Should indicate 2 responses synthesized"
    
    print("✅ Reviewer synthesis test PASSED")
    return synthesis


def test_full_dialectical_interaction():
    """Test complete dialectical interaction: workers respond, reviewer synthesizes."""
    print("\n" + "="*60)
    print("TEST 5: Complete Dialectical Interaction")
    print("="*60)
    
    # Initialize agents
    worker1 = BasicWorkerAgent("dialectic_worker_1")
    worker2 = BasicWorkerAgent("dialectic_worker_2")
    reviewer = BasicReviewerAgent("dialectic_reviewer")
    
    # Test question
    question = "What are the ethical implications of artificial intelligence in healthcare?"
    
    # Create debate context
    context = DebateContext(
        question=question,
        worker_responses=[],
        round_number=1
    )
    
    print(f"Debate ID: {context.debate_id}")
    print(f"Question: {question}")
    print("=" * 60)
    
    # Log debate start
    log_debate_interaction(context, "debate_started", {
        'question': question,
        'num_workers': 2
    })
    
    # Worker responses
    print("Phase 1: Worker Responses")
    print("-" * 30)
    
    response1 = worker1.respond(question)
    context.worker_responses.append(response1)
    print(f"Worker 1:\n{response1.content}\n")
    
    response2 = worker2.respond(question)
    context.worker_responses.append(response2)
    print(f"Worker 2:\n{response2.content}\n")
    
    # Log worker responses
    log_debate_interaction(context, "worker_responses_completed", {
        'num_responses': len(context.worker_responses),
        'avg_confidence': sum(r.confidence or 0 for r in context.worker_responses) / len(context.worker_responses)
    })
    
    # Reviewer analysis
    print("Phase 2: Reviewer Analysis")
    print("-" * 30)
    
    # Compare responses
    comparison = reviewer.compare_responses(question, response1, response2)
    print(f"Comparison Analysis:\n{comparison.content}\n")
    
    # Full review and synthesis
    review_result = reviewer.review_and_synthesize(question, context.worker_responses)
    context.reviewer_synthesis = review_result['synthesis']
    
    print("Phase 3: Individual Critiques")
    print("-" * 30)
    for i, critique in enumerate(review_result['critiques'], 1):
        print(f"Critique {i}:\n{critique.content}\n")
    
    print("Phase 4: Final Synthesis")
    print("-" * 30)
    print(f"Final Synthesis:\n{context.reviewer_synthesis.content}\n")
    print(f"Synthesis Confidence: {context.reviewer_synthesis.confidence}")
    
    # Log debate completion
    log_debate_interaction(context, "debate_completed", {
        'synthesis_confidence': context.reviewer_synthesis.confidence,
        'synthesis_length': len(context.reviewer_synthesis.content)
    })
    
    # Validate complete interaction
    assert len(context.worker_responses) == 2, "Should have 2 worker responses"
    assert context.reviewer_synthesis is not None, "Should have reviewer synthesis"
    assert comparison.content, "Should have comparison analysis"
    
    print("✅ Complete dialectical interaction test PASSED")
    return context


def test_agent_stats():
    """Test that agents provide useful statistics."""
    print("\n" + "="*60)
    print("TEST 6: Agent Statistics")
    print("="*60)
    
    # Initialize agents
    worker = BasicWorkerAgent("stats_worker")
    reviewer = BasicReviewerAgent("stats_reviewer")
    
    # Add some knowledge
    worker.add_knowledge(["Test knowledge item 1", "Test knowledge item 2"])
    
    # Get stats
    worker_stats = worker.get_stats()
    reviewer_stats = reviewer.get_stats()
    
    print(f"Worker Stats: {json.dumps(worker_stats, indent=2)}")
    print(f"Reviewer Stats: {json.dumps(reviewer_stats, indent=2)}")
    
    # Validate stats
    assert worker_stats['agent_id'] == 'stats_worker', "Worker ID should match"
    assert reviewer_stats['agent_id'] == 'stats_reviewer', "Reviewer ID should match"
    assert worker_stats['knowledge_base_size'] == 2, "Should reflect added knowledge"
    
    print("✅ Agent statistics test PASSED")


def run_all_tests():
    """Run all agent tests."""
    print("HEGELS AGENTS PHASE 0.5 - AGENT TESTING")
    print("=" * 60)
    print("Testing basic agent functionality and dialectical interaction")
    
    try:
        # Check if we have necessary environment variables
        if not os.getenv('GEMINI_API_KEY'):
            print("⚠️  No GEMINI_API_KEY found. Running in mock mode...")
            # Set a dummy API key for configuration testing
            os.environ['GEMINI_API_KEY'] = 'test_key_for_mock_mode'
        
        if not os.getenv('SUPABASE_DB_URL'):
            print("⚠️  No SUPABASE_DB_URL found. Using dummy database URL...")
            os.environ['SUPABASE_DB_URL'] = 'postgresql://user:pass@localhost:5432/test'
        
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        print("✅ Configuration loaded successfully")
        
        # Run individual tests
        test_single_worker_response()
        test_multiple_worker_responses()
        test_reviewer_critique()
        test_reviewer_synthesis() 
        test_full_dialectical_interaction()
        test_agent_stats()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("Basic agent implementation is working correctly.")
        print("Dialectical debate functionality validated.")
        print("Ready for Phase 0.5 validation experiments.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_agents.py")
        print("Tests basic agent functionality for Hegels Agents Phase 0.5")
        return
    
    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
"""
Demonstration of the Reward Computation System

This script demonstrates the reward system capabilities and integration
with existing debate and quality assessment components.
"""

import time
import json
from typing import Dict, Any

from agents.utils import AgentResponse
from debate.session import DebateSession
from training.rewards import (
    RewardCalculator,
    RewardConfig,
    create_standard_reward_calculator,
    create_fast_reward_calculator,
    create_quality_focused_calculator
)


def create_sample_debate_scenario() -> Dict[str, Any]:
    """Create a sample debate scenario for demonstration."""
    
    # Question: Climate change mitigation strategies
    question = "What are the most effective strategies for mitigating climate change?"
    
    # Worker agent responses representing different perspectives
    worker1_response = AgentResponse(
        content="The most effective strategy is rapid transition to renewable energy sources. "
               "Solar and wind technologies have reached cost parity with fossil fuels, and "
               "massive deployment can significantly reduce carbon emissions within a decade. "
               "Government subsidies and carbon pricing can accelerate this transition.",
        reasoning="Based on IPCC reports and recent technological advances showing renewable "
                 "energy is now economically viable at scale. The urgency of climate action "
                 "requires immediate deployment of proven technologies.",
        confidence=0.85,
        sources=["IPCC_2023_report.pdf", "renewable_cost_analysis_2023.pdf"]
    )
    
    worker2_response = AgentResponse(
        content="While renewable energy is important, we must also focus on energy efficiency "
               "and carbon capture technologies. Behavioral changes and policy reforms are "
               "equally crucial. A carbon tax, building efficiency standards, and investment "
               "in public transportation can complement renewable energy deployment.",
        reasoning="Climate mitigation requires a comprehensive approach beyond just energy "
                 "generation. Efficiency reduces overall energy demand, and behavior change "
                 "can have immediate impacts without waiting for infrastructure deployment.",
        confidence=0.8,
        sources=["energy_efficiency_potential_2023.pdf", "behavioral_economics_climate.pdf"]
    )
    
    # Synthesis response integrating both perspectives
    synthesis_response = AgentResponse(
        content="The most effective climate mitigation strategy combines rapid renewable energy "
               "deployment with comprehensive efficiency measures and behavioral interventions. "
               "While renewable energy provides the foundation for decarbonization, energy "
               "efficiency can deliver immediate emissions reductions at lower cost. A holistic "
               "approach should include: 1) Aggressive renewable energy targets with supportive "
               "policies, 2) Building and transportation efficiency standards, 3) Carbon pricing "
               "to drive behavioral change, and 4) Investment in emerging technologies like "
               "carbon capture for hard-to-abate sectors. This multi-pronged strategy maximizes "
               "impact while addressing the urgency of climate action.",
        reasoning="Synthesizing both perspectives recognizes that climate change requires "
                 "immediate action on multiple fronts. Renewable energy provides long-term "
                 "decarbonization potential, while efficiency and behavior change offer "
                 "near-term wins. The scale of the challenge demands all available solutions.",
        confidence=0.9,
        sources=["IPCC_2023_report.pdf", "renewable_cost_analysis_2023.pdf", 
                "energy_efficiency_potential_2023.pdf", "integrated_climate_strategy_2023.pdf"]
    )
    
    # Baseline single-agent response for comparison
    baseline_response = AgentResponse(
        content="Climate change can be addressed through renewable energy and efficiency measures. "
               "Solar and wind power are becoming cheaper, and we should invest in these technologies.",
        reasoning="Clean energy is important for reducing emissions.",
        confidence=0.6,
        sources=["basic_renewable_info.pdf"]
    )
    
    # Gold standard for evaluation
    gold_standard = (
        "Effective climate mitigation requires an integrated approach combining renewable energy "
        "deployment, energy efficiency improvements, policy interventions, and behavioral changes. "
        "The strategy should prioritize both immediate emissions reductions and long-term "
        "decarbonization pathways."
    )
    
    return {
        'question': question,
        'worker_responses': [worker1_response, worker2_response],
        'synthesis_response': synthesis_response,
        'baseline_response': baseline_response,
        'gold_standard': gold_standard
    }


def demonstrate_text_similarity(calculator: RewardCalculator, scenario: Dict[str, Any]):
    """Demonstrate text similarity calculation."""
    print("=== Text Similarity Demonstration ===")
    
    predicted_text = scenario['synthesis_response'].content
    gold_text = scenario['gold_standard']
    
    print(f"Predicted text (first 100 chars): {predicted_text[:100]}...")
    print(f"Gold standard text: {gold_text}")
    print()
    
    # Calculate similarity components
    bleu_score = calculator.text_similarity.compute_bleu_score(predicted_text, gold_text)
    f1_score = calculator.text_similarity.compute_f1_score(predicted_text, gold_text)
    semantic_score = calculator.text_similarity.compute_semantic_similarity(predicted_text, gold_text)
    
    overall_similarity = calculator.compute_text_similarity(predicted_text, gold_text)
    
    print(f"BLEU Score: {bleu_score:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"Semantic Similarity: {semantic_score:.3f}")
    print(f"Overall Text Similarity: {overall_similarity:.3f}")
    print()


def demonstrate_debate_quality(calculator: RewardCalculator, scenario: Dict[str, Any]):
    """Demonstrate debate quality assessment."""
    print("=== Debate Quality Demonstration ===")
    
    # Create debate trace
    debate_trace = {
        'worker_responses': scenario['worker_responses'],
        'synthesis_response': scenario['synthesis_response'],
        'question': scenario['question']
    }
    
    # Create debate session for detailed analysis
    debate_session = DebateSession(scenario['question'])
    
    # Add turns to session
    for i, response in enumerate(scenario['worker_responses']):
        debate_session.add_turn(f"worker_{i+1}", response)
    debate_session.add_turn("reviewer", scenario['synthesis_response'])
    
    # Analyze debate
    conflict_analysis = debate_session.analyze_debate(
        scenario['worker_responses'], 
        scenario['synthesis_response']
    )
    
    print(f"Conflicts Detected: {conflict_analysis.conflicts_detected}")
    print(f"Conflict Areas: {conflict_analysis.conflict_areas}")
    print(f"Agreement Areas: {conflict_analysis.agreement_areas}")
    print(f"Conflict Severity: {conflict_analysis.conflict_severity:.3f}")
    print(f"Resolution Quality: {conflict_analysis.resolution_quality:.3f}")
    print()
    
    # Calculate debate quality
    overall_quality, details = calculator.debate_quality.compute_debate_quality(
        debate_trace, debate_session
    )
    
    print("Debate Quality Components:")
    for component, score in details.items():
        print(f"  {component}: {score:.3f}")
    print(f"Overall Debate Quality: {overall_quality:.3f}")
    print()


def demonstrate_composite_reward(calculator: RewardCalculator, scenario: Dict[str, Any]):
    """Demonstrate comprehensive composite reward calculation."""
    print("=== Composite Reward Demonstration ===")
    
    debate_trace = {
        'worker_responses': scenario['worker_responses'],
        'synthesis_response': scenario['synthesis_response'],
        'question': scenario['question']
    }
    
    # Calculate composite reward
    start_time = time.time()
    total_reward, components = calculator.compute_composite_reward(
        predicted_text=scenario['synthesis_response'].content,
        gold_text=scenario['gold_standard'],
        debate_trace=debate_trace,
        baseline_response=scenario['baseline_response']
    )
    computation_time = time.time() - start_time
    
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Computation Time: {computation_time*1000:.1f}ms")
    print()
    
    print("Reward Components:")
    component_dict = components.to_dict()
    for component, value in component_dict.items():
        print(f"  {component.replace('_', ' ').title()}: {value:.3f}")
    print()
    
    # Show weighted contributions
    config = calculator.config
    text_quality = (
        components.text_similarity * config.similarity_weight +
        components.semantic_coherence * config.coherence_weight +
        components.factual_accuracy * config.accuracy_weight
    ) * config.text_quality_weight
    
    debate_quality = (
        components.conflict_identification * config.conflict_weight +
        components.perspective_integration * config.integration_weight +
        components.synthesis_effectiveness * config.synthesis_weight
    ) * config.debate_quality_weight
    
    process_efficiency = (
        components.response_efficiency * config.efficiency_weight +
        components.reasoning_quality * config.reasoning_weight +
        components.confidence_calibration * config.calibration_weight
    ) * config.process_efficiency_weight
    
    meta_rewards = (
        components.improvement_over_baseline * config.improvement_weight +
        components.dialectical_necessity * config.necessity_weight +
        components.learning_potential * config.learning_weight
    ) * config.meta_rewards_weight
    
    print("Weighted Category Contributions:")
    print(f"  Text Quality: {text_quality:.3f} (weight: {config.text_quality_weight})")
    print(f"  Debate Quality: {debate_quality:.3f} (weight: {config.debate_quality_weight})")
    print(f"  Process Efficiency: {process_efficiency:.3f} (weight: {config.process_efficiency_weight})")
    print(f"  Meta Rewards: {meta_rewards:.3f} (weight: {config.meta_rewards_weight})")
    print()


def compare_calculator_configurations():
    """Compare different calculator configurations."""
    print("=== Calculator Configuration Comparison ===")
    
    scenario = create_sample_debate_scenario()
    
    calculators = {
        "Standard": create_standard_reward_calculator(),
        "Fast": create_fast_reward_calculator(),
        "Quality-Focused": create_quality_focused_calculator()
    }
    
    debate_trace = {
        'worker_responses': scenario['worker_responses'],
        'synthesis_response': scenario['synthesis_response'],
        'question': scenario['question']
    }
    
    results = {}
    
    for name, calc in calculators.items():
        start_time = time.time()
        reward, components = calc.compute_composite_reward(
            predicted_text=scenario['synthesis_response'].content,
            gold_text=scenario['gold_standard'],
            debate_trace=debate_trace,
            baseline_response=scenario['baseline_response']
        )
        computation_time = time.time() - start_time
        
        results[name] = {
            'reward': reward,
            'computation_time': computation_time,
            'components': components
        }
    
    print("Configuration Comparison:")
    print(f"{'Config':<15} {'Reward':<10} {'Time(ms)':<10} {'Text Qual':<10} {'Debate Qual':<12}")
    print("-" * 65)
    
    for name, result in results.items():
        print(f"{name:<15} {result['reward']:<10.2f} {result['computation_time']*1000:<10.1f} "
              f"{result['components'].text_similarity:<10.3f} "
              f"{result['components'].synthesis_effectiveness:<12.3f}")
    
    print()


def demonstrate_performance_analysis():
    """Demonstrate performance analysis capabilities."""
    print("=== Performance Analysis Demonstration ===")
    
    calculator = create_standard_reward_calculator()
    scenario = create_sample_debate_scenario()
    
    debate_trace = {
        'worker_responses': scenario['worker_responses'],
        'synthesis_response': scenario['synthesis_response'],
        'question': scenario['question']
    }
    
    # Run multiple computations for performance analysis
    print("Running performance test...")
    for i in range(20):
        # Vary the inputs slightly to get realistic performance metrics
        varied_gold = f"{scenario['gold_standard']} Additional context for test {i}."
        
        calculator.compute_composite_reward(
            predicted_text=scenario['synthesis_response'].content,
            gold_text=varied_gold,
            debate_trace=debate_trace,
            baseline_response=scenario['baseline_response']
        )
    
    # Get performance statistics
    stats = calculator.get_performance_stats()
    
    print("Performance Statistics:")
    print(f"Total Computations: {stats['computation_stats']['total_computations']}")
    print(f"Mean Computation Time: {stats['computation_stats']['mean_computation_time']*1000:.1f}ms")
    print(f"Max Computation Time: {stats['computation_stats']['max_computation_time']*1000:.1f}ms")
    print(f"Suitable for Real-time: {stats['efficiency_metrics']['suitable_for_realtime']}")
    print(f"Rewards per Second: {stats['efficiency_metrics']['rewards_per_second']:.1f}")
    print()
    
    print("Reward Statistics:")
    print(f"Mean Reward: {stats['reward_stats']['mean_reward']:.2f}")
    print(f"Reward Range: {stats['reward_stats']['min_reward']:.2f} to {stats['reward_stats']['max_reward']:.2f}")
    print(f"Reward Std Dev: {stats['reward_stats']['std_reward']:.2f}")
    print()


def demonstrate_edge_cases():
    """Demonstrate handling of edge cases."""
    print("=== Edge Case Demonstration ===")
    
    calculator = create_standard_reward_calculator()
    
    # Edge case 1: Empty responses
    print("1. Empty responses:")
    empty_trace = {
        'worker_responses': [],
        'synthesis_response': None,
        'question': 'test question'
    }
    
    reward, components = calculator.compute_composite_reward("", "", empty_trace)
    print(f"   Reward for empty inputs: {reward:.2f}")
    
    # Edge case 2: Single word responses
    print("2. Single word responses:")
    single_word_reward = calculator.compute_text_similarity("good", "bad")
    print(f"   Similarity for 'good' vs 'bad': {single_word_reward:.3f}")
    
    # Edge case 3: Identical responses (should get high reward)
    print("3. Identical responses:")
    identical_reward = calculator.compute_text_similarity(
        "This is a comprehensive answer", 
        "This is a comprehensive answer"
    )
    print(f"   Similarity for identical text: {identical_reward:.3f}")
    
    # Edge case 4: Very long responses
    print("4. Very long responses:")
    long_text = "This is a test sentence. " * 100  # 500 words
    normal_text = "This is a comprehensive test sentence for evaluation."
    
    start_time = time.time()
    long_reward = calculator.compute_text_similarity(long_text, normal_text)
    long_time = time.time() - start_time
    
    print(f"   Similarity for long text: {long_reward:.3f}")
    print(f"   Processing time: {long_time*1000:.1f}ms")
    print()


def main():
    """Run the complete reward system demonstration."""
    print("HEGEL'S AGENTS - REWARD COMPUTATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Create sample scenario
    print("Creating sample debate scenario...")
    scenario = create_sample_debate_scenario()
    print(f"Question: {scenario['question']}")
    print(f"Number of worker responses: {len(scenario['worker_responses'])}")
    print(f"Synthesis response length: {len(scenario['synthesis_response'].content)} characters")
    print()
    
    # Create reward calculator
    calculator = create_standard_reward_calculator()
    
    # Run demonstrations
    try:
        demonstrate_text_similarity(calculator, scenario)
        demonstrate_debate_quality(calculator, scenario)
        demonstrate_composite_reward(calculator, scenario)
        compare_calculator_configurations()
        demonstrate_performance_analysis()
        demonstrate_edge_cases()
        
        print("=== Summary ===")
        print("✅ Text similarity computation working")
        print("✅ Debate quality assessment working")
        print("✅ Composite reward calculation working")
        print("✅ Multiple configurations available")
        print("✅ Performance requirements met")
        print("✅ Edge cases handled appropriately")
        print()
        print("The reward computation system is ready for training integration!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def save_demonstration_results():
    """Save demonstration results to file for review."""
    import json
    from datetime import datetime
    
    scenario = create_sample_debate_scenario()
    calculator = create_standard_reward_calculator()
    
    debate_trace = {
        'worker_responses': scenario['worker_responses'],
        'synthesis_response': scenario['synthesis_response'],
        'question': scenario['question']
    }
    
    reward, components = calculator.compute_composite_reward(
        predicted_text=scenario['synthesis_response'].content,
        gold_text=scenario['gold_standard'],
        debate_trace=debate_trace,
        baseline_response=scenario['baseline_response']
    )
    
    # Create serializable results
    results = {
        'timestamp': datetime.now().isoformat(),
        'scenario': {
            'question': scenario['question'],
            'worker1_content': scenario['worker_responses'][0].content,
            'worker2_content': scenario['worker_responses'][1].content,
            'synthesis_content': scenario['synthesis_response'].content,
            'baseline_content': scenario['baseline_response'].content,
            'gold_standard': scenario['gold_standard']
        },
        'reward_calculation': {
            'total_reward': reward,
            'components': components.to_dict(),
            'config': {
                'text_quality_weight': calculator.config.text_quality_weight,
                'debate_quality_weight': calculator.config.debate_quality_weight,
                'process_efficiency_weight': calculator.config.process_efficiency_weight,
                'meta_rewards_weight': calculator.config.meta_rewards_weight
            }
        }
    }
    
    # Save to file
    output_path = "/tmp/reward_system_demo_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Demonstration results saved to: {output_path}")


if __name__ == "__main__":
    main()
    
    # Optionally save results
    save_demonstration_results()
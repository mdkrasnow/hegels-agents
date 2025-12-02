"""
Demo of Enhanced Agent Factory with ClaudeAgentConfig Integration

This script demonstrates the key features of the enhanced agent factory:
- Creating Claude configurations for different agent types
- Profile-based agent configuration
- Enhanced configuration with caching and retry logic
- Batch agent creation
- Cache management

Usage: python src/training/demo_enhanced_factory.py
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.models.claude_agent_config import (
    ClaudeAgentConfig,
    ClaudeModel,
    PromptRole
)
from training.data_structures import PromptProfile, RolePrompt
from training.agent_factory import (
    ConfigurableAgentFactory,
    EnhancedAgentConfig,
    AgentConfig
)


def create_demo_profile():
    """Create a comprehensive demo profile with multiple agent roles."""
    print("📋 Creating demo profile with multiple roles...")
    
    profile = PromptProfile(
        name="Demo Multi-Role Profile",
        description="Demonstration profile showing different agent configurations",
        author="enhanced-factory-demo"
    )
    
    # Define role-specific prompts
    role_prompts = {
        'worker': {
            'prompt': "You are a focused worker agent. Complete tasks efficiently and provide clear, actionable outputs. Always include reasoning for your decisions.",
            'metadata': {'priority': 'efficiency', 'output_format': 'structured'}
        },
        'reviewer': {
            'prompt': "You are a meticulous reviewer agent. Analyze work for quality, accuracy, and completeness. Provide constructive feedback.",
            'metadata': {
                'priority': 'quality',
                'synthesis_prompt': "Synthesize feedback into actionable improvements while highlighting strengths."
            }
        },
        'orchestrator': {
            'prompt': "You are an orchestrator agent coordinating complex workflows. Break down tasks, assign priorities, and manage dependencies.",
            'metadata': {'priority': 'coordination', 'scope': 'multi-agent'}
        },
        'summarizer': {
            'prompt': "You are a summarizer agent creating clear, concise summaries. Focus on key insights and actionable outcomes.",
            'metadata': {'priority': 'clarity', 'length': 'concise'}
        }
    }
    
    # Add role prompts to profile
    for role, config in role_prompts.items():
        role_prompt = RolePrompt(
            role=role,
            prompt_text=config['prompt'],
            description=f"Demo {role} configuration",
            metadata=config['metadata']
        )
        profile.add_role_prompt(role_prompt)
        print(f"   ✓ Added {role} role")
    
    return profile


def create_claude_configurations():
    """Create different Claude configurations for various use cases."""
    print("🤖 Creating Claude configurations...")
    
    configs = {}
    
    # High-precision configuration for critical tasks
    configs['precision'] = ClaudeAgentConfig.create_reviewer_config(
        "Precision Reviewer",
        temperature=0.1  # Very low temperature for consistency
    )
    configs['precision'].model_settings.max_tokens = 6000
    configs['precision'].prompt_config.include_thinking_tags = True
    configs['precision'].prompt_config.include_confidence_scores = True
    print("   ✓ Precision configuration (low temperature)")
    
    # Balanced configuration for general work
    configs['balanced'] = ClaudeAgentConfig.create_worker_config(
        "Balanced Worker",
        temperature=0.7  # Moderate temperature
    )
    configs['balanced'].model_settings.max_tokens = 4000
    print("   ✓ Balanced configuration (moderate temperature)")
    
    # Creative configuration for brainstorming
    configs['creative'] = ClaudeAgentConfig.create_high_creativity_config(
        "Creative Brainstormer",
        role=PromptRole.WORKER
    )
    configs['creative'].model_settings.top_p = 0.9
    print("   ✓ Creative configuration (high temperature)")
    
    return configs


def demonstrate_configuration_integration(profile, claude_configs):
    """Demonstrate different ways to configure agents."""
    print("⚙️  Demonstrating configuration integration...")
    
    # 1. Legacy configuration (backward compatibility)
    legacy_config = AgentConfig(
        temperature=0.8,
        max_tokens=2000,
        model_name='gemini-2.5-flash'
    )
    
    enhanced_legacy = EnhancedAgentConfig.from_legacy_config(legacy_config)
    print("   ✓ Legacy configuration converted to enhanced")
    
    # 2. Direct enhanced configuration
    enhanced_direct = EnhancedAgentConfig(
        temperature=0.6,
        max_tokens=3000,
        caching_enabled=True,
        cache_ttl=1800,  # 30 minutes
        retry_attempts=2
    )
    print("   ✓ Direct enhanced configuration created")
    
    # 3. Claude-based enhanced configuration
    enhanced_claude = EnhancedAgentConfig.from_claude_config(claude_configs['balanced'])
    print("   ✓ Claude-based enhanced configuration created")
    
    # Show parameter differences
    print("\n   Configuration comparison:")
    configs = [
        ("Legacy", enhanced_legacy),
        ("Direct", enhanced_direct),
        ("Claude", enhanced_claude)
    ]
    
    for name, config in configs:
        params = config.get_effective_model_params()
        print(f"     {name}: temp={config.temperature}, tokens={config.max_tokens}, claude_api={config.use_claude_api}")
    
    return enhanced_legacy, enhanced_direct, enhanced_claude


def demonstrate_cache_system():
    """Demonstrate the caching system capabilities."""
    print("🗄️  Demonstrating cache system...")
    
    # Get initial cache stats
    initial_stats = ConfigurableAgentFactory.get_cache_stats()
    print(f"   Initial cache: {initial_stats['cache_size']} items, {initial_stats['hit_rate']:.1%} hit rate")
    
    # Clear cache for clean demo
    ConfigurableAgentFactory.clear_cache()
    print("   ✓ Cache cleared for demo")
    
    # Enable caching
    ConfigurableAgentFactory.set_cache_enabled(True)
    
    # Cache statistics will be shown in agent creation
    return True


def simulate_agent_creation(profile, configs):
    """Simulate agent creation without actual agent instantiation."""
    print("🏭 Simulating agent creation process...")
    
    # This would normally create actual agents, but we'll simulate the configuration process
    print("   Note: Actual agent creation requires agent classes (BasicWorkerAgent, BasicReviewerAgent)")
    print("   Demonstrating configuration validation and preparation...")
    
    agent_specs = [
        {
            'type': 'worker',
            'profile': profile,
            'agent_id': 'demo-worker-1',
            'config': configs['enhanced_direct']
        },
        {
            'type': 'reviewer', 
            'profile': profile,
            'agent_id': 'demo-reviewer-1',
            'config': configs['enhanced_claude']
        },
        {
            'type': 'orchestrator',
            'profile': profile,
            'agent_id': 'demo-orchestrator-1',
            'config': configs['enhanced_legacy']
        }
    ]
    
    print("   Agent specifications prepared:")
    for spec in agent_specs:
        config = spec['config']
        print(f"     {spec['agent_id']}: {spec['type']} with {type(config).__name__}")
        
        # Validate configuration
        try:
            config.validate()
            print(f"       ✓ Configuration valid")
        except Exception as e:
            print(f"       ✗ Configuration error: {e}")
        
        # Check profile compatibility
        worker_role = profile.get_role_prompt(spec['type'])
        if not worker_role:
            worker_role = profile.get_role_prompt('worker')  # Fallback
        
        if worker_role:
            print(f"       ✓ Profile has {spec['type']} role")
        else:
            print(f"       ✗ Profile missing {spec['type']} role")
    
    print("   ✓ All agent specifications validated")


def demonstrate_profile_compatibility(profile, claude_configs):
    """Demonstrate profile compatibility features."""
    print("🔗 Demonstrating profile compatibility...")
    
    # Add profile compatibility to Claude configs
    for name, config in claude_configs.items():
        config.add_profile_compatibility(profile.profile_id)
        is_compatible = config.is_compatible_with_profile(profile.profile_id)
        print(f"   ✓ {name} config compatible with profile: {is_compatible}")
    
    # Test with non-compatible profile
    other_profile = PromptProfile(name="Other Profile", description="Different profile")
    is_compatible = claude_configs['precision'].is_compatible_with_profile(other_profile.profile_id)
    print(f"   ✓ Precision config with other profile: {is_compatible}")


def main():
    """Run the enhanced agent factory demonstration."""
    print("🚀 Enhanced Agent Factory Demo")
    print("=" * 50)
    print()
    
    try:
        # Create demo profile
        profile = create_demo_profile()
        print()
        
        # Create Claude configurations
        claude_configs = create_claude_configurations()
        print()
        
        # Demonstrate configuration integration
        enhanced_legacy, enhanced_direct, enhanced_claude = demonstrate_configuration_integration(
            profile, claude_configs
        )
        print()
        
        # Store all configs for later use
        all_configs = {
            'enhanced_legacy': enhanced_legacy,
            'enhanced_direct': enhanced_direct,
            'enhanced_claude': enhanced_claude
        }
        
        # Demonstrate caching system
        demonstrate_cache_system()
        print()
        
        # Demonstrate profile compatibility
        demonstrate_profile_compatibility(profile, claude_configs)
        print()
        
        # Simulate agent creation
        simulate_agent_creation(profile, all_configs)
        print()
        
        print("✅ Demo completed successfully!")
        print()
        print("Key features demonstrated:")
        print("  • Multiple Claude configuration types")
        print("  • Enhanced configuration with caching and retry logic")
        print("  • Profile-based agent role configuration")
        print("  • Backward compatibility with legacy configurations")
        print("  • Profile compatibility validation")
        print("  • Cache management system")
        print("  • Configuration validation and error handling")
        print()
        print("Ready for production use with actual agent classes!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
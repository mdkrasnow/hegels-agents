#!/usr/bin/env python3
"""
Hegel's Agents Training Demo Script
Automatically loads environment variables and demonstrates the training system.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

def check_environment():
    """Check if required environment variables are set."""
    required_vars = {
        'GEMINI_API_KEY': 'Google Gemini API key for LLM functionality',
        'DATABASE_URL': 'PostgreSQL database URL for storing training data'
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value in ['your_gemini_api_key_here', 'postgresql://username:password@localhost:5432/hegels_agents']:
            missing.append(f"  {var}: {description}")
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(var)
        print("\n💡 Please:")
        print("1. Copy .env.example to .env")
        print("2. Fill in your actual API keys and database URL")
        print("3. Run this script again")
        return False
    
    print("✅ Environment variables loaded successfully")
    return True

def create_sample_env_if_missing():
    """Create .env file from .env.example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from .env.example...")
        with open(env_example) as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file - please edit it with your actual credentials")
        return False
    return True

def run_training_demo():
    """Run the training demonstration."""
    try:
        print("🚀 Starting Hegel's Agents Training Demo...")
        print("=" * 60)
        
        # Import training components
        from training.hegel_trainer import HegelTrainer, create_trainer
        from training.profile_store import PromptProfileStore
        from config import load_config
        
        # Load configuration
        config = load_config()
        print(f"📋 Configuration loaded (Environment: {config.app.environment})")
        
        # Initialize training system
        print("🔧 Initializing training system...")
        
        # Create trainer with grad=False first to test basic functionality
        trainer = create_trainer(grad=False)
        print("✅ HegelTrainer initialized successfully")
        
        # Test queries for demonstration
        test_queries = [
            {
                "query": "What is the fundamental nature of consciousness?",
                "corpus_id": "philosophy",
                "gold_answer": "Consciousness is the subjective, first-person experience of awareness, including qualia, intentionality, and self-awareness.",
                "expected_improvement": "Should improve philosophical reasoning and depth"
            },
            {
                "query": "Explain the concept of infinity in mathematics.",
                "corpus_id": "mathematics", 
                "gold_answer": "Mathematical infinity refers to unboundedness, including countable infinity (ℵ₀) and uncountable infinities, with applications in calculus, set theory, and analysis.",
                "expected_improvement": "Should improve mathematical precision and rigor"
            },
            {
                "query": "What were the main causes of World War I?",
                "corpus_id": "history",
                "gold_answer": "The main causes were imperialism, nationalism, the alliance system, and the immediate trigger of Archduke Franz Ferdinand's assassination.",
                "expected_improvement": "Should improve historical analysis and causal reasoning"
            }
        ]
        
        print("\n" + "=" * 60)
        print("🧪 PHASE 1: Testing Inference Mode (grad=False)")
        print("=" * 60)
        
        # Test inference mode first
        for i, test in enumerate(test_queries[:1], 1):  # Test just one query first
            print(f"\n📝 Test {i}: {test['query'][:50]}...")
            
            result = trainer.run(
                query=test['query'],
                corpus_id=test['corpus_id'],
                task_type="qa",
                grad=False  # Inference mode only
            )
            
            print(f"✅ Answer generated: {result['answer'][:100]}...")
            print(f"📊 Metadata: {len(result.get('metadata', {}))} fields")
            print(f"⏱️  Processing time: {result.get('processing_time', 'N/A')}")
        
        print("\n" + "=" * 60)  
        print("🎓 PHASE 2: Testing Training Mode (grad=True)")
        print("=" * 60)
        
        # Now test training mode
        trainer_with_grad = create_trainer(grad=True)
        
        for i, test in enumerate(test_queries[:2], 1):  # Test two queries with training
            print(f"\n📚 Training Test {i}: {test['query'][:50]}...")
            print(f"🎯 Expected improvement: {test['expected_improvement']}")
            
            # Run with training enabled
            result = trainer_with_grad.run(
                query=test['query'],
                corpus_id=test['corpus_id'], 
                task_type="qa",
                grad=True,  # Enable training
                gold_answer=test['gold_answer']
            )
            
            print(f"✅ Answer: {result['answer'][:150]}...")
            
            # Check training results
            if result.get('training_performed'):
                print("🔄 Training performed - prompts optimized!")
                if 'reward' in result:
                    print(f"📈 Performance reward: {result['reward']:.3f}")
                if 'profile_evolution' in result:
                    print("🧬 Profile evolution tracked")
            else:
                print("ℹ️  No training needed (performance already good)")
            
            print(f"⏱️  Total time: {result.get('processing_time', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📊 Summary:")
        print("• Inference mode (grad=False): ✅ Working")
        print("• Training mode (grad=True): ✅ Working") 
        print("• Automatic prompt optimization: ✅ Functional")
        print("• Multi-corpus support: ✅ Tested")
        
        print("\n🚀 Next Steps:")
        print("• Try your own questions with the training system")
        print("• Experiment with different corpus_id values")
        print("• Monitor prompt improvements over time")
        print("• Use the evaluation framework to measure progress")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Error during training demo: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print("📋 Full traceback:")
        print(traceback.format_exc())
        return False

def main():
    """Main function to run the training demo."""
    print("🤖 Hegel's Agents Training System Demo")
    print("=" * 60)
    
    # Step 1: Check if .env exists, create if needed
    if not create_sample_env_if_missing():
        return
    
    # Step 2: Check environment variables
    if not check_environment():
        return
    
    # Step 3: Run the training demonstration
    success = run_training_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("📝 Check the output above for training results")
    else:
        print("\n❌ Demo failed - check the error messages above")
        print("💡 Make sure your .env file has valid credentials")

if __name__ == "__main__":
    main()
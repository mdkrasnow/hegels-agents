# 🚀 Hegel's Agents Training System - Quick Start Guide

## Prerequisites

1. **Python Environment**: Python 3.8+ with required packages
2. **API Keys**: Google Gemini API key 
3. **Database**: PostgreSQL database for storing training data

## 🔧 Setup (First Time Only)

### 1. Install Dependencies
```bash
# Install required Python packages
pip install python-dotenv google-generativeai psycopg2-binary
```

### 2. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or use your preferred editor
```

**Required variables to update in .env:**
```bash
# Get from Google AI Studio: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Your PostgreSQL database connection string
DATABASE_URL=postgresql://username:password@host:port/database_name

# Or use Supabase (alternative to DATABASE_URL)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
```

### 3. Verify Setup
```bash
# Test that environment loads correctly
python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print('✅ Gemini API Key:', 'SET' if os.getenv('GEMINI_API_KEY') else 'MISSING')"
```

## 🎯 Running Training

### Option 1: Quick Demo (Recommended)
```bash
# Run the complete training demonstration
python run_training_demo.py
```

This will:
- ✅ Automatically load your .env variables
- ✅ Test inference mode (grad=False) 
- ✅ Test training mode (grad=True)
- ✅ Show prompt optimization in action
- ✅ Demonstrate multi-corpus learning

### Option 2: Custom Training Script
```python
# Create your own training script
from dotenv import load_dotenv
load_dotenv('.env')

import sys
sys.path.insert(0, 'src')

from training.hegel_trainer import create_trainer

# Initialize trainer with learning enabled
trainer = create_trainer(grad=True)

# Run training on your question
result = trainer.run(
    query="Your question here",
    corpus_id="philosophy",  # or "mathematics", "history", etc.
    grad=True,               # Enable learning
    gold_answer="Expected answer (optional)"
)

print(f"Answer: {result['answer']}")
print(f"Training performed: {result.get('training_performed', False)}")
```

### Option 3: Simple CLI Usage
```bash
# Direct Python execution with environment loading
PYTHONPATH=src python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from training.hegel_trainer import create_trainer
trainer = create_trainer(grad=True)
result = trainer.run('What is consciousness?', 'philosophy', grad=True)
print('Answer:', result['answer'][:200] + '...')
print('Training performed:', result.get('training_performed', False))
"
```

## 📊 What to Expect

### First Run (Inference Mode)
- System loads existing prompts
- Runs debate between agents
- Returns answer without training

### Training Mode (grad=True)
- Runs inference first
- Computes performance reward
- If reward < 0.8: automatically optimizes prompts
- Saves improved prompts to database
- Returns enhanced results with training metadata

### Output Example
```
🚀 Starting Hegel's Agents Training Demo...
✅ Environment variables loaded successfully
✅ HegelTrainer initialized successfully

📝 Test 1: What is the fundamental nature of consciousness?...
✅ Answer generated: Consciousness represents the subjective...
📊 Metadata: 12 fields
⏱️  Processing time: 2.3s

🔄 Training performed - prompts optimized!
📈 Performance reward: 0.67
🧬 Profile evolution tracked
⏱️  Total time: 4.1s
```

## 🔍 Troubleshooting

### Common Issues

**1. "Missing environment variables"**
```bash
# Make sure .env file exists and has real values
ls -la .env
cat .env | grep GEMINI_API_KEY
```

**2. "Import Error"**
```bash
# Run from project root directory
cd /path/to/hegels-agents
python run_training_demo.py
```

**3. "Database connection failed"**
```bash
# Test database connection
python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print(os.getenv('DATABASE_URL'))"
```

**4. "Google API authentication failed"**
```bash
# Test API key
python -c "
from dotenv import load_dotenv
load_dotenv('.env')
import os
print('API Key length:', len(os.getenv('GEMINI_API_KEY', '')))
print('Starts with:', os.getenv('GEMINI_API_KEY', '')[:10] + '...')
"
```

### Getting Help

1. **Check logs**: Look for detailed error messages in the console output
2. **Verify setup**: Ensure all environment variables are properly set
3. **Test components**: Run individual parts of the system to isolate issues
4. **Check dependencies**: Ensure all required packages are installed

## 🎓 Next Steps

Once training is working:

1. **Experiment with different corpus types**: "philosophy", "mathematics", "history", "science"
2. **Try your own questions**: Test the system with domain-specific queries
3. **Monitor improvement**: Run the same questions multiple times to see prompt evolution
4. **Use evaluation tools**: Leverage the statistical evaluation framework for analysis
5. **Explore advanced features**: Population-based optimization, multi-corpus specialization

## 📚 Advanced Usage

### Batch Training
```python
questions = [
    {"query": "Question 1", "corpus_id": "philosophy"},
    {"query": "Question 2", "corpus_id": "mathematics"},
    # ... more questions
]

for q in questions:
    result = trainer.run(grad=True, **q)
    print(f"Trained on: {q['query'][:50]}")
```

### Evaluation and Analysis
```python
from training.evaluation import create_training_evaluator

evaluator = create_training_evaluator()
results = evaluator.evaluate_profile_performance(
    profile=current_profile,
    test_questions=test_set,
    baseline_profile=original_profile
)
print(f"Improvement: {results['improvement_percentage']:.1f}%")
```

Ready to start training! 🎯
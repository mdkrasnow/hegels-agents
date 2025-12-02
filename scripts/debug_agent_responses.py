#!/usr/bin/env python3
"""
Debug script to inspect agent responses and quality evaluation.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import os
from config.settings import load_config
from agents.worker import BasicWorkerAgent
from debate.dialectical_tester import DialecticalTester

def load_env_file():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"Loading .env from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove 'export ' prefix if present
                    if line.startswith("export "):
                        line = line[7:]
                    
                    if "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes if present
                        value = value.strip("'\"")
                        os.environ[key] = value

def debug_response():
    print("Loading configuration...")
    load_env_file()
    load_config()
    
    print("Initializing agents...")
    worker = BasicWorkerAgent("debug_worker")
    tester = DialecticalTester("corpus_data") # Corpus dir is required but we can ignore it for this test if we don't use retrieval
    
    question = "What should be the primary focus of future solar system exploration: searching for life on Mars, studying Jupiter's moons, or asteroid mining preparation?"
    print(f"\nQuestion: {question}")
    
    print("\n--- Running Single Agent Test (with retrieval) ---")
    response, time_taken = tester.run_single_agent_test(question)
    
    print(f"\n[Raw Response Content]:\n{response.content}")
    print(f"\n[Parsed Reasoning]:\n{response.reasoning}")
    print(f"\n[Retrieved Context]:\n{response.metadata.get('has_external_context')}")
    
    print("\n--- Evaluating Quality ---")
    score = tester.evaluate_response_quality(response)
    print(f"Quality Score: {score}")
    
    # Also try to evaluate with a better prompt (Chain of Thought)
    print("\n--- Testing Better Evaluation Prompt ---")
    better_prompt = """
    Please evaluate the quality of this response on a scale from 1-10 based on:
    1. Accuracy and factual correctness
    2. Comprehensiveness and depth of analysis  
    3. Clarity and organization of reasoning
    4. Use of evidence and supporting information
    5. Acknowledgment of limitations or uncertainties
    
    First, provide a brief explanation of your evaluation.
    Then, on a new line, provide the score in the format: "Score: X/10".
    """
    
    eval_prompt = f"""
    {better_prompt}
    
    Response to evaluate:
    {response.content}
    
    Reasoning provided: {response.reasoning or 'None'}
    """
    
    eval_response = tester.reviewer._make_gemini_call(eval_prompt)
    print(f"\n[Better Evaluation Response]:\n{eval_response}")

if __name__ == "__main__":
    try:
        debug_response()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

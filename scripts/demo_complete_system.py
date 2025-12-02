#!/usr/bin/env python3
"""
Phase 0.5 Complete System Demonstration

This script provides an interactive demonstration of the complete dialectical
reasoning system. Shows the full workflow: corpus retrieval → agent debate → synthesis
with real examples that highlight the system's capabilities and dialectical improvements.

This is designed as both a validation tool and a showcase for stakeholders.

Usage:
    python scripts/demo_complete_system.py [--mock] [--interactive] [--question "custom question"]
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

from config.settings import load_config
from agents.worker import BasicWorkerAgent  
from agents.reviewer import BasicReviewerAgent
from agents.utils import AgentResponse
from corpus.file_retriever import FileCorpusRetriever
from debate.dialectical_tester import DialecticalTester
from test_questions.dialectical_test_questions import get_question_set


class DialecticalSystemDemo:
    """
    Interactive demonstration of the complete dialectical reasoning system.
    """
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize the demo system.
        
        Args:
            mock_mode: Whether to use mock agents (no API calls)
        """
        self.mock_mode = mock_mode
        
        # Handle configuration loading with graceful fallback for mock mode
        try:
            self.config = load_config()
        except Exception as e:
            if self.mock_mode:
                print(f"⚠️  Configuration loading failed, but continuing in mock mode: {e}")
                # Create minimal mock config
                self.config = {"mock_mode": True}
            else:
                raise
        
        # Initialize components
        self.setup_system_components()
        
        # Demo questions - carefully selected to showcase dialectical benefits
        self.demo_questions = [
            {
                "question": "What is the most compelling interpretation of quantum mechanics and why?",
                "category": "Physics/Quantum Mechanics", 
                "why_dialectical": "Multiple valid interpretations create natural dialectical tension",
                "expected_perspectives": ["Copenhagen interpretation", "Many-worlds", "Hidden variables"],
                "synthesis_opportunity": "Comparative analysis of strengths/weaknesses across interpretations"
            },
            {
                "question": "Which ethical framework provides the most practical guidance for modern moral dilemmas: utilitarianism, deontological ethics, or virtue ethics?",
                "category": "Philosophy/Ethics",
                "why_dialectical": "Fundamental disagreement between ethical approaches with practical implications",
                "expected_perspectives": ["Utilitarian calculus", "Duty-based moral imperatives", "Character-based virtue"],
                "synthesis_opportunity": "Pluralistic approach combining insights from each framework"
            },
            {
                "question": "What is the most important factor driving biological evolution: natural selection, genetic drift, gene flow, or mutation?",
                "category": "Biology/Evolution",
                "why_dialectical": "Different evolutionary forces operate at different scales and contexts",
                "expected_perspectives": ["Natural selection as primary driver", "Genetic drift in small populations", "Gene flow between populations"],
                "synthesis_opportunity": "Context-dependent importance of different evolutionary mechanisms"
            }
        ]
    
    def setup_system_components(self):
        """Set up all system components for the demo."""
        print("🔧 Initializing dialectical reasoning system...")
        
        # Corpus retrieval system
        self.corpus_retriever = FileCorpusRetriever(
            corpus_dir=str(project_root / "corpus_data")
        )
        
        # Load and index corpus
        print("  ✓ Loading corpus...")
        self.corpus_retriever.load_corpus()
        print("  ✓ Building search index...")
        self.corpus_retriever.build_search_index()
        
        if self.mock_mode:
            # Mock agents for demonstration
            from unittest.mock import MagicMock
            from agents.utils import AgentResponse
            
            self.worker_1 = MagicMock()
            self.worker_2 = MagicMock() 
            self.reviewer = MagicMock()
            
            # Configure realistic mock responses
            def mock_worker_1_response(question, external_context=None):
                return AgentResponse(
                    content="I believe the Copenhagen interpretation of quantum mechanics is most compelling because it provides a pragmatic framework that focuses on what we can measure and predict rather than speculating about unmeasurable hidden realities.",
                    reasoning="The Copenhagen interpretation has been extraordinarily successful in practical applications and avoids philosophical complications about the nature of reality that cannot be empirically tested.",
                    confidence=0.75,
                    sources=["quantum_physics_textbook", "experimental_results"],
                    metadata={"agent": "worker_1", "perspective": "pragmatic_realist"}
                )
            
            def mock_worker_2_response(question, external_context=None):
                return AgentResponse(
                    content="I argue that the Many-worlds interpretation offers the most compelling explanation because it maintains deterministic evolution of the universal wave function and eliminates the problematic wave function collapse.",
                    reasoning="Many-worlds interpretation preserves the mathematical elegance of quantum mechanics without introducing arbitrary collapse mechanisms, making it more theoretically consistent.",
                    confidence=0.80,
                    sources=["quantum_mechanics_many_worlds", "theoretical_physics"],
                    metadata={"agent": "worker_2", "perspective": "theoretical_purist"}
                )
                
            def mock_synthesis_response(question, responses):
                return AgentResponse(
                    content="Both interpretations offer valuable insights. The Copenhagen interpretation excels in practical applications and experimental design, while Many-worlds provides theoretical elegance and mathematical consistency. The choice between them may depend on whether we prioritize empirical utility or theoretical completeness. A pluralistic approach recognizing the strengths of each interpretation for different purposes may be most productive.",
                    reasoning="Rather than viewing these interpretations as mutually exclusive, we can recognize that they serve different scientific purposes - Copenhagen for practical quantum mechanics and experimental design, Many-worlds for theoretical understanding and foundational questions.",
                    confidence=0.85,
                    sources=["synthesis_analysis", "interpretation_comparison"],
                    metadata={"agent": "reviewer", "synthesis_type": "pluralistic_integration"}
                )
            
            self.worker_1.respond = mock_worker_1_response
            self.worker_2.respond = mock_worker_2_response  
            self.reviewer.synthesize_responses = mock_synthesis_response
            self.reviewer._make_gemini_call = lambda x: "8"
            
            print("  ✓ Mock agents initialized for safe demonstration")
            
        else:
            # Real agents for live demonstration
            self.worker_1 = BasicWorkerAgent(
                name="Demo_Worker_1",
                config=self.config
            )
            
            self.worker_2 = BasicWorkerAgent(
                name="Demo_Worker_2", 
                config=self.config
            )
            
            self.reviewer = BasicReviewerAgent(
                name="Demo_Reviewer",
                config=self.config
            )
            
            print("  ✓ Live agents initialized - API calls will be made")
        
        # Dialectical testing system - only needed for evaluation, not demo
        # We'll handle dialectical workflow manually in demo
        self.dialectical_tester = None
        
        print("  ✓ Dialectical testing system ready")
        print()
    
    def _evaluate_response_heuristic(self, response: AgentResponse) -> float:
        """Simple heuristic evaluation of response quality for demo purposes."""
        score = 5.0  # Base score
        
        # Length bonus (longer responses tend to be more detailed)
        word_count = len(response.content.split())
        if word_count > 100:
            score += 1.5
        elif word_count > 50:
            score += 1.0
        
        # Reasoning bonus
        if response.reasoning and len(response.reasoning.strip()) > 20:
            score += 1.0
        
        # Confidence consideration
        if response.confidence and response.confidence > 0.7:
            score += 0.5
        
        # Sources bonus
        if response.sources and len(response.sources) > 0:
            score += 1.0
        
        return min(score, 10.0)
    
    def print_section_header(self, title: str, subtitle: str = None):
        """Print a formatted section header."""
        print("=" * 80)
        print(f"🎯 {title}")
        if subtitle:
            print(f"   {subtitle}")
        print("=" * 80)
        print()
    
    def print_step_header(self, step: int, title: str):
        """Print a formatted step header.""" 
        print(f"\n{'─' * 60}")
        print(f"📍 STEP {step}: {title}")
        print(f"{'─' * 60}")
    
    def demonstrate_corpus_integration(self, question: str):
        """Demonstrate corpus retrieval and integration."""
        self.print_step_header(1, "CORPUS KNOWLEDGE RETRIEVAL")
        
        print(f"Question: {question}")
        print()
        
        print("🔍 Searching knowledge corpus for relevant information...")
        context = self.corpus_retriever.retrieve_for_question(question, max_results=3)
        
        if context:
            context_preview = context[:500] + "..." if len(context) > 500 else context
            print(f"📚 Retrieved {len(context)} characters of relevant context")
            print(f"📄 Context preview:\n{context_preview}")
        else:
            print("⚠️  No specific corpus context found - agents will rely on general knowledge")
        
        print()
        return context
    
    def demonstrate_single_agent_response(self, question: str, context: str):
        """Demonstrate single agent response generation."""
        self.print_step_header(2, "SINGLE AGENT BASELINE RESPONSE")
        
        print("🤖 Generating single agent baseline response...")
        print()
        
        start_time = time.time()
        single_response = self.worker_1.respond(question, external_context=context)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"🎯 Confidence: {single_response.confidence:.2f}")
        print()
        
        print("📝 SINGLE AGENT RESPONSE:")
        print(f"{single_response.content}")
        print()
        
        if single_response.reasoning:
            print("🧠 REASONING:")
            print(f"{single_response.reasoning}")
            print()
        
        return single_response
    
    def demonstrate_dialectical_process(self, question: str, context: str):
        """Demonstrate the full dialectical process."""
        self.print_step_header(3, "DIALECTICAL DEBATE PROCESS")
        
        print("🥊 Initiating dialectical debate between multiple agents...")
        print()
        
        start_time = time.time()
        
        # Worker 1 - Thesis
        print("🤖 AGENT 1 - INITIAL THESIS:")
        worker1_response = self.worker_1.respond(question, external_context=context)
        print(f"Perspective: {worker1_response.metadata.get('perspective', 'Unknown')}")
        print(f"Confidence: {worker1_response.confidence:.2f}")
        print(f"Response: {worker1_response.content}")
        print()
        
        # Worker 2 - Antithesis
        print("🤖 AGENT 2 - ALTERNATIVE PERSPECTIVE:")
        worker2_context = f"Previous response to consider: {worker1_response.content}\n\n{context if context else ''}"
        worker2_response = self.worker_2.respond(question, external_context=worker2_context)
        print(f"Perspective: {worker2_response.metadata.get('perspective', 'Unknown')}")
        print(f"Confidence: {worker2_response.confidence:.2f}")
        print(f"Response: {worker2_response.content}")
        print()
        
        # Reviewer - Synthesis
        print("🧠 REVIEWER - DIALECTICAL SYNTHESIS:")
        synthesis_response = self.reviewer.synthesize_responses(
            question, [worker1_response, worker2_response]
        )
        print(f"Synthesis Type: {synthesis_response.metadata.get('synthesis_type', 'Unknown')}")
        print(f"Confidence: {synthesis_response.confidence:.2f}")
        print(f"Synthesis: {synthesis_response.content}")
        
        end_time = time.time()
        
        print()
        print(f"⏱️  Total dialectical process time: {end_time - start_time:.2f} seconds")
        print()
        
        return worker1_response, worker2_response, synthesis_response
    
    def demonstrate_quality_comparison(self, single_response, synthesis_response):
        """Demonstrate quality comparison between approaches."""
        self.print_step_header(4, "QUALITY ASSESSMENT & COMPARISON")
        
        print("📊 Evaluating response quality...")
        print()
        
        # Evaluate quality scores using simple heuristic
        single_quality = self._evaluate_response_heuristic(single_response)
        synthesis_quality = self._evaluate_response_heuristic(synthesis_response)
        
        improvement = ((synthesis_quality - single_quality) / single_quality) * 100
        
        print("QUALITY SCORES:")
        print(f"🎯 Single Agent:     {single_quality:.1f}/10")
        print(f"🎯 Dialectical:      {synthesis_quality:.1f}/10")
        print(f"📈 Improvement:      {improvement:+.1f}%")
        print()
        
        # Quality analysis
        if improvement > 10:
            quality_assessment = "🟢 SIGNIFICANT IMPROVEMENT"
        elif improvement > 5:
            quality_assessment = "🟡 MODERATE IMPROVEMENT" 
        elif improvement > 0:
            quality_assessment = "🟠 SLIGHT IMPROVEMENT"
        else:
            quality_assessment = "🔴 NO IMPROVEMENT"
        
        print(f"Assessment: {quality_assessment}")
        print()
        
        # Detailed comparison
        print("COMPARATIVE ANALYSIS:")
        
        single_length = len(single_response.content.split())
        synthesis_length = len(synthesis_response.content.split())
        
        print(f"📏 Response length:   Single={single_length} words, Dialectical={synthesis_length} words")
        print(f"🎯 Confidence:        Single={single_response.confidence:.2f}, Dialectical={synthesis_response.confidence:.2f}")
        
        # Sources comparison
        single_sources = len(single_response.sources) if single_response.sources else 0
        synthesis_sources = len(synthesis_response.sources) if synthesis_response.sources else 0
        print(f"📚 Sources cited:     Single={single_sources}, Dialectical={synthesis_sources}")
        
        return {
            "single_quality": single_quality,
            "synthesis_quality": synthesis_quality, 
            "improvement": improvement,
            "assessment": quality_assessment
        }
    
    def run_demo_question(self, demo_data: Dict[str, Any], interactive: bool = False):
        """Run complete demonstration for a single question."""
        
        question = demo_data["question"]
        category = demo_data["category"]
        why_dialectical = demo_data["why_dialectical"]
        
        self.print_section_header(
            f"DIALECTICAL REASONING DEMONSTRATION",
            f"Category: {category}"
        )
        
        print(f"❓ QUESTION: {question}")
        print()
        print(f"🎯 Why this question benefits from dialectical reasoning:")
        print(f"   {why_dialectical}")
        print()
        
        if interactive:
            input("Press Enter to continue...")
            print()
        
        # Step 1: Corpus Integration
        context = self.demonstrate_corpus_integration(question)
        
        if interactive:
            input("\nPress Enter to see single agent response...")
            print()
        
        # Step 2: Single Agent Response  
        single_response = self.demonstrate_single_agent_response(question, context)
        
        if interactive:
            input("Press Enter to see dialectical process...")
            print()
        
        # Step 3: Dialectical Process
        worker1_response, worker2_response, synthesis_response = self.demonstrate_dialectical_process(question, context)
        
        if interactive:
            input("Press Enter to see quality comparison...")
            print()
        
        # Step 4: Quality Comparison
        quality_results = self.demonstrate_quality_comparison(single_response, synthesis_response)
        
        # Summary
        print("🏁 DEMONSTRATION SUMMARY:")
        print(f"✓ Question category: {category}")
        print(f"✓ Dialectical improvement: {quality_results['improvement']:+.1f}%")
        print(f"✓ Assessment: {quality_results['assessment']}")
        print()
        
        return {
            "question": question,
            "category": category,
            "single_response": single_response,
            "synthesis_response": synthesis_response,
            "quality_results": quality_results
        }
    
    def run_complete_demonstration(self, interactive: bool = False, custom_question: str = None):
        """Run the complete system demonstration."""
        
        print("🚀 DIALECTICAL REASONING SYSTEM - COMPLETE DEMONSTRATION")
        print(f"Mode: {'Interactive' if interactive else 'Automated'}")
        print(f"Agents: {'Mock (Safe Demo)' if self.mock_mode else 'Live API'}")
        print()
        
        if custom_question:
            # Custom question demo
            demo_data = {
                "question": custom_question,
                "category": "Custom Question",
                "why_dialectical": "Custom question to test system capabilities"
            }
            
            results = [self.run_demo_question(demo_data, interactive)]
            
        else:
            # Standard demo questions
            results = []
            
            for i, demo_data in enumerate(self.demo_questions):
                print(f"\n\n🔄 DEMONSTRATION {i+1} of {len(self.demo_questions)}")
                
                if interactive and i > 0:
                    continue_demo = input(f"\nContinue with demonstration {i+1}? (y/n): ").lower().startswith('y')
                    if not continue_demo:
                        break
                
                result = self.run_demo_question(demo_data, interactive)
                results.append(result)
                
                if interactive:
                    input(f"\nDemonstration {i+1} complete. Press Enter to continue...")
        
        # Overall summary
        self.print_section_header("OVERALL DEMONSTRATION SUMMARY")
        
        if results:
            improvements = [r["quality_results"]["improvement"] for r in results]
            avg_improvement = sum(improvements) / len(improvements)
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            
            print("📊 AGGREGATE RESULTS:")
            print(f"✓ Questions demonstrated: {len(results)}")
            print(f"✓ Average improvement: {avg_improvement:.1f}%")
            print(f"✓ Questions with improvement: {positive_improvements}/{len(results)} ({positive_improvements/len(results)*100:.0f}%)")
            print()
            
            print("📋 INDIVIDUAL RESULTS:")
            for i, result in enumerate(results, 1):
                improvement = result["quality_results"]["improvement"]
                print(f"{i}. {result['category']}: {improvement:+.1f}% {result['quality_results']['assessment']}")
            
        print()
        
        # System assessment
        if len(results) > 0 and sum(r["quality_results"]["improvement"] for r in results) / len(results) > 5:
            system_assessment = "✅ SYSTEM DEMONSTRATES CLEAR DIALECTICAL BENEFITS"
        elif len(results) > 0:
            system_assessment = "⚠️  SYSTEM SHOWS MIXED RESULTS"
        else:
            system_assessment = "❌ DEMONSTRATION INCOMPLETE"
        
        print(f"🎯 SYSTEM ASSESSMENT: {system_assessment}")
        print()
        
        print("🏆 DEMONSTRATION COMPLETE")
        print("   The dialectical reasoning system has been fully demonstrated.")
        print("   This showcase illustrates how multi-agent debate can improve reasoning quality.")
        print()
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Phase 0.5 Complete System Demonstration")
    parser.add_argument("--mock", action="store_true", help="Use mock agents (no API calls)")
    parser.add_argument("--interactive", action="store_true", help="Interactive demonstration mode")
    parser.add_argument("--question", type=str, help="Custom question to demonstrate")
    
    args = parser.parse_args()
    
    # Initialize demo system
    demo = DialecticalSystemDemo(mock_mode=args.mock)
    
    # Run demonstration
    results = demo.run_complete_demonstration(
        interactive=args.interactive,
        custom_question=args.question
    )
    
    # Exit successfully if demo completed
    sys.exit(0)


if __name__ == "__main__":
    main()
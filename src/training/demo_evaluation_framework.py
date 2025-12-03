#!/usr/bin/env python3
"""
Demonstration of T2.5 Basic Evaluation Framework
===============================================

This script demonstrates the comprehensive evaluation capabilities including:
- Statistical evaluation with baseline comparison
- Learning curve analysis
- Integration with corpus data
- Report generation
- Performance benchmarking

Run with: python demo_evaluation_framework.py
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.evaluation import (
    TrainingEvaluator, create_training_evaluator, create_evaluation_test_suite,
    EvaluationMetrics, LearningCurveAnalysis, BaselineComparison
)
from src.training.data_structures import PromptProfile, RolePrompt
from src.training.database.prompt_profile_store import PromptProfileStore
from src.training.hegel_trainer import HegelTrainer


class EvaluationFrameworkDemo:
    """
    Demonstration of the training evaluation framework capabilities.
    """
    
    def __init__(self):
        """Initialize the demonstration."""
        print("🚀 Initializing T2.5 Basic Evaluation Framework Demo")
        print("=" * 60)
        
        # Setup temporary directory for demo
        self.temp_dir = tempfile.mkdtemp()
        self.demo_corpus_dir = Path(self.temp_dir) / "demo_corpus"
        self.demo_corpus_dir.mkdir()
        
        # Create demo corpus files
        self._create_demo_corpus()
        
        # Initialize components (using mocks for this demo)
        self._setup_evaluation_components()
        
        print(f"✅ Demo environment setup complete")
        print(f"📁 Demo directory: {self.temp_dir}")
        print()
    
    def _create_demo_corpus(self):
        """Create sample corpus files for demonstration."""
        print("📚 Creating demo corpus files...")
        
        # Philosophy corpus
        philosophy_file = self.demo_corpus_dir / "philosophy_demo.txt"
        with open(philosophy_file, 'w') as f:
            f.write("""
# Philosophy: Ethics and Morality

Ethics is the branch of philosophy concerned with moral principles and values.
It examines questions about what is right and wrong, good and evil.

## Key Ethical Theories

### Utilitarianism
Utilitarianism judges actions by their consequences, seeking the greatest good 
for the greatest number of people.

### Deontological Ethics  
Deontological ethics focuses on duties and rules rather than consequences.
Kant's categorical imperative is a famous example.

### Virtue Ethics
Virtue ethics emphasizes character traits and asks what kind of person 
one should be rather than what one should do.
""")
        
        # Science corpus
        science_file = self.demo_corpus_dir / "science_demo.txt"
        with open(science_file, 'w') as f:
            f.write("""
# Science: Quantum Physics

Quantum physics describes the behavior of matter and energy at the atomic scale.
It challenges our classical understanding of reality.

## Key Principles

### Wave-Particle Duality
Particles exhibit both wave and particle properties depending on observation.

### Uncertainty Principle
Position and momentum cannot be simultaneously measured with perfect accuracy.

### Quantum Entanglement
Particles can be correlated in ways that seem to defy classical physics.
""")
        
        print(f"✅ Created {len(list(self.demo_corpus_dir.glob('*.txt')))} corpus files")
    
    def _setup_evaluation_components(self):
        """Setup evaluation components (using mocks for demo)."""
        print("🔧 Setting up evaluation components...")
        
        # For this demo, we'll use simplified mock implementations
        # In production, these would be real instances
        
        class MockProfileStore:
            def __init__(self):
                self.profiles = {}
            
            def create(self, profile, corpus_id, task_type):
                profile_id = f"profile_{len(self.profiles)}"
                self.profiles[profile_id] = profile
                return profile_id
            
            def get_by_id(self, profile_id):
                return self.profiles.get(profile_id)
        
        class MockHegelTrainer:
            def create_training_session(self):
                return f"session_{datetime.now().timestamp()}"
        
        self.mock_profile_store = MockProfileStore()
        self.mock_trainer = MockHegelTrainer()
        
        # Create the evaluator
        self.evaluator = create_training_evaluator(
            profile_store=self.mock_profile_store,
            hegel_trainer=self.mock_trainer,
            corpus_dir=str(self.demo_corpus_dir)
        )
        
        print("✅ Evaluation components initialized")
    
    def demo_evaluation_metrics(self):
        """Demonstrate evaluation metrics functionality."""
        print("📊 Demonstrating Evaluation Metrics")
        print("-" * 40)
        
        # Create sample metrics
        metrics = EvaluationMetrics(
            accuracy=0.85,
            f1_score=0.82,
            bleu_score=0.78,
            debate_quality_score=0.90,
            synthesis_effectiveness=0.88,
            conflict_resolution_quality=0.86,
            improvement_over_baseline=0.15,
            confidence_interval=(0.80, 0.90),
            p_value=0.02,
            effect_size=0.65,
            sample_size=50,
            evaluation_time=45.2
        )
        
        print(f"Primary Score: {metrics.get_primary_score():.3f}")
        print(f"Accuracy: {metrics.accuracy:.3f}")
        print(f"F1 Score: {metrics.f1_score:.3f}")
        print(f"Debate Quality: {metrics.debate_quality_score:.3f}")
        print(f"Confidence Interval: ({metrics.confidence_interval[0]:.3f}, {metrics.confidence_interval[1]:.3f})")
        print(f"Statistical Significance (p-value): {metrics.p_value}")
        print(f"Effect Size: {metrics.effect_size:.3f}")
        print()
        
        # Show JSON serialization
        print("📋 Metrics as JSON:")
        print(json.dumps(metrics.to_dict(), indent=2))
        print()
    
    def demo_baseline_comparison(self):
        """Demonstrate baseline comparison functionality."""
        print("⚖️  Demonstrating Baseline Comparison")
        print("-" * 40)
        
        # Create baseline and trained metrics
        baseline_metrics = EvaluationMetrics(
            accuracy=0.70,
            f1_score=0.68,
            bleu_score=0.65,
            debate_quality_score=0.72,
            synthesis_effectiveness=0.69,
            confidence_interval=(0.65, 0.75),
            sample_size=50
        )
        
        trained_metrics = EvaluationMetrics(
            accuracy=0.85,
            f1_score=0.82,
            bleu_score=0.78,
            debate_quality_score=0.90,
            synthesis_effectiveness=0.88,
            confidence_interval=(0.80, 0.90),
            sample_size=50
        )
        
        # Create comparison
        comparison = BaselineComparison(
            baseline_profile_id="baseline_001",
            trained_profile_id="trained_001", 
            corpus_id="philosophy_demo",
            task_type="qa",
            baseline_metrics=baseline_metrics,
            trained_metrics=trained_metrics,
            improvement_metrics={
                'accuracy_improvement': 0.15,
                'f1_improvement': 0.14,
                'debate_quality_improvement': 0.18
            },
            statistical_significance={
                'accuracy': 0.01,
                'f1_score': 0.02,
                'debate_quality': 0.005
            },
            effect_sizes={
                'accuracy': 0.75,
                'f1_score': 0.68,
                'debate_quality': 0.82
            },
            overall_improvement=0.156,
            statistically_significant=True,
            practically_significant=True,
            test_questions_count=50
        )
        
        print(f"Baseline vs Trained Comparison:")
        print(f"Overall Improvement: {comparison.overall_improvement:.3f} ({comparison.overall_improvement*100:.1f}%)")
        print(f"Statistically Significant: {comparison.statistically_significant}")
        print(f"Practically Significant: {comparison.practically_significant}")
        print()
        
        print("Metric-by-Metric Improvements:")
        for metric, improvement in comparison.improvement_metrics.items():
            p_value = comparison.statistical_significance.get(metric.replace('_improvement', ''), 'N/A')
            effect_size = comparison.effect_sizes.get(metric.replace('_improvement', ''), 'N/A')
            print(f"  {metric}: +{improvement:.3f} (p={p_value}, d={effect_size})")
        
        print()
    
    def demo_learning_curve_analysis(self):
        """Demonstrate learning curve analysis."""
        print("📈 Demonstrating Learning Curve Analysis")
        print("-" * 40)
        
        # Create learning curve analysis
        analysis = LearningCurveAnalysis(
            profile_id="learning_profile_001",
            corpus_id="philosophy_demo",
            task_type="qa",
            training_steps=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            performance_scores=[0.50, 0.58, 0.65, 0.71, 0.76, 0.79, 0.82, 0.84, 0.85, 0.85, 0.85],
            convergence_detected=True,
            convergence_step=35,
            final_performance=0.85,
            learning_rate=0.007,
            performance_variance=0.012,
            peak_performance=0.85,
            peak_performance_step=40,
            early_stopping_recommendation=40,
            total_training_time=1250.0
        )
        
        print(f"Training Profile: {analysis.profile_id}")
        print(f"Corpus: {analysis.corpus_id}")
        print(f"Training Steps: {len(analysis.training_steps)}")
        print(f"Final Performance: {analysis.final_performance:.3f}")
        print(f"Peak Performance: {analysis.peak_performance:.3f} (at step {analysis.peak_performance_step})")
        print(f"Learning Rate: {analysis.learning_rate:.4f}")
        print(f"Convergence Detected: {analysis.convergence_detected}")
        if analysis.convergence_detected:
            print(f"Convergence Step: {analysis.convergence_step}")
        print(f"Early Stopping Recommended: Step {analysis.early_stopping_recommendation}")
        print(f"Total Training Time: {analysis.total_training_time:.1f}s")
        print()
        
        # Show learning progression
        print("Learning Progression:")
        for i, (step, score) in enumerate(zip(analysis.training_steps, analysis.performance_scores)):
            marker = " ←convergence" if step == analysis.convergence_step else ""
            marker = " ←peak" if step == analysis.peak_performance_step else marker
            print(f"  Step {step:2d}: {score:.3f}{marker}")
        print()
    
    def demo_statistical_analysis(self):
        """Demonstrate statistical analysis capabilities."""
        print("📊 Demonstrating Statistical Analysis")
        print("-" * 40)
        
        # Generate sample evaluation data
        import random
        random.seed(42)  # For reproducible results
        
        evaluation_data = []
        for i in range(30):
            # Simulate improving performance with noise
            base_performance = 0.6 + (i / 30) * 0.3  # Improve from 0.6 to 0.9
            noise = random.uniform(-0.05, 0.05)
            performance = max(0.0, min(1.0, base_performance + noise))
            
            evaluation_data.append({
                'accuracy': performance,
                'f1_score': performance * 0.95,
                'debate_quality': performance * 1.1,
                'timestamp': datetime.now().isoformat()
            })
        
        # Calculate statistics
        accuracies = [d['accuracy'] for d in evaluation_data]
        f1_scores = [d['f1_score'] for d in evaluation_data]
        
        accuracy_stats = self.evaluator.statistical_analyzer.calculate_summary_statistics(accuracies)
        f1_stats = self.evaluator.statistical_analyzer.calculate_summary_statistics(f1_scores)
        
        print(f"Accuracy Statistics (n={accuracy_stats.count}):")
        print(f"  Mean: {accuracy_stats.mean:.3f}")
        print(f"  Median: {accuracy_stats.median:.3f}")
        print(f"  Std Dev: {accuracy_stats.std_dev:.3f}")
        print(f"  95% CI: [{accuracy_stats.confidence_interval[0]:.3f}, {accuracy_stats.confidence_interval[1]:.3f}]")
        print()
        
        print(f"F1 Score Statistics (n={f1_stats.count}):")
        print(f"  Mean: {f1_stats.mean:.3f}")
        print(f"  Median: {f1_stats.median:.3f}")
        print(f"  Std Dev: {f1_stats.std_dev:.3f}")
        print(f"  95% CI: [{f1_stats.confidence_interval[0]:.3f}, {f1_stats.confidence_interval[1]:.3f}]")
        print()
        
        # Correlation analysis
        try:
            correlation = self.evaluator.statistical_analyzer.analyze_correlation(
                evaluation_data, 'accuracy', 'f1_score'
            )
            print(f"Accuracy vs F1 Correlation:")
            print(f"  Correlation Coefficient: {correlation.correlation_coefficient:.3f}")
            print(f"  Relationship: {correlation.relationship_strength} {correlation.relationship_direction}")
            print(f"  Sample Size: {correlation.sample_size}")
            if correlation.p_value:
                print(f"  P-value: {correlation.p_value:.3f}")
            print()
        except Exception as e:
            print(f"Correlation analysis skipped: {e}")
            print()
    
    def demo_test_suite_creation(self):
        """Demonstrate test suite creation."""
        print("🧪 Demonstrating Test Suite Creation")
        print("-" * 40)
        
        # Create test suites for different corpora and tasks
        test_suites = {
            'philosophy_qa': create_evaluation_test_suite(
                corpus_id="philosophy_demo",
                task_type="qa",
                num_questions=5
            ),
            'science_qa': create_evaluation_test_suite(
                corpus_id="science_demo", 
                task_type="qa",
                num_questions=5
            ),
            'philosophy_analysis': create_evaluation_test_suite(
                corpus_id="philosophy_demo",
                task_type="analysis",
                num_questions=3
            )
        }
        
        for suite_name, test_suite in test_suites.items():
            print(f"{suite_name.upper()} Test Suite ({len(test_suite)} questions):")
            for i, question in enumerate(test_suite[:2], 1):  # Show first 2 questions
                print(f"  Question {i}: {question['question'][:60]}...")
                print(f"  Expected: {question['expected_answer'][:60]}...")
                print()
        
        print(f"Total test questions generated: {sum(len(suite) for suite in test_suites.values())}")
        print()
    
    def demo_profile_creation_and_evaluation(self):
        """Demonstrate profile creation and evaluation."""
        print("👤 Demonstrating Profile Creation and Evaluation")
        print("-" * 50)
        
        # Create sample profiles
        baseline_profile = PromptProfile(
            name="Baseline_Philosophy_QA",
            description="Baseline profile for philosophy question answering",
            metadata={'corpus_id': 'philosophy_demo', 'task_type': 'qa'}
        )
        
        # Add role prompts
        worker_prompt = RolePrompt(
            role="worker",
            prompt_text="Answer questions about philosophy clearly and accurately.",
            description="Basic philosophy Q&A worker"
        )
        
        reviewer_prompt = RolePrompt(
            role="reviewer",
            prompt_text="Review philosophical answers for accuracy and completeness.",
            description="Philosophy answer reviewer"
        )
        
        baseline_profile.add_role_prompt(worker_prompt)
        baseline_profile.add_role_prompt(reviewer_prompt)
        
        # Store profile
        baseline_id = self.mock_profile_store.create(
            baseline_profile, "philosophy_demo", "qa"
        )
        
        print(f"Created Profile: {baseline_profile.name}")
        print(f"Profile ID: {baseline_id}")
        print(f"Roles: {', '.join(baseline_profile.get_roles())}")
        print(f"Metadata: {baseline_profile.metadata}")
        print()
        
        # Create test questions
        test_questions = create_evaluation_test_suite(
            corpus_id="philosophy_demo",
            task_type="qa", 
            num_questions=3
        )
        
        print(f"Generated {len(test_questions)} test questions for evaluation")
        for i, q in enumerate(test_questions, 1):
            print(f"  {i}. {q['question']}")
        print()
        
        print("📋 Evaluation Framework Ready!")
        print("In a full implementation, this would:")
        print("  - Generate responses using the profile")
        print("  - Calculate accuracy, F1, BLEU scores")
        print("  - Assess debate quality and synthesis effectiveness")
        print("  - Perform statistical analysis")
        print("  - Generate comprehensive reports")
        print()
    
    def demo_export_capabilities(self):
        """Demonstrate result export capabilities."""
        print("💾 Demonstrating Export Capabilities")
        print("-" * 40)
        
        # Create sample evaluation results
        sample_results = {
            'evaluation_id': 'demo_eval_001',
            'profile_id': 'baseline_philosophy_001',
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_metrics': {
                'accuracy': 0.85,
                'f1_score': 0.82,
                'bleu_score': 0.78,
                'debate_quality_score': 0.90
            },
            'statistical_validation': {
                'statistical_rigor_level': 'adequate',
                'confidence_level': 0.95,
                'sample_size_adequate': True,
                'recommendations': ['Results show statistical significance']
            }
        }
        
        # Add to evaluator history
        self.evaluator.evaluation_history.append(sample_results)
        
        # Export results
        export_dir = Path(self.temp_dir) / "exports"
        try:
            exported_files = self.evaluator.export_evaluation_results(
                output_dir=export_dir,
                format='json',
                include_raw_data=True
            )
            
            print("Export completed successfully!")
            for file_type, file_path in exported_files.items():
                file_size = Path(file_path).stat().st_size
                print(f"  {file_type}: {file_path} ({file_size} bytes)")
            
        except Exception as e:
            print(f"Export demonstration (expected in mock environment): {e}")
        
        # Show statistics
        stats = self.evaluator.get_evaluation_statistics()
        print()
        print("📊 Evaluation Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        print()
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        try:
            print("🎯 Starting Complete T2.5 Evaluation Framework Demo")
            print("=" * 60)
            print()
            
            # Run all demonstration components
            self.demo_evaluation_metrics()
            self.demo_baseline_comparison()
            self.demo_learning_curve_analysis()
            self.demo_statistical_analysis()
            self.demo_test_suite_creation()
            self.demo_profile_creation_and_evaluation()
            self.demo_export_capabilities()
            
            print("🎉 Demo completed successfully!")
            print("=" * 60)
            print()
            print("Key Features Demonstrated:")
            print("✅ Comprehensive evaluation metrics with statistical rigor")
            print("✅ Baseline comparison with significance testing")
            print("✅ Learning curve analysis and convergence detection")
            print("✅ Statistical analysis with confidence intervals")
            print("✅ Test suite generation for different corpora")
            print("✅ Profile creation and evaluation workflow")
            print("✅ Export capabilities for research analysis")
            print()
            print("The evaluation framework is ready for integration with:")
            print("  - Existing HegelTrainer training system")
            print("  - PromptProfileStore for profile management")
            print("  - Corpus data for contextual evaluation")
            print("  - Statistical analysis for research validation")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up demo resources."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up demo directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    print("🚀 T2.5 Basic Evaluation Framework - Comprehensive Demo")
    print("This demo showcases statistical evaluation and baseline comparison capabilities")
    print()
    
    demo = EvaluationFrameworkDemo()
    demo.run_complete_demo()
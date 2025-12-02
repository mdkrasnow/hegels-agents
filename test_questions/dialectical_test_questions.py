"""
Dialectical Test Questions for Phase 0.5.3 Validation

This module contains 10 carefully crafted questions designed to test whether
dialectical debate improves AI reasoning quality. Questions are chosen to:

1. Cover topics available in the file-based corpus
2. Have multiple valid perspectives or approaches
3. Allow for substantive disagreement and synthesis
4. Test different types of reasoning (factual, analytical, evaluative)
5. Be answerable with the available knowledge base

Each question is designed to elicit different responses from multiple agents,
enabling meaningful dialectical synthesis.
"""

# Test questions for dialectical validation
DIALECTICAL_TEST_QUESTIONS = [
    # Physics/Quantum Mechanics - Interpretational question
    "What is the most compelling interpretation of quantum mechanics and why: "
    "the Copenhagen interpretation, Many-worlds theory, or Hidden variables theory?",
    
    # Philosophy/Ethics - Moral framework comparison
    "Which ethical framework provides the most practical guidance for modern moral decisions: "
    "utilitarianism, deontological ethics, or virtue ethics?",
    
    # Biology/Evolution - Mechanism emphasis
    "What is the most important driving force in evolution: natural selection, "
    "genetic drift, gene flow, or mutation? Explain your reasoning.",
    
    # History/WWII - Causal analysis
    "What was the most decisive factor that led to the Allied victory in World War II: "
    "industrial capacity, strategic decisions, technological advantages, or Soviet involvement?",
    
    # Economics/Macroeconomics - Policy evaluation
    "Should governments prioritize controlling inflation or reducing unemployment "
    "during economic downturns, and what are the trade-offs?",
    
    # Computer Science/Algorithms - Optimization approach
    "For large-scale data processing, what is more important: time complexity optimization "
    "or space complexity optimization? Consider practical applications.",
    
    # Psychology/Cognitive Science - Learning theory
    "What is the most effective approach to understanding human learning: "
    "behaviorist conditioning, cognitive information processing, or constructivist learning?",
    
    # Literature/Shakespeare - Analytical interpretation
    "What is the central theme of Hamlet: the corruption of power, the paralysis of thought, "
    "or the inevitability of fate? Support your interpretation.",
    
    # Mathematics/Calculus - Application priority
    "In teaching calculus, should educators emphasize rigorous theoretical foundations "
    "or practical problem-solving applications? What approach better serves students?",
    
    # Astronomy/Solar System - Scientific priority
    "What should be the primary focus of future solar system exploration: "
    "searching for life on Mars, studying Jupiter's moons, or asteroid mining preparation?"
]

# Additional metadata for each question
QUESTION_METADATA = [
    {
        "id": 1,
        "domain": "Physics",
        "corpus_file": "physics_quantum_mechanics.txt",
        "question_type": "evaluative",
        "expected_conflicts": ["interpretation_preferences", "scientific_philosophy"],
        "synthesis_opportunities": ["combine_complementary_aspects", "acknowledge_contextual_validity"]
    },
    {
        "id": 2,
        "domain": "Philosophy", 
        "corpus_file": "philosophy_ethics.txt",
        "question_type": "evaluative",
        "expected_conflicts": ["moral_framework_preferences", "practical_vs_theoretical"],
        "synthesis_opportunities": ["situational_ethics", "multi_framework_approach"]
    },
    {
        "id": 3,
        "domain": "Biology",
        "corpus_file": "biology_evolution.txt", 
        "question_type": "analytical",
        "expected_conflicts": ["mechanism_emphasis", "temporal_vs_population_factors"],
        "synthesis_opportunities": ["integrated_evolutionary_theory", "context_dependent_importance"]
    },
    {
        "id": 4,
        "domain": "History",
        "corpus_file": "history_world_war_two.txt",
        "question_type": "analytical", 
        "expected_conflicts": ["single_vs_multiple_causation", "material_vs_strategic_factors"],
        "synthesis_opportunities": ["multi_causal_analysis", "factor_interaction_effects"]
    },
    {
        "id": 5,
        "domain": "Economics",
        "corpus_file": "economics_macroeconomics.txt",
        "question_type": "evaluative",
        "expected_conflicts": ["policy_priority_preferences", "short_vs_long_term"],
        "synthesis_opportunities": ["contextual_policy_framework", "balanced_approach"]
    },
    {
        "id": 6,
        "domain": "Computer Science",
        "corpus_file": "computer_science_algorithms.txt",
        "question_type": "evaluative",
        "expected_conflicts": ["optimization_priorities", "theoretical_vs_practical"],
        "synthesis_opportunities": ["application_specific_optimization", "hybrid_approaches"]
    },
    {
        "id": 7,
        "domain": "Psychology", 
        "corpus_file": "psychology_cognitive_science.txt",
        "question_type": "analytical",
        "expected_conflicts": ["learning_theory_preferences", "mechanism_emphasis"],
        "synthesis_opportunities": ["integrated_learning_model", "context_dependent_effectiveness"]
    },
    {
        "id": 8,
        "domain": "Literature",
        "corpus_file": "literature_shakespeare.txt",
        "question_type": "analytical",
        "expected_conflicts": ["thematic_interpretation", "textual_evidence_emphasis"],
        "synthesis_opportunities": ["multi_layered_interpretation", "thematic_interconnection"]
    },
    {
        "id": 9,
        "domain": "Mathematics",
        "corpus_file": "mathematics_calculus.txt", 
        "question_type": "evaluative",
        "expected_conflicts": ["pedagogy_philosophy", "theory_vs_application"],
        "synthesis_opportunities": ["integrated_teaching_approach", "student_need_based_adaptation"]
    },
    {
        "id": 10,
        "domain": "Astronomy",
        "corpus_file": "astronomy_solar_system.txt",
        "question_type": "evaluative", 
        "expected_conflicts": ["exploration_priorities", "resource_allocation"],
        "synthesis_opportunities": ["multi_target_strategy", "phase_based_exploration"]
    }
]

# Question analysis framework
QUESTION_ANALYSIS_FRAMEWORK = {
    "conflict_potential": {
        "high": "Questions designed to elicit strong disagreements",
        "medium": "Questions with multiple reasonable approaches", 
        "low": "Questions with some variation in emphasis"
    },
    "synthesis_difficulty": {
        "hard": "Requires sophisticated integration of competing perspectives",
        "medium": "Clear opportunities for combining viewpoints",
        "easy": "Straightforward combination of complementary aspects"
    },
    "knowledge_requirements": {
        "deep": "Requires substantial domain knowledge",
        "moderate": "Basic to intermediate domain familiarity",
        "broad": "General knowledge across multiple areas"
    }
}

# Validation criteria for dialectical improvement
VALIDATION_CRITERIA = {
    "improvement_indicators": [
        "More comprehensive coverage of the topic",
        "Integration of multiple valid perspectives", 
        "Acknowledgment of nuances and trade-offs",
        "Higher quality reasoning and evidence",
        "Better structured argumentation",
        "More balanced and fair analysis"
    ],
    "conflict_resolution_quality": [
        "Clear identification of disagreement areas",
        "Thoughtful analysis of why disagreements exist",
        "Reasonable synthesis that preserves valid elements",
        "Acknowledgment of irreconcilable differences where appropriate"
    ],
    "dialectical_process_indicators": [
        "Evidence that agents built upon each other's responses",
        "Clear progression from initial positions to synthesis",
        "Meaningful engagement with opposing viewpoints",
        "Synthesis that goes beyond simple averaging or combination"
    ]
}

def get_question_set() -> list:
    """
    Get the complete set of dialectical test questions.
    
    Returns:
        List of test questions
    """
    return DIALECTICAL_TEST_QUESTIONS

def get_question_metadata() -> list:
    """
    Get metadata for all test questions.
    
    Returns:
        List of question metadata dictionaries
    """
    return QUESTION_METADATA

def get_question_by_domain(domain: str) -> list:
    """
    Get questions filtered by domain.
    
    Args:
        domain: Domain to filter by
        
    Returns:
        List of questions in the specified domain
    """
    filtered_questions = []
    for i, metadata in enumerate(QUESTION_METADATA):
        if metadata["domain"].lower() == domain.lower():
            filtered_questions.append(DIALECTICAL_TEST_QUESTIONS[i])
    return filtered_questions

def get_question_analysis(question_id: int) -> dict:
    """
    Get analysis framework for a specific question.
    
    Args:
        question_id: ID of question to analyze
        
    Returns:
        Dictionary with question analysis information
    """
    if 1 <= question_id <= len(QUESTION_METADATA):
        return QUESTION_METADATA[question_id - 1]
    else:
        raise ValueError(f"Question ID {question_id} out of range (1-{len(QUESTION_METADATA)})")

def validate_question_coverage() -> dict:
    """
    Validate that questions provide good coverage of available corpus.
    
    Returns:
        Dictionary with coverage analysis
    """
    covered_domains = set(meta["domain"] for meta in QUESTION_METADATA)
    question_types = [meta["question_type"] for meta in QUESTION_METADATA]
    
    return {
        "total_questions": len(DIALECTICAL_TEST_QUESTIONS),
        "covered_domains": list(covered_domains),
        "domain_count": len(covered_domains),
        "question_type_distribution": {
            qtype: question_types.count(qtype) for qtype in set(question_types)
        },
        "expected_conflicts": sum(len(meta["expected_conflicts"]) for meta in QUESTION_METADATA),
        "synthesis_opportunities": sum(len(meta["synthesis_opportunities"]) for meta in QUESTION_METADATA)
    }

if __name__ == "__main__":
    # Print question set summary
    print("Dialectical Test Questions for Phase 0.5.3")
    print("=" * 50)
    
    coverage = validate_question_coverage()
    print(f"Total Questions: {coverage['total_questions']}")
    print(f"Domains Covered: {coverage['domain_count']}")
    print(f"Question Types: {coverage['question_type_distribution']}")
    
    print("\nTest Questions:")
    print("-" * 30)
    
    for i, question in enumerate(DIALECTICAL_TEST_QUESTIONS, 1):
        metadata = QUESTION_METADATA[i-1]
        print(f"\n{i}. [{metadata['domain']}] {question}")
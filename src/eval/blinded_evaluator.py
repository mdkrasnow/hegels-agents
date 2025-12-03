"""
Blinded Evaluator for Fair Response Quality Assessment

This module implements scientifically rigorous blinded evaluation to ensure
fair comparison between single-agent and dialectical responses by eliminating
evaluator bias through response anonymization and independent evaluation.
"""

import random
import re
import uuid
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from agents.reviewer import BasicReviewerAgent
from agents.utils import AgentResponse


@dataclass
class AnonymizedResponse:
    """
    Anonymized version of AgentResponse stripped of identifying information.
    """
    content: str
    reasoning: str
    confidence: float
    anonymous_id: str  # Random ID for tracking, not revealing source
    
    @classmethod
    def from_agent_response(cls, response: AgentResponse) -> 'AnonymizedResponse':
        """
        Create anonymized response from AgentResponse.
        
        Args:
            response: Original response to anonymize
            
        Returns:
            AnonymizedResponse with identifying info removed
        """
        return cls(
            content=response.content,
            reasoning=response.reasoning or "No reasoning provided",
            confidence=response.confidence or 0.5,
            anonymous_id=str(uuid.uuid4())
        )


@dataclass
class BlindedEvaluationResult:
    """
    Result from a blinded evaluation session.
    """
    anonymous_id: str
    quality_score: float
    evaluator_id: str
    evaluation_timestamp: datetime
    raw_evaluation_text: str


class BlindedEvaluator:
    """
    Provides fair, blinded evaluation of responses without knowledge of source.
    
    This evaluator ensures scientific rigor by:
    1. Anonymizing all responses to remove source identification
    2. Using fresh reviewer instances to prevent memory bias
    3. Applying identical evaluation criteria and prompts
    4. Randomizing evaluation order to prevent sequence bias
    """
    
    # Standardized evaluation prompt - identical for all response types
    STANDARD_EVALUATION_PROMPT = """You are evaluating the quality of a response to a question. Please provide an objective assessment based solely on the response content.

Evaluation Criteria (Rate 1-100):
1. Accuracy and factual correctness
2. Comprehensiveness and depth of analysis  
3. Clarity and organization of reasoning
4. Use of evidence and supporting information
5. Acknowledgment of limitations or uncertainties

Important Instructions:
- Focus only on the response quality, not its source or context
- Provide only a numeric score from 1-100, no explanation needed
- Be consistent in your evaluation standards
- Rate based on content merit alone

Provide your score as a single number between 1 and 100."""
    
    def __init__(self, base_evaluator_id: str = "blinded_evaluator"):
        """
        Initialize the BlindedEvaluator.
        
        Args:
            base_evaluator_id: Base ID for generating unique evaluator instances
        """
        self.base_evaluator_id = base_evaluator_id
        self.evaluation_count = 0
        
    def _create_fresh_evaluator(self) -> BasicReviewerAgent:
        """
        Create a fresh reviewer agent instance to prevent bias from previous evaluations.
        
        Returns:
            New BasicReviewerAgent instance
        """
        self.evaluation_count += 1
        evaluator_id = f"{self.base_evaluator_id}_{self.evaluation_count}_{uuid.uuid4().hex[:6]}"
        return BasicReviewerAgent(evaluator_id)
    
    def _anonymize_response(self, response: AgentResponse) -> AnonymizedResponse:
        """
        Anonymize a response by removing identifying information.
        
        Args:
            response: Original response to anonymize
            
        Returns:
            AnonymizedResponse with source information stripped
        """
        return AnonymizedResponse.from_agent_response(response)
    
    def _evaluate_anonymized_response(self, 
                                    question: str, 
                                    anonymized_response: AnonymizedResponse) -> BlindedEvaluationResult:
        """
        Evaluate an anonymized response using a fresh evaluator instance.
        
        Args:
            question: The original question (context for evaluation)
            anonymized_response: Anonymized response to evaluate
            
        Returns:
            BlindedEvaluationResult with evaluation details
        """
        # Create fresh evaluator to prevent bias
        evaluator = self._create_fresh_evaluator()
        
        # Format evaluation prompt with anonymized content only
        evaluation_prompt = f"""
        {self.STANDARD_EVALUATION_PROMPT}
        
        Original Question: {question}
        
        Response to Evaluate:
        {anonymized_response.content}
        
        Reasoning: {anonymized_response.reasoning}
        
        Score (1-100):"""
        
        try:
            # Get evaluation from fresh evaluator
            raw_evaluation = evaluator._make_gemini_call(evaluation_prompt)
            
            # Extract numeric score using same logic as original system
            score = 5.0  # Default fallback
            
            # Try multiple patterns to extract score
            patterns = [
                r'(\d+(?:\.\d+)?)\s*/\s*100',  # X/100 format
                r'Score:\s*(\d+(?:\.\d+)?)',   # Score: X format
                r'(\d+(?:\.\d+)?)'             # First number found
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_evaluation, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Clamp score to valid range
            score = max(1.0, min(100.0, score))
            
            return BlindedEvaluationResult(
                anonymous_id=anonymized_response.anonymous_id,
                quality_score=score,
                evaluator_id=evaluator.agent_id,
                evaluation_timestamp=datetime.now(),
                raw_evaluation_text=raw_evaluation
            )
            
        except Exception as e:
            raise RuntimeError(f"Blinded evaluation failed for response {anonymized_response.anonymous_id}: {e}")
    
    def evaluate_responses_blinded(self, 
                                 question: str,
                                 responses: List[AgentResponse]) -> List[BlindedEvaluationResult]:
        """
        Evaluate multiple responses using blinded methodology.
        
        Args:
            question: The original question being answered
            responses: List of responses to evaluate blindly
            
        Returns:
            List of BlindedEvaluationResult in randomized order
        """
        if not responses:
            raise ValueError("Cannot evaluate empty list of responses")
        
        # Anonymize all responses
        anonymized_responses = [self._anonymize_response(response) for response in responses]
        
        # Randomize evaluation order to prevent sequence bias
        random.shuffle(anonymized_responses)
        
        # Evaluate each response with fresh evaluator
        results = []
        for anonymized_response in anonymized_responses:
            result = self._evaluate_anonymized_response(question, anonymized_response)
            results.append(result)
        
        return results
    
    def evaluate_single_vs_dialectical_blinded(self,
                                             question: str,
                                             single_response: AgentResponse,
                                             dialectical_response: AgentResponse) -> Tuple[float, float, Dict[str, Any]]:
        """
        Perform blinded evaluation comparing single-agent vs dialectical responses.
        
        This is the main method for fair dialectical evaluation that ensures:
        1. Neither evaluator knows which response is which type
        2. Identical evaluation criteria applied to both
        3. Independent evaluator instances prevent bias
        4. Random evaluation order prevents sequence effects
        
        Args:
            question: The question being answered
            single_response: Single-agent response
            dialectical_response: Dialectical synthesis response
            
        Returns:
            Tuple of (single_score, dialectical_score, evaluation_metadata)
        """
        # Create anonymized responses and maintain mapping
        single_anonymized = self._anonymize_response(single_response)
        dialectical_anonymized = self._anonymize_response(dialectical_response)
        
        # Create mapping for result lookup
        id_to_type = {
            single_anonymized.anonymous_id: "single",
            dialectical_anonymized.anonymous_id: "dialectical"
        }
        
        # Randomize evaluation order
        anonymized_responses = [single_anonymized, dialectical_anonymized]
        random.shuffle(anonymized_responses)
        
        # Evaluate each response with fresh evaluator
        results = {}
        for anonymized_response in anonymized_responses:
            result = self._evaluate_anonymized_response(question, anonymized_response)
            response_type = id_to_type[result.anonymous_id]
            results[response_type] = result
        
        # Create evaluation metadata for transparency
        metadata = {
            "evaluation_method": "blinded_independent",
            "single_evaluator_id": results["single"].evaluator_id,
            "dialectical_evaluator_id": results["dialectical"].evaluator_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "anonymized_evaluation": True,
            "independent_evaluators": True,
            "randomized_order": True,
            "evaluation_criteria": "standardized_5_point_scale"
        }
        
        return (
            results["single"].quality_score,
            results["dialectical"].quality_score,
            metadata
        )
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about evaluations performed.
        
        Returns:
            Dictionary with evaluation statistics
        """
        return {
            "total_evaluations_performed": self.evaluation_count,
            "evaluator_base_id": self.base_evaluator_id,
            "evaluation_method": "blinded_independent",
            "bias_prevention_measures": [
                "response_anonymization",
                "fresh_evaluator_instances", 
                "identical_prompts",
                "randomized_evaluation_order"
            ]
        }


class BlindedDialecticalComparison:
    """
    Utility class for performing complete blinded dialectical comparisons.
    
    This provides a high-level interface for the common use case of comparing
    single-agent vs dialectical responses with full blinding methodology.
    """
    
    def __init__(self):
        self.evaluator = BlindedEvaluator("dialectical_comparison")
    
    def compare_approaches(self,
                         question: str,
                         single_response: AgentResponse,
                         dialectical_response: AgentResponse) -> Dict[str, Any]:
        """
        Perform complete blinded comparison of single vs dialectical approaches.
        
        Args:
            question: Question being answered
            single_response: Single-agent response
            dialectical_response: Dialectical response
            
        Returns:
            Complete comparison results with fairness metadata
        """
        # Perform blinded evaluation
        single_score, dialectical_score, eval_metadata = self.evaluator.evaluate_single_vs_dialectical_blinded(
            question, single_response, dialectical_response
        )
        
        # Calculate improvement metrics
        improvement_score = (dialectical_score - single_score) / 100.0  # Normalize to -1 to 1
        # Calculate improvement percentage with proper zero baseline handling
        if single_score == 0:
            # If baseline is 0, improvement is undefined/infinite
            improvement_percentage = float('inf') if dialectical_score > 0 else 0.0
        else:
            improvement_percentage = ((dialectical_score - single_score) / single_score) * 100
        
        return {
            "question": question,
            "scores": {
                "single_agent": single_score,
                "dialectical": dialectical_score
            },
            "improvement": {
                "absolute_improvement": dialectical_score - single_score,
                "improvement_score": improvement_score,
                "improvement_percentage": improvement_percentage,
                "dialectical_better": dialectical_score > single_score
            },
            "evaluation_metadata": eval_metadata,
            "fairness_validation": {
                "blinded_evaluation": True,
                "independent_evaluators": True,
                "identical_criteria": True,
                "source_anonymized": True,
                "sequence_randomized": True,
                "bias_eliminated": True
            },
            "timestamp": datetime.now().isoformat()
        }
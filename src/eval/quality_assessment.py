"""
Quality Assessment Framework for Phase 0.5

This module implements systematic quality metrics and evaluation frameworks
beyond the basic mock testing approach. It provides comprehensive assessment
of response quality, dialectical effectiveness, and reasoning improvement.

This framework enables rigorous evaluation of whether dialectical debate
actually improves AI reasoning quality.
"""

import re
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter

from agents.utils import AgentResponse
from debate.session import DebateSession


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for response evaluation.
    """
    # Content Quality
    depth_score: float           # How thoroughly the response explores the topic
    clarity_score: float         # How clearly the response is expressed  
    accuracy_score: float        # Factual accuracy and correctness
    coherence_score: float       # Logical flow and consistency
    
    # Reasoning Quality  
    reasoning_depth: float       # Sophistication of reasoning process
    evidence_usage: float        # Effective use of evidence and examples
    logical_structure: float     # Quality of logical argumentation
    nuance_recognition: float    # Recognition of complexity and nuance
    
    # Dialectical Effectiveness (for synthesis responses)
    perspective_integration: float   # How well different perspectives are integrated
    conflict_resolution: float      # Effectiveness at resolving disagreements
    synthesis_originality: float    # Novel insights from combining perspectives
    
    # Meta-Quality
    confidence_calibration: float    # How well confidence matches actual quality
    uncertainty_acknowledgment: float  # Recognition of limitations and unknowns
    
    # Overall
    overall_quality: float          # Composite quality score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_scores(cls, scores: Dict[str, float]) -> 'QualityMetrics':
        """Create from a dictionary of scores."""
        return cls(**scores)


@dataclass
class DialecticalAssessment:
    """
    Assessment of dialectical process effectiveness.
    """
    # Process Quality
    debate_engagement: float        # How well agents engaged with each other's points
    perspective_diversity: float    # Diversity of perspectives presented
    conflict_identification: float  # How well conflicts were identified
    
    # Synthesis Quality  
    synthesis_effectiveness: float  # Quality of the synthesis process
    improvement_demonstration: float  # Clear evidence of improvement over single responses
    
    # Knowledge Integration
    corpus_utilization: float      # Effective use of corpus knowledge
    cross_domain_insights: float   # Integration of insights across domains
    
    # Meta-Assessment
    dialectical_necessity: float   # Whether dialectical process was actually needed
    process_efficiency: float     # Efficiency of the dialectical process
    
    overall_dialectical_score: float  # Composite dialectical effectiveness
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ResponseAnalyzer:
    """
    Analyzes individual responses for quality metrics.
    """
    
    def __init__(self):
        # Common quality indicators
        self.depth_indicators = [
            r'\b(?:because|since|therefore|thus|consequently|as a result)\b',
            r'\b(?:however|nevertheless|on the other hand|alternatively)\b', 
            r'\b(?:furthermore|moreover|additionally|in addition)\b',
            r'\b(?:for example|for instance|specifically|particularly)\b'
        ]
        
        self.reasoning_patterns = [
            r'\b(?:if.*then|given that|assuming|suppose)\b',
            r'\b(?:evidence suggests|research shows|studies indicate)\b',
            r'\b(?:this implies|this suggests|this means)\b',
            r'\b(?:in conclusion|to summarize|overall)\b'
        ]
        
        self.uncertainty_indicators = [
            r'\b(?:might|may|could|possibly|perhaps|likely)\b',
            r'\b(?:uncertain|unclear|unknown|debatable)\b',
            r'\b(?:it depends|context matters|varies)\b',
            r'\b(?:more research|further study|additional evidence)\b'
        ]
    
    def analyze_response(self, response: AgentResponse) -> QualityMetrics:
        """
        Analyze a single response for comprehensive quality metrics.
        
        Args:
            response: The response to analyze
            
        Returns:
            QualityMetrics object with all scores
        """
        content = response.content.lower()
        reasoning = (response.reasoning or "").lower()
        full_text = f"{content} {reasoning}".strip()
        
        # Content Quality Analysis
        depth_score = self._analyze_depth(full_text)
        clarity_score = self._analyze_clarity(full_text)
        accuracy_score = self._estimate_accuracy(response)
        coherence_score = self._analyze_coherence(full_text)
        
        # Reasoning Quality Analysis  
        reasoning_depth = self._analyze_reasoning_depth(full_text)
        evidence_usage = self._analyze_evidence_usage(response)
        logical_structure = self._analyze_logical_structure(full_text)
        nuance_recognition = self._analyze_nuance(full_text)
        
        # Dialectical metrics (basic for single responses)
        perspective_integration = 0.5  # Neutral for single responses
        conflict_resolution = 0.5     # Neutral for single responses
        synthesis_originality = 0.5   # Neutral for single responses
        
        # Meta-Quality
        confidence_calibration = self._analyze_confidence_calibration(response)
        uncertainty_acknowledgment = self._analyze_uncertainty_acknowledgment(full_text)
        
        # Calculate overall quality
        content_avg = (depth_score + clarity_score + accuracy_score + coherence_score) / 4
        reasoning_avg = (reasoning_depth + evidence_usage + logical_structure + nuance_recognition) / 4
        meta_avg = (confidence_calibration + uncertainty_acknowledgment) / 2
        
        overall_quality = (content_avg * 0.4 + reasoning_avg * 0.4 + meta_avg * 0.2)
        
        return QualityMetrics(
            depth_score=depth_score,
            clarity_score=clarity_score,
            accuracy_score=accuracy_score,
            coherence_score=coherence_score,
            reasoning_depth=reasoning_depth,
            evidence_usage=evidence_usage,
            logical_structure=logical_structure,
            nuance_recognition=nuance_recognition,
            perspective_integration=perspective_integration,
            conflict_resolution=conflict_resolution,
            synthesis_originality=synthesis_originality,
            confidence_calibration=confidence_calibration,
            uncertainty_acknowledgment=uncertainty_acknowledgment,
            overall_quality=overall_quality
        )
    
    def _analyze_depth(self, text: str) -> float:
        """Analyze the depth of exploration in the response."""
        depth_matches = 0
        for pattern in self.depth_indicators:
            depth_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize by text length (per 100 words)
        word_count = len(text.split())
        depth_density = (depth_matches / max(word_count, 1)) * 100
        
        # Convert to 0-1 scale (assume 5+ indicators per 100 words = high depth)
        return min(depth_density / 5.0, 1.0)
    
    def _analyze_clarity(self, text: str) -> float:
        """Analyze clarity of expression."""
        words = text.split()
        if not words:
            return 0.0
        
        # Sentence length analysis (avoid run-on sentences)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
        
        # Prefer moderate sentence length (15-25 words)
        sentence_score = 1.0 - abs(avg_sentence_length - 20) / 50.0
        sentence_score = max(0.0, min(1.0, sentence_score))
        
        # Vocabulary complexity (avoid overly complex or overly simple)
        avg_word_length = sum(len(word) for word in words) / len(words)
        vocabulary_score = 1.0 - abs(avg_word_length - 6.0) / 10.0
        vocabulary_score = max(0.0, min(1.0, vocabulary_score))
        
        return (sentence_score + vocabulary_score) / 2.0
    
    def _estimate_accuracy(self, response: AgentResponse) -> float:
        """Estimate factual accuracy (heuristic-based)."""
        # This is a simplified heuristic approach
        # In production, this would integrate with fact-checking services
        
        content = response.content.lower()
        
        # Presence of hedging language (good for accuracy)
        hedge_words = ['might', 'may', 'could', 'possibly', 'likely', 'generally', 'typically']
        hedging_count = sum(1 for word in hedge_words if word in content)
        hedging_score = min(hedging_count / 5.0, 1.0)
        
        # Presence of specific claims (risk for accuracy, but necessary)
        specific_patterns = [r'\d+%', r'\b\d{4}\b', r'\b(?:always|never|all|none)\b']
        specificity_count = sum(len(re.findall(pattern, content)) for pattern in specific_patterns)
        specificity_score = min(specificity_count / 3.0, 1.0)
        
        # Sources and evidence references (good for accuracy)
        source_score = min(len(response.sources) / 3.0, 1.0) if response.sources else 0.0
        
        # Balanced accuracy estimate
        return (hedging_score * 0.3 + specificity_score * 0.3 + source_score * 0.4)
    
    def _analyze_coherence(self, text: str) -> float:
        """Analyze logical coherence and flow."""
        # Transition word analysis
        transitions = [
            'therefore', 'thus', 'consequently', 'however', 'moreover',
            'furthermore', 'in addition', 'on the other hand', 'similarly', 'likewise'
        ]
        
        transition_count = sum(1 for word in transitions if word in text)
        sentences = len(re.split(r'[.!?]+', text))
        
        # Good coherence has appropriate transitions
        transition_density = transition_count / max(sentences - 1, 1)
        coherence_score = min(transition_density / 0.5, 1.0)  # 1 transition per 2 sentences ideal
        
        return coherence_score
    
    def _analyze_reasoning_depth(self, text: str) -> float:
        """Analyze sophistication of reasoning."""
        reasoning_matches = 0
        for pattern in self.reasoning_patterns:
            reasoning_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        word_count = len(text.split())
        reasoning_density = (reasoning_matches / max(word_count, 1)) * 100
        
        return min(reasoning_density / 3.0, 1.0)
    
    def _analyze_evidence_usage(self, response: AgentResponse) -> float:
        """Analyze effective use of evidence."""
        content = response.content.lower()
        
        # Evidence indicators
        evidence_patterns = [
            r'\b(?:research shows|studies indicate|evidence suggests)\b',
            r'\b(?:according to|based on|as shown by)\b',
            r'\b(?:for example|for instance|specifically)\b',
            r'\b(?:data shows|analysis reveals|experiments demonstrate)\b'
        ]
        
        evidence_count = sum(len(re.findall(pattern, content)) for pattern in evidence_patterns)
        source_count = len(response.sources) if response.sources else 0
        
        evidence_score = min(evidence_count / 3.0, 1.0)
        source_score = min(source_count / 3.0, 1.0)
        
        return (evidence_score + source_score) / 2.0
    
    def _analyze_logical_structure(self, text: str) -> float:
        """Analyze logical structure and argumentation."""
        # Look for logical argument structure
        structure_patterns = [
            r'\b(?:first|firstly|to begin|initially)\b',
            r'\b(?:second|secondly|next|then|furthermore)\b',
            r'\b(?:finally|lastly|in conclusion|to conclude)\b',
            r'\b(?:premise|assumption|given that|if we accept)\b'
        ]
        
        structure_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structure_patterns)
        sentences = len(re.split(r'[.!?]+', text))
        
        structure_score = min(structure_matches / max(sentences * 0.3, 1), 1.0)
        
        return structure_score
    
    def _analyze_nuance(self, text: str) -> float:
        """Analyze recognition of complexity and nuance."""
        nuance_patterns = [
            r'\b(?:complex|complicated|nuanced|multifaceted)\b',
            r'\b(?:depends on|varies|contextual|situational)\b',
            r'\b(?:both.*and|not only.*but also|on one hand.*on the other)\b',
            r'\b(?:trade-off|balance|competing|tension)\b'
        ]
        
        nuance_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in nuance_patterns)
        word_count = len(text.split())
        
        nuance_density = (nuance_count / max(word_count, 1)) * 100
        return min(nuance_density / 2.0, 1.0)
    
    def _analyze_confidence_calibration(self, response: AgentResponse) -> float:
        """Analyze how well confidence matches response quality."""
        content_length = len(response.content.split())
        reasoning_present = bool(response.reasoning and len(response.reasoning.strip()) > 10)
        sources_present = bool(response.sources and len(response.sources) > 0)
        
        # Expected confidence based on response characteristics
        expected_confidence = 0.5  # Base
        
        if content_length > 100:
            expected_confidence += 0.15  # Detailed response
        if reasoning_present:
            expected_confidence += 0.15  # Explicit reasoning
        if sources_present:
            expected_confidence += 0.2   # Evidence provided
        
        expected_confidence = min(expected_confidence, 1.0)
        
        # How close is actual confidence to expected?
        confidence_diff = abs(response.confidence - expected_confidence)
        calibration_score = 1.0 - confidence_diff
        
        return max(0.0, calibration_score)
    
    def _analyze_uncertainty_acknowledgment(self, text: str) -> float:
        """Analyze acknowledgment of limitations and uncertainty."""
        uncertainty_matches = 0
        for pattern in self.uncertainty_indicators:
            uncertainty_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        word_count = len(text.split())
        uncertainty_density = (uncertainty_matches / max(word_count, 1)) * 100
        
        # Good to acknowledge uncertainty, but not be overly uncertain
        return min(uncertainty_density / 3.0, 1.0)


class DialecticalEvaluator:
    """
    Evaluates the effectiveness of dialectical processes.
    """
    
    def __init__(self):
        self.response_analyzer = ResponseAnalyzer()
    
    def evaluate_dialectical_process(
        self,
        question: str,
        worker_responses: List[AgentResponse], 
        synthesis_response: AgentResponse,
        debate_session: DebateSession
    ) -> DialecticalAssessment:
        """
        Evaluate the effectiveness of a complete dialectical process.
        
        Args:
            question: The question being addressed
            worker_responses: Responses from worker agents
            synthesis_response: The synthesis response
            debate_session: The debate session with metadata
            
        Returns:
            DialecticalAssessment with detailed evaluation
        """
        # Analyze debate engagement
        debate_engagement = self._analyze_debate_engagement(worker_responses)
        
        # Analyze perspective diversity
        perspective_diversity = self._analyze_perspective_diversity(worker_responses)
        
        # Analyze conflict identification
        conflict_identification = self._analyze_conflict_identification(worker_responses, debate_session)
        
        # Analyze synthesis effectiveness
        synthesis_effectiveness = self._analyze_synthesis_effectiveness(
            worker_responses, synthesis_response
        )
        
        # Analyze improvement demonstration
        improvement_demonstration = self._analyze_improvement_demonstration(
            worker_responses, synthesis_response
        )
        
        # Analyze corpus utilization
        corpus_utilization = self._analyze_corpus_utilization(worker_responses + [synthesis_response])
        
        # Analyze cross-domain insights
        cross_domain_insights = self._analyze_cross_domain_insights(synthesis_response)
        
        # Meta-assessments
        dialectical_necessity = self._assess_dialectical_necessity(worker_responses)
        process_efficiency = self._assess_process_efficiency(worker_responses, synthesis_response)
        
        # Overall dialectical score
        process_scores = [debate_engagement, perspective_diversity, conflict_identification]
        synthesis_scores = [synthesis_effectiveness, improvement_demonstration]
        knowledge_scores = [corpus_utilization, cross_domain_insights]
        meta_scores = [dialectical_necessity, process_efficiency]
        
        overall_dialectical_score = (
            statistics.mean(process_scores) * 0.3 +
            statistics.mean(synthesis_scores) * 0.4 +
            statistics.mean(knowledge_scores) * 0.2 +
            statistics.mean(meta_scores) * 0.1
        )
        
        return DialecticalAssessment(
            debate_engagement=debate_engagement,
            perspective_diversity=perspective_diversity,
            conflict_identification=conflict_identification,
            synthesis_effectiveness=synthesis_effectiveness,
            improvement_demonstration=improvement_demonstration,
            corpus_utilization=corpus_utilization,
            cross_domain_insights=cross_domain_insights,
            dialectical_necessity=dialectical_necessity,
            process_efficiency=process_efficiency,
            overall_dialectical_score=overall_dialectical_score
        )
    
    def _analyze_debate_engagement(self, worker_responses: List[AgentResponse]) -> float:
        """Analyze how well agents engaged with each other's points."""
        if len(worker_responses) < 2:
            return 0.0
        
        # Look for references to other perspectives
        engagement_indicators = [
            r'\b(?:however|but|although|while|whereas)\b',
            r'\b(?:in contrast|on the other hand|alternatively)\b',
            r'\b(?:previous|earlier|above|mentioned|stated)\b',
            r'\b(?:disagree|agree|build on|expand|challenge)\b'
        ]
        
        total_engagement = 0
        for i, response in enumerate(worker_responses[1:], 1):  # Skip first response
            content = response.content.lower()
            engagement_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                                 for pattern in engagement_indicators)
            total_engagement += engagement_count
        
        # Normalize by number of responses and length
        avg_length = statistics.mean(len(r.content.split()) for r in worker_responses[1:])
        engagement_density = total_engagement / max(avg_length / 100, 0.1)
        
        return min(engagement_density / 3.0, 1.0)
    
    def _analyze_perspective_diversity(self, worker_responses: List[AgentResponse]) -> float:
        """Analyze diversity of perspectives presented."""
        if len(worker_responses) < 2:
            return 0.0
        
        # Simple text similarity analysis
        response_texts = [r.content.lower() for r in worker_responses]
        
        # Calculate overlap in key terms
        all_words = []
        response_word_sets = []
        
        for text in response_texts:
            words = set(word for word in text.split() if len(word) > 4)  # Focus on content words
            response_word_sets.append(words)
            all_words.extend(words)
        
        if not all_words:
            return 0.5  # Neutral if no content
        
        # Calculate average pairwise diversity
        diversity_scores = []
        for i in range(len(response_word_sets)):
            for j in range(i + 1, len(response_word_sets)):
                overlap = len(response_word_sets[i] & response_word_sets[j])
                union = len(response_word_sets[i] | response_word_sets[j])
                jaccard_similarity = overlap / max(union, 1)
                diversity = 1.0 - jaccard_similarity
                diversity_scores.append(diversity)
        
        return statistics.mean(diversity_scores) if diversity_scores else 0.5
    
    def _analyze_conflict_identification(self, worker_responses: List[AgentResponse], 
                                       debate_session: DebateSession) -> float:
        """Analyze how well conflicts were identified."""
        # Use debate session conflict analysis if available
        if hasattr(debate_session, 'conflicts_detected') and debate_session.conflicts_detected is not None:
            return 1.0 if debate_session.conflicts_detected else 0.0
        
        # Manual conflict detection
        conflict_indicators = [
            r'\b(?:disagree|dispute|challenge|contradict|oppose)\b',
            r'\b(?:different|alternative|competing|conflicting)\b',
            r'\b(?:however|but|although|nevertheless|nonetheless)\b'
        ]
        
        if len(worker_responses) < 2:
            return 0.0
        
        conflict_count = 0
        for response in worker_responses[1:]:  # Skip first response
            content = response.content.lower()
            for pattern in conflict_indicators:
                conflict_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Normalize by response count
        return min(conflict_count / max(len(worker_responses) - 1, 1) / 3.0, 1.0)
    
    def _analyze_synthesis_effectiveness(self, worker_responses: List[AgentResponse], 
                                       synthesis_response: AgentResponse) -> float:
        """Analyze quality of the synthesis process."""
        synthesis_indicators = [
            r'\b(?:both|combine|integrate|synthesize|merge)\b',
            r'\b(?:together|collectively|jointly|unified)\b',
            r'\b(?:balance|reconcile|bridge|connect)\b',
            r'\b(?:overall|comprehensive|holistic)\b'
        ]
        
        content = synthesis_response.content.lower()
        synthesis_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                            for pattern in synthesis_indicators)
        
        word_count = len(synthesis_response.content.split())
        synthesis_density = (synthesis_count / max(word_count, 1)) * 100
        
        return min(synthesis_density / 2.0, 1.0)
    
    def _analyze_improvement_demonstration(self, worker_responses: List[AgentResponse],
                                         synthesis_response: AgentResponse) -> float:
        """Analyze evidence of improvement over individual responses."""
        # Length comparison (synthesis should be more comprehensive)
        worker_lengths = [len(r.content.split()) for r in worker_responses]
        synthesis_length = len(synthesis_response.content.split())
        
        avg_worker_length = statistics.mean(worker_lengths)
        length_improvement = min((synthesis_length - avg_worker_length) / avg_worker_length, 1.0) if avg_worker_length > 0 else 0.0
        length_improvement = max(0.0, length_improvement)  # No penalty for shorter but better
        
        # Confidence comparison (synthesis should be more confident if truly better)
        worker_confidences = [r.confidence for r in worker_responses if r.confidence is not None]
        if worker_confidences and synthesis_response.confidence is not None:
            avg_worker_confidence = statistics.mean(worker_confidences)
            confidence_improvement = max(0.0, (synthesis_response.confidence - avg_worker_confidence) / avg_worker_confidence)
        else:
            confidence_improvement = 0.0
        
        # Source integration (synthesis should integrate sources)
        all_worker_sources = set()
        for response in worker_responses:
            if response.sources:
                all_worker_sources.update(response.sources)
        
        synthesis_sources = set(synthesis_response.sources) if synthesis_response.sources else set()
        source_integration = len(synthesis_sources & all_worker_sources) / max(len(all_worker_sources), 1) if all_worker_sources else 0.0
        
        return (length_improvement * 0.3 + confidence_improvement * 0.3 + source_integration * 0.4)
    
    def _analyze_corpus_utilization(self, responses: List[AgentResponse]) -> float:
        """Analyze effective utilization of corpus knowledge."""
        corpus_indicators = [
            r'\b(?:according to|based on|research shows|studies indicate)\b',
            r'\b(?:evidence|data|analysis|findings)\b',
            r'\b(?:literature|scholarship|academic|scientific)\b'
        ]
        
        total_corpus_usage = 0
        total_words = 0
        
        for response in responses:
            content = response.content.lower()
            words = len(content.split())
            total_words += words
            
            for pattern in corpus_indicators:
                total_corpus_usage += len(re.findall(pattern, content, re.IGNORECASE))
        
        if total_words == 0:
            return 0.0
        
        corpus_density = (total_corpus_usage / total_words) * 100
        return min(corpus_density / 2.0, 1.0)
    
    def _analyze_cross_domain_insights(self, synthesis_response: AgentResponse) -> float:
        """Analyze integration of insights across domains."""
        cross_domain_indicators = [
            r'\b(?:interdisciplinary|cross-field|multidisciplinary)\b',
            r'\b(?:similar to|analogous|parallel|comparable)\b',
            r'\b(?:connections|relationships|links|bridges)\b',
            r'\b(?:broader|wider|general|universal)\b'
        ]
        
        content = synthesis_response.content.lower()
        cross_domain_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                               for pattern in cross_domain_indicators)
        
        word_count = len(synthesis_response.content.split())
        cross_domain_density = (cross_domain_count / max(word_count, 1)) * 100
        
        return min(cross_domain_density / 1.5, 1.0)
    
    def _assess_dialectical_necessity(self, worker_responses: List[AgentResponse]) -> float:
        """Assess whether dialectical process was actually needed."""
        if len(worker_responses) < 2:
            return 0.0
        
        # High necessity if responses are sufficiently different
        diversity_score = self._analyze_perspective_diversity(worker_responses)
        
        # High necessity if responses show uncertainty
        uncertainty_scores = []
        for response in worker_responses:
            uncertainty_indicators = ['might', 'may', 'could', 'possibly', 'unclear', 'uncertain']
            content = response.content.lower()
            uncertainty_count = sum(1 for word in uncertainty_indicators if word in content)
            uncertainty_scores.append(min(uncertainty_count / 5.0, 1.0))
        
        avg_uncertainty = statistics.mean(uncertainty_scores)
        
        # Combine diversity and uncertainty for necessity assessment
        return (diversity_score * 0.7 + avg_uncertainty * 0.3)
    
    def _assess_process_efficiency(self, worker_responses: List[AgentResponse],
                                 synthesis_response: AgentResponse) -> float:
        """Assess efficiency of the dialectical process."""
        # Simple heuristic: efficiency decreases with excessive length
        total_worker_words = sum(len(r.content.split()) for r in worker_responses)
        synthesis_words = len(synthesis_response.content.split())
        
        total_words = total_worker_words + synthesis_words
        
        # Ideal range: 300-800 words total for good efficiency
        if 300 <= total_words <= 800:
            efficiency = 1.0
        elif total_words < 300:
            efficiency = total_words / 300.0  # Penalize too brief
        else:
            efficiency = max(0.2, 800.0 / total_words)  # Penalize too verbose
        
        return efficiency


class ComprehensiveQualityFramework:
    """
    Comprehensive quality assessment framework combining all evaluation approaches.
    """
    
    def __init__(self):
        self.response_analyzer = ResponseAnalyzer()
        self.dialectical_evaluator = DialecticalEvaluator()
    
    def evaluate_single_response(self, response: AgentResponse) -> QualityMetrics:
        """
        Evaluate a single response comprehensively.
        
        Args:
            response: The response to evaluate
            
        Returns:
            Comprehensive quality metrics
        """
        return self.response_analyzer.analyze_response(response)
    
    def evaluate_dialectical_session(
        self,
        question: str,
        single_response: AgentResponse,
        worker_responses: List[AgentResponse],
        synthesis_response: AgentResponse,
        debate_session: DebateSession
    ) -> Dict[str, Any]:
        """
        Evaluate a complete dialectical session.
        
        Args:
            question: The question being addressed
            single_response: Baseline single-agent response
            worker_responses: Multi-agent worker responses
            synthesis_response: Final synthesis response
            debate_session: The debate session data
            
        Returns:
            Comprehensive evaluation results
        """
        # Evaluate individual responses
        single_metrics = self.evaluate_single_response(single_response)
        worker_metrics = [self.evaluate_single_response(r) for r in worker_responses]
        synthesis_metrics = self.evaluate_single_response(synthesis_response)
        
        # Evaluate dialectical process
        dialectical_assessment = self.dialectical_evaluator.evaluate_dialectical_process(
            question, worker_responses, synthesis_response, debate_session
        )
        
        # Calculate improvement metrics
        quality_improvement = (
            synthesis_metrics.overall_quality - single_metrics.overall_quality
        ) / max(single_metrics.overall_quality, 0.1) * 100
        
        # Comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "single_response_metrics": single_metrics.to_dict(),
            "worker_response_metrics": [m.to_dict() for m in worker_metrics],
            "synthesis_response_metrics": synthesis_metrics.to_dict(),
            "dialectical_assessment": dialectical_assessment.to_dict(),
            "improvement_analysis": {
                "quality_improvement_percentage": quality_improvement,
                "single_quality_score": single_metrics.overall_quality,
                "synthesis_quality_score": synthesis_metrics.overall_quality,
                "dialectical_effectiveness": dialectical_assessment.overall_dialectical_score,
                "improvement_significance": self._assess_improvement_significance(quality_improvement)
            },
            "summary": {
                "dialectical_successful": quality_improvement > 5.0,
                "process_efficient": dialectical_assessment.process_efficiency > 0.7,
                "high_quality_synthesis": synthesis_metrics.overall_quality > 0.75,
                "recommendation": self._generate_recommendation(quality_improvement, dialectical_assessment)
            }
        }
        
        return results
    
    def _assess_improvement_significance(self, improvement_percentage: float) -> str:
        """Assess the significance of quality improvement."""
        if improvement_percentage > 20:
            return "HIGHLY_SIGNIFICANT"
        elif improvement_percentage > 10:
            return "SIGNIFICANT"
        elif improvement_percentage > 5:
            return "MODERATE"
        elif improvement_percentage > 0:
            return "MINIMAL"
        else:
            return "NO_IMPROVEMENT"
    
    def _generate_recommendation(self, improvement_percentage: float,
                               dialectical_assessment: DialecticalAssessment) -> str:
        """Generate recommendation based on evaluation results."""
        if improvement_percentage > 10 and dialectical_assessment.overall_dialectical_score > 0.7:
            return "STRONGLY_RECOMMEND_DIALECTICAL"
        elif improvement_percentage > 5 and dialectical_assessment.overall_dialectical_score > 0.6:
            return "RECOMMEND_DIALECTICAL"
        elif improvement_percentage > 0:
            return "CONSIDER_DIALECTICAL"
        else:
            return "USE_SINGLE_AGENT"
    
    def evaluate_test_suite_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate results from a complete test suite.
        
        Args:
            test_results: List of individual test evaluation results
            
        Returns:
            Aggregate test suite evaluation
        """
        if not test_results:
            return {"error": "No test results provided"}
        
        # Aggregate metrics
        improvements = [r["improvement_analysis"]["quality_improvement_percentage"] for r in test_results]
        dialectical_scores = [r["dialectical_assessment"]["overall_dialectical_score"] for r in test_results]
        single_scores = [r["improvement_analysis"]["single_quality_score"] for r in test_results]
        synthesis_scores = [r["improvement_analysis"]["synthesis_quality_score"] for r in test_results]
        
        # Calculate statistics
        suite_evaluation = {
            "timestamp": datetime.now().isoformat(),
            "test_count": len(test_results),
            "aggregate_metrics": {
                "mean_improvement": statistics.mean(improvements),
                "median_improvement": statistics.median(improvements),
                "improvement_std": statistics.stdev(improvements) if len(improvements) > 1 else 0.0,
                "min_improvement": min(improvements),
                "max_improvement": max(improvements),
                "positive_improvement_rate": sum(1 for i in improvements if i > 0) / len(improvements) * 100,
                "significant_improvement_rate": sum(1 for i in improvements if i > 5) / len(improvements) * 100,
                "mean_dialectical_score": statistics.mean(dialectical_scores),
                "mean_single_quality": statistics.mean(single_scores),
                "mean_synthesis_quality": statistics.mean(synthesis_scores)
            },
            "hypothesis_validation": {
                "improvement_hypothesis_supported": statistics.mean(improvements) > 5.0,
                "consistency_hypothesis_supported": sum(1 for i in improvements if i > 0) / len(improvements) > 0.6,
                "dialectical_process_effective": statistics.mean(dialectical_scores) > 0.6,
                "overall_hypothesis_validation": None  # Will be set below
            },
            "recommendations": []
        }
        
        # Overall hypothesis validation
        hypothesis_checks = suite_evaluation["hypothesis_validation"]
        hypothesis_support_count = sum(1 for v in [
            hypothesis_checks["improvement_hypothesis_supported"],
            hypothesis_checks["consistency_hypothesis_supported"],
            hypothesis_checks["dialectical_process_effective"]
        ] if v)
        
        if hypothesis_support_count >= 3:
            hypothesis_checks["overall_hypothesis_validation"] = "STRONGLY_SUPPORTED"
        elif hypothesis_support_count >= 2:
            hypothesis_checks["overall_hypothesis_validation"] = "SUPPORTED"
        elif hypothesis_support_count >= 1:
            hypothesis_checks["overall_hypothesis_validation"] = "PARTIALLY_SUPPORTED"
        else:
            hypothesis_checks["overall_hypothesis_validation"] = "NOT_SUPPORTED"
        
        # Generate recommendations
        if hypothesis_checks["overall_hypothesis_validation"] in ["STRONGLY_SUPPORTED", "SUPPORTED"]:
            suite_evaluation["recommendations"] = [
                "Proceed with dialectical approach implementation",
                "Scale up to larger test sets for further validation",
                "Begin Phase 1 infrastructure development",
                "Consider production deployment planning"
            ]
        elif hypothesis_checks["overall_hypothesis_validation"] == "PARTIALLY_SUPPORTED":
            suite_evaluation["recommendations"] = [
                "Refine dialectical process implementation",
                "Investigate factors affecting inconsistent results",
                "Expand test coverage to identify optimal use cases",
                "Consider conditional dialectical deployment"
            ]
        else:
            suite_evaluation["recommendations"] = [
                "Do not proceed with current dialectical approach",
                "Investigate fundamental issues with implementation",
                "Consider alternative multi-agent architectures",
                "Return to single-agent optimization"
            ]
        
        return suite_evaluation
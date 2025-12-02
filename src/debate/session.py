"""
DebateSession - Debate Session Management for Dialectical Testing

This module manages debate sessions, tracking agent interactions, turn management,
conflict identification, and synthesis quality assessment for the dialectical
testing framework.
"""

import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agents.utils import AgentResponse


class TurnType(Enum):
    """Types of turns in a debate session."""
    WORKER_RESPONSE = "worker_response"
    REVIEWER_CRITIQUE = "reviewer_critique"
    REVIEWER_SYNTHESIS = "reviewer_synthesis"
    REVIEWER_COMPARISON = "reviewer_comparison"


@dataclass
class DebateTurn:
    """
    Represents a single turn in a debate session.
    """
    turn_id: int
    agent_id: str
    turn_type: TurnType
    response: AgentResponse
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictAnalysis:
    """
    Analysis of conflicts between agent responses.
    """
    conflicts_detected: bool
    conflict_areas: List[str] = field(default_factory=list)
    agreement_areas: List[str] = field(default_factory=list)
    conflict_severity: float = 0.0  # 0-1 scale
    resolution_quality: float = 0.0  # 0-1 scale
    analysis_details: str = ""


class DebateSession:
    """
    Manages a single debate session between agents.
    
    Tracks agent interactions, identifies conflicts, and assesses
    synthesis quality for dialectical validation.
    """
    
    def __init__(self, question: str, session_id: Optional[str] = None):
        """
        Initialize a new debate session.
        
        Args:
            question: The question being debated
            session_id: Optional unique identifier for the session
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.question = question
        self.turns: List[DebateTurn] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        # Analysis results
        self.conflict_analysis: Optional[ConflictAnalysis] = None
        self.synthesis_effectiveness: float = 0.0
        self.conflicts_identified: bool = False
        
        # Metadata
        self.metadata: Dict[str, Any] = {
            'question_length': len(question),
            'question_type': self._classify_question(question)
        }
    
    def _classify_question(self, question: str) -> str:
        """
        Classify the type of question being asked.
        
        Args:
            question: Question to classify
            
        Returns:
            Question type category
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'definition']):
            return 'definitional'
        elif any(word in question_lower for word in ['why', 'how', 'explain']):
            return 'explanatory'
        elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
            return 'comparative'
        elif any(word in question_lower for word in ['should', 'better', 'best', 'evaluate']):
            return 'evaluative'
        elif question_lower.endswith('?'):
            return 'interrogative'
        else:
            return 'analytical'
    
    def add_turn(self, agent_id: str, response: AgentResponse, 
                 turn_type: TurnType = TurnType.WORKER_RESPONSE, 
                 metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a turn to the debate session.
        
        Args:
            agent_id: ID of the agent making the turn
            response: Agent's response
            turn_type: Type of turn being made
            metadata: Optional additional metadata
            
        Returns:
            Turn ID for the added turn
        """
        turn_id = len(self.turns)
        
        turn = DebateTurn(
            turn_id=turn_id,
            agent_id=agent_id,
            turn_type=turn_type,
            response=response,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.turns.append(turn)
        return turn_id
    
    def get_turns_by_agent(self, agent_id: str) -> List[DebateTurn]:
        """
        Get all turns made by a specific agent.
        
        Args:
            agent_id: Agent ID to filter by
            
        Returns:
            List of turns made by the agent
        """
        return [turn for turn in self.turns if turn.agent_id == agent_id]
    
    def get_turns_by_type(self, turn_type: TurnType) -> List[DebateTurn]:
        """
        Get all turns of a specific type.
        
        Args:
            turn_type: Type of turn to filter by
            
        Returns:
            List of turns of the specified type
        """
        return [turn for turn in self.turns if turn.turn_type == turn_type]
    
    def get_worker_responses(self) -> List[DebateTurn]:
        """
        Get all worker response turns.
        
        Returns:
            List of worker response turns
        """
        return self.get_turns_by_type(TurnType.WORKER_RESPONSE)
    
    def analyze_debate(self, worker_responses: List[AgentResponse], 
                      synthesis_response: AgentResponse,
                      reviewer_agent=None) -> ConflictAnalysis:
        """
        Analyze the debate to identify conflicts and assess synthesis quality.
        
        Args:
            worker_responses: List of worker agent responses
            synthesis_response: Reviewer's synthesis response
            reviewer_agent: Optional reviewer agent for detailed analysis
            
        Returns:
            ConflictAnalysis with detailed conflict assessment
        """
        # Basic conflict detection using response comparison
        conflicts_detected = False
        conflict_areas = []
        agreement_areas = []
        
        if len(worker_responses) >= 2:
            # Simple heuristic conflict detection
            response1_content = worker_responses[0].content.lower()
            response2_content = worker_responses[1].content.lower()
            
            # Look for contradictory keywords
            contradictory_pairs = [
                (['yes', 'true', 'correct', 'is'], ['no', 'false', 'incorrect', 'not']),
                (['increase', 'more', 'higher'], ['decrease', 'less', 'lower']),
                (['positive', 'beneficial'], ['negative', 'harmful']),
                (['agree', 'support'], ['disagree', 'oppose']),
            ]
            
            for positive_terms, negative_terms in contradictory_pairs:
                response1_has_positive = any(term in response1_content for term in positive_terms)
                response1_has_negative = any(term in response1_content for term in negative_terms)
                response2_has_positive = any(term in response2_content for term in positive_terms)
                response2_has_negative = any(term in response2_content for term in negative_terms)
                
                # Check for contradictions
                if ((response1_has_positive and response2_has_negative) or 
                    (response1_has_negative and response2_has_positive)):
                    conflicts_detected = True
                    conflict_areas.append(f"Contradiction in {'/'.join(positive_terms)} vs {'/'.join(negative_terms)}")
                
                # Check for agreements
                if ((response1_has_positive and response2_has_positive) or 
                    (response1_has_negative and response2_has_negative)):
                    agreement_areas.append(f"Agreement on {'/'.join(positive_terms) if response1_has_positive else '/'.join(negative_terms)}")
        
        # Assess conflict severity (simple heuristic)
        conflict_severity = min(len(conflict_areas) / 5.0, 1.0)  # 0-1 scale
        
        # Assess synthesis quality based on response characteristics
        synthesis_effectiveness = self._assess_synthesis_effectiveness(
            worker_responses, synthesis_response
        )
        
        # Store analysis results
        self.conflict_analysis = ConflictAnalysis(
            conflicts_detected=conflicts_detected,
            conflict_areas=conflict_areas,
            agreement_areas=agreement_areas,
            conflict_severity=conflict_severity,
            resolution_quality=synthesis_effectiveness,
            analysis_details=f"Analyzed {len(worker_responses)} worker responses for conflicts and synthesis quality"
        )
        
        self.conflicts_identified = conflicts_detected
        self.synthesis_effectiveness = synthesis_effectiveness
        
        return self.conflict_analysis
    
    def _assess_synthesis_effectiveness(self, worker_responses: List[AgentResponse],
                                      synthesis_response: AgentResponse) -> float:
        """
        Assess how effectively the synthesis combines worker responses.
        
        Args:
            worker_responses: Original worker responses
            synthesis_response: Synthesis response to evaluate
            
        Returns:
            Effectiveness score (0-1)
        """
        if not worker_responses or not synthesis_response:
            return 0.0
        
        synthesis_content = synthesis_response.content.lower()
        effectiveness_indicators = 0
        max_indicators = 6
        
        # Check for synthesis quality indicators
        
        # 1. Length and comprehensiveness
        avg_worker_length = sum(len(r.content) for r in worker_responses) / len(worker_responses)
        if len(synthesis_response.content) >= avg_worker_length * 0.8:
            effectiveness_indicators += 1
        
        # 2. Integration keywords
        integration_terms = ['combine', 'both', 'however', 'while', 'although', 'together', 'synthesis', 'integrate']
        if any(term in synthesis_content for term in integration_terms):
            effectiveness_indicators += 1
        
        # 3. Reference to multiple perspectives
        perspective_terms = ['perspective', 'view', 'approach', 'response', 'argument']
        if any(term in synthesis_content for term in perspective_terms):
            effectiveness_indicators += 1
        
        # 4. Balanced analysis
        balance_terms = ['on one hand', 'on the other hand', 'conversely', 'alternatively', 'in contrast']
        if any(term in synthesis_content for term in balance_terms):
            effectiveness_indicators += 1
        
        # 5. Conclusion or resolution
        conclusion_terms = ['conclusion', 'therefore', 'overall', 'in summary', 'ultimately']
        if any(term in synthesis_content for term in conclusion_terms):
            effectiveness_indicators += 1
        
        # 6. Higher confidence or quality indicators
        if (synthesis_response.confidence and 
            any(r.confidence for r in worker_responses) and
            synthesis_response.confidence >= max(r.confidence for r in worker_responses if r.confidence)):
            effectiveness_indicators += 1
        
        return effectiveness_indicators / max_indicators
    
    def get_session_duration(self) -> Optional[float]:
        """
        Get the duration of the debate session in seconds.
        
        Returns:
            Duration in seconds, or None if session not ended
        """
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        else:
            return (datetime.now() - self.start_time).total_seconds()
    
    def end_session(self):
        """Mark the debate session as ended."""
        self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the debate session.
        
        Returns:
            Dictionary with session summary information
        """
        worker_turns = len(self.get_worker_responses())
        total_turns = len(self.turns)
        
        return {
            'session_id': self.session_id,
            'question': self.question,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.get_session_duration(),
            'total_turns': total_turns,
            'worker_turns': worker_turns,
            'conflicts_identified': self.conflicts_identified,
            'synthesis_effectiveness': self.synthesis_effectiveness,
            'question_type': self.metadata.get('question_type'),
            'agents_participated': list(set(turn.agent_id for turn in self.turns)),
            'conflict_areas': (
                self.conflict_analysis.conflict_areas 
                if self.conflict_analysis else []
            ),
            'agreement_areas': (
                self.conflict_analysis.agreement_areas 
                if self.conflict_analysis else []
            )
        }
    
    def export_transcript(self) -> str:
        """
        Export the debate session as a readable transcript.
        
        Returns:
            Formatted transcript string
        """
        transcript = f"""
Debate Session Transcript
========================
Session ID: {self.session_id}
Question: {self.question}
Start Time: {self.start_time}
Duration: {self.get_session_duration():.2f} seconds

"""
        
        for turn in self.turns:
            transcript += f"""
--- Turn {turn.turn_id} ({turn.turn_type.value}) ---
Agent: {turn.agent_id}
Time: {turn.timestamp}

Content:
{turn.response.content}

Reasoning:
{turn.response.reasoning or "Not provided"}

Confidence: {turn.response.confidence or "Not specified"}

"""
        
        if self.conflict_analysis:
            transcript += f"""
--- Conflict Analysis ---
Conflicts Detected: {self.conflict_analysis.conflicts_detected}
Conflict Areas: {', '.join(self.conflict_analysis.conflict_areas) or "None"}
Agreement Areas: {', '.join(self.conflict_analysis.agreement_areas) or "None"}
Synthesis Effectiveness: {self.synthesis_effectiveness:.2f}

"""
        
        return transcript
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the debate session to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the session
        """
        return {
            'session_id': self.session_id,
            'question': self.question,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'turns': [
                {
                    'turn_id': turn.turn_id,
                    'agent_id': turn.agent_id,
                    'turn_type': turn.turn_type.value,
                    'response': {
                        'content': turn.response.content,
                        'reasoning': turn.response.reasoning,
                        'confidence': turn.response.confidence,
                        'sources': turn.response.sources,
                        'metadata': turn.response.metadata
                    },
                    'timestamp': turn.timestamp.isoformat(),
                    'metadata': turn.metadata
                }
                for turn in self.turns
            ],
            'conflict_analysis': (
                {
                    'conflicts_detected': self.conflict_analysis.conflicts_detected,
                    'conflict_areas': self.conflict_analysis.conflict_areas,
                    'agreement_areas': self.conflict_analysis.agreement_areas,
                    'conflict_severity': self.conflict_analysis.conflict_severity,
                    'resolution_quality': self.conflict_analysis.resolution_quality,
                    'analysis_details': self.conflict_analysis.analysis_details
                }
                if self.conflict_analysis else None
            ),
            'synthesis_effectiveness': self.synthesis_effectiveness,
            'conflicts_identified': self.conflicts_identified,
            'metadata': self.metadata
        }
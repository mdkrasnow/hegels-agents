"""
Agent utilities for Hegels Agents Phase 0.5

Simple utilities for agent response formatting, basic logging, and data structures
to support minimal agent implementation for dialectical debate validation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
from pathlib import Path

from config.logging import get_logger, LogCategory


@dataclass
class AgentResponse:
    """Basic agent response data structure."""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'content': self.content,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'sources': self.sources or [],
            'metadata': self.metadata or {},
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DebateContext:
    """Context for dialectical debate between agents."""
    question: str
    worker_responses: List[AgentResponse]
    reviewer_synthesis: Optional[AgentResponse] = None
    round_number: int = 1
    debate_id: str = ""
    
    def __post_init__(self):
        """Generate debate ID if not provided."""
        if not self.debate_id:
            self.debate_id = f"debate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


class AgentLogger:
    """Simple logger for agent interactions."""
    
    def __init__(self, agent_name: str):
        """
        Initialize agent logger.
        
        Args:
            agent_name: Name of the agent for logging identification
        """
        self.agent_name = agent_name
        self.logger = get_logger(f"hegels_agents.{agent_name}")
        
        # Create agent-specific log directory
        log_dir = Path(__file__).parent.parent.parent / "logs" / "agents"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.agent_log_file = log_dir / f"{agent_name}.jsonl"
    
    def log_response(self, 
                    question: str, 
                    response: AgentResponse, 
                    context: Optional[Dict[str, Any]] = None):
        """
        Log an agent response to both structured logger and file.
        
        Args:
            question: The input question
            response: Agent's response
            context: Additional context information
        """
        # Log to structured logger
        self.logger.log_agent_response(
            agent_id=self.agent_name,
            message=f"Responding to: {question[:100]}...",
            response=response.content,
            confidence=response.confidence,
            sources=response.sources,
            reasoning_length=len(response.reasoning) if response.reasoning else 0
        )
        
        # Log to agent-specific file
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent': self.agent_name,
            'event_type': 'response',
            'question': question,
            'response': response.to_dict(),
            'context': context or {}
        }
        
        try:
            with open(self.agent_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write agent log: {e}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an agent error."""
        self.logger.log_exception(
            exception=error,
            context=f"{self.agent_name}: {context}",
            agent_id=self.agent_name
        )
    
    def log_debug(self, message: str, **kwargs):
        """Log debug information."""
        self.logger.debug(
            f"{self.agent_name}: {message}",
            category=LogCategory.AGENT,
            agent_id=self.agent_name,
            **kwargs
        )


def format_prompt_with_context(base_prompt: str, 
                              question: str, 
                              context: Optional[str] = None,
                              previous_responses: Optional[List[AgentResponse]] = None) -> str:
    """
    Format a prompt with question and optional context.
    
    Args:
        base_prompt: Base system prompt for the agent
        question: The question to answer
        context: Optional context information
        previous_responses: Previous agent responses for synthesis
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = [base_prompt, "", f"Question: {question}"]
    
    if context:
        prompt_parts.extend(["", f"Context: {context}"])
    
    if previous_responses:
        prompt_parts.extend(["", "Previous responses to consider:"])
        for i, resp in enumerate(previous_responses, 1):
            prompt_parts.extend([
                f"Response {i}:",
                f"Content: {resp.content}",
                f"Reasoning: {resp.reasoning or 'Not provided'}",
                ""
            ])
    
    prompt_parts.append("\nPlease provide your response:")
    
    return "\n".join(prompt_parts)


def simple_text_search(query: str, text_corpus: List[str], max_results: int = 3) -> List[str]:
    """
    Simple text search for basic retrieval capability.
    
    Args:
        query: Search query
        text_corpus: List of text documents to search
        max_results: Maximum number of results to return
        
    Returns:
        List of relevant text snippets
    """
    query_lower = query.lower()
    
    # Simple relevance scoring based on keyword matches
    scored_texts = []
    for text in text_corpus:
        text_lower = text.lower()
        score = 0
        
        # Count exact word matches
        query_words = query_lower.split()
        for word in query_words:
            score += text_lower.count(word)
        
        if score > 0:
            scored_texts.append((text, score))
    
    # Sort by score and return top results
    scored_texts.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scored_texts[:max_results]]


def validate_agent_response(response: AgentResponse) -> List[str]:
    """
    Validate an agent response for basic requirements.
    
    Args:
        response: Agent response to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not response.content or not response.content.strip():
        errors.append("Response content cannot be empty")
    
    if response.confidence is not None:
        if not (0.0 <= response.confidence <= 1.0):
            errors.append("Confidence must be between 0.0 and 1.0")
    
    if len(response.content) > 10000:
        errors.append("Response content is too long (max 10,000 characters)")
    
    return errors


# Module-level logger for utilities
_logger = get_logger("hegels_agents.utils")

def log_debate_interaction(context: DebateContext, event: str, details: Dict[str, Any]):
    """Log a debate interaction event."""
    _logger.log_debate_event(
        event_type=event,
        message=f"Debate {context.debate_id}: {event}",
        debate_id=context.debate_id,
        participants=["worker", "reviewer"],
        turn_index=context.round_number,
        **details
    )
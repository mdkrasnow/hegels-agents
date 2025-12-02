"""
BasicWorkerAgent for Hegels Agents Phase 0.5

Simple worker agent implementation that makes direct Gemini API calls
with a focus on proposing answers using available information.
"""

import google.genai as genai
from typing import Optional, List, Dict, Any

from config.settings import get_config
from agents.utils import AgentResponse, AgentLogger, format_prompt_with_context, simple_text_search


class BasicWorkerAgent:
    """
    Simple worker agent that proposes answers using direct Gemini API calls.
    
    This agent focuses on providing well-reasoned responses to questions,
    using basic retrieval capabilities when context is available.
    """
    
    SYSTEM_PROMPT = """You are a helpful and thorough research assistant focused on proposing well-reasoned answers to questions.

Your role is to:
1. Carefully analyze the question and any provided context
2. Use available information to construct a comprehensive answer
3. Provide clear reasoning for your conclusions
4. Acknowledge limitations or uncertainties in your knowledge
5. Be precise and factual in your responses

When answering:
- Start with a direct answer to the question
- Provide supporting reasoning and evidence
- If using context, cite specific relevant information
- If uncertain, explain what additional information would be helpful
- Structure your response clearly with main points and supporting details

Focus on being helpful, accurate, and thorough in your analysis."""

    def __init__(self, agent_id: str = "worker_agent"):
        """
        Initialize the BasicWorkerAgent.
        
        Args:
            agent_id: Unique identifier for this agent instance
        """
        self.agent_id = agent_id
        self.logger = AgentLogger(agent_id)
        
        # Configure Gemini API
        config = get_config()
        self.client = genai.Client(api_key=config.get_gemini_api_key())
        self.model_name = 'gemini-2.5-flash'
        
        # Simple text corpus for basic retrieval (can be extended)
        self.knowledge_base: List[str] = []
        
        self.logger.log_debug("BasicWorkerAgent initialized")
    
    def add_knowledge(self, texts: List[str]):
        """
        Add texts to the agent's knowledge base for retrieval.
        
        Args:
            texts: List of text documents to add
        """
        self.knowledge_base.extend(texts)
        self.logger.log_debug(f"Added {len(texts)} texts to knowledge base")
    
    def _retrieve_context(self, question: str, max_results: int = 3) -> Optional[str]:
        """
        Simple retrieval from knowledge base.
        
        Args:
            question: Question to find relevant context for
            max_results: Maximum number of results to retrieve
            
        Returns:
            Concatenated relevant context or None if no knowledge base
        """
        if not self.knowledge_base:
            return None
        
        relevant_texts = simple_text_search(question, self.knowledge_base, max_results)
        
        if relevant_texts:
            context = "\n\n".join(f"Context {i+1}: {text}" 
                                 for i, text in enumerate(relevant_texts))
            self.logger.log_debug(f"Retrieved {len(relevant_texts)} context snippets")
            return context
        
        return None
    
    def _make_gemini_call(self, prompt: str) -> str:
        """
        Make a direct call to Gemini API.
        
        Args:
            prompt: Full prompt to send to Gemini
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=2000,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            self.logger.log_error(e, "Gemini API call failed")
            raise
    
    def respond(self, question: str, external_context: Optional[str] = None) -> AgentResponse:
        """
        Generate a response to a question.
        
        Args:
            question: The question to answer
            external_context: Optional external context to include
            
        Returns:
            AgentResponse with content and reasoning
        """
        try:
            self.logger.log_debug(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant context from knowledge base
            retrieved_context = self._retrieve_context(question)
            
            # Combine contexts
            contexts = []
            if external_context:
                contexts.append(f"External context: {external_context}")
            if retrieved_context:
                contexts.append(f"Retrieved information:\n{retrieved_context}")
            
            combined_context = "\n\n".join(contexts) if contexts else None
            
            # Format the prompt
            full_prompt = format_prompt_with_context(
                base_prompt=self.SYSTEM_PROMPT,
                question=question,
                context=combined_context
            )
            
            # Make API call
            response_text = self._make_gemini_call(full_prompt)
            
            # Parse response to extract reasoning (simple heuristic)
            content_lines = response_text.split('\n')
            
            # Try to identify reasoning sections
            reasoning_keywords = ['reasoning:', 'because', 'therefore', 'analysis:', 'rationale:']
            reasoning_parts = []
            content_parts = []
            
            current_is_reasoning = False
            for line in content_lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in reasoning_keywords):
                    current_is_reasoning = True
                    reasoning_parts.append(line)
                elif current_is_reasoning and line.strip():
                    reasoning_parts.append(line)
                elif line.strip():
                    content_parts.append(line)
            
            # If no clear reasoning section found, use the full response as content
            if not reasoning_parts:
                final_content = response_text
                final_reasoning = None
            else:
                final_content = '\n'.join(content_parts) if content_parts else response_text
                final_reasoning = '\n'.join(reasoning_parts)
            
            # Create response object
            agent_response = AgentResponse(
                content=final_content.strip(),
                reasoning=final_reasoning.strip() if final_reasoning else None,
                confidence=0.8,  # Default confidence for basic implementation
                sources=[self.model_name],
                metadata={
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'has_external_context': bool(external_context),
                    'has_retrieved_context': bool(retrieved_context),
                    'prompt_length': len(full_prompt)
                }
            )
            
            # Log the response
            self.logger.log_response(question, agent_response, {
                'context_provided': bool(combined_context),
                'response_length': len(response_text)
            })
            
            return agent_response
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to generate response for question: {question[:50]}...")
            
            # Return error response
            return AgentResponse(
                content=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                reasoning="Error occurred during response generation",
                confidence=0.0,
                sources=[],
                metadata={
                    'error': str(e), 
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'has_external_context': bool(external_context),
                    'has_retrieved_context': False,
                    'prompt_length': 0
                }
            )
    
    def batch_respond(self, questions: List[str], 
                     contexts: Optional[List[str]] = None) -> List[AgentResponse]:
        """
        Respond to multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            contexts: Optional list of contexts (must match questions length)
            
        Returns:
            List of agent responses
        """
        if contexts and len(contexts) != len(questions):
            raise ValueError("Contexts list must match questions list length")
        
        responses = []
        for i, question in enumerate(questions):
            context = contexts[i] if contexts else None
            response = self.respond(question, context)
            responses.append(response)
        
        self.logger.log_debug(f"Completed batch processing of {len(questions)} questions")
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the agent.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_id': self.agent_id,
            'model': self.model_name,
            'knowledge_base_size': len(self.knowledge_base),
            'system_prompt_length': len(self.SYSTEM_PROMPT)
        }
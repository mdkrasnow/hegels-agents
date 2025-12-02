"""
BasicReviewerAgent for Hegels Agents Phase 0.5

Simple reviewer agent implementation that critiques and synthesizes responses
from worker agents to enable dialectical debate validation.
"""

import google.genai as genai
from typing import List, Optional, Dict, Any

from config.settings import get_config
from agents.utils import AgentResponse, AgentLogger, format_prompt_with_context


class BasicReviewerAgent:
    """
    Simple reviewer agent that critiques worker responses and produces synthesis.
    
    This agent focuses on comparing multiple responses, identifying conflicts,
    and producing improved synthesized answers through dialectical analysis.
    """
    
    CRITIQUE_PROMPT = """You are a critical reviewer and synthesizer focused on analyzing and improving responses to questions.

Your role is to:
1. Carefully analyze each provided response for strengths and weaknesses
2. Identify areas of agreement and conflict between responses
3. Evaluate the reasoning and evidence presented
4. Synthesize insights from multiple perspectives into an improved answer
5. Highlight any limitations, gaps, or areas needing further investigation

When reviewing responses:
- Assess the logical consistency and evidence quality of each response
- Identify the strongest points from each perspective
- Note any contradictions or disagreements between responses
- Consider which response provides better reasoning or evidence
- Look for complementary insights that can be combined

Your synthesis should:
- Integrate the best insights from all responses
- Resolve contradictions where possible or acknowledge irreconcilable differences
- Provide a more comprehensive answer than any individual response
- Maintain intellectual honesty about limitations and uncertainties
- Structure the final answer clearly with supporting reasoning

Be thorough, fair, and focus on improving the overall quality of the answer through critical analysis."""

    SYNTHESIS_PROMPT = """You are a dialectical synthesizer creating improved answers by combining multiple perspectives.

Your role is to:
1. Identify the strongest elements from each provided response
2. Resolve apparent contradictions through deeper analysis
3. Combine complementary insights into a more complete answer
4. Acknowledge genuine disagreements and explain their sources
5. Produce a final synthesis that represents the best collective understanding

When synthesizing:
- Start with areas of agreement as a foundation
- Address contradictions by examining underlying assumptions
- Integrate different types of evidence or reasoning approaches
- Build upon the strongest arguments from each response
- Maintain nuance and avoid oversimplification

Your final synthesis should be more comprehensive, accurate, and well-reasoned than any individual input response."""

    def __init__(self, agent_id: str = "reviewer_agent"):
        """
        Initialize the BasicReviewerAgent.
        
        Args:
            agent_id: Unique identifier for this agent instance
        """
        self.agent_id = agent_id
        self.logger = AgentLogger(agent_id)
        
        # Configure Gemini API
        config = get_config()
        self.client = genai.Client(api_key=config.get_gemini_api_key())
        self.model_name = 'gemini-1.5-flash'
        
        self.logger.log_debug("BasicReviewerAgent initialized")
    
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
                    max_output_tokens=2500,
                    temperature=0.6,  # Slightly lower temperature for more focused analysis
                )
            )
            return response.text
        except Exception as e:
            self.logger.log_error(e, "Gemini API call failed")
            raise
    
    def critique_response(self, question: str, response: AgentResponse) -> AgentResponse:
        """
        Critique a single worker response.
        
        Args:
            question: Original question being answered
            response: Worker agent response to critique
            
        Returns:
            AgentResponse containing the critique
        """
        try:
            self.logger.log_debug(f"Critiquing response to: {question[:100]}...")
            
            # Format critique prompt
            critique_context = f"""Original Question: {question}

Response to Critique:
Content: {response.content}
Reasoning: {response.reasoning or 'Not provided'}
Confidence: {response.confidence or 'Not specified'}

Please provide a thorough critique of this response."""
            
            full_prompt = format_prompt_with_context(
                base_prompt=self.CRITIQUE_PROMPT,
                question="Critique the following response:",
                context=critique_context
            )
            
            # Make API call
            critique_text = self._make_gemini_call(full_prompt)
            
            # Create critique response
            critique_response = AgentResponse(
                content=critique_text,
                reasoning=f"Critical analysis of response from {response.metadata.get('agent_id', 'unknown')}",
                confidence=0.7,
                sources=[self.model_name],
                metadata={
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'critique_type': 'single_response',
                    'original_response_id': response.metadata.get('agent_id'),
                    'prompt_length': len(full_prompt)
                }
            )
            
            self.logger.log_response(f"Critique of: {question}", critique_response, {
                'critique_type': 'single_response',
                'original_confidence': response.confidence
            })
            
            return critique_response
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to critique response for question: {question[:50]}...")
            
            return AgentResponse(
                content=f"I apologize, but I encountered an error while critiquing the response: {str(e)}",
                reasoning="Error occurred during critique generation",
                confidence=0.0,
                sources=[],
                metadata={
                    'error': str(e), 
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'critique_type': 'error',
                    'original_response_id': response.metadata.get('agent_id', 'unknown') if response.metadata else 'unknown',
                    'prompt_length': 0
                }
            )
    
    def synthesize_responses(self, question: str, responses: List[AgentResponse]) -> AgentResponse:
        """
        Synthesize multiple worker responses into an improved answer.
        
        Args:
            question: Original question being answered
            responses: List of worker responses to synthesize
            
        Returns:
            AgentResponse containing the synthesis
        """
        if not responses:
            raise ValueError("Cannot synthesize empty list of responses")
        
        try:
            self.logger.log_debug(f"Synthesizing {len(responses)} responses to: {question[:100]}...")
            
            # Format synthesis context
            synthesis_context = f"Original Question: {question}\n\n"
            
            for i, response in enumerate(responses, 1):
                synthesis_context += f"Response {i}:\n"
                synthesis_context += f"Content: {response.content}\n"
                synthesis_context += f"Reasoning: {response.reasoning or 'Not provided'}\n"
                synthesis_context += f"Confidence: {response.confidence or 'Not specified'}\n"
                synthesis_context += f"Sources: {', '.join(response.sources or [])}\n\n"
            
            synthesis_context += "Please synthesize these responses into an improved, comprehensive answer."
            
            full_prompt = format_prompt_with_context(
                base_prompt=self.SYNTHESIS_PROMPT,
                question="Synthesize the following responses:",
                context=synthesis_context
            )
            
            # Make API call
            synthesis_text = self._make_gemini_call(full_prompt)
            
            # Calculate average confidence (simple approach)
            confidences = [r.confidence for r in responses if r.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            # Collect all sources
            all_sources = set()
            for response in responses:
                if response.sources:
                    all_sources.update(response.sources)
            all_sources.add(self.model_name)  # Add synthesis model
            
            # Create synthesis response
            synthesis_response = AgentResponse(
                content=synthesis_text,
                reasoning=f"Dialectical synthesis of {len(responses)} worker responses",
                confidence=min(avg_confidence + 0.1, 1.0) if avg_confidence else 0.8,  # Slight boost for synthesis
                sources=list(all_sources),
                metadata={
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'synthesis_type': 'multi_response',
                    'num_responses_synthesized': len(responses),
                    'input_agent_ids': [r.metadata.get('agent_id') for r in responses],
                    'prompt_length': len(full_prompt),
                    'avg_input_confidence': avg_confidence
                }
            )
            
            self.logger.log_response(f"Synthesis for: {question}", synthesis_response, {
                'synthesis_type': 'multi_response',
                'num_inputs': len(responses),
                'avg_input_confidence': avg_confidence
            })
            
            return synthesis_response
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to synthesize responses for question: {question[:50]}...")
            
            return AgentResponse(
                content=f"I apologize, but I encountered an error while synthesizing responses: {str(e)}",
                reasoning="Error occurred during synthesis generation",
                confidence=0.0,
                sources=[],
                metadata={
                    'error': str(e), 
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'synthesis_type': 'error',
                    'num_responses_synthesized': len(responses) if responses else 0,
                    'input_agent_ids': [r.metadata.get('agent_id', 'unknown') if r.metadata else 'unknown' for r in responses] if responses else [],
                    'prompt_length': 0,
                    'avg_input_confidence': None
                }
            )
    
    def compare_responses(self, question: str, 
                         response1: AgentResponse, 
                         response2: AgentResponse) -> AgentResponse:
        """
        Compare two responses and identify key differences/agreements.
        
        Args:
            question: Original question being answered
            response1: First response to compare
            response2: Second response to compare
            
        Returns:
            AgentResponse containing the comparison analysis
        """
        try:
            self.logger.log_debug(f"Comparing two responses to: {question[:100]}...")
            
            # Format comparison context
            comparison_context = f"""Original Question: {question}

Response A:
Content: {response1.content}
Reasoning: {response1.reasoning or 'Not provided'}
Confidence: {response1.confidence or 'Not specified'}

Response B:
Content: {response2.content}
Reasoning: {response2.reasoning or 'Not provided'}
Confidence: {response2.confidence or 'Not specified'}

Please provide a detailed comparison identifying agreements, differences, strengths, and weaknesses of each response."""
            
            full_prompt = format_prompt_with_context(
                base_prompt=self.CRITIQUE_PROMPT,
                question="Compare and analyze these two responses:",
                context=comparison_context
            )
            
            # Make API call
            comparison_text = self._make_gemini_call(full_prompt)
            
            # Create comparison response
            comparison_response = AgentResponse(
                content=comparison_text,
                reasoning="Comparative analysis of two responses",
                confidence=0.8,
                sources=[self.model_name],
                metadata={
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'analysis_type': 'comparison',
                    'compared_agents': [
                        response1.metadata.get('agent_id'),
                        response2.metadata.get('agent_id')
                    ],
                    'prompt_length': len(full_prompt)
                }
            )
            
            self.logger.log_response(f"Comparison for: {question}", comparison_response, {
                'analysis_type': 'comparison',
                'compared_confidences': [response1.confidence, response2.confidence]
            })
            
            return comparison_response
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to compare responses for question: {question[:50]}...")
            
            return AgentResponse(
                content=f"I apologize, but I encountered an error while comparing responses: {str(e)}",
                reasoning="Error occurred during comparison generation",
                confidence=0.0,
                sources=[],
                metadata={
                    'error': str(e), 
                    'agent_id': self.agent_id,
                    'model': self.model_name,
                    'analysis_type': 'error',
                    'compared_agents': [
                        response1.metadata.get('agent_id', 'unknown') if response1.metadata else 'unknown',
                        response2.metadata.get('agent_id', 'unknown') if response2.metadata else 'unknown'
                    ],
                    'prompt_length': 0
                }
            )
    
    def review_and_synthesize(self, question: str, responses: List[AgentResponse]) -> Dict[str, AgentResponse]:
        """
        Perform both critique and synthesis of responses.
        
        Args:
            question: Original question being answered
            responses: List of worker responses to review
            
        Returns:
            Dictionary with 'critiques' (list) and 'synthesis' (AgentResponse)
        """
        self.logger.log_debug(f"Full review process for: {question[:100]}...")
        
        # Generate critiques for each response
        critiques = []
        for i, response in enumerate(responses):
            critique = self.critique_response(question, response)
            critiques.append(critique)
            self.logger.log_debug(f"Completed critique {i+1}/{len(responses)}")
        
        # Generate synthesis
        synthesis = self.synthesize_responses(question, responses)
        
        self.logger.log_debug(f"Completed full review process with {len(critiques)} critiques and 1 synthesis")
        
        return {
            'critiques': critiques,
            'synthesis': synthesis
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the agent.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_id': self.agent_id,
            'model': self.model_name,
            'critique_prompt_length': len(self.CRITIQUE_PROMPT),
            'synthesis_prompt_length': len(self.SYNTHESIS_PROMPT)
        }
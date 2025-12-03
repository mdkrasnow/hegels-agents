"""
ReflectionOptimizer Implementation

Uses LLM-based reflection to generate targeted prompt improvements when performance is poor.
This is the core optimizer that enables online learning in the Hegel's Agents training system.
"""

import json
import re
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

import google.genai as genai

from src.config.settings import get_config
from src.training.data_structures import PromptProfile, RolePrompt
from src.training.optimizers.base import PromptOptimizer, OptimizationResult
from src.agents.utils import AgentLogger, format_prompt_with_context


class EditInstruction:
    """Represents a single edit instruction for a prompt."""
    
    def __init__(
        self,
        role: str,
        edit_type: str,
        target: str,
        replacement: str,
        reasoning: str,
        confidence: float = 0.8
    ):
        """
        Initialize edit instruction.
        
        Args:
            role: Role name (e.g., 'worker', 'reviewer')
            edit_type: Type of edit ('replace', 'append', 'prepend', 'insert')
            target: Text to find/target for edit
            replacement: New text to use
            reasoning: Explanation of why this edit improves the prompt
            confidence: Confidence in this edit (0.0 to 1.0)
        """
        self.role = role
        self.edit_type = edit_type
        self.target = target
        self.replacement = replacement
        self.reasoning = reasoning
        self.confidence = confidence
        self.edit_id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "edit_id": self.edit_id,
            "role": self.role,
            "edit_type": self.edit_type,
            "target": self.target,
            "replacement": self.replacement,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }
    
    def validate(self) -> List[str]:
        """Validate edit instruction."""
        errors = []
        
        if not self.role:
            errors.append("Role cannot be empty")
        
        if self.edit_type not in ['replace', 'append', 'prepend', 'insert']:
            errors.append(f"Invalid edit_type: {self.edit_type}")
        
        if not self.replacement:
            errors.append("Replacement text cannot be empty")
        
        if not (0.0 <= self.confidence <= 1.0):
            errors.append("Confidence must be between 0.0 and 1.0")
        
        if len(self.replacement) > 1000:
            errors.append("Replacement text too long (max 1000 chars)")
        
        return errors


class ReflectionOptimizer(PromptOptimizer):
    """
    LLM-based reflection optimizer for prompt improvement.
    
    This optimizer uses the LLM itself to analyze poor performance and suggest
    targeted improvements to prompts. It focuses on generating incremental,
    well-reasoned edits rather than wholesale prompt replacement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ReflectionOptimizer.
        
        Args:
            config: Optional configuration dictionary
        """
        # Configuration
        self.config = config or {}
        self.no_change_threshold = self.config.get('no_change_threshold', 0.8)
        self.max_edit_length = self.config.get('max_edit_length', 500)
        self.max_edits_per_prompt = self.config.get('max_edits_per_prompt', 3)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        
        # Initialize Gemini client using existing patterns
        app_config = get_config()
        self.client = genai.Client(api_key=app_config.get_gemini_api_key())
        self.model_name = 'gemini-2.0-flash-exp'
        
        # Logging
        self.logger = AgentLogger("reflection_optimizer")
        
        # Performance tracking
        self.optimization_count = 0
        self.successful_optimizations = 0
        self.total_improvement = 0.0
        self.optimization_history = []
        
        self.logger.log_debug("ReflectionOptimizer initialized")
    
    def update_profile(
        self, 
        profile: PromptProfile,
        query: str,
        answer: str,
        gold_answer: Optional[str],
        reward: float,
        trace: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptProfile:
        """
        Update a prompt profile using LLM-based reflection.
        
        Args:
            profile: Current prompt profile to optimize
            query: The input query that was processed
            answer: The generated answer from current profile
            gold_answer: Optional reference answer
            reward: Performance reward score (0.0 to 1.0)
            trace: Detailed execution trace
            metadata: Additional context
            
        Returns:
            Optimized PromptProfile (may be unchanged if no improvement identified)
        """
        try:
            self.optimization_count += 1
            start_time = time.time()
            
            self.logger.log_debug(
                f"Starting optimization for profile {profile.profile_id}, reward: {reward:.3f}"
            )
            
            # Check if optimization is needed
            if not self.should_optimize(reward, self.no_change_threshold):
                self.logger.log_debug(f"Reward {reward:.3f} above threshold {self.no_change_threshold}, skipping optimization")
                return profile
            
            # Generate reflection context
            reflection_context = self._build_reflection_context(
                profile, query, answer, gold_answer, reward, trace, metadata
            )
            
            # Get edit suggestions from LLM
            edit_suggestions = self._generate_edit_suggestions(reflection_context)
            
            if not edit_suggestions:
                self.logger.log_debug("No edit suggestions generated")
                return profile
            
            # Apply edits to create optimized profile
            optimized_profile = self._apply_edits(profile, edit_suggestions)
            
            # Update metadata
            optimization_metadata = {
                'optimization_strategy': 'reflection',
                'parent_profile_id': profile.profile_id,
                'parent_reward': reward,
                'optimization_timestamp': datetime.utcnow().isoformat(),
                'edit_count': len(edit_suggestions),
                'original_query': query,
                'expected_improvement': self._estimate_improvement(edit_suggestions)
            }
            
            optimized_profile.metadata.update(optimization_metadata)
            
            # Track performance
            optimization_time = time.time() - start_time
            self.optimization_history.append({
                'profile_id': profile.profile_id,
                'original_reward': reward,
                'edit_count': len(edit_suggestions),
                'optimization_time': optimization_time,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            self.successful_optimizations += 1
            
            self.logger.log_debug(
                f"Optimization completed in {optimization_time:.2f}s, "
                f"applied {len(edit_suggestions)} edits"
            )
            
            return optimized_profile
            
        except Exception as e:
            self.logger.log_error(e, f"Failed to optimize profile {profile.profile_id}")
            return profile
    
    def _build_reflection_context(
        self,
        profile: PromptProfile,
        query: str,
        answer: str,
        gold_answer: Optional[str],
        reward: float,
        trace: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Build the reflection context prompt for the LLM."""
        
        # Extract role prompts
        role_prompts = {}
        for role, role_prompt in profile.role_prompts.items():
            role_prompts[role] = role_prompt.prompt_text
        
        # Build context sections
        context_sections = [
            "PERFORMANCE ANALYSIS CONTEXT",
            "=" * 40,
            "",
            f"Question: {query}",
            f"Generated Answer: {answer}",
            f"Performance Score: {reward:.3f} (on 0.0-1.0 scale)",
            ""
        ]
        
        if gold_answer:
            context_sections.extend([
                f"Reference Answer: {gold_answer}",
                ""
            ])
        
        # Add current prompts
        context_sections.extend([
            "CURRENT PROMPT CONFIGURATION",
            "=" * 40,
            ""
        ])
        
        for role, prompt_text in role_prompts.items():
            context_sections.extend([
                f"{role.upper()} PROMPT:",
                prompt_text,
                ""
            ])
        
        # Add execution trace information
        if trace:
            context_sections.extend([
                "EXECUTION DETAILS",
                "=" * 40,
                ""
            ])
            
            worker_responses = trace.get('worker_responses', [])
            if worker_responses:
                context_sections.append(f"Worker responses generated: {len(worker_responses)}")
                for i, resp in enumerate(worker_responses[:2]):  # Show first 2
                    if hasattr(resp, 'content'):
                        context_sections.append(f"Worker {i+1}: {resp.content[:200]}...")
            
            synthesis_response = trace.get('synthesis_response')
            if synthesis_response and hasattr(synthesis_response, 'content'):
                context_sections.extend([
                    f"Final synthesis: {synthesis_response.content[:200]}...",
                    ""
                ])
        
        return "\n".join(context_sections)
    
    def _generate_edit_suggestions(self, reflection_context: str) -> List[EditInstruction]:
        """Generate edit suggestions using LLM reflection."""
        
        analysis_prompt = f"""You are an expert prompt engineer analyzing poor AI performance to suggest improvements.

{reflection_context}

TASK: Analyze the performance issue and suggest specific, targeted edits to improve the prompts.

GUIDELINES:
1. Focus on edits that address the specific performance gap
2. Suggest 1-3 high-impact edits maximum
3. Each edit should be small, focused, and generalizable
4. Preserve the overall prompt structure and style
5. Provide clear reasoning for each edit

RESPONSE FORMAT (JSON):
{{
    "analysis": "Brief analysis of the performance issue",
    "edits": [
        {{
            "role": "worker|reviewer",
            "edit_type": "replace|append|prepend",
            "target": "exact text to find/replace (for replace type)",
            "replacement": "new text to use",
            "reasoning": "why this edit will improve performance",
            "confidence": 0.85
        }}
    ],
    "expected_improvement": "Brief description of expected improvement"
}}

Focus on edits that improve reasoning quality, accuracy, and dialectical effectiveness."""
        
        try:
            # Make LLM call for edit suggestions
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=analysis_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=1500,
                    temperature=0.3,
                )
            )
            
            if not response or not response.text:
                self.logger.log_error(None, "Empty response from LLM for edit suggestions")
                return []
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean up response if it has markdown code blocks
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.log_error(e, f"Failed to parse LLM response as JSON: {response_text[:200]}...")
                return []
            
            # Convert to EditInstruction objects
            edit_instructions = []
            edits_data = parsed_response.get('edits', [])
            
            for edit_data in edits_data:
                try:
                    edit_instruction = EditInstruction(
                        role=edit_data.get('role', ''),
                        edit_type=edit_data.get('edit_type', ''),
                        target=edit_data.get('target', ''),
                        replacement=edit_data.get('replacement', ''),
                        reasoning=edit_data.get('reasoning', ''),
                        confidence=edit_data.get('confidence', 0.8)
                    )
                    
                    # Validate edit
                    validation_errors = edit_instruction.validate()
                    if validation_errors:
                        self.logger.log_debug(f"Invalid edit instruction: {validation_errors}")
                        continue
                    
                    # Apply length and confidence filters
                    if len(edit_instruction.replacement) > self.max_edit_length:
                        self.logger.log_debug(f"Edit replacement too long: {len(edit_instruction.replacement)}")
                        continue
                    
                    if edit_instruction.confidence < self.min_confidence_threshold:
                        self.logger.log_debug(f"Edit confidence too low: {edit_instruction.confidence}")
                        continue
                    
                    edit_instructions.append(edit_instruction)
                    
                except Exception as e:
                    self.logger.log_error(e, f"Failed to create edit instruction from: {edit_data}")
                    continue
            
            # Limit number of edits
            if len(edit_instructions) > self.max_edits_per_prompt:
                edit_instructions = edit_instructions[:self.max_edits_per_prompt]
                self.logger.log_debug(f"Limited edits to {self.max_edits_per_prompt}")
            
            self.logger.log_debug(f"Generated {len(edit_instructions)} valid edit instructions")
            
            return edit_instructions
            
        except Exception as e:
            self.logger.log_error(e, "Failed to generate edit suggestions")
            return []
    
    def _apply_edits(
        self, 
        original_profile: PromptProfile, 
        edit_instructions: List[EditInstruction]
    ) -> PromptProfile:
        """Apply edit instructions to create a new optimized profile."""
        
        # Create a copy of the original profile
        optimized_profile = PromptProfile(
            name=f"{original_profile.name}_optimized_{int(time.time())}",
            description=f"Optimized version of {original_profile.name}",
            version=original_profile.version,
            author=original_profile.author,
            tags=original_profile.tags.copy(),
            metadata=original_profile.metadata.copy()
        )
        
        # Copy and potentially modify each role prompt
        for role, original_role_prompt in original_profile.role_prompts.items():
            # Start with original prompt text
            modified_prompt_text = original_role_prompt.prompt_text
            applied_edits = []
            
            # Apply relevant edits for this role
            for edit in edit_instructions:
                if edit.role != role:
                    continue
                
                try:
                    if edit.edit_type == 'replace':
                        if edit.target in modified_prompt_text:
                            modified_prompt_text = modified_prompt_text.replace(
                                edit.target, edit.replacement, 1
                            )
                            applied_edits.append(edit.edit_id)
                        else:
                            self.logger.log_debug(f"Target text not found for replace edit: {edit.target[:50]}...")
                    
                    elif edit.edit_type == 'append':
                        modified_prompt_text += f"\n\n{edit.replacement}"
                        applied_edits.append(edit.edit_id)
                    
                    elif edit.edit_type == 'prepend':
                        modified_prompt_text = f"{edit.replacement}\n\n{modified_prompt_text}"
                        applied_edits.append(edit.edit_id)
                    
                    elif edit.edit_type == 'insert':
                        # For insert, we'll append for simplicity
                        # More sophisticated insert logic could be added later
                        modified_prompt_text += f"\n\n{edit.replacement}"
                        applied_edits.append(edit.edit_id)
                    
                except Exception as e:
                    self.logger.log_error(e, f"Failed to apply edit {edit.edit_id} to {role}")
                    continue
            
            # Create new role prompt
            optimized_role_prompt = RolePrompt(
                role=original_role_prompt.role,
                prompt_text=modified_prompt_text,
                description=f"Optimized {original_role_prompt.description or role} prompt",
                version=f"{original_role_prompt.version}_opt",
                author=original_role_prompt.author,
                metadata={
                    **original_role_prompt.metadata,
                    'optimization_applied': True,
                    'applied_edits': applied_edits,
                    'original_length': len(original_role_prompt.prompt_text),
                    'optimized_length': len(modified_prompt_text)
                }
            )
            
            optimized_profile.add_role_prompt(optimized_role_prompt)
        
        return optimized_profile
    
    def _estimate_improvement(self, edit_instructions: List[EditInstruction]) -> float:
        """Estimate expected improvement based on edit instructions."""
        if not edit_instructions:
            return 0.0
        
        # Simple heuristic: average of edit confidences, weighted by number of edits
        avg_confidence = sum(edit.confidence for edit in edit_instructions) / len(edit_instructions)
        edit_factor = min(len(edit_instructions) / 3.0, 1.0)  # More edits up to 3
        
        return avg_confidence * edit_factor * 0.2  # Max 20% expected improvement
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        if self.optimization_count == 0:
            return {
                "optimizations_performed": 0,
                "success_rate": 0.0,
                "average_improvement": 0.0
            }
        
        success_rate = self.successful_optimizations / self.optimization_count
        avg_improvement = self.total_improvement / max(self.successful_optimizations, 1)
        
        recent_optimizations = self.optimization_history[-10:] if self.optimization_history else []
        avg_optimization_time = sum(opt['optimization_time'] for opt in recent_optimizations) / max(len(recent_optimizations), 1)
        
        return {
            "optimizations_performed": self.optimization_count,
            "successful_optimizations": self.successful_optimizations,
            "success_rate": success_rate,
            "average_improvement": avg_improvement,
            "average_optimization_time_seconds": avg_optimization_time,
            "recent_optimizations": len(recent_optimizations)
        }
    
    def reset_stats(self):
        """Reset optimization statistics."""
        self.optimization_count = 0
        self.successful_optimizations = 0
        self.total_improvement = 0.0
        self.optimization_history.clear()
        self.logger.log_debug("Optimization statistics reset")
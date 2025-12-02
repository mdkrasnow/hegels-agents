"""
Core Data Structures for Hegel's Agents Training Layer

This module provides foundational data structures for the training system,
ensuring backward compatibility with existing AgentResponse while adding
comprehensive training capabilities.

Design Features:
- Backward compatible with existing AgentResponse structure
- Comprehensive JSON serialization/deserialization  
- UUID-based identifiers for all profiles
- Validation methods for data integrity
- Extensible architecture for future training features
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import existing AgentResponse for compatibility
from agents.utils import AgentResponse


@dataclass
class RolePrompt:
    """
    Individual prompt definition for a specific agent role.
    
    Represents a single prompt template that can be used to configure
    an agent's behavior for a particular role (e.g., worker, reviewer).
    """
    
    role: str
    prompt_text: str
    description: Optional[str] = None
    version: str = "1.0"
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values after construction."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Validate role name
        if not self.role or not isinstance(self.role, str):
            raise ValueError("Role must be a non-empty string")
        
        # Validate prompt text
        if not self.prompt_text or not isinstance(self.prompt_text, str):
            raise ValueError("Prompt text must be a non-empty string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'role': self.role,
            'prompt_text': self.prompt_text,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RolePrompt':
        """Create RolePrompt from dictionary."""
        # Parse datetime if present
        created_at = None
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            elif isinstance(data['created_at'], datetime):
                created_at = data['created_at']
        
        return cls(
            role=data['role'],
            prompt_text=data['prompt_text'],
            description=data.get('description'),
            version=data.get('version', '1.0'),
            author=data.get('author'),
            created_at=created_at,
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RolePrompt':
        """Create RolePrompt from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """
        Validate the RolePrompt structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.role or not self.role.strip():
            errors.append("Role cannot be empty")
        
        if not self.prompt_text or not self.prompt_text.strip():
            errors.append("Prompt text cannot be empty")
        
        if len(self.prompt_text) > 50000:  # Reasonable limit
            errors.append("Prompt text is too long (max 50,000 characters)")
        
        if self.version and not isinstance(self.version, str):
            errors.append("Version must be a string")
        
        return errors


@dataclass
class PromptProfile:
    """
    Collection of RolePrompts with metadata for training experiments.
    
    A PromptProfile represents a complete configuration for training,
    containing prompts for all necessary agent roles and experiment metadata.
    """
    
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: Optional[str] = None
    role_prompts: Dict[str, RolePrompt] = field(default_factory=dict)
    version: str = "1.0"
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values after construction."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Validate profile ID is a valid UUID
        try:
            uuid.UUID(self.profile_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid UUID for profile_id: {self.profile_id}")
        
        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Profile name must be a non-empty string")
    
    def add_role_prompt(self, role_prompt: RolePrompt) -> None:
        """Add a RolePrompt to this profile."""
        if not isinstance(role_prompt, RolePrompt):
            raise ValueError("Must provide a valid RolePrompt instance")
        
        # Validate the role prompt before adding
        validation_errors = role_prompt.validate()
        if validation_errors:
            raise ValueError(f"Invalid RolePrompt: {'; '.join(validation_errors)}")
        
        self.role_prompts[role_prompt.role] = role_prompt
    
    def remove_role_prompt(self, role: str) -> bool:
        """Remove a RolePrompt by role name. Returns True if removed."""
        return self.role_prompts.pop(role, None) is not None
    
    def get_role_prompt(self, role: str) -> Optional[RolePrompt]:
        """Get a RolePrompt by role name."""
        return self.role_prompts.get(role)
    
    def get_roles(self) -> List[str]:
        """Get list of available roles in this profile."""
        return list(self.role_prompts.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'profile_id': self.profile_id,
            'name': self.name,
            'description': self.description,
            'role_prompts': {role: prompt.to_dict() for role, prompt in self.role_prompts.items()},
            'version': self.version,
            'author': self.author,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptProfile':
        """Create PromptProfile from dictionary."""
        # Parse datetime if present
        created_at = None
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            elif isinstance(data['created_at'], datetime):
                created_at = data['created_at']
        
        # Parse role prompts
        role_prompts = {}
        for role, prompt_data in data.get('role_prompts', {}).items():
            role_prompts[role] = RolePrompt.from_dict(prompt_data)
        
        return cls(
            profile_id=data.get('profile_id', str(uuid.uuid4())),
            name=data['name'],
            description=data.get('description'),
            role_prompts=role_prompts,
            version=data.get('version', '1.0'),
            author=data.get('author'),
            created_at=created_at,
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PromptProfile':
        """Create PromptProfile from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """
        Validate the PromptProfile structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic fields
        if not self.profile_id:
            errors.append("Profile ID cannot be empty")
        
        try:
            uuid.UUID(self.profile_id)
        except (ValueError, TypeError):
            errors.append(f"Profile ID must be a valid UUID: {self.profile_id}")
        
        if not self.name or not self.name.strip():
            errors.append("Profile name cannot be empty")
        
        if len(self.name) > 200:
            errors.append("Profile name is too long (max 200 characters)")
        
        # Validate role prompts
        for role, prompt in self.role_prompts.items():
            if not isinstance(prompt, RolePrompt):
                errors.append(f"Invalid RolePrompt for role '{role}'")
                continue
            
            prompt_errors = prompt.validate()
            for error in prompt_errors:
                errors.append(f"Role '{role}': {error}")
        
        return errors
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save PromptProfile to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'PromptProfile':
        """Load PromptProfile from JSON file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


@dataclass
class TrainingStep:
    """
    Represents a single step in a training process.
    
    TrainingStep captures the inputs, outputs, and metadata for one
    iteration of agent training or evaluation.
    """
    
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int = 0
    step_type: str = "training"  # "training", "evaluation", "validation"
    prompt_profile_id: str = ""
    question: str = ""
    expected_response: Optional[str] = None
    agent_responses: List[AgentResponse] = field(default_factory=list)
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    
    def __post_init__(self):
        """Initialize default values after construction."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Validate step ID is a valid UUID
        try:
            uuid.UUID(self.step_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid UUID for step_id: {self.step_id}")
        
        # Validate step type
        valid_types = ["training", "evaluation", "validation"]
        if self.step_type not in valid_types:
            raise ValueError(f"Step type must be one of: {valid_types}")
        
        # Validate status
        valid_statuses = ["pending", "running", "completed", "failed"]
        if self.status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
    
    def add_agent_response(self, response: AgentResponse) -> None:
        """Add an agent response to this training step."""
        if not isinstance(response, AgentResponse):
            raise ValueError("Must provide a valid AgentResponse instance")
        
        self.agent_responses.append(response)
    
    def set_evaluation_score(self, metric: str, score: float) -> None:
        """Set an evaluation score for a specific metric."""
        if not isinstance(score, (int, float)):
            raise ValueError("Score must be a number")
        
        if not (0.0 <= score <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        
        self.evaluation_scores[metric] = float(score)
    
    def mark_completed(self) -> None:
        """Mark this training step as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str = "") -> None:
        """Mark this training step as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        if error_message:
            self.training_metadata['error'] = error_message
    
    def get_average_score(self) -> Optional[float]:
        """Calculate average evaluation score across all metrics."""
        if not self.evaluation_scores:
            return None
        
        return sum(self.evaluation_scores.values()) / len(self.evaluation_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_id': self.step_id,
            'step_number': self.step_number,
            'step_type': self.step_type,
            'prompt_profile_id': self.prompt_profile_id,
            'question': self.question,
            'expected_response': self.expected_response,
            'agent_responses': [resp.to_dict() for resp in self.agent_responses],
            'evaluation_scores': self.evaluation_scores,
            'training_metadata': self.training_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingStep':
        """Create TrainingStep from dictionary."""
        # Parse datetime fields
        created_at = None
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            elif isinstance(data['created_at'], datetime):
                created_at = data['created_at']
        
        completed_at = None
        if data.get('completed_at'):
            if isinstance(data['completed_at'], str):
                completed_at = datetime.fromisoformat(data['completed_at'].replace('Z', '+00:00'))
            elif isinstance(data['completed_at'], datetime):
                completed_at = data['completed_at']
        
        # Parse agent responses
        agent_responses = []
        for resp_data in data.get('agent_responses', []):
            # Recreate AgentResponse from dict data
            response = AgentResponse(
                content=resp_data['content'],
                reasoning=resp_data.get('reasoning'),
                confidence=resp_data.get('confidence'),
                sources=resp_data.get('sources', []),
                metadata=resp_data.get('metadata', {}),
                timestamp=datetime.fromisoformat(resp_data['timestamp'].replace('Z', '+00:00')) 
                         if resp_data.get('timestamp') else None
            )
            agent_responses.append(response)
        
        return cls(
            step_id=data.get('step_id', str(uuid.uuid4())),
            step_number=data.get('step_number', 0),
            step_type=data.get('step_type', 'training'),
            prompt_profile_id=data.get('prompt_profile_id', ''),
            question=data.get('question', ''),
            expected_response=data.get('expected_response'),
            agent_responses=agent_responses,
            evaluation_scores=data.get('evaluation_scores', {}),
            training_metadata=data.get('training_metadata', {}),
            created_at=created_at,
            completed_at=completed_at,
            status=data.get('status', 'pending')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TrainingStep':
        """Create TrainingStep from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """
        Validate the TrainingStep structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate step ID
        if not self.step_id:
            errors.append("Step ID cannot be empty")
        
        try:
            uuid.UUID(self.step_id)
        except (ValueError, TypeError):
            errors.append(f"Step ID must be a valid UUID: {self.step_id}")
        
        # Validate step type
        valid_types = ["training", "evaluation", "validation"]
        if self.step_type not in valid_types:
            errors.append(f"Step type must be one of: {valid_types}")
        
        # Validate status
        valid_statuses = ["pending", "running", "completed", "failed"]
        if self.status not in valid_statuses:
            errors.append(f"Status must be one of: {valid_statuses}")
        
        # Validate question
        if not self.question or not self.question.strip():
            errors.append("Question cannot be empty")
        
        if len(self.question) > 10000:
            errors.append("Question is too long (max 10,000 characters)")
        
        # Validate evaluation scores
        for metric, score in self.evaluation_scores.items():
            if not isinstance(score, (int, float)):
                errors.append(f"Score for '{metric}' must be a number")
            elif not (0.0 <= score <= 1.0):
                errors.append(f"Score for '{metric}' must be between 0.0 and 1.0")
        
        # Validate agent responses
        for i, response in enumerate(self.agent_responses):
            if not isinstance(response, AgentResponse):
                errors.append(f"Agent response {i} must be an AgentResponse instance")
        
        return errors


# Backward compatibility utilities
def enhance_agent_response_with_training(response: AgentResponse, 
                                       training_metadata: Dict[str, Any]) -> AgentResponse:
    """
    Enhance an existing AgentResponse with training metadata while maintaining compatibility.
    
    Args:
        response: Existing AgentResponse instance
        training_metadata: Additional metadata for training context
    
    Returns:
        Enhanced AgentResponse with training metadata merged into existing metadata
    """
    # Create a copy of the existing metadata
    enhanced_metadata = (response.metadata or {}).copy()
    
    # Add training metadata under a specific key to avoid conflicts
    enhanced_metadata['training'] = training_metadata
    
    # Create new AgentResponse with enhanced metadata
    return AgentResponse(
        content=response.content,
        reasoning=response.reasoning,
        confidence=response.confidence,
        sources=response.sources,
        metadata=enhanced_metadata,
        timestamp=response.timestamp
    )


def extract_training_metadata(response: AgentResponse) -> Dict[str, Any]:
    """
    Extract training metadata from an AgentResponse.
    
    Args:
        response: AgentResponse instance to extract from
    
    Returns:
        Training metadata dictionary, empty if none found
    """
    if not response.metadata:
        return {}
    
    return response.metadata.get('training', {})


# Validation utilities
def validate_all_structures(*structures) -> Dict[str, List[str]]:
    """
    Validate multiple data structures and return consolidated errors.
    
    Args:
        *structures: Variable number of data structure instances to validate
    
    Returns:
        Dictionary mapping structure type to list of validation errors
    """
    validation_results = {}
    
    for i, structure in enumerate(structures):
        if hasattr(structure, 'validate'):
            errors = structure.validate()
            if errors:
                structure_type = type(structure).__name__
                key = f"{structure_type}_{i}" if structure_type in validation_results else structure_type
                validation_results[key] = errors
    
    return validation_results


# Export all public classes and functions
__all__ = [
    'RolePrompt',
    'PromptProfile', 
    'TrainingStep',
    'enhance_agent_response_with_training',
    'extract_training_metadata',
    'validate_all_structures'
]
"""
SQLAlchemy ORM models for the training database schema.

This module defines SQLAlchemy models that map to the database tables
defined in schema.sql, providing ORM access to training data.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Integer, Text, TIMESTAMP, ForeignKey, DECIMAL, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class PromptProfileModel(Base):
    """
    SQLAlchemy model for hegel_prompt_profiles table.
    
    Maps to the PromptProfile data structure for database persistence.
    """
    __tablename__ = 'hegel_prompt_profiles'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key relationships
    base_profile_id = Column(UUID(as_uuid=True), ForeignKey('hegel_prompt_profiles.id'), nullable=True)
    
    # Core identification fields
    corpus_id = Column(String(255), nullable=False, index=True)
    task_type = Column(String(50), nullable=False, default='qa', index=True)
    
    # Profile data stored as JSONB
    profile = Column(JSONB, nullable=False)
    
    # Performance and metadata
    performance_stats = Column(JSONB, nullable=True, default={})
    profile_metadata = Column('metadata', JSONB, nullable=True, default={})
    
    # Timestamp fields
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    
    # Self-referential relationship for profile lineage
    base_profile = relationship("PromptProfileModel", remote_side=[id], backref="derived_profiles")
    
    def __repr__(self) -> str:
        return f"<PromptProfileModel(id={self.id}, corpus_id='{self.corpus_id}', task_type='{self.task_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            'id': str(self.id),
            'base_profile_id': str(self.base_profile_id) if self.base_profile_id else None,
            'corpus_id': self.corpus_id,
            'task_type': self.task_type,
            'profile': self.profile,
            'performance_stats': self.performance_stats,
            'metadata': self.profile_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TrainingStepModel(Base):
    """
    SQLAlchemy model for hegel_training_steps table.
    
    Records training iterations with full context for analysis.
    """
    __tablename__ = 'hegel_training_steps'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Profile evolution tracking
    old_profile_id = Column(UUID(as_uuid=True), ForeignKey('hegel_prompt_profiles.id'), nullable=False)
    new_profile_id = Column(UUID(as_uuid=True), ForeignKey('hegel_prompt_profiles.id'), nullable=False)
    
    # Training context
    corpus_id = Column(String(255), nullable=False, index=True)
    task_type = Column(String(50), nullable=False, default='qa', index=True)
    
    # Training data
    query = Column(Text, nullable=False)
    gold_answer = Column(Text, nullable=True)
    predicted_answer = Column(Text, nullable=False)
    reward = Column(DECIMAL(precision=10, scale=6), nullable=True)
    
    # Rich metadata for analysis
    metrics = Column(JSONB, nullable=True, default={})
    debate_trace = Column(JSONB, nullable=True, default={})
    optimization_strategy = Column(String(100), nullable=True)
    
    # Timing and metadata
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    old_profile = relationship("PromptProfileModel", foreign_keys=[old_profile_id])
    new_profile = relationship("PromptProfileModel", foreign_keys=[new_profile_id])
    
    def __repr__(self) -> str:
        return f"<TrainingStepModel(id={self.id}, corpus_id='{self.corpus_id}', reward={self.reward})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            'id': str(self.id),
            'old_profile_id': str(self.old_profile_id),
            'new_profile_id': str(self.new_profile_id),
            'corpus_id': self.corpus_id,
            'task_type': self.task_type,
            'query': self.query,
            'gold_answer': self.gold_answer,
            'predicted_answer': self.predicted_answer,
            'reward': float(self.reward) if self.reward else None,
            'metrics': self.metrics,
            'debate_trace': self.debate_trace,
            'optimization_strategy': self.optimization_strategy,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processing_time_ms': self.processing_time_ms
        }


class ProfilePopulationModel(Base):
    """
    SQLAlchemy model for hegel_profile_populations table.
    
    Manages population-based optimization state.
    """
    __tablename__ = 'hegel_profile_populations'
    
    # Composite primary key
    corpus_id = Column(String(255), nullable=False, primary_key=True)
    task_type = Column(String(50), nullable=False, default='qa', primary_key=True)
    profile_id = Column(UUID(as_uuid=True), ForeignKey('hegel_prompt_profiles.id'), nullable=False, primary_key=True)
    
    # Population fitness metrics
    fitness_score = Column(DECIMAL(precision=10, scale=6), nullable=True)
    selection_count = Column(Integer, nullable=False, default=0)
    generation = Column(Integer, nullable=False, default=1)
    
    # Population metadata
    population_metadata = Column(JSONB, nullable=True, default={})
    last_updated = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship to profile
    profile = relationship("PromptProfileModel")
    
    def __repr__(self) -> str:
        return f"<ProfilePopulationModel(corpus_id='{self.corpus_id}', task_type='{self.task_type}', profile_id={self.profile_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            'corpus_id': self.corpus_id,
            'task_type': self.task_type,
            'profile_id': str(self.profile_id),
            'fitness_score': float(self.fitness_score) if self.fitness_score else None,
            'selection_count': self.selection_count,
            'generation': self.generation,
            'population_metadata': self.population_metadata,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
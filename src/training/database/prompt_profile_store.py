"""
PromptProfileStore implementation for database persistence.

This module provides comprehensive CRUD operations for PromptProfile objects
with full database persistence, error handling, and performance optimization.
"""

import logging
import uuid
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.data_structures import PromptProfile, RolePrompt
from training.database.base import get_db_session, DatabaseError
from training.database.models import PromptProfileModel

logger = logging.getLogger(__name__)


class PromptProfileStoreError(Exception):
    """Base exception for PromptProfileStore operations."""
    pass


class ProfileNotFoundError(PromptProfileStoreError):
    """Exception raised when a profile is not found."""
    pass


class ProfileValidationError(PromptProfileStoreError):
    """Exception raised when profile validation fails."""
    pass


class PromptProfileStore:
    """
    Database store for PromptProfile persistence with comprehensive CRUD operations.
    
    Provides:
    - Full CRUD operations (Create, Read, Update, Delete)
    - Advanced querying with filtering and pagination
    - Profile versioning and lineage tracking
    - Performance optimization with proper indexing
    - Comprehensive error handling and logging
    - Backward compatibility with existing PromptProfile structure
    """
    
    def __init__(self):
        """Initialize the PromptProfileStore."""
        self.logger = logger
    
    def _prompt_profile_to_model(self, profile: PromptProfile) -> PromptProfileModel:
        """
        Convert PromptProfile to SQLAlchemy model.
        
        Args:
            profile: PromptProfile instance to convert
            
        Returns:
            PromptProfileModel instance
        """
        # Convert role prompts to database format
        role_prompts_dict = {}
        for role, role_prompt in profile.role_prompts.items():
            role_prompts_dict[role] = role_prompt.to_dict()
        
        # Create profile data structure
        profile_data = {
            'name': profile.name,
            'description': profile.description,
            'version': profile.version,
            'author': profile.author,
            'tags': profile.tags,
            'role_prompts': role_prompts_dict
        }
        
        return PromptProfileModel(
            id=uuid.UUID(profile.profile_id),
            corpus_id=profile.metadata.get('corpus_id', 'default'),
            task_type=profile.metadata.get('task_type', 'qa'),
            profile=profile_data,
            performance_stats=profile.metadata.get('performance_stats', {}),
            profile_metadata=profile.metadata,
            created_at=profile.created_at,
            updated_at=datetime.utcnow()
        )
    
    def _model_to_prompt_profile(self, model: PromptProfileModel) -> PromptProfile:
        """
        Convert SQLAlchemy model to PromptProfile.
        
        Args:
            model: PromptProfileModel instance to convert
            
        Returns:
            PromptProfile instance
        """
        # Extract profile data
        profile_data = model.profile
        
        # Reconstruct role prompts
        role_prompts = {}
        for role, prompt_data in profile_data.get('role_prompts', {}).items():
            role_prompts[role] = RolePrompt.from_dict(prompt_data)
        
        # Merge metadata
        metadata = model.profile_metadata.copy() if model.profile_metadata else {}
        metadata.update({
            'corpus_id': model.corpus_id,
            'task_type': model.task_type,
            'performance_stats': model.performance_stats or {}
        })
        
        return PromptProfile(
            profile_id=str(model.id),
            name=profile_data.get('name', ''),
            description=profile_data.get('description'),
            role_prompts=role_prompts,
            version=profile_data.get('version', '1.0'),
            author=profile_data.get('author'),
            created_at=model.created_at,
            tags=profile_data.get('tags', []),
            metadata=metadata
        )
    
    def create(self, profile: PromptProfile, corpus_id: str, task_type: str = 'qa') -> str:
        """
        Create a new prompt profile in the database.
        
        Args:
            profile: PromptProfile instance to store
            corpus_id: Corpus identifier for the profile
            task_type: Task type (default: 'qa')
            
        Returns:
            UUID string of the created profile
            
        Raises:
            ProfileValidationError: If profile validation fails
            PromptProfileStoreError: If database operation fails
        """
        try:
            # Validate profile
            validation_errors = profile.validate()
            if validation_errors:
                raise ProfileValidationError(f"Profile validation failed: {validation_errors}")
            
            # Add corpus and task metadata
            profile.metadata.update({
                'corpus_id': corpus_id,
                'task_type': task_type
            })
            
            with get_db_session() as session:
                # Convert to model
                model = self._prompt_profile_to_model(profile)
                model.corpus_id = corpus_id
                model.task_type = task_type
                
                # Save to database
                session.add(model)
                session.commit()
                
                profile_id = str(model.id)
                self.logger.info(f"Created prompt profile {profile_id} for corpus '{corpus_id}', task '{task_type}'")
                return profile_id
                
        except ProfileValidationError:
            raise
        except IntegrityError as e:
            raise PromptProfileStoreError(f"Profile creation failed due to constraint violation: {e}")
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during profile creation: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during profile creation: {e}")
    
    def get_by_id(self, profile_id: Union[str, uuid.UUID]) -> PromptProfile:
        """
        Retrieve a prompt profile by its ID.
        
        Args:
            profile_id: UUID of the profile to retrieve
            
        Returns:
            PromptProfile instance
            
        Raises:
            ProfileNotFoundError: If profile is not found
            PromptProfileStoreError: If database operation fails
        """
        try:
            if isinstance(profile_id, str):
                profile_id = uuid.UUID(profile_id)
            
            with get_db_session() as session:
                model = session.query(PromptProfileModel).filter(
                    PromptProfileModel.id == profile_id
                ).first()
                
                if not model:
                    raise ProfileNotFoundError(f"Profile with ID {profile_id} not found")
                
                profile = self._model_to_prompt_profile(model)
                self.logger.debug(f"Retrieved prompt profile {profile_id}")
                return profile
                
        except ProfileNotFoundError:
            raise
        except ValueError as e:
            raise PromptProfileStoreError(f"Invalid profile ID format: {e}")
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during profile retrieval: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during profile retrieval: {e}")
    
    def update(self, profile: PromptProfile) -> None:
        """
        Update an existing prompt profile.
        
        Args:
            profile: PromptProfile instance with updated data
            
        Raises:
            ProfileNotFoundError: If profile is not found
            ProfileValidationError: If profile validation fails
            PromptProfileStoreError: If database operation fails
        """
        try:
            # Validate profile
            validation_errors = profile.validate()
            if validation_errors:
                raise ProfileValidationError(f"Profile validation failed: {validation_errors}")
            
            profile_uuid = uuid.UUID(profile.profile_id)
            
            with get_db_session() as session:
                model = session.query(PromptProfileModel).filter(
                    PromptProfileModel.id == profile_uuid
                ).first()
                
                if not model:
                    raise ProfileNotFoundError(f"Profile with ID {profile.profile_id} not found")
                
                # Update model with new data
                updated_model = self._prompt_profile_to_model(profile)
                
                model.profile = updated_model.profile
                model.performance_stats = updated_model.performance_stats
                model.profile_metadata = updated_model.profile_metadata
                model.updated_at = datetime.utcnow()
                
                session.commit()
                
                self.logger.info(f"Updated prompt profile {profile.profile_id}")
                
        except (ProfileNotFoundError, ProfileValidationError):
            raise
        except ValueError as e:
            raise PromptProfileStoreError(f"Invalid profile ID format: {e}")
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during profile update: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during profile update: {e}")
    
    def delete(self, profile_id: Union[str, uuid.UUID]) -> bool:
        """
        Delete a prompt profile by its ID.
        
        Args:
            profile_id: UUID of the profile to delete
            
        Returns:
            True if profile was deleted, False if not found
            
        Raises:
            PromptProfileStoreError: If database operation fails
        """
        try:
            if isinstance(profile_id, str):
                profile_id = uuid.UUID(profile_id)
            
            with get_db_session() as session:
                model = session.query(PromptProfileModel).filter(
                    PromptProfileModel.id == profile_id
                ).first()
                
                if not model:
                    self.logger.warning(f"Attempted to delete non-existent profile {profile_id}")
                    return False
                
                session.delete(model)
                session.commit()
                
                self.logger.info(f"Deleted prompt profile {profile_id}")
                return True
                
        except ValueError as e:
            raise PromptProfileStoreError(f"Invalid profile ID format: {e}")
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during profile deletion: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during profile deletion: {e}")
    
    def list_by_corpus_and_task(self, 
                                corpus_id: str, 
                                task_type: str = 'qa',
                                limit: int = 100,
                                offset: int = 0,
                                order_by: str = 'created_at') -> List[PromptProfile]:
        """
        List prompt profiles for a specific corpus and task type.
        
        Args:
            corpus_id: Corpus identifier
            task_type: Task type (default: 'qa')
            limit: Maximum number of profiles to return (default: 100)
            offset: Number of profiles to skip (default: 0)
            order_by: Field to order by (default: 'created_at')
            
        Returns:
            List of PromptProfile instances
            
        Raises:
            PromptProfileStoreError: If database operation fails
        """
        try:
            with get_db_session() as session:
                query = session.query(PromptProfileModel).filter(
                    and_(
                        PromptProfileModel.corpus_id == corpus_id,
                        PromptProfileModel.task_type == task_type
                    )
                )
                
                # Apply ordering
                if order_by == 'created_at':
                    query = query.order_by(desc(PromptProfileModel.created_at))
                elif order_by == 'updated_at':
                    query = query.order_by(desc(PromptProfileModel.updated_at))
                else:
                    query = query.order_by(desc(PromptProfileModel.created_at))
                
                # Apply pagination
                query = query.offset(offset).limit(limit)
                
                models = query.all()
                profiles = [self._model_to_prompt_profile(model) for model in models]
                
                self.logger.debug(f"Retrieved {len(profiles)} profiles for corpus '{corpus_id}', task '{task_type}'")
                return profiles
                
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during profile listing: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during profile listing: {e}")
    
    def get_latest_by_corpus_and_task(self, corpus_id: str, task_type: str = 'qa') -> Optional[PromptProfile]:
        """
        Get the most recently created profile for a corpus and task type.
        
        Args:
            corpus_id: Corpus identifier
            task_type: Task type (default: 'qa')
            
        Returns:
            Latest PromptProfile instance or None if no profiles found
            
        Raises:
            PromptProfileStoreError: If database operation fails
        """
        try:
            with get_db_session() as session:
                model = session.query(PromptProfileModel).filter(
                    and_(
                        PromptProfileModel.corpus_id == corpus_id,
                        PromptProfileModel.task_type == task_type
                    )
                ).order_by(desc(PromptProfileModel.created_at)).first()
                
                if not model:
                    self.logger.debug(f"No profiles found for corpus '{corpus_id}', task '{task_type}'")
                    return None
                
                profile = self._model_to_prompt_profile(model)
                self.logger.debug(f"Retrieved latest profile {profile.profile_id} for corpus '{corpus_id}', task '{task_type}'")
                return profile
                
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during latest profile retrieval: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during latest profile retrieval: {e}")
    
    def search_profiles(self, 
                       search_criteria: Dict[str, Any],
                       limit: int = 100,
                       offset: int = 0) -> Tuple[List[PromptProfile], int]:
        """
        Search profiles based on various criteria.
        
        Args:
            search_criteria: Dictionary with search parameters:
                - corpus_ids: List of corpus IDs to filter by
                - task_types: List of task types to filter by
                - author: Author name to filter by
                - tags: List of tags to filter by (any match)
                - created_after: DateTime to filter profiles created after
                - created_before: DateTime to filter profiles created before
                - name_contains: Text to search in profile names
            limit: Maximum number of profiles to return
            offset: Number of profiles to skip
            
        Returns:
            Tuple of (list of PromptProfile instances, total count)
            
        Raises:
            PromptProfileStoreError: If database operation fails
        """
        try:
            with get_db_session() as session:
                query = session.query(PromptProfileModel)
                count_query = session.query(func.count(PromptProfileModel.id))
                
                # Apply filters
                filters = []
                
                # Corpus IDs filter
                if 'corpus_ids' in search_criteria and search_criteria['corpus_ids']:
                    filters.append(PromptProfileModel.corpus_id.in_(search_criteria['corpus_ids']))
                
                # Task types filter
                if 'task_types' in search_criteria and search_criteria['task_types']:
                    filters.append(PromptProfileModel.task_type.in_(search_criteria['task_types']))
                
                # Author filter (searches in profile JSON)
                if 'author' in search_criteria and search_criteria['author']:
                    filters.append(PromptProfileModel.profile['author'].astext == search_criteria['author'])
                
                # Tags filter (searches in profile JSON)
                if 'tags' in search_criteria and search_criteria['tags']:
                    for tag in search_criteria['tags']:
                        filters.append(PromptProfileModel.profile['tags'].astext.contains(f'"{tag}"'))
                
                # Date range filters
                if 'created_after' in search_criteria and search_criteria['created_after']:
                    filters.append(PromptProfileModel.created_at >= search_criteria['created_after'])
                
                if 'created_before' in search_criteria and search_criteria['created_before']:
                    filters.append(PromptProfileModel.created_at <= search_criteria['created_before'])
                
                # Name contains filter
                if 'name_contains' in search_criteria and search_criteria['name_contains']:
                    filters.append(PromptProfileModel.profile['name'].astext.ilike(
                        f"%{search_criteria['name_contains']}%"
                    ))
                
                # Apply all filters
                if filters:
                    query = query.filter(and_(*filters))
                    count_query = count_query.filter(and_(*filters))
                
                # Get total count
                total_count = count_query.scalar()
                
                # Apply ordering and pagination
                query = query.order_by(desc(PromptProfileModel.created_at))
                query = query.offset(offset).limit(limit)
                
                models = query.all()
                profiles = [self._model_to_prompt_profile(model) for model in models]
                
                self.logger.debug(f"Search returned {len(profiles)} profiles out of {total_count} total")
                return profiles, total_count
                
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during profile search: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during profile search: {e}")
    
    def get_profile_lineage(self, profile_id: Union[str, uuid.UUID]) -> List[PromptProfile]:
        """
        Get the lineage of a profile (base profile and all derived profiles).
        
        Args:
            profile_id: UUID of the profile to get lineage for
            
        Returns:
            List of PromptProfile instances in the lineage chain
            
        Raises:
            ProfileNotFoundError: If profile is not found
            PromptProfileStoreError: If database operation fails
        """
        try:
            if isinstance(profile_id, str):
                profile_id = uuid.UUID(profile_id)
            
            with get_db_session() as session:
                # Check if profile exists
                base_model = session.query(PromptProfileModel).filter(
                    PromptProfileModel.id == profile_id
                ).first()
                
                if not base_model:
                    raise ProfileNotFoundError(f"Profile with ID {profile_id} not found")
                
                # Get the entire lineage
                lineage_models = []
                
                # Get all ancestors (traverse up the base_profile_id chain)
                current = base_model
                while current:
                    lineage_models.insert(0, current)  # Insert at beginning to maintain order
                    if current.base_profile_id:
                        current = session.query(PromptProfileModel).filter(
                            PromptProfileModel.id == current.base_profile_id
                        ).first()
                    else:
                        current = None
                
                # Get all descendants (all profiles that have this as base)
                descendants = session.query(PromptProfileModel).filter(
                    PromptProfileModel.base_profile_id == profile_id
                ).order_by(PromptProfileModel.created_at).all()
                
                lineage_models.extend(descendants)
                
                # Convert to PromptProfile objects
                profiles = [self._model_to_prompt_profile(model) for model in lineage_models]
                
                self.logger.debug(f"Retrieved lineage of {len(profiles)} profiles for {profile_id}")
                return profiles
                
        except ProfileNotFoundError:
            raise
        except ValueError as e:
            raise PromptProfileStoreError(f"Invalid profile ID format: {e}")
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during lineage retrieval: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during lineage retrieval: {e}")
    
    def create_derived_profile(self, 
                             base_profile_id: Union[str, uuid.UUID],
                             new_profile: PromptProfile) -> str:
        """
        Create a new profile derived from an existing base profile.
        
        Args:
            base_profile_id: UUID of the base profile
            new_profile: New PromptProfile instance to create
            
        Returns:
            UUID string of the created derived profile
            
        Raises:
            ProfileNotFoundError: If base profile is not found
            ProfileValidationError: If new profile validation fails
            PromptProfileStoreError: If database operation fails
        """
        try:
            if isinstance(base_profile_id, str):
                base_profile_id = uuid.UUID(base_profile_id)
            
            # Validate new profile
            validation_errors = new_profile.validate()
            if validation_errors:
                raise ProfileValidationError(f"New profile validation failed: {validation_errors}")
            
            with get_db_session() as session:
                # Check that base profile exists
                base_model = session.query(PromptProfileModel).filter(
                    PromptProfileModel.id == base_profile_id
                ).first()
                
                if not base_model:
                    raise ProfileNotFoundError(f"Base profile with ID {base_profile_id} not found")
                
                # Create derived profile model
                model = self._prompt_profile_to_model(new_profile)
                model.base_profile_id = base_profile_id
                model.corpus_id = base_model.corpus_id  # Inherit corpus from base
                model.task_type = base_model.task_type   # Inherit task type from base
                
                # Save to database
                session.add(model)
                session.commit()
                
                profile_id = str(model.id)
                self.logger.info(f"Created derived profile {profile_id} from base {base_profile_id}")
                return profile_id
                
        except (ProfileNotFoundError, ProfileValidationError):
            raise
        except ValueError as e:
            raise PromptProfileStoreError(f"Invalid profile ID format: {e}")
        except IntegrityError as e:
            raise PromptProfileStoreError(f"Derived profile creation failed due to constraint violation: {e}")
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during derived profile creation: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during derived profile creation: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring and debugging.
        
        Returns:
            Dictionary with various statistics about stored profiles
            
        Raises:
            PromptProfileStoreError: If database operation fails
        """
        try:
            with get_db_session() as session:
                # Total profiles count
                total_profiles = session.query(func.count(PromptProfileModel.id)).scalar()
                
                # Profiles by corpus
                corpus_counts = session.query(
                    PromptProfileModel.corpus_id,
                    func.count(PromptProfileModel.id)
                ).group_by(PromptProfileModel.corpus_id).all()
                
                # Profiles by task type
                task_counts = session.query(
                    PromptProfileModel.task_type,
                    func.count(PromptProfileModel.id)
                ).group_by(PromptProfileModel.task_type).all()
                
                # Recent activity
                recent_count = session.query(func.count(PromptProfileModel.id)).filter(
                    PromptProfileModel.created_at >= func.now() - func.interval('7 days')
                ).scalar()
                
                stats = {
                    'total_profiles': total_profiles,
                    'profiles_by_corpus': dict(corpus_counts),
                    'profiles_by_task': dict(task_counts),
                    'profiles_created_last_7_days': recent_count,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.logger.debug("Retrieved profile statistics")
                return stats
                
        except SQLAlchemyError as e:
            raise PromptProfileStoreError(f"Database error during statistics retrieval: {e}")
        except Exception as e:
            raise PromptProfileStoreError(f"Unexpected error during statistics retrieval: {e}")
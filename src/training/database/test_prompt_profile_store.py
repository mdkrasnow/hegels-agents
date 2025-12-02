"""
Comprehensive unit tests for PromptProfileStore.

This module provides extensive test coverage for all PromptProfileStore
functionality including CRUD operations, error handling, and edge cases.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from training.data_structures import PromptProfile, RolePrompt
from training.database.prompt_profile_store import (
    PromptProfileStore,
    PromptProfileStoreError,
    ProfileNotFoundError,
    ProfileValidationError
)
from training.database.models import PromptProfileModel


class TestPromptProfileStore:
    """Test suite for PromptProfileStore functionality."""
    
    @pytest.fixture
    def store(self):
        """Create a PromptProfileStore instance."""
        return PromptProfileStore()
    
    @pytest.fixture
    def sample_role_prompt(self):
        """Create a sample RolePrompt for testing."""
        return RolePrompt(
            role="worker",
            prompt_text="You are a helpful worker agent.",
            description="Worker agent prompt",
            author="test_user"
        )
    
    @pytest.fixture
    def sample_prompt_profile(self, sample_role_prompt):
        """Create a sample PromptProfile for testing."""
        profile = PromptProfile(
            name="Test Profile",
            description="A test profile",
            author="test_user",
            tags=["test", "sample"]
        )
        profile.add_role_prompt(sample_role_prompt)
        return profile
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = Mock(spec=Session)
        session.query.return_value = session
        session.filter.return_value = session
        session.order_by.return_value = session
        session.offset.return_value = session
        session.limit.return_value = session
        session.first.return_value = None
        session.all.return_value = []
        session.scalar.return_value = 0
        return session
    
    @pytest.fixture
    def mock_db_session(self, mock_session):
        """Mock the get_db_session context manager."""
        with patch('training.database.prompt_profile_store.get_db_session') as mock:
            mock.return_value.__enter__.return_value = mock_session
            mock.return_value.__exit__.return_value = None
            yield mock_session


class TestPromptProfileStoreCreation(TestPromptProfileStore):
    """Test profile creation functionality."""
    
    def test_create_valid_profile_success(self, store, sample_prompt_profile, mock_db_session):
        """Test successful creation of a valid profile."""
        # Arrange
        mock_db_session.add = Mock()
        mock_db_session.commit = Mock()
        
        # Act
        result = store.create(sample_prompt_profile, "test_corpus", "qa")
        
        # Assert
        assert result is not None
        assert isinstance(result, str)
        # Verify UUID format
        uuid.UUID(result)
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_create_invalid_profile_validation_error(self, store, mock_db_session):
        """Test creation with invalid profile raises validation error."""
        # Arrange
        invalid_profile = PromptProfile(name="")  # Empty name should fail validation
        
        # Act & Assert
        with pytest.raises(ProfileValidationError):
            store.create(invalid_profile, "test_corpus", "qa")
        
        # Verify no database operations
        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_not_called()
    
    def test_create_database_integrity_error(self, store, sample_prompt_profile, mock_db_session):
        """Test creation with database integrity error."""
        # Arrange
        mock_db_session.commit.side_effect = IntegrityError("constraint violation", None, None)
        
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.create(sample_prompt_profile, "test_corpus", "qa")
        
        assert "constraint violation" in str(exc_info.value)
    
    def test_create_unexpected_database_error(self, store, sample_prompt_profile, mock_db_session):
        """Test creation with unexpected database error."""
        # Arrange
        mock_db_session.commit.side_effect = SQLAlchemyError("database error")
        
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.create(sample_prompt_profile, "test_corpus", "qa")
        
        assert "database error" in str(exc_info.value)
    
    def test_create_adds_corpus_metadata(self, store, sample_prompt_profile, mock_db_session):
        """Test that creation properly adds corpus and task metadata."""
        # Arrange
        original_metadata = sample_prompt_profile.metadata.copy()
        
        # Act
        store.create(sample_prompt_profile, "test_corpus", "custom_task")
        
        # Assert
        assert sample_prompt_profile.metadata['corpus_id'] == "test_corpus"
        assert sample_prompt_profile.metadata['task_type'] == "custom_task"
    
    def test_create_with_default_task_type(self, store, sample_prompt_profile, mock_db_session):
        """Test creation with default task type."""
        # Act
        store.create(sample_prompt_profile, "test_corpus")
        
        # Assert
        assert sample_prompt_profile.metadata['task_type'] == "qa"


class TestPromptProfileStoreRetrieval(TestPromptProfileStore):
    """Test profile retrieval functionality."""
    
    def test_get_by_id_existing_profile(self, store, sample_prompt_profile, mock_db_session):
        """Test successful retrieval of existing profile."""
        # Arrange
        test_id = uuid.uuid4()
        mock_model = Mock(spec=PromptProfileModel)
        mock_model.id = test_id
        mock_model.corpus_id = "test_corpus"
        mock_model.task_type = "qa"
        mock_model.profile = {
            'name': sample_prompt_profile.name,
            'description': sample_prompt_profile.description,
            'version': sample_prompt_profile.version,
            'author': sample_prompt_profile.author,
            'tags': sample_prompt_profile.tags,
            'role_prompts': {}
        }
        mock_model.performance_stats = {}
        mock_model.metadata = {}
        mock_model.created_at = datetime.utcnow()
        
        mock_db_session.first.return_value = mock_model
        
        # Act
        result = store.get_by_id(str(test_id))
        
        # Assert
        assert isinstance(result, PromptProfile)
        assert result.name == sample_prompt_profile.name
        assert result.profile_id == str(test_id)
    
    def test_get_by_id_nonexistent_profile(self, store, mock_db_session):
        """Test retrieval of non-existent profile raises error."""
        # Arrange
        test_id = uuid.uuid4()
        mock_db_session.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ProfileNotFoundError):
            store.get_by_id(str(test_id))
    
    def test_get_by_id_invalid_uuid_format(self, store, mock_db_session):
        """Test retrieval with invalid UUID format."""
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.get_by_id("invalid-uuid")
        
        assert "Invalid profile ID format" in str(exc_info.value)
    
    def test_get_by_id_database_error(self, store, mock_db_session):
        """Test retrieval with database error."""
        # Arrange
        test_id = uuid.uuid4()
        mock_db_session.query.side_effect = SQLAlchemyError("database error")
        
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.get_by_id(str(test_id))
        
        assert "database error" in str(exc_info.value)
    
    def test_get_by_id_accepts_uuid_object(self, store, mock_db_session):
        """Test that get_by_id accepts UUID objects as well as strings."""
        # Arrange
        test_id = uuid.uuid4()
        mock_db_session.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ProfileNotFoundError):
            store.get_by_id(test_id)  # Pass UUID object directly


class TestPromptProfileStoreUpdate(TestPromptProfileStore):
    """Test profile update functionality."""
    
    def test_update_existing_profile_success(self, store, sample_prompt_profile, mock_db_session):
        """Test successful update of existing profile."""
        # Arrange
        test_id = uuid.uuid4()
        sample_prompt_profile.profile_id = str(test_id)
        
        mock_model = Mock(spec=PromptProfileModel)
        mock_model.id = test_id
        mock_db_session.first.return_value = mock_model
        
        # Act
        store.update(sample_prompt_profile)
        
        # Assert
        mock_db_session.commit.assert_called_once()
        assert mock_model.updated_at is not None
    
    def test_update_nonexistent_profile(self, store, sample_prompt_profile, mock_db_session):
        """Test update of non-existent profile raises error."""
        # Arrange
        test_id = uuid.uuid4()
        sample_prompt_profile.profile_id = str(test_id)
        mock_db_session.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ProfileNotFoundError):
            store.update(sample_prompt_profile)
    
    def test_update_invalid_profile_validation_error(self, store, mock_db_session):
        """Test update with invalid profile raises validation error."""
        # Arrange
        invalid_profile = PromptProfile(name="", profile_id=str(uuid.uuid4()))
        
        # Act & Assert
        with pytest.raises(ProfileValidationError):
            store.update(invalid_profile)
    
    def test_update_database_error(self, store, sample_prompt_profile, mock_db_session):
        """Test update with database error."""
        # Arrange
        test_id = uuid.uuid4()
        sample_prompt_profile.profile_id = str(test_id)
        
        mock_model = Mock(spec=PromptProfileModel)
        mock_db_session.first.return_value = mock_model
        mock_db_session.commit.side_effect = SQLAlchemyError("database error")
        
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.update(sample_prompt_profile)
        
        assert "database error" in str(exc_info.value)


class TestPromptProfileStoreDeletion(TestPromptProfileStore):
    """Test profile deletion functionality."""
    
    def test_delete_existing_profile_success(self, store, mock_db_session):
        """Test successful deletion of existing profile."""
        # Arrange
        test_id = uuid.uuid4()
        mock_model = Mock(spec=PromptProfileModel)
        mock_db_session.first.return_value = mock_model
        
        # Act
        result = store.delete(str(test_id))
        
        # Assert
        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_model)
        mock_db_session.commit.assert_called_once()
    
    def test_delete_nonexistent_profile(self, store, mock_db_session):
        """Test deletion of non-existent profile returns False."""
        # Arrange
        test_id = uuid.uuid4()
        mock_db_session.first.return_value = None
        
        # Act
        result = store.delete(str(test_id))
        
        # Assert
        assert result is False
        mock_db_session.delete.assert_not_called()
        mock_db_session.commit.assert_not_called()
    
    def test_delete_invalid_uuid_format(self, store, mock_db_session):
        """Test deletion with invalid UUID format."""
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.delete("invalid-uuid")
        
        assert "Invalid profile ID format" in str(exc_info.value)
    
    def test_delete_database_error(self, store, mock_db_session):
        """Test deletion with database error."""
        # Arrange
        test_id = uuid.uuid4()
        mock_model = Mock(spec=PromptProfileModel)
        mock_db_session.first.return_value = mock_model
        mock_db_session.commit.side_effect = SQLAlchemyError("database error")
        
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.delete(str(test_id))
        
        assert "database error" in str(exc_info.value)


class TestPromptProfileStoreListing(TestPromptProfileStore):
    """Test profile listing functionality."""
    
    def test_list_by_corpus_and_task_success(self, store, mock_db_session):
        """Test successful listing of profiles by corpus and task."""
        # Arrange
        mock_models = [Mock(spec=PromptProfileModel) for _ in range(3)]
        for i, model in enumerate(mock_models):
            model.id = uuid.uuid4()
            model.corpus_id = "test_corpus"
            model.task_type = "qa"
            model.profile = {'name': f'Profile {i}', 'role_prompts': {}}
            model.performance_stats = {}
            model.metadata = {}
            model.created_at = datetime.utcnow()
        
        mock_db_session.all.return_value = mock_models
        
        # Act
        result = store.list_by_corpus_and_task("test_corpus", "qa")
        
        # Assert
        assert len(result) == 3
        assert all(isinstance(profile, PromptProfile) for profile in result)
    
    def test_list_by_corpus_and_task_with_pagination(self, store, mock_db_session):
        """Test listing with pagination parameters."""
        # Arrange
        mock_db_session.all.return_value = []
        
        # Act
        store.list_by_corpus_and_task("test_corpus", "qa", limit=50, offset=10)
        
        # Assert
        mock_db_session.offset.assert_called_with(10)
        mock_db_session.limit.assert_called_with(50)
    
    def test_list_by_corpus_and_task_with_ordering(self, store, mock_db_session):
        """Test listing with different ordering options."""
        # Arrange
        mock_db_session.all.return_value = []
        
        # Act
        store.list_by_corpus_and_task("test_corpus", "qa", order_by="updated_at")
        
        # Assert
        mock_db_session.order_by.assert_called()
    
    def test_list_by_corpus_and_task_database_error(self, store, mock_db_session):
        """Test listing with database error."""
        # Arrange
        mock_db_session.query.side_effect = SQLAlchemyError("database error")
        
        # Act & Assert
        with pytest.raises(PromptProfileStoreError) as exc_info:
            store.list_by_corpus_and_task("test_corpus", "qa")
        
        assert "database error" in str(exc_info.value)


class TestPromptProfileStoreAdvanced(TestPromptProfileStore):
    """Test advanced functionality."""
    
    def test_get_latest_by_corpus_and_task_success(self, store, mock_db_session):
        """Test successful retrieval of latest profile."""
        # Arrange
        mock_model = Mock(spec=PromptProfileModel)
        mock_model.id = uuid.uuid4()
        mock_model.corpus_id = "test_corpus"
        mock_model.task_type = "qa"
        mock_model.profile = {'name': 'Latest Profile', 'role_prompts': {}}
        mock_model.performance_stats = {}
        mock_model.metadata = {}
        mock_model.created_at = datetime.utcnow()
        
        mock_db_session.first.return_value = mock_model
        
        # Act
        result = store.get_latest_by_corpus_and_task("test_corpus", "qa")
        
        # Assert
        assert result is not None
        assert isinstance(result, PromptProfile)
        assert result.name == "Latest Profile"
    
    def test_get_latest_by_corpus_and_task_no_profiles(self, store, mock_db_session):
        """Test retrieval when no profiles exist."""
        # Arrange
        mock_db_session.first.return_value = None
        
        # Act
        result = store.get_latest_by_corpus_and_task("test_corpus", "qa")
        
        # Assert
        assert result is None
    
    def test_search_profiles_with_criteria(self, store, mock_db_session):
        """Test profile search with various criteria."""
        # Arrange
        mock_db_session.all.return_value = []
        mock_db_session.scalar.return_value = 0
        
        search_criteria = {
            'corpus_ids': ['corpus1', 'corpus2'],
            'task_types': ['qa', 'classification'],
            'author': 'test_user',
            'tags': ['test'],
            'created_after': datetime.utcnow() - timedelta(days=7),
            'name_contains': 'test'
        }
        
        # Act
        profiles, total_count = store.search_profiles(search_criteria)
        
        # Assert
        assert isinstance(profiles, list)
        assert isinstance(total_count, int)
        # Verify that filters were applied
        assert mock_db_session.filter.called
    
    def test_get_profile_lineage_success(self, store, mock_db_session):
        """Test successful retrieval of profile lineage."""
        # Arrange
        base_id = uuid.uuid4()
        derived_id = uuid.uuid4()
        
        base_model = Mock(spec=PromptProfileModel)
        base_model.id = base_id
        base_model.base_profile_id = None
        base_model.corpus_id = "test_corpus"
        base_model.task_type = "qa"
        base_model.profile = {'name': 'Base Profile', 'role_prompts': {}}
        base_model.performance_stats = {}
        base_model.metadata = {}
        base_model.created_at = datetime.utcnow()
        
        derived_model = Mock(spec=PromptProfileModel)
        derived_model.id = derived_id
        derived_model.base_profile_id = base_id
        derived_model.corpus_id = "test_corpus"
        derived_model.task_type = "qa"
        derived_model.profile = {'name': 'Derived Profile', 'role_prompts': {}}
        derived_model.performance_stats = {}
        derived_model.metadata = {}
        derived_model.created_at = datetime.utcnow()
        
        # Mock the query sequence for lineage retrieval
        mock_db_session.first.side_effect = [base_model, None]  # Base profile, then no parent
        mock_db_session.all.return_value = [derived_model]  # Derived profiles
        
        # Act
        result = store.get_profile_lineage(str(base_id))
        
        # Assert
        assert len(result) >= 1
        assert all(isinstance(profile, PromptProfile) for profile in result)
    
    def test_create_derived_profile_success(self, store, sample_prompt_profile, mock_db_session):
        """Test successful creation of derived profile."""
        # Arrange
        base_id = uuid.uuid4()
        base_model = Mock(spec=PromptProfileModel)
        base_model.id = base_id
        base_model.corpus_id = "test_corpus"
        base_model.task_type = "qa"
        
        mock_db_session.first.return_value = base_model
        
        # Act
        result = store.create_derived_profile(str(base_id), sample_prompt_profile)
        
        # Assert
        assert result is not None
        assert isinstance(result, str)
        uuid.UUID(result)  # Verify UUID format
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_create_derived_profile_nonexistent_base(self, store, sample_prompt_profile, mock_db_session):
        """Test creation of derived profile with non-existent base."""
        # Arrange
        base_id = uuid.uuid4()
        mock_db_session.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ProfileNotFoundError):
            store.create_derived_profile(str(base_id), sample_prompt_profile)
    
    def test_get_statistics_success(self, store, mock_db_session):
        """Test successful retrieval of database statistics."""
        # Arrange
        mock_db_session.scalar.side_effect = [100, 50]  # Total profiles, recent profiles
        mock_db_session.all.side_effect = [
            [('corpus1', 30), ('corpus2', 70)],  # Corpus counts
            [('qa', 80), ('classification', 20)]  # Task counts
        ]
        
        # Act
        result = store.get_statistics()
        
        # Assert
        assert isinstance(result, dict)
        assert 'total_profiles' in result
        assert 'profiles_by_corpus' in result
        assert 'profiles_by_task' in result
        assert 'profiles_created_last_7_days' in result
        assert 'timestamp' in result


class TestPromptProfileStoreConversion(TestPromptProfileStore):
    """Test data conversion between PromptProfile and PromptProfileModel."""
    
    def test_prompt_profile_to_model_conversion(self, store, sample_prompt_profile):
        """Test conversion from PromptProfile to PromptProfileModel."""
        # Act
        model = store._prompt_profile_to_model(sample_prompt_profile)
        
        # Assert
        assert isinstance(model, PromptProfileModel)
        assert model.profile['name'] == sample_prompt_profile.name
        assert model.profile['description'] == sample_prompt_profile.description
        assert model.profile['author'] == sample_prompt_profile.author
        assert 'role_prompts' in model.profile
    
    def test_model_to_prompt_profile_conversion(self, store):
        """Test conversion from PromptProfileModel to PromptProfile."""
        # Arrange
        test_id = uuid.uuid4()
        mock_model = Mock(spec=PromptProfileModel)
        mock_model.id = test_id
        mock_model.corpus_id = "test_corpus"
        mock_model.task_type = "qa"
        mock_model.profile = {
            'name': 'Test Profile',
            'description': 'Test description',
            'version': '1.0',
            'author': 'test_user',
            'tags': ['test'],
            'role_prompts': {
                'worker': {
                    'role': 'worker',
                    'prompt_text': 'Test prompt',
                    'description': 'Test role',
                    'version': '1.0',
                    'author': 'test_user',
                    'created_at': datetime.utcnow().isoformat(),
                    'metadata': {}
                }
            }
        }
        mock_model.performance_stats = {}
        mock_model.metadata = {}
        mock_model.created_at = datetime.utcnow()
        
        # Act
        profile = store._model_to_prompt_profile(mock_model)
        
        # Assert
        assert isinstance(profile, PromptProfile)
        assert profile.profile_id == str(test_id)
        assert profile.name == 'Test Profile'
        assert profile.description == 'Test description'
        assert 'worker' in profile.role_prompts
        assert isinstance(profile.role_prompts['worker'], RolePrompt)
    
    def test_roundtrip_conversion_preserves_data(self, store, sample_prompt_profile):
        """Test that converting to model and back preserves data."""
        # Arrange
        original_profile = sample_prompt_profile
        
        # Act - Convert to model and back
        model = store._prompt_profile_to_model(original_profile)
        model.corpus_id = "test_corpus"
        model.task_type = "qa"
        model.performance_stats = {}
        model.metadata = {}
        model.created_at = original_profile.created_at
        
        converted_profile = store._model_to_prompt_profile(model)
        
        # Assert
        assert converted_profile.name == original_profile.name
        assert converted_profile.description == original_profile.description
        assert converted_profile.author == original_profile.author
        assert converted_profile.version == original_profile.version
        assert converted_profile.tags == original_profile.tags
        assert len(converted_profile.role_prompts) == len(original_profile.role_prompts)


# Integration test helpers
@pytest.fixture(scope="session")
def test_database_url():
    """Provide test database URL for integration tests."""
    return "postgresql://test:test@localhost:5432/test_db"


class TestPromptProfileStoreIntegration:
    """Integration tests requiring actual database connection."""
    
    @pytest.mark.integration
    def test_full_crud_cycle_integration(self, test_database_url):
        """
        Integration test for full CRUD cycle.
        Note: Requires actual database connection.
        """
        # This test would require a real database setup
        # Skip if INTEGRATION environment variable is not set
        import os
        if not os.getenv('INTEGRATION'):
            pytest.skip("Integration tests disabled")
        
        # Implementation would test actual database operations
        pass
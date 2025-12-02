"""
Database base configuration and session management.

This module provides database connection management, session handling,
and initialization utilities for the training system.
"""

import logging
from contextlib import contextmanager
from typing import Iterator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import get_config

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass


class ConnectionError(DatabaseError):
    """Exception raised for database connection issues."""
    pass


class DatabaseSession:
    """
    Database session manager with connection pooling and error handling.
    
    Provides centralized database session management with:
    - Connection pooling for performance
    - Proper error handling and logging
    - Transaction management
    - Session lifecycle management
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database session manager.
        
        Args:
            database_url: Optional database URL. If not provided, uses config.
        """
        self.database_url = database_url or get_config().get_database_url()
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing database connection...")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Validate connections before use
                pool_recycle=3600,   # Recycle connections every hour
                echo=False           # Set to True for SQL debugging
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False  # Keep objects usable after commit
            )
            
            # Test connection
            self._test_connection()
            
            self._initialized = True
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise ConnectionError(f"Database initialization failed: {e}")
    
    def _test_connection(self) -> None:
        """Test database connection with a simple query."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except OperationalError as e:
            raise ConnectionError(f"Database connection test failed: {e}")
    
    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """
        Get a database session with proper error handling and cleanup.
        
        Yields:
            SQLAlchemy Session object
            
        Raises:
            DatabaseError: If session creation or operation fails
        """
        if not self._initialized:
            self.initialize()
        
        session = None
        try:
            session = self.session_factory()
            yield session
            
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Unexpected session error: {e}")
            raise DatabaseError(f"Unexpected database error: {e}")
            
        finally:
            if session:
                session.close()
    
    def close(self) -> None:
        """Close database engine and cleanup connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database session instance
_db_session = None


def init_database(database_url: Optional[str] = None) -> DatabaseSession:
    """
    Initialize global database session.
    
    Args:
        database_url: Optional database URL override
        
    Returns:
        Initialized DatabaseSession instance
    """
    global _db_session
    
    if _db_session is None:
        _db_session = DatabaseSession(database_url)
        _db_session.initialize()
    
    return _db_session


def get_database_session() -> DatabaseSession:
    """
    Get the global database session instance.
    
    Returns:
        DatabaseSession instance
        
    Raises:
        DatabaseError: If database not initialized
    """
    if _db_session is None:
        raise DatabaseError("Database not initialized. Call init_database() first.")
    
    return _db_session


@contextmanager
def get_db_session() -> Iterator[Session]:
    """
    Convenience function to get a database session.
    
    Yields:
        SQLAlchemy Session object
    """
    db = get_database_session()
    with db.get_session() as session:
        yield session
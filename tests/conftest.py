"""
Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires API keys)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Return the project root path."""
    return project_root


@pytest.fixture(scope="session")
def data_path() -> Path:
    """Return the data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def corpus_path() -> Path:
    """Return the corpus directory path."""
    return project_root / "data" / "corpus"


@pytest.fixture
def api_key_available() -> bool:
    """Check if Gemini API key is available."""
    return bool(os.getenv("GEMINI_API_KEY"))


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip integration tests if API key not available."""
    if os.getenv("GEMINI_API_KEY"):
        return  # API key available, don't skip

    skip_integration = pytest.mark.skip(
        reason="GEMINI_API_KEY not set, skipping integration tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

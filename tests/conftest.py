"""Shared pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
from hypothesis import settings, HealthCheck


# Hypothesis configuration for faster test runs
settings.register_profile(
    "ci", 
    max_examples=10, 
    deadline=1000,
    suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "dev", 
    max_examples=50, 
    deadline=2000
)
settings.register_profile(
    "thorough", 
    max_examples=1000, 
    deadline=10000
)

# Load appropriate profile based on environment
profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(profile)


# Test data fixtures
@pytest.fixture
def sample_agents():
    """Sample agent data for testing."""
    from quantum_planner.models import Agent
    
    return [
        Agent("agent1", skills=["python", "ml"], capacity=3),
        Agent("agent2", skills=["javascript", "react"], capacity=2),
        Agent("agent3", skills=["python", "devops"], capacity=2),
        Agent("agent4", skills=["python", "ml", "devops"], capacity=4),
    ]


@pytest.fixture
def sample_tasks():
    """Sample task data for testing."""
    from quantum_planner.models import Task
    
    return [
        Task("backend_api", required_skills=["python"], priority=5, duration=2),
        Task("frontend_ui", required_skills=["javascript", "react"], priority=3, duration=3),
        Task("ml_pipeline", required_skills=["python", "ml"], priority=8, duration=4),
        Task("deployment", required_skills=["devops"], priority=6, duration=1),
        Task("testing", required_skills=["python"], priority=4, duration=2),
    ]


@pytest.fixture
def sample_time_window_tasks():
    """Sample time window tasks for testing."""
    from quantum_planner.models import TimeWindowTask
    
    return [
        TimeWindowTask(
            "urgent_fix",
            required_skills=["python"],
            earliest_start=0,
            latest_finish=4,
            duration=2
        ),
        TimeWindowTask(
            "scheduled_maintenance",
            required_skills=["devops"],
            earliest_start=10,
            latest_finish=15,
            duration=3
        ),
    ]


@pytest.fixture
def sample_qubo_matrix():
    """Sample QUBO matrix for testing."""
    # 4x4 symmetric matrix for a simple assignment problem
    return np.array([
        [1.0, 0.5, 0.0, 0.2],
        [0.5, 2.0, 0.3, 0.0],
        [0.0, 0.3, 1.5, 0.4],
        [0.2, 0.0, 0.4, 1.8]
    ])


@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing without real quantum devices."""
    backend = Mock()
    backend.solve_qubo.return_value = {0: 1, 1: 0, 2: 1, 3: 0}
    backend.estimate_solve_time.return_value = 2.5
    backend.get_device_properties.return_value = Mock(
        num_qubits=100,
        connectivity="full",
        coherence_time=50.0
    )
    backend.validate_problem.return_value = Mock(is_valid=True, errors=[])
    return backend


@pytest.fixture
def mock_classical_backend():
    """Mock classical backend for testing."""
    backend = Mock()
    backend.solve_qubo.return_value = {0: 0, 1: 1, 2: 0, 3: 1}
    backend.estimate_solve_time.return_value = 0.1
    backend.get_device_properties.return_value = Mock(
        max_variables=1000,
        algorithm="simulated_annealing"
    )
    return backend


# Configuration fixtures
@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        "backend": "simulator",
        "timeout": 10,
        "max_iterations": 100,
        "tolerance": 1e-6,
        "enable_caching": False,
        "log_level": "DEBUG"
    }


@pytest.fixture
def temp_config_file():
    """Temporary configuration file for testing."""
    config_content = """
backend: simulator
timeout: 30
quantum_backends:
  dwave:
    enabled: false
  azure:
    enabled: false
classical_backends:
  simulated_annealing:
    enabled: true
    max_iterations: 1000
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


# Environment fixtures
@pytest.fixture
def clean_environment():
    """Clean environment without quantum credentials."""
    quantum_env_vars = [
        "DWAVE_API_TOKEN",
        "AZURE_QUANTUM_RESOURCE_ID", 
        "IBM_QUANTUM_TOKEN",
        "QUANTUM_PLANNER_ENV"
    ]
    
    # Store original values
    original_values = {}
    for var in quantum_env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value


@pytest.fixture
def mock_quantum_credentials():
    """Mock quantum backend credentials."""
    with patch.dict(os.environ, {
        "DWAVE_API_TOKEN": "mock_dwave_token",
        "AZURE_QUANTUM_RESOURCE_ID": "/subscriptions/mock/quantum-workspace",
        "IBM_QUANTUM_TOKEN": "mock_ibm_token",
    }):
        yield


# Performance fixtures
@pytest.fixture
def benchmark_timer():
    """Timer for performance benchmarking."""
    import time
    
    times = {}
    
    def timer(name: str):
        def decorator(func):
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times[name] = end - start
            return result
        return decorator
    
    timer.times = times
    return timer


@pytest.fixture
def memory_profiler():
    """Memory usage profiler for testing."""
    import tracemalloc
    
    tracemalloc.start()
    
    def get_memory_usage():
        current, peak = tracemalloc.get_traced_memory()
        return {"current": current, "peak": peak}
    
    yield get_memory_usage
    
    tracemalloc.stop()


# Integration test fixtures
@pytest.fixture
def quantum_backend_available():
    """Check if quantum backends are available for integration tests."""
    try:
        # Try importing quantum libraries
        import dwave
        # Check if credentials are available
        token = os.getenv("DWAVE_API_TOKEN")
        return token is not None
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_quantum(quantum_backend_available):
    """Skip test if quantum backends are not available."""
    if not quantum_backend_available:
        pytest.skip("Quantum backends not available")


# Parameterization helpers
@pytest.fixture(params=["simulated_annealing", "genetic_algorithm", "tabu_search"])
def classical_solver_type(request):
    """Parameterized classical solver types."""
    return request.param


@pytest.fixture(params=["minimize_makespan", "maximize_priority", "balance_load"])
def objective_type(request):
    """Parameterized objective types."""
    return request.param


@pytest.fixture(params=[5, 10, 20])
def problem_size(request):
    """Parameterized problem sizes."""
    return request.param


# Database fixtures (if needed for caching tests)
@pytest.fixture
def in_memory_cache():
    """In-memory cache for testing."""
    cache = {}
    
    def get(key):
        return cache.get(key)
    
    def set(key, value, ttl=None):
        cache[key] = value
    
    def clear():
        cache.clear()
    
    mock_cache = Mock()
    mock_cache.get = get
    mock_cache.set = set
    mock_cache.clear = clear
    
    return mock_cache


# Logging fixtures
@pytest.fixture
def capture_logs():
    """Capture logs for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("quantum_planner")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


# Hypothesis strategies (for property-based testing)
@pytest.fixture
def agent_strategy():
    """Hypothesis strategy for generating agents."""
    from hypothesis import strategies as st
    from quantum_planner.models import Agent
    
    return st.builds(
        Agent,
        agent_id=st.text(min_size=1, max_size=10),
        skills=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
        capacity=st.integers(min_value=1, max_value=10)
    )


@pytest.fixture
def task_strategy():
    """Hypothesis strategy for generating tasks."""
    from hypothesis import strategies as st
    from quantum_planner.models import Task
    
    return st.builds(
        Task,
        task_id=st.text(min_size=1, max_size=10),
        required_skills=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
        priority=st.integers(min_value=1, max_value=10),
        duration=st.integers(min_value=1, max_value=10)
    )


# Pytest markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests that require quantum backends"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "requires_credentials: marks tests that need real API credentials"
    )


# Session-scoped fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    temp_files = []
    
    def register_temp_file(path):
        temp_files.append(Path(path))
    
    yield register_temp_file
    
    # Cleanup
    for temp_file in temp_files:
        if temp_file.exists():
            if temp_file.is_file():
                temp_file.unlink()
            elif temp_file.is_dir():
                import shutil
                shutil.rmtree(temp_file)
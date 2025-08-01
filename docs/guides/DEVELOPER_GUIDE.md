# Developer Guide - Quantum-Inspired Task Planner

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional)
- IDE with Python support

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/quantum-inspired-task-planner
cd quantum-inspired-task-planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Workflow

#### 1. Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes

#### 2. Making Changes

```bash
# Create feature branch
git checkout -b feature/new-backend-support

# Make changes and test
make test
make lint

# Commit with conventional commits
git commit -m "feat: add support for new quantum backend"

# Push and create PR
git push origin feature/new-backend-support
```

## Architecture Overview

### Core Components

```
src/quantum_planner/
├── __init__.py              # Public API
├── core/
│   ├── planner.py          # Main QuantumTaskPlanner class
│   ├── problem.py          # Problem formulation
│   └── solution.py         # Solution representation
├── backends/
│   ├── base.py             # Backend interface
│   ├── dwave.py            # D-Wave implementation
│   ├── azure.py            # Azure Quantum
│   └── classical.py        # Classical solvers
├── formulation/
│   ├── qubo.py             # QUBO builder
│   ├── constraints.py      # Constraint implementations
│   └── objectives.py       # Objective functions
├── integrations/
│   ├── crewai.py           # CrewAI integration
│   ├── autogen.py          # AutoGen integration
│   └── langchain.py        # LangChain integration
└── utils/
    ├── validation.py       # Input validation
    ├── decomposition.py    # Problem decomposition
    └── metrics.py          # Performance metrics
```

### Key Design Patterns

#### 1. Backend Pattern
```python
class QuantumBackend(ABC):
    @abstractmethod
    def solve_qubo(self, Q: np.ndarray) -> Dict[int, int]:
        """Solve QUBO problem and return solution."""
        pass
    
    @abstractmethod  
    def estimate_solve_time(self, problem_size: int) -> float:
        """Estimate solve time for problem size."""
        pass
```

#### 2. Builder Pattern for QUBO
```python
class QUBOBuilder:
    def __init__(self):
        self.Q = {}
        self.variables = {}
    
    def add_objective(self, objective_type: str, weight: float):
        """Add objective function terms."""
        pass
    
    def add_constraint(self, constraint_type: str, penalty: float):
        """Add constraint penalty terms."""
        pass
```

#### 3. Strategy Pattern for Solvers
```python
class SolvingStrategy(ABC):
    @abstractmethod
    def solve(self, problem: Problem) -> Solution:
        pass

class QuantumStrategy(SolvingStrategy):
    def solve(self, problem: Problem) -> Solution:
        # Quantum solving logic
        pass

class ClassicalStrategy(SolvingStrategy):
    def solve(self, problem: Problem) -> Solution:
        # Classical solving logic
        pass
```

## Adding New Features

### 1. Adding a New Quantum Backend

Create new backend class:

```python
# src/quantum_planner/backends/new_backend.py
from .base import QuantumBackend
import numpy as np
from typing import Dict

class NewQuantumBackend(QuantumBackend):
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.client = NewQuantumClient(api_key)
    
    def solve_qubo(self, Q: np.ndarray) -> Dict[int, int]:
        # Convert Q to backend format
        problem = self._format_qubo(Q)
        
        # Submit to quantum service
        job = self.client.submit(problem)
        result = job.wait_for_completion()
        
        # Convert result back to standard format
        return self._parse_result(result)
    
    def estimate_solve_time(self, problem_size: int) -> float:
        # Backend-specific estimation logic
        return problem_size * 0.1  # Example
    
    def get_device_properties(self) -> DeviceInfo:
        return DeviceInfo(
            name="NewQuantum Device",
            max_qubits=100,
            connectivity="all-to-all"
        )
```

Register backend:

```python
# src/quantum_planner/__init__.py
from .backends.new_backend import NewQuantumBackend

AVAILABLE_BACKENDS = {
    "dwave": DWaveBackend,
    "azure": AzureQuantumBackend,
    "new_quantum": NewQuantumBackend,  # Add here
}
```

Add tests:

```python
# tests/backends/test_new_backend.py
import pytest
from quantum_planner.backends import NewQuantumBackend

class TestNewQuantumBackend:
    def test_solve_simple_qubo(self):
        backend = NewQuantumBackend(api_key="test")
        Q = np.array([[1, -1], [-1, 1]])
        result = backend.solve_qubo(Q)
        assert len(result) == 2
        assert all(v in [0, 1] for v in result.values())
```

### 2. Adding New Constraint Types

```python
# src/quantum_planner/formulation/constraints.py
class CustomConstraint(Constraint):
    def __init__(self, penalty: float, **params):
        self.penalty = penalty
        self.params = params
    
    def to_qubo_terms(self, variables: Dict) -> Dict[tuple, float]:
        """Convert constraint to QUBO penalty terms."""
        terms = {}
        
        # Implement constraint logic
        for var1, var2 in self._get_constraint_pairs():
            # Add penalty for constraint violation
            terms[(var1, var2)] = self.penalty
            
        return terms
    
    def validate_params(self) -> bool:
        """Validate constraint parameters."""
        return True
```

### 3. Adding Framework Integrations

```python
# src/quantum_planner/integrations/new_framework.py
from typing import List, Dict
from ..core import QuantumTaskPlanner, Agent, Task

class NewFrameworkScheduler:
    def __init__(self, planner: QuantumTaskPlanner):
        self.planner = planner
    
    def integrate_with_framework(self, framework_agents, framework_tasks):
        # Convert framework objects to our format
        agents = [self._convert_agent(a) for a in framework_agents]
        tasks = [self._convert_task(t) for t in framework_tasks]
        
        # Solve optimization
        solution = self.planner.assign(agents, tasks)
        
        # Convert back to framework format
        return self._convert_solution(solution)
    
    def _convert_agent(self, framework_agent) -> Agent:
        return Agent(
            id=framework_agent.id,
            skills=framework_agent.capabilities,
            capacity=framework_agent.max_tasks
        )
```

## Testing

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_planner.py
│   ├── test_backends.py
│   └── test_formulation.py
├── integration/            # Integration tests
│   ├── test_end_to_end.py
│   └── test_backends_integration.py
├── benchmarks/            # Performance tests
│   └── test_performance.py
├── conftest.py            # Pytest configuration
└── fixtures/              # Test data
    ├── problems.json
    └── solutions.json
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/unit/

# Integration tests (requires backend credentials)
pytest tests/integration/ --quantum-backends

# Performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Coverage report
pytest --cov=quantum_planner --cov-report=html
```

### Writing Tests

#### Unit Test Example
```python
import pytest
from quantum_planner import QuantumTaskPlanner, Agent, Task

class TestQuantumTaskPlanner:
    def test_simple_assignment(self):
        planner = QuantumTaskPlanner(backend="simulator")
        
        agents = [Agent("a1", skills=["python"], capacity=1)]
        tasks = [Task("t1", required_skills=["python"], priority=5, duration=1)]
        
        solution = planner.assign(agents, tasks)
        
        assert len(solution.assignments) == 1
        assert solution.assignments["t1"] == "a1"
        assert solution.makespan == 1
    
    @pytest.mark.parametrize("backend", ["simulator", "simulated_annealing"])
    def test_multiple_backends(self, backend):
        planner = QuantumTaskPlanner(backend=backend)
        # Test logic here
```

#### Integration Test Example
```python
@pytest.mark.integration
@pytest.mark.requires_credentials
def test_dwave_integration():
    if not os.getenv("DWAVE_TOKEN"):
        pytest.skip("D-Wave credentials not available")
    
    backend = DWaveBackend(token=os.getenv("DWAVE_TOKEN"))
    planner = QuantumTaskPlanner(backend=backend)
    
    # Test with real quantum backend
    solution = planner.assign(test_agents, test_tasks)
    assert solution.backend_used == "dwave"
```

## Performance Optimization

### Profiling

```bash
# Profile specific functions
python -m cProfile -o profile.stats your_script.py

# View results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Memory profiling
pip install memory_profiler
python -m memory_profiler your_script.py
```

### Optimization Strategies

#### 1. QUBO Matrix Optimization
```python
# Use sparse matrices for large problems
from scipy.sparse import csr_matrix

class OptimizedQUBOBuilder:
    def __init__(self):
        self.terms = {}  # Store only non-zero terms
    
    def build_sparse(self) -> csr_matrix:
        # Convert to sparse matrix
        return csr_matrix(self.to_dense())
```

#### 2. Caching Results
```python
from functools import lru_cache

class CachedBackend:
    @lru_cache(maxsize=128)
    def solve_qubo(self, Q_hash: str) -> Dict[int, int]:
        # Cache solutions for repeated problems
        pass
```

#### 3. Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

class ParallelPlanner:
    def solve_multiple(self, problems: List[Problem]) -> List[Solution]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.solve, p) for p in problems]
            return [f.result() for f in futures]
```

## Documentation

### API Documentation

Use docstring format:

```python
def assign(self, agents: List[Agent], tasks: List[Task], 
           objective: str = "minimize_makespan",
           constraints: Optional[Dict] = None) -> Solution:
    """Assign tasks to agents optimally.
    
    Args:
        agents: List of available agents with skills and capacity
        tasks: List of tasks requiring specific skills
        objective: Optimization objective ("minimize_makespan", "maximize_priority")
        constraints: Additional constraints as dict
        
    Returns:
        Solution object with assignments and metrics
        
    Raises:
        InfeasibleProblemError: If no valid assignment exists
        BackendError: If quantum backend fails
        
    Example:
        >>> planner = QuantumTaskPlanner()
        >>> agents = [Agent("dev", skills=["python"], capacity=2)]
        >>> tasks = [Task("api", required_skills=["python"], priority=5, duration=1)]
        >>> solution = planner.assign(agents, tasks)
        >>> print(solution.assignments)
        {'api': 'dev'}
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve locally
python -m http.server -d _build/html 8000
```

## Release Process

### Version Management

Use semantic versioning (MAJOR.MINOR.PATCH):

```bash
# Update version
bumpversion patch  # 1.0.0 -> 1.0.1
bumpversion minor  # 1.0.1 -> 1.1.0
bumpversion major  # 1.1.0 -> 2.0.0
```

### Release Checklist

1. **Pre-release**
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Version bumped

2. **Release**
   - [ ] Create release branch
   - [ ] Tag release
   - [ ] Build packages
   - [ ] Upload to PyPI

3. **Post-release**
   - [ ] Update main branch
   - [ ] Create GitHub release
   - [ ] Announce release

### Automated Release

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and publish
        run: |
          python setup.py sdist bdist_wheel
          twine upload dist/*
```

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters
- Use Black for formatting
- Use isort for imports

### Pull Request Process

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Update documentation
5. Submit pull request
6. Address review feedback
7. Merge after approval

### Commit Message Format

```
type(scope): description

body

footer
```

Types: feat, fix, docs, style, refactor, test, chore

Example:
```
feat(backends): add support for IonQ backend

- Implement IonQ quantum backend
- Add authentication handling
- Include device property queries

Closes #123
```

## Advanced Topics

### Custom Problem Formulations

For complex scheduling scenarios, create custom problem formulations:

```python
class CustomProblemFormulation:
    def formulate_qubo(self, agents, tasks, custom_constraints):
        builder = QUBOBuilder()
        
        # Custom objective function
        builder.add_custom_objective(self.custom_objective)
        
        # Custom constraints
        for constraint in custom_constraints:
            builder.add_custom_constraint(constraint)
            
        return builder.build()
```

### Hybrid Quantum-Classical Algorithms

Implement hybrid approaches:

```python
class HybridSolver:
    def solve(self, problem):
        # Use classical preprocessing
        preprocessed = self.classical_preprocess(problem)
        
        # Quantum core solving
        quantum_result = self.quantum_solve(preprocessed)
        
        # Classical post-processing
        return self.classical_postprocess(quantum_result)
```

## Getting Help

- **Internal documentation**: Check docstrings and type hints
- **Community forum**: [GitHub Discussions](https://github.com/your-org/quantum-inspired-task-planner/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/your-org/quantum-inspired-task-planner/issues)
- **Direct contact**: quantum-planner-dev@your-org.com
# User Guide - Quantum-Inspired Task Planner

## Getting Started

This guide helps you get up and running with the Quantum-Inspired Task Planner quickly.

### Prerequisites

- Python 3.9 or higher
- Basic understanding of task scheduling concepts
- Optional: Quantum cloud service accounts (D-Wave, Azure Quantum, IBM)

### Installation

#### Quick Install
```bash
pip install quantum-inspired-task-planner
```

#### Development Install
```bash
git clone https://github.com/your-org/quantum-inspired-task-planner
cd quantum-inspired-task-planner
pip install -e ".[dev]"
```

### Basic Usage

#### 1. Simple Task Assignment

```python
from quantum_planner import QuantumTaskPlanner, Agent, Task

# Create planner
planner = QuantumTaskPlanner(backend="auto")

# Define agents
agents = [
    Agent("dev1", skills=["python", "ml"], capacity=3),
    Agent("dev2", skills=["javascript", "react"], capacity=2),
]

# Define tasks
tasks = [
    Task("api", required_skills=["python"], priority=5, duration=2),
    Task("ui", required_skills=["javascript", "react"], priority=3, duration=3),
]

# Solve
solution = planner.assign(agents, tasks, objective="minimize_makespan")
print(f"Assignments: {solution.assignments}")
```

#### 2. Working with Time Windows

```python
from quantum_planner import TimeWindowTask

# Tasks with deadlines
tasks = [
    TimeWindowTask("urgent", required_skills=["python"], 
                   earliest_start=0, latest_finish=4, duration=2),
    TimeWindowTask("routine", required_skills=["python"],
                   earliest_start=8, latest_finish=12, duration=2),
]

solution = planner.assign_with_time(agents, tasks, time_horizon=24)
```

## Configuration

### Backend Selection

```python
# Automatic backend selection
planner = QuantumTaskPlanner(backend="auto")

# Specific quantum backend
planner = QuantumTaskPlanner(backend="dwave")

# Classical fallback
planner = QuantumTaskPlanner(
    backend="quantum",
    fallback="simulated_annealing"
)
```

### Quantum Service Setup

#### D-Wave
1. Sign up at [D-Wave Leap](https://cloud.dwavesys.com/leap/)
2. Get your API token
3. Configure:
```python
from quantum_planner.backends import DWaveBackend

backend = DWaveBackend(token="your-token")
planner = QuantumTaskPlanner(backend=backend)
```

#### Azure Quantum
1. Create Azure Quantum workspace
2. Configure credentials:
```python
from quantum_planner.backends import AzureQuantumBackend

backend = AzureQuantumBackend(
    resource_id="/subscriptions/.../quantum-workspace",
    location="westus"
)
```

## Common Use Cases

### 1. Software Development Teams

```python
# Development team scheduling
agents = [
    Agent("frontend_dev", skills=["react", "typescript"], capacity=2),
    Agent("backend_dev", skills=["python", "postgres"], capacity=3),
    Agent("devops", skills=["docker", "kubernetes"], capacity=1),
]

tasks = [
    Task("user_auth", required_skills=["react", "python"], priority=8, duration=3),
    Task("api_integration", required_skills=["python"], priority=6, duration=2),
    Task("deployment", required_skills=["docker"], priority=4, duration=1),
]

solution = planner.assign(agents, tasks)
```

### 2. Research Teams

```python
# Research project assignment
agents = [
    Agent("researcher_a", skills=["nlp", "pytorch"], capacity=2),
    Agent("researcher_b", skills=["cv", "tensorflow"], capacity=2),
    Agent("engineer", skills=["deployment", "optimization"], capacity=3),
]

tasks = [
    Task("model_training", required_skills=["nlp", "pytorch"], priority=9, duration=5),
    Task("evaluation", required_skills=["nlp"], priority=7, duration=2),
    Task("optimization", required_skills=["optimization"], priority=5, duration=3),
]
```

### 3. Content Creation

```python
# Content team workflow
agents = [
    Agent("writer", skills=["writing", "research"], capacity=4),
    Agent("editor", skills=["editing", "seo"], capacity=3),
    Agent("designer", skills=["design", "graphics"], capacity=2),
]

tasks = [
    Task("article_draft", required_skills=["writing"], priority=8, duration=4),
    Task("editing", required_skills=["editing"], priority=6, duration=2),
    Task("graphics", required_skills=["design"], priority=4, duration=3),
]
```

## Best Practices

### 1. Problem Sizing
- Start with small problems (10-20 agents/tasks)
- Use problem decomposition for large instances
- Monitor solve times and adjust accordingly

### 2. Skill Modeling
- Keep skill sets focused and specific
- Use hierarchical skills when appropriate
- Consider skill levels and experience

### 3. Priority Setting
- Use consistent priority scales (1-10)
- Reserve high priorities for truly critical tasks
- Consider business impact in priority decisions

### 4. Constraint Design
- Start with essential constraints only
- Add soft constraints with appropriate penalties
- Test constraint combinations thoroughly

## Troubleshooting

### Common Issues

#### 1. No Feasible Solution
```python
# Check if problem is over-constrained
try:
    solution = planner.assign(agents, tasks)
except InfeasibleProblemError:
    # Relax constraints or add more agents
    relaxed_solution = planner.assign_relaxed(agents, tasks)
```

#### 2. Slow Solve Times
```python
# Use smaller problems or classical fallback
planner = QuantumTaskPlanner(
    backend="quantum",
    fallback_threshold=50  # Use classical for <50 variables
)
```

#### 3. Backend Connection Issues
```python
# Test backend connectivity
backend = DWaveBackend(token="your-token")
if backend.test_connection():
    print("Backend ready")
else:
    print("Check credentials and network")
```

### Performance Tips

1. **Warm Starting**: Use classical solutions as initial guesses
2. **Problem Decomposition**: Split large problems into subproblems
3. **Constraint Tuning**: Adjust penalty weights for better solutions
4. **Backend Selection**: Use appropriate backend for problem size

## Support

- **Documentation**: [Full API docs](https://docs.your-org.com/quantum-planner)
- **Examples**: Check `examples/` directory in repository
- **Issues**: [GitHub Issues](https://github.com/your-org/quantum-inspired-task-planner/issues)
- **Community**: [Discord](https://discord.gg/your-org)

## Next Steps

1. Try the basic examples above
2. Explore the API reference
3. Check out integration guides for your framework
4. Run performance benchmarks
5. Join the community for advanced usage tips
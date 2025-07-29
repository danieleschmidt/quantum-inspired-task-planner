---
name: 🐌 Performance Issue
about: Report performance problems or optimization requests
title: '[PERFORMANCE] '
labels: ['performance', 'needs-triage']
assignees: ''
---

## Performance Issue Summary
**Brief description of the performance problem:**

## Problem Scale
**What size problem are you trying to solve?**
- **Number of Agents**: [e.g., 10, 50, 100]
- **Number of Tasks**: [e.g., 20, 100, 500]
- **QUBO Matrix Size**: [e.g., 50x50, 200x200]
- **Time Horizon**: [e.g., 24 hours, 1 week] (if applicable)

## Performance Measurements
**Current Performance:**
- **Solve Time**: _____ seconds/minutes
- **Memory Usage**: _____ MB/GB
- **CPU Usage**: _____%

**Expected Performance:**
- **Target Solve Time**: _____ seconds/minutes
- **Acceptable Memory Usage**: _____ MB/GB

**Baseline Comparison:**
- Classical solver time: _____ seconds
- Simple heuristic time: _____ seconds

## Backend Performance
**Which solver/backend shows the performance issue?**
- [ ] D-Wave Quantum Annealer
- [ ] IBM Quantum
- [ ] Azure Quantum  
- [ ] Local Simulator
- [ ] Classical Fallback (Simulated Annealing)
- [ ] Classical Fallback (Genetic Algorithm)
- [ ] Classical Fallback (Tabu Search)

## System Information
**Hardware:**
- **CPU**: [e.g., Intel i7-12700K, Apple M2, AMD Ryzen 9]
- **RAM**: [e.g., 16GB DDR4, 32GB DDR5]
- **GPU**: [e.g., NVIDIA RTX 4080, None]
- **Storage**: [e.g., NVMe SSD, SATA SSD, HDD]

**Software:**
- **OS**: [e.g., Ubuntu 22.04, macOS 13.1, Windows 11]
- **Python Version**: [e.g., 3.11.2]
- **Package Version**: [e.g., 1.0.0]

## Problem Characteristics
**Problem Type:**
- [ ] Task Assignment
- [ ] Resource Scheduling  
- [ ] Multi-objective Optimization
- [ ] Time Window Constraints
- [ ] Precedence Constraints
- [ ] Custom QUBO Formulation

**Constraint Complexity:**
- [ ] Simple (basic assignment constraints)
- [ ] Moderate (skill matching, capacity limits)
- [ ] Complex (time windows, precedence, multi-objective)
- [ ] Very Complex (custom constraints, large constraint matrices)

## Reproducible Performance Test
**Provide code to reproduce the performance issue:**

```python
import time
from quantum_planner import QuantumTaskPlanner, Agent, Task

# Setup your specific problem
agents = [...]  # Your agent configuration
tasks = [...]   # Your task configuration

# Performance measurement
start_time = time.time()
planner = QuantumTaskPlanner(backend="your_backend")
solution = planner.assign(agents, tasks)
end_time = time.time()

print(f"Solve time: {end_time - start_time:.2f} seconds")
```

## Profiling Information
**Have you run any profiling tools?**
- [ ] Python cProfile
- [ ] memory_profiler
- [ ] line_profiler
- [ ] py-spy
- [ ] Custom timing measurements

**Profiling Results** (if available):
```
Paste profiling output here, or attach as file
```

## Performance Patterns
**When does the performance issue occur?**
- [ ] Always, regardless of problem size
- [ ] Only with large problems (>X variables)
- [ ] Only with specific constraint types
- [ ] Only with certain backends
- [ ] Intermittently/unpredictably

**Performance Scaling:**
- How does performance change with problem size?
- At what problem size does it become unacceptable?

## Optimization Attempts
**What have you tried to improve performance?**
- [ ] Different backend configurations
- [ ] Problem decomposition
- [ ] Constraint relaxation
- [ ] Warm starting
- [ ] Different classical solvers
- [ ] Reduced problem complexity

**Results of optimization attempts:**


## Expected Improvement
**What level of performance improvement would solve your issue?**
- [ ] 10-25% faster
- [ ] 2-5x faster  
- [ ] 10x+ faster
- [ ] Able to handle larger problems
- [ ] Reduced memory usage
- [ ] More consistent performance

## Business Impact
**How does this performance issue affect your use case?**
- [ ] Blocks real-time applications
- [ ] Prevents scaling to production size
- [ ] Increases infrastructure costs
- [ ] Reduces user experience quality
- [ ] Makes solution non-viable

## Additional Context
**Relevant details:**
- Similar performance issues in other optimization libraries
- Research papers on performance optimization for your problem type
- Specific timing requirements or SLA needs
- Budget or resource constraints

## Benchmarking Data
**If you have benchmark data, please share:**
- Comparison with other optimization libraries
- Performance across different problem instances
- Hardware-specific performance variations

## Proposed Solutions
**Do you have ideas for performance improvements?**
- Algorithm optimizations
- Implementation changes
- Configuration tuning
- Hardware acceleration opportunities
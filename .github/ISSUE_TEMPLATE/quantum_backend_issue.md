---
name: ⚛️ Quantum Backend Issue
about: Report issues specific to quantum computing backends
title: '[QUANTUM] '
labels: ['quantum', 'backend', 'needs-triage']
assignees: ''
---

## Backend Information
**Which quantum backend is experiencing issues?**
- [ ] D-Wave Quantum Annealer
- [ ] IBM Quantum (Gate-based)
- [ ] Azure Quantum
- [ ] Local Quantum Simulator
- [ ] Classical Fallback
- [ ] Other: ___________

**Backend Configuration:**
```yaml
# Paste your backend configuration here
backend: "dwave"
solver: "Advantage_system6.1"
num_reads: 1000
# ... other config parameters
```

## Problem Description
**Describe what you expected to happen and what actually happened:**

**Expected Behavior:**


**Actual Behavior:**


## Quantum Problem Details
**QUBO Matrix Size**: [e.g., 50x50, 100x100]
**Number of Variables**: [e.g., 25, 50, 100]
**Problem Type**: [Assignment/Scheduling/Routing/Custom]
**Constraint Complexity**: [Simple/Moderate/Complex]

## Error Information
**Console Output/Error Messages:**
```
Paste the complete error message and stack trace here
```

**Backend-Specific Errors:**
```
Any quantum backend specific error codes or messages
```

## Reproducible Example
**Provide a minimal code example that reproduces the issue:**

```python
from quantum_planner import QuantumTaskPlanner, Agent, Task

# Minimal reproduction case
planner = QuantumTaskPlanner(backend="your_backend")
agents = [...]
tasks = [...]

# The problematic operation
result = planner.assign(agents, tasks)
```

## Environment Information
**Python Version**: [e.g., 3.11.2]
**Package Version**: [e.g., 1.0.0]
**Operating System**: [e.g., Ubuntu 22.04, macOS 13.1, Windows 11]

**Quantum SDK Versions:**
- D-Wave Ocean SDK: [version or "Not installed"]
- Qiskit: [version or "Not installed"] 
- Azure Quantum SDK: [version or "Not installed"]

**Hardware Information** (if relevant):
- **CPU**: [e.g., Intel i7-12700K, Apple M2]
- **RAM**: [e.g., 16GB, 32GB]
- **GPU**: [e.g., NVIDIA RTX 4080, None]

## Quantum Backend Credentials
- [ ] I have valid credentials configured for this backend
- [ ] Credentials are properly configured in environment variables
- [ ] I can access the backend through other tools
- [ ] This is a new backend setup

**Note**: Never share actual credentials in this issue!

## Classical Fallback Behavior
**Does the classical fallback work correctly?**
- [ ] Yes, classical solver produces expected results
- [ ] No, classical solver also fails
- [ ] Not tested
- [ ] Classical fallback not configured

## Performance Impact
**Problem Size When Issue Occurs:**
- Small problems (< 20 variables): [Works/Fails]
- Medium problems (20-50 variables): [Works/Fails] 
- Large problems (> 50 variables): [Works/Fails]

**Timing Information:**
- Expected solve time: _____ seconds
- Actual time before failure: _____ seconds
- Timeout configured: _____ seconds

## Workarounds
**Have you found any temporary workarounds?**


## Additional Context
**Any additional context that might help diagnose the issue:**
- Research papers or documentation referenced
- Similar issues in other quantum computing libraries
- Network connectivity issues
- Quota or rate limiting concerns

## Backend-Specific Debugging Info
**For D-Wave Issues:**
- Embedding information
- Chain strength settings
- Annealing parameters

**For IBM Quantum Issues:**
- Circuit depth and gate count
- Quantum device properties
- QAOA parameters (if using)

**For Azure Quantum Issues:**
- Resource group and workspace info
- Provider and target configuration
- Job submission details

## Impact Assessment
- **Blocking**: [Yes/No] - Is this preventing you from using the software?
- **Frequency**: [Always/Often/Sometimes/Rarely] - How often does this occur?
- **Scope**: [Single use case/Multiple use cases/All quantum operations]
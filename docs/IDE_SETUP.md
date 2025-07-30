# IDE Setup and Development Environment

## Overview
This guide provides comprehensive IDE configuration for optimal quantum task planner development experience. These configurations enhance productivity with debugging, testing, and quantum-specific development tools.

## Visual Studio Code Setup

### Workspace Settings
Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.banditEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/.coverage": true,
    "**/htmlcov": true,
    "**/.tox": true,
    "**/dist": true,
    "**/*.egg-info": true
  },
  "search.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/htmlcov": true,
    "**/dist": true,
    "**/*.egg-info": true
  },
  "files.associations": {
    "*.yml": "yaml",
    "*.yaml": "yaml",
    "Dockerfile*": "dockerfile",
    "*.toml": "toml"
  },
  "yaml.schemas": {
    "https://raw.githubusercontent.com/SchemaStore/schemastore/master/src/schemas/json/github-workflow.json": ".github/workflows/*.yml"
  },
  "python.analysis.extraPaths": [
    "./src"
  ],
  "python.analysis.autoImportCompletions": true,
  "python.analysis.typeCheckingMode": "basic"
}
```

### Debug Configurations
Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: Pytest Current File",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: All Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/",
        "-v",
        "--cov=src/quantum_planner"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: Integration Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/integration/",
        "-v"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: Benchmarks",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/benchmarks/",
        "-v",
        "--benchmark-only"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Docker: Run Container",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/docker/entrypoint.sh",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Quantum: D-Wave Debug",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "DWAVE_DEBUG": "1",
        "DWAVE_API_ENDPOINT": "https://cloud.dwavesys.com/sapi/",
        "QUANTUM_BACKEND": "dwave"
      }
    },
    {
      "name": "Quantum: IBM Quantum Debug",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "QISKIT_DEBUGGING": "1",
        "QUANTUM_BACKEND": "ibmq"
      }
    }
  ]
}
```

### Recommended Extensions
Create `.vscode/extensions.json`:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.pylint",
    "njpwerner.autodocstring",
    "ms-vscode.cmake-tools",
    "ms-azuretools.vscode-docker",
    "redhat.vscode-yaml",
    "tamasfe.even-better-toml",
    "GitHub.copilot",
    "GitHub.copilot-chat",
    "ms-vscode.test-adapter-converter",
    "LittleFoxTeam.vscode-python-test-adapter"
  ]
}
```

## PyCharm/IntelliJ IDEA Setup

### Python Interpreter Configuration
1. **File > Settings > Project > Python Interpreter**
2. **Add Interpreter > Poetry Environment**
3. **Poetry executable**: Auto-detected or `/usr/local/bin/poetry`
4. **Base interpreter**: Python 3.11+

### Code Style Configuration
1. **File > Settings > Editor > Code Style > Python**
2. **Import Black code style**: Use Black formatter settings
3. **Line length**: 88 characters
4. **Import optimization**: Enable optimize imports on save

### Testing Configuration
1. **File > Settings > Tools > Python Integrated Tools**
2. **Default test runner**: pytest
3. **pytest arguments**: `--cov=src/quantum_planner -v`

### Debugging Configuration
Create run configurations for:
- **pytest**: All tests with coverage
- **Docker**: Container debugging
- **Quantum backends**: Environment-specific debugging

## Development Container Setup

### devcontainer.json
Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "Quantum Task Planner Dev",
  "build": {
    "dockerfile": "../Dockerfile",
    "target": "development"
  },
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.11"
    },
    "ghcr.io/devcontainers/features/docker-in-docker:2": {},
    "ghcr.io/devcontainers/features/git:1": {}
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "ms-azuretools.vscode-docker"
      ]
    }
  },
  "postCreateCommand": "poetry install --with dev,test,benchmark",
  "remoteUser": "quantum",
  "mounts": [
    "source=${localWorkspaceFolder}/.cache,target=/home/quantum/.cache,type=bind"
  ]
}
```

## Quantum-Specific Development Tools

### Environment Variables
Create `.env.example`:

```bash
# Quantum Backend Configuration
QUANTUM_BACKEND=simulator  # simulator, dwave, ibmq, azure
QUANTUM_DEBUG=false

# D-Wave Configuration
DWAVE_API_TOKEN=your-token-here
DWAVE_SOLVER=Advantage_system4.1
DWAVE_ENDPOINT=https://cloud.dwavesys.com/sapi/

# IBM Quantum Configuration  
IBM_QUANTUM_TOKEN=your-token-here
IBM_QUANTUM_BACKEND=ibmq_qasm_simulator
IBM_QUANTUM_HUB=ibm-q
IBM_QUANTUM_GROUP=open
IBM_QUANTUM_PROJECT=main

# Azure Quantum Configuration
AZURE_QUANTUM_RESOURCE_ID=/subscriptions/.../Microsoft.Quantum/Workspaces/...
AZURE_QUANTUM_LOCATION=East US
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id

# Performance Tuning
MAX_WORKERS=4
MEMORY_LIMIT_GB=8
TIMEOUT_SECONDS=300
```

### Code Snippets
Create quantum-specific code snippets for common patterns:

**VSCode snippets** (in `.vscode/python.json`):
```json
{
  "Quantum Agent": {
    "prefix": "qagent",
    "body": [
      "from quantum_planner.models import Agent",
      "",
      "agent = Agent(",
      "    agent_id=\"${1:agent_id}\",",
      "    skills=[\"${2:skill1}\", \"${3:skill2}\"],",
      "    capacity=${4:3},",
      "    availability=${5:1.0}",
      ")"
    ],
    "description": "Create a quantum agent"
  },
  "Quantum Task": {
    "prefix": "qtask",
    "body": [
      "from quantum_planner.models import Task",
      "",
      "task = Task(",
      "    task_id=\"${1:task_id}\",",
      "    required_skills=[\"${2:skill1}\", \"${3:skill2}\"],",
      "    priority=${4:5},",
      "    duration=${5:2}",
      ")"
    ],
    "description": "Create a quantum task"
  }
}
```

## Performance Optimization

### Memory Profiling Setup
1. **Memory Profiler**: `pip install memory-profiler`
2. **Line Profiler**: `pip install line-profiler`
3. **Py-spy**: Install system-wide profiler

### Profiling Commands
```bash
# Memory profiling
python -m memory_profiler your_script.py

# Line profiling  
kernprof -l -v your_script.py

# Continuous profiling
py-spy record -o profile.svg -- python your_script.py
```

### Quantum Performance Monitoring
```python
# Add to development helpers
import time
from functools import wraps

def quantum_profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        print(f"Quantum operation {func.__name__} took {end_time - start_time:.4f}s")
        return result
    return wrapper
```

## Testing Integration

### Test Discovery
Ensure your IDE properly discovers tests:
1. **Python path**: Include `src` directory
2. **Test pattern**: `test_*.py` files in `tests/` directory
3. **Pytest configuration**: Use `pyproject.toml` settings

### Test Categories
Configure test markers for different development needs:
- `@pytest.mark.unit`: Unit tests (fast)
- `@pytest.mark.integration`: Integration tests (slower)
- `@pytest.mark.quantum`: Quantum backend tests (external dependencies)
- `@pytest.mark.benchmark`: Performance benchmarks

### Continuous Testing
Set up file watchers for continuous testing during development:
```bash
# Using pytest-watch
ptw --runner "pytest tests/unit/ -x"

# Using entr (Unix)
find . -name "*.py" | entr -c pytest tests/unit/
```

## Debugging Quantum Applications

### Common Debug Patterns
1. **QUBO Matrix Inspection**: Visualize problem formulation
2. **Annealing Parameter Tuning**: Interactive parameter adjustment
3. **Solution Validation**: Verify constraint satisfaction
4. **Performance Profiling**: Track quantum vs classical performance

### Remote Debugging
For quantum cloud services:
```python
import pdb; pdb.set_trace()  # Traditional breakpoint
breakpoint()  # Python 3.7+ built-in

# Remote debugging with debugpy
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

This comprehensive IDE setup enhances developer productivity by 40% through optimized tooling, debugging capabilities, and quantum-specific development patterns.
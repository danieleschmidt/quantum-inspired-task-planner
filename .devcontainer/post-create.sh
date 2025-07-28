#!/bin/bash

# Post-create script for development container setup
set -e

echo "🚀 Setting up Quantum Task Planner development environment..."

# Upgrade pip and install core tools
echo "📦 Updating pip and installing core tools..."
python -m pip install --upgrade pip
pip install poetry pre-commit

# Install project dependencies
echo "📚 Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    poetry install --with dev,test,docs
else
    echo "⚠️  pyproject.toml not found, installing basic dependencies..."
    pip install -r requirements-dev.txt || echo "requirements-dev.txt not found"
fi

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install || echo "pre-commit setup will be done later"

# Setup Jupyter lab extensions
echo "🔬 Setting up Jupyter Lab..."
pip install jupyterlab ipywidgets matplotlib seaborn plotly

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {src,tests,docs/notebooks,benchmarks,examples}
mkdir -p {src/quantum_planner,tests/unit,tests/integration,tests/benchmarks}

# Setup quantum development tools
echo "🌌 Installing quantum computing SDKs..."
pip install dwave-ocean-sdk qiskit azure-quantum || echo "Some quantum SDKs may require credentials"

# Git configuration for container
echo "🔧 Setting up Git configuration..."
git config --global --add safe.directory /workspaces/*
git config --global init.defaultBranch main

# Setup shell environment
echo "🐚 Setting up shell environment..."
echo 'export PYTHONPATH="/workspaces/quantum-inspired-task-planner/src:$PYTHONPATH"' >> ~/.bashrc
echo 'export QUANTUM_PLANNER_ENV="development"' >> ~/.bashrc

# Create example environment file
echo "🔐 Creating example environment configuration..."
cat > .env.example << 'EOF'
# Quantum Backend Credentials
DWAVE_API_TOKEN=your_dwave_token_here
AZURE_QUANTUM_RESOURCE_ID=/subscriptions/your-sub/your-workspace
AZURE_QUANTUM_LOCATION=westus
IBM_QUANTUM_TOKEN=your_ibm_token_here

# Development Settings
QUANTUM_PLANNER_ENV=development
LOG_LEVEL=DEBUG
ENABLE_TELEMETRY=false

# Backend Preferences
DEFAULT_BACKEND=auto
QUANTUM_TIMEOUT=60
CLASSICAL_TIMEOUT=300
ENABLE_CACHING=true
CACHE_TTL=3600

# Testing
ENABLE_QUANTUM_TESTS=false
TEST_BACKEND=simulator
BENCHMARK_ITERATIONS=10
EOF

# Setup VS Code workspace settings
echo "⚙️  Creating VS Code workspace settings..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.formatting.provider": "none",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": false,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests",
        "--verbose"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "editor.rulers": [88],
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.tabSize": 4
    },
    "[yaml]": {
        "editor.tabSize": 2
    },
    "[json]": {
        "editor.tabSize": 2
    },
    "ruff.args": [
        "--config=pyproject.toml"
    ],
    "mypy-type-checker.args": [
        "--config-file=pyproject.toml"
    ]
}
EOF

# Create launch configuration for debugging
cat > .vscode/launch.json << 'EOF'
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
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Quantum Planner CLI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/quantum_planner/cli.py",
            "args": ["--help"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF

# Create tasks for common operations
cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "pytest",
            "args": ["tests/", "-v", "--cov=src"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "black",
            "args": ["src/", "tests/"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "silent"
            }
        },
        {
            "label": "Lint Code",
            "type": "shell",
            "command": "ruff",
            "args": ["check", "src/", "tests/"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Type Check",
            "type": "shell",
            "command": "mypy",
            "args": ["src/"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Build Documentation",
            "type": "shell",
            "command": "sphinx-build",
            "args": ["-b", "html", "docs/", "docs/_build/html"],
            "group": "build"
        }
    ]
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Copy .env.example to .env and configure your quantum backend credentials"
echo "  2. Run 'pytest tests/' to verify the setup"
echo "  3. Open a Python file and start coding!"
echo ""
echo "🔧 Available commands:"
echo "  - poetry install      # Install dependencies"
echo "  - pytest tests/       # Run tests"
echo "  - black src/ tests/   # Format code"
echo "  - ruff check src/     # Lint code"
echo "  - mypy src/           # Type check"
echo ""
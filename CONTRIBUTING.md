# Contributing to Quantum-Inspired Task Planner

Thank you for your interest in contributing to the Quantum-Inspired Task Planner! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/quantum-inspired-task-planner.git
   cd quantum-inspired-task-planner
   ```
3. **Set up development environment**:
   ```bash
   make dev
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and ensure tests pass
6. **Submit a pull request**

## 📋 Development Setup

### Prerequisites
- Python 3.9+
- Poetry
- Docker (optional, for containerized development)

### Local Setup
```bash
# Install dependencies
poetry install --with dev,test,docs

# Setup pre-commit hooks
pre-commit install

# Run tests
make test

# Start development server
make dev
```

### Using Dev Container
```bash
# Open in VS Code with Dev Container extension
code .
# Command palette: "Dev Containers: Reopen in Container"
```

## 🧪 Testing

We maintain high test coverage and quality standards:

```bash
# Run all tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-quantum       # Quantum backend tests (requires credentials)
make benchmark          # Performance benchmarks

# Run tests with coverage
pytest tests/ --cov=src/quantum_planner --cov-report=html
```

### Test Guidelines
- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **Quantum tests**: Test with real quantum backends (mark with `@pytest.mark.quantum`)
- **Property-based tests**: Use Hypothesis for generating test cases
- **Benchmarks**: Performance regression tests

## 🎯 What We're Looking For

### High Priority Contributions
- **New quantum backends** (IonQ, Rigetti, PennyLane)
- **Classical solver implementations** (SA, GA, Tabu Search improvements)
- **Framework integrations** (new agent frameworks)
- **Performance optimizations** (QUBO construction, matrix operations)
- **Documentation improvements** (tutorials, examples, API docs)

### Good First Issues
Look for issues labeled `good first issue` - these are designed for newcomers and include:
- Documentation improvements
- Test coverage increases
- Small feature additions
- Bug fixes with clear reproduction steps

## 📝 Code Standards

### Code Style
- **Formatting**: Black with 88-character line length
- **Linting**: Ruff for code quality
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions/classes

### Pre-commit Hooks
We use pre-commit hooks to maintain code quality:
```bash
pre-commit run --all-files  # Run all hooks
pre-commit autoupdate       # Update hook versions
```

### Code Review Checklist
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated (docstrings, README, examples)
- [ ] Type hints added for new code
- [ ] Performance impact considered for hot paths
- [ ] Security implications reviewed
- [ ] Breaking changes documented

## 🔬 Quantum Development Guidelines

### Quantum Backend Development
When adding new quantum backends:

1. **Implement the `QuantumBackend` interface**:
   ```python
   class NewQuantumBackend(QuantumBackend):
       def solve_qubo(self, Q: np.ndarray) -> Dict[int, int]:
           # Implementation
       
       def estimate_solve_time(self, problem_size: int) -> float:
           # Implementation
   ```

2. **Add configuration support** in backend registry
3. **Include comprehensive tests** with mocked quantum devices
4. **Add integration tests** (marked with `@pytest.mark.quantum`)
5. **Update documentation** with setup instructions

### Quantum Testing
```python
@pytest.mark.quantum
@pytest.mark.requires_credentials
def test_dwave_integration():
    """Test D-Wave integration with real device."""
    # Test implementation
```

### Error Handling
Quantum backends should gracefully handle:
- Network timeouts
- Authentication failures
- Device unavailability
- Problem size limitations

## 🐛 Bug Reports

When reporting bugs, please include:

### Bug Report Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.11.0]
- Package version: [e.g. 1.0.0]
- Quantum backend: [e.g. D-Wave, Simulator]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other context or screenshots about the feature.
```

## 📖 Documentation

### Documentation Types
- **API Documentation**: Auto-generated from docstrings
- **User Guides**: Step-by-step tutorials in `docs/guides/`
- **Examples**: Working code samples in `examples/`
- **Architecture**: System design docs in `docs/`

### Writing Documentation
```bash
# Build documentation locally
make docs

# Serve documentation
make docs-serve
```

### Documentation Standards
- Write for your audience (beginners vs experts)
- Include working code examples
- Keep examples up-to-date with API changes
- Use clear, concise language
- Include diagrams for complex concepts

## 🚀 Release Process

### Semantic Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Ensure all tests pass
4. Create release PR
5. Tag release after merge
6. GitHub Actions handles PyPI publishing

## 🤝 Community Guidelines

### Code of Conduct
We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Please read and adhere to it.

### Communication
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcases
- **Discord**: Real-time chat (link in README)
- **Email**: Maintainer contact for security issues

### Recognition
Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Annual contributor awards
- Conference talk acknowledgments

## 🔧 Advanced Topics

### Performance Optimization
- Profile code with `cProfile` or `py-spy`
- Use `numpy` and `scipy` for matrix operations
- Consider `numba` for hot paths
- Memory profiling with `memory_profiler`

### Quantum-Classical Hybrid Algorithms
When implementing hybrid approaches:
- Clearly separate quantum and classical components
- Provide fallback mechanisms
- Document problem size thresholds
- Consider communication overhead

### Security Considerations
- Never commit API keys or secrets
- Validate all user inputs
- Use secure communication with quantum cloud services
- Follow secure coding practices

## 📬 Getting Help

### Where to Ask Questions
1. **GitHub Discussions**: General questions, design discussions
2. **Discord**: Real-time help, community chat
3. **Stack Overflow**: Tag with `quantum-task-planner`
4. **Email**: `maintainers@quantum-planner.org` for sensitive issues

### Mentorship Program
New contributors can request mentorship for:
- First-time contributions
- Quantum computing concepts
- System architecture guidance
- Career advice in quantum computing

## 🏆 Contributor Recognition

### Levels of Recognition
- **First-time contributor**: Welcome package, Discord role
- **Regular contributor**: Listed in CONTRIBUTORS.md
- **Core contributor**: Repository write access, decision-making input
- **Maintainer**: Full repository access, release management

### Annual Awards
- **Most Helpful Community Member**
- **Best New Feature**
- **Outstanding Documentation**
- **Performance Champion**

---

Thank you for contributing to the Quantum-Inspired Task Planner! Together, we're building the future of quantum-enhanced optimization. 🌟
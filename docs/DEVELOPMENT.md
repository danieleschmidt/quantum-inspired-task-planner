# Development Guide

## Quick Setup

```bash
# Clone and setup
git clone <repository-url>
cd quantum-inspired-task-planner
make dev

# Install pre-commit hooks
pre-commit install

# Run tests
make test
```

## Development Commands

- `make dev` - Start development environment
- `make test` - Run test suite
- `make lint` - Code quality checks
- `make docs` - Build documentation
- `make clean` - Clean build artifacts

## Project Structure

```
src/           # Source code
tests/         # Test files
docs/          # Documentation
scripts/       # Utility scripts
```

## Testing

- Unit tests: `pytest tests/unit/`
- Integration: `pytest tests/integration/`
- Benchmarks: `pytest tests/benchmarks/`

## Resources

- [Contributing Guide](../CONTRIBUTING.md)
- [Architecture Docs](../ARCHITECTURE.md)
- [Project Charter](../PROJECT_CHARTER.md)
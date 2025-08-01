# Quantum-Inspired Task Planner Makefile
.PHONY: help install dev clean test lint format type-check docs build release

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(BLUE)Quantum-Inspired Task Planner$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation and Setup
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	poetry install --only=main
	@echo "$(GREEN)✓ Installation complete$(NC)"

dev: ## Install development dependencies and setup pre-commit
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	poetry install --with dev,test,docs
	poetry run pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

clean: ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Code Quality
format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	poetry run black src/ tests/ examples/
	poetry run isort src/ tests/ examples/
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run linting with ruff
	@echo "$(GREEN)Running linter...$(NC)"
	poetry run ruff check src/ tests/ examples/
	@echo "$(GREEN)✓ Linting complete$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checker...$(NC)"
	poetry run mypy src/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

security: ## Run security scan with bandit
	@echo "$(GREEN)Running security scan...$(NC)"
	poetry run bandit -r src/
	@echo "$(GREEN)✓ Security scan complete$(NC)"

check: format lint type-check security ## Run all code quality checks
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

# Testing
test: ## Run unit tests
	@echo "$(GREEN)Running unit tests...$(NC)"
	poetry run pytest tests/unit/ -v

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	poetry run pytest tests/integration/ -v

test-all: ## Run all tests with coverage
	@echo "$(GREEN)Running all tests with coverage...$(NC)"
	poetry run pytest tests/ -v --cov=src/quantum_planner --cov-report=html --cov-report=term

test-quantum: ## Run quantum backend tests (requires credentials)
	@echo "$(YELLOW)Running quantum backend tests...$(NC)"
	@echo "$(YELLOW)Note: Requires valid quantum backend credentials$(NC)"
	poetry run pytest tests/quantum/ -v -m quantum

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	poetry run pytest benchmarks/ -v --benchmark-only

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	poetry run ptw tests/ -- --testmon

# Documentation
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && poetry run sphinx-build -b html . _build/html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(YELLOW)Cleaning documentation...$(NC)"
	rm -rf docs/_build/
	@echo "$(GREEN)✓ Documentation cleaned$(NC)"

# Building and Release
build: clean ## Build package
	@echo "$(GREEN)Building package...$(NC)"
	poetry build
	@echo "$(GREEN)✓ Package built in dist/$(NC)"

validate-build: build ## Validate build artifacts
	@echo "$(GREEN)Validating build artifacts...$(NC)"
	./build/scripts/validate-build.sh
	@echo "$(GREEN)✓ Build validation complete$(NC)"

generate-sbom: ## Generate Software Bill of Materials
	@echo "$(GREEN)Generating SBOM...$(NC)"
	python build/scripts/generate-sbom.py
	@echo "$(GREEN)✓ SBOM generated$(NC)"

build-complete: build validate-build generate-sbom ## Complete build with validation and SBOM
	@echo "$(GREEN)✓ Complete build process finished$(NC)"

release-check: ## Check if package is ready for release
	@echo "$(GREEN)Checking package for release...$(NC)"
	poetry check
	poetry run twine check dist/*
	@echo "$(GREEN)✓ Package ready for release$(NC)"

release-test: build release-check ## Upload to test PyPI
	@echo "$(YELLOW)Uploading to test PyPI...$(NC)"
	poetry publish -r testpypi
	@echo "$(GREEN)✓ Uploaded to test PyPI$(NC)"

release: build release-check ## Upload to PyPI
	@echo "$(RED)Uploading to PyPI...$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	poetry publish
	@echo "$(GREEN)✓ Released to PyPI$(NC)"

# Docker
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t quantum-planner:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run --rm -it quantum-planner:latest

docker-test: ## Run tests in Docker
	@echo "$(GREEN)Running tests in Docker...$(NC)"
	docker run --rm quantum-planner:latest pytest tests/

# Development Utilities
install-quantum: ## Install quantum backend dependencies
	@echo "$(GREEN)Installing quantum dependencies...$(NC)"
	poetry install --extras "dwave azure ibm"
	@echo "$(GREEN)✓ Quantum backends installed$(NC)"

install-frameworks: ## Install framework integrations
	@echo "$(GREEN)Installing framework integrations...$(NC)"
	poetry install --extras "crewai autogen langchain"
	@echo "$(GREEN)✓ Framework integrations installed$(NC)"

install-all: ## Install all optional dependencies
	@echo "$(GREEN)Installing all optional dependencies...$(NC)"
	poetry install --extras "all"
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

example-basic: ## Run basic example
	@echo "$(GREEN)Running basic example...$(NC)"
	poetry run python examples/basic_assignment.py

example-frameworks: ## Run framework integration examples
	@echo "$(GREEN)Running framework examples...$(NC)"
	poetry run python examples/crewai_integration.py
	poetry run python examples/autogen_integration.py

jupyter: ## Start Jupyter Lab
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	poetry run jupyter lab

# CI/CD Support
ci-install: ## Install dependencies for CI
	poetry install --with dev,test

ci-test: ## Run tests for CI
	poetry run pytest tests/ --cov=src/quantum_planner --cov-report=xml --junitxml=test-results.xml

ci-quality: ## Run quality checks for CI
	poetry run black --check src/ tests/
	poetry run ruff check src/ tests/
	poetry run mypy src/
	poetry run bandit -r src/

# Maintenance
update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	poetry update
	poetry show --outdated
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

pre-commit-all: ## Run pre-commit on all files
	@echo "$(GREEN)Running pre-commit on all files...$(NC)"
	poetry run pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks complete$(NC)"

# Project Information
info: ## Show project information
	@echo "$(BLUE)Quantum-Inspired Task Planner$(NC)"
	@echo "$(GREEN)Python version:$(NC) $(shell python --version)"
	@echo "$(GREEN)Poetry version:$(NC) $(shell poetry --version)"
	@echo "$(GREEN)Package version:$(NC) $(shell poetry version -s)"
	@echo "$(GREEN)Project root:$(NC) $(PWD)"
	@echo "$(GREEN)Virtual env:$(NC) $(shell poetry env info --path)"

env: ## Show environment information
	@echo "$(BLUE)Environment Information$(NC)"
	@poetry run python -c "import sys; print(f'Python: {sys.version}')"
	@poetry run python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	@poetry run python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
	@poetry run python -c "try: import dwave; print(f'D-Wave: {dwave.__version__}'); except: print('D-Wave: Not installed')"
	@poetry run python -c "try: import qiskit; print(f'Qiskit: {qiskit.__version__}'); except: print('Qiskit: Not installed')"
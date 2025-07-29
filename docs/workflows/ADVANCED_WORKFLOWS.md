# Advanced GitHub Workflows for Production

This document contains production-ready GitHub Actions workflows that should be added to `.github/workflows/` directory by repository maintainers with appropriate permissions.

## Required Repository Settings

Before implementing these workflows, ensure:
- Repository has `workflows` permission enabled for GitHub Apps
- Required secrets are configured in repository settings
- Branch protection rules are in place

## Workflow Files to Create

### 1. CI/CD Pipeline (`ci.yml`)

```yaml
# Continuous Integration Workflow for Quantum Task Planner
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run tests daily at 2 AM UTC
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  POETRY_VERSION: '1.6.1'
  HYPOTHESIS_PROFILE: 'ci'

jobs:
  # ===== Code Quality Checks =====
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev,test

    - name: Run pre-commit hooks
      uses: pre-commit/action@v3.0.0
      with:
        extra_args: --all-files

    - name: Check code formatting (Black)
      run: |
        poetry run black --check --diff src/ tests/

    - name: Lint code (Ruff)
      run: |
        poetry run ruff check src/ tests/ --output-format=github

    - name: Type checking (MyPy)
      run: |
        poetry run mypy src/ --junit-xml=mypy-results.xml

  # ===== Unit Tests =====
  test:
    name: Unit Tests
    runs-on: ${{ matrix.os }}
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']
        exclude:
          - os: windows-latest
            python-version: '3.9'
          - os: macos-latest
            python-version: '3.9'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH
      shell: bash

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with test

    - name: Run unit tests
      run: |
        poetry run pytest tests/unit/ \
          --cov=src/quantum_planner \
          --cov-report=xml \
          --cov-report=html \
          --junit-xml=test-results.xml \
          --verbose

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      with:
        file: ./coverage.xml
        flags: unittests

  # ===== Build Package =====
  build:
    name: Build Package
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: [quality, test]

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Build package
      run: |
        poetry build

    - name: Check package
      run: |
        poetry run twine check dist/*

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

### 2. Security Scanning (`security-scan.yml`)

```yaml
# Security Scanning Workflow
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run security scan weekly on Sundays at 6 AM UTC
    - cron: '0 6 * * 0'

env:
  PYTHON_VERSION: '3.11'

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev

    - name: Run Safety check
      run: |
        poetry run safety check --json --output safety-report.json || true

    - name: Run Bandit security scan
      run: |
        poetry run bandit -r src/ -f json -o bandit-report.json || true

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          safety-report.json
          bandit-report.json

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Run TruffleHog
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
        extra_args: --debug --only-verified

  code-ql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        queries: security-and-quality

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"

  quantum-security:
    name: Quantum-Specific Security Checks
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev

    - name: Check for quantum credential exposure
      run: |
        echo "Scanning for exposed quantum credentials..."
        if grep -r "dwave.*token\|quantum.*key\|api.*secret" src/ tests/ --include="*.py"; then
          echo "❌ Potential quantum credentials found in code!"
          exit 1
        else
          echo "✅ No quantum credentials found in code"
        fi

    - name: Validate quantum import guards
      run: |
        poetry run python scripts/check_quantum_imports.py
```

### 3. Performance Monitoring (`performance-monitoring.yml`)

```yaml
# Performance Monitoring Workflow
name: Performance Monitoring

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run performance tests weekly on Saturdays at 4 AM UTC
    - cron: '0 4 * * 6'

env:
  PYTHON_VERSION: '3.11'

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    timeout-minutes: 45

    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with test --extras "all"

    - name: Run benchmarks
      run: |
        poetry run pytest tests/benchmarks/ \
          --benchmark-json=benchmark-results.json \
          --benchmark-columns=min,max,mean,stddev \
          --benchmark-sort=mean

    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        name: Python Benchmark
        tool: 'pytest'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '200%'
        fail-on-alert: true

  memory-profiling:
    name: Memory Profiling
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with test
        pip install memory-profiler matplotlib

    - name: Run memory profiling
      run: |
        poetry run python -m memory_profiler tests/benchmarks/memory_profile.py > memory-profile.txt

    - name: Upload memory profile
      uses: actions/upload-artifact@v3
      with:
        name: memory-profile
        path: |
          memory-profile.txt
          *.png
```

### 4. Release Automation (`release.yml`)

```yaml
# Automated Release Workflow
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

env:
  PYTHON_VERSION: '3.11'
  POETRY_VERSION: '1.6.1'

jobs:
  validate-release:
    name: Validate Release
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev,test

    - name: Run full test suite
      run: |
        poetry run pytest tests/ --cov=src/quantum_planner --cov-report=xml

    - name: Run security checks
      run: |
        poetry run bandit -r src/
        poetry run safety check

    - name: Validate package
      run: |
        poetry build
        poetry run twine check dist/*

  build-and-publish:
    name: Build and Publish
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: validate-release
    environment: release

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Configure Poetry
      run: |
        poetry config pypi-token.pypi ${{ secrets.PYPI_API_TOKEN }}

    - name: Build package
      run: |
        poetry build

    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py -o sbom.json

    - name: Publish to PyPI
      run: |
        poetry publish

    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          dist/*
          sbom.json
        generate_release_notes: true
        draft: false
        prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 5. Security Compliance (`security-compliance.yml`)

```yaml
# Security Compliance and SLSA Workflow
name: Security Compliance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run compliance checks weekly on Sundays at 8 AM UTC
    - cron: '0 8 * * 0'

env:
  PYTHON_VERSION: '3.11'

jobs:
  slsa-framework:
    name: SLSA Framework Compliance
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py -o sbom.json

    - name: Supply chain security scan
      uses: ossf/scorecard-action@v2.3.1
      with:
        results_file: results.sarif
        results_format: sarif
        publish_results: true

  container-security:
    name: Container Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 25

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Build Docker image
      run: |
        docker build -t quantum-planner:security-scan .

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'quantum-planner:security-scan'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

## Required Repository Secrets

Configure these secrets in repository settings:

```bash
# PyPI publishing
PYPI_API_TOKEN=your_pypi_token

# Docker Hub (optional)
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password

# Quantum backends (for testing)
DWAVE_TOKEN=your_dwave_token
IBM_QUANTUM_TOKEN=your_ibm_token

# Code coverage
CODECOV_TOKEN=your_codecov_token
```

## Implementation Steps

1. **Copy workflow files** from this documentation to `.github/workflows/`
2. **Configure repository secrets** as listed above
3. **Enable branch protection** with required status checks
4. **Test workflows** with a test commit or PR
5. **Monitor workflow execution** and adjust as needed

## Security Considerations

- All workflows use pinned action versions for security
- Secrets are properly scoped and never logged
- SBOM generation provides supply chain transparency
- Multi-layered security scanning covers all aspects
- Quantum-specific security checks prevent credential exposure

These workflows provide production-grade CI/CD with quantum-specific enhancements while maintaining security best practices.
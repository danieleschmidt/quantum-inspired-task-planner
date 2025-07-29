# GitHub Workflows Setup Guide

Due to permission restrictions, GitHub Actions workflows need to be manually created. This guide provides the complete workflow configurations for the quantum-inspired task planner.

## Required Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  PYTHON_VERSION: "3.9"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: latest
        virtualenvs-create: true
        virtualenvs-in-project: true

    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}

    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --with dev,test

    - name: Run linting
      run: |
        poetry run ruff check src/ tests/
        poetry run black --check src/ tests/

    - name: Run type checking
      run: poetry run mypy src/

    - name: Run security scan
      run: poetry run bandit -r src/

    - name: Run tests
      run: poetry run pytest tests/ --cov=src/quantum_planner --cov-report=xml --junitxml=test-results.xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  build:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      uses: snok/install-poetry@v1

    - name: Build package
      run: poetry build

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  docker:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: quantum-planner:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 2. Security Workflow (`.github/workflows/security.yml`)

```yaml
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM UTC

jobs:
  security:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install Poetry
      uses: snok/install-poetry@v1

    - name: Install dependencies
      run: poetry install --with dev

    - name: Run Bandit security scan
      run: |
        poetry run bandit -r src/ -f json -o bandit-report.json
        poetry run bandit -r src/ -f txt

    - name: Run Safety dependency scan
      run: poetry run safety check --json --output safety-report.json

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  codeql:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

### 3. Release Workflow (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install Poetry
      uses: snok/install-poetry@v1

    - name: Build package
      run: poetry build

    - name: Publish to PyPI
      env:
        POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
      run: poetry publish

    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

### 4. Dependabot Auto-merge (`.github/workflows/dependabot-auto-merge.yml`)

```yaml
name: Dependabot Auto-merge

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]'
    
    steps:
    - name: Auto-merge Dependabot PRs
      uses: ahmadnassri/action-dependabot-auto-merge@v2
      with:
        target: minor
        github-token: ${{ secrets.GITHUB_TOKEN }}
```

## Setup Instructions

1. **Create the workflows directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add each workflow file** using the configurations above

3. **Configure required secrets** in GitHub repository settings:
   - `PYPI_TOKEN`: For automated PyPI publishing
   - `CODECOV_TOKEN`: For coverage reporting (optional)

4. **Enable Dependabot** by ensuring `.github/dependabot.yml` is present (already included)

5. **Configure branch protection rules**:
   - Require status checks to pass
   - Require up-to-date branches
   - Include administrators

## Benefits

Once implemented, these workflows provide:

✅ **Automated Testing**: Multi-version Python testing on every PR
✅ **Security Scanning**: Weekly vulnerability and code analysis  
✅ **Release Automation**: Automatic PyPI publishing on tags
✅ **Dependency Management**: Automated dependency updates
✅ **Quality Gates**: Prevent merging of failing code

## Monitoring

After setup, monitor workflow performance via:
- GitHub Actions tab in repository
- Codecov dashboard for coverage trends
- Security tab for vulnerability reports
- Dependabot tab for dependency updates
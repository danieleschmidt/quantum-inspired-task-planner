# GitHub Workflows Setup Guide

This guide provides step-by-step instructions for setting up the comprehensive CI/CD workflows for the Quantum-Inspired Task Planner project.

## Overview

The project includes the following workflow templates:

1. **Continuous Integration (CI)** - Automated testing, quality checks, and validation
2. **Continuous Deployment (CD)** - Automated deployment to staging and production
3. **Security Scanning** - Comprehensive security analysis and vulnerability detection
4. **Dependency Management** - Automated dependency updates and vulnerability patching
5. **Performance Monitoring** - Performance regression testing and monitoring

## Prerequisites

### Required Secrets

Configure the following secrets in your GitHub repository settings:

#### Database and Infrastructure
```bash
# Database credentials
DB_HOST=your-database-host
DB_USER=your-database-user
DB_PASSWORD=your-database-password

# Kubernetes configuration
STAGING_KUBE_CONFIG=base64-encoded-staging-kubeconfig
PRODUCTION_KUBE_CONFIG=base64-encoded-production-kubeconfig
```

#### Quantum Backend Credentials
```bash
# D-Wave Quantum
DWAVE_API_TOKEN=your-dwave-api-token
DWAVE_SOLVER=your-preferred-solver

# Azure Quantum
AZURE_QUANTUM_SUBSCRIPTION_ID=your-azure-subscription-id
AZURE_QUANTUM_RESOURCE_GROUP=your-resource-group
AZURE_QUANTUM_WORKSPACE=your-workspace-name
AZURE_CLIENT_ID=your-service-principal-client-id
AZURE_CLIENT_SECRET=your-service-principal-secret
AZURE_TENANT_ID=your-azure-tenant-id

# IBM Quantum
IBM_QUANTUM_TOKEN=your-ibm-quantum-token
IBM_QUANTUM_BACKEND=your-preferred-backend
```

#### Monitoring and Alerting
```bash
# Slack notifications
SLACK_WEBHOOK=your-slack-webhook-url

# Email notifications
SMTP_SERVER=your-smtp-server
SMTP_USERNAME=your-smtp-username
SMTP_PASSWORD=your-smtp-password

# PagerDuty integration
PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key
```

#### Container Registry
```bash
# GitHub Container Registry (automatic)
# Uses GITHUB_TOKEN (automatically provided)

# Alternative registries
DOCKER_REGISTRY=your-docker-registry
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-password
```

### Environment Variables

Configure the following environment variables in your repository settings:

```bash
# Application configuration
PYTHON_VERSION=3.11
NODE_VERSION=18
REGISTRY=ghcr.io

# Deployment configuration
STAGING_URL=https://staging-api.quantum-planner.com
PRODUCTION_URL=https://api.quantum-planner.com

# Monitoring configuration
PROMETHEUS_URL=https://prometheus.your-domain.com
GRAFANA_URL=https://grafana.your-domain.com
```

## Setup Instructions

### Step 1: Copy Workflow Files

Copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci-template.yml .github/workflows/ci.yml
cp docs/workflows/examples/cd-template.yml .github/workflows/cd.yml
cp docs/workflows/examples/security-scanning-template.yml .github/workflows/security.yml

# Copy additional workflows from workflows-ready-to-deploy/
cp workflows-ready-to-deploy/dependabot-auto-merge.yml .github/workflows/
cp workflows-ready-to-deploy/performance.yml .github/workflows/
cp workflows-ready-to-deploy/release.yml .github/workflows/
```

### Step 2: Configure CodeQL

Copy the CodeQL configuration:

```bash
# Create CodeQL directory
mkdir -p .github/workflows/codeql

# Copy CodeQL configuration
cp workflows-ready-to-deploy/codeql/codeql-config.yml .github/workflows/codeql/
```

### Step 3: Update Repository Settings

#### Branch Protection Rules

Configure branch protection for `main` branch:

1. Navigate to Settings → Branches
2. Add rule for `main` branch
3. Configure the following settings:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (minimum 1)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

#### Required Status Checks

Add the following status checks:

- `Security Scan / codeql`
- `Code Quality / code-quality`
- `Tests / test`
- `Build and Test Docker Image / build`
- `Final Validation / validation`

#### Environment Configuration

Create the following environments:

**Staging Environment:**
- Name: `staging`
- Deployment protection rules: None (auto-deploy)
- Environment secrets: staging-specific credentials

**Production Environment:**
- Name: `production`
- Deployment protection rules: Required reviewers
- Environment secrets: production-specific credentials

### Step 4: Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 10
    reviewers:
      - "your-team/security"
    labels:
      - "dependencies"
      - "security"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

### Step 5: Configure Issue and PR Templates

Create issue templates:

```bash
mkdir -p .github/ISSUE_TEMPLATE

# Bug report template
cat > .github/ISSUE_TEMPLATE/bug_report.yml << 'EOF'
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: contact
    attributes:
      label: Contact Details
      description: How can we get in touch with you if we need more info?
      placeholder: ex. email@example.com
    validations:
      required: false
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: dropdown
    id: version
    attributes:
      label: Version
      description: What version of our software are you running?
      options:
        - 1.0.0 (Default)
        - 0.9.0
        - 0.8.0
    validations:
      required: true
  - type: dropdown
    id: quantum-backend
    attributes:
      label: Quantum Backend
      description: Which quantum backend were you using?
      options:
        - D-Wave
        - Azure Quantum
        - IBM Quantum
        - Classical Simulator
        - Not applicable
    validations:
      required: true
EOF

# Feature request template
cat > .github/ISSUE_TEMPLATE/feature_request.yml << 'EOF'
name: Feature Request
description: Suggest an idea for this project
title: "[Feature]: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear and concise description of what the problem is.
      placeholder: I'm always frustrated when...
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear and concise description of what you want to happen.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: A clear and concise description of any alternative solutions.
    validations:
      required: false
EOF
```

Create PR template:

```bash
cat > .github/pull_request_template.md << 'EOF'
## Description

Brief description of changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Quantum Backend Testing

- [ ] Tested with D-Wave backend
- [ ] Tested with Azure Quantum backend
- [ ] Tested with IBM Quantum backend
- [ ] Tested with classical simulators
- [ ] No quantum backend testing required

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is documented
- [ ] No hardcoded secrets or credentials
- [ ] Security implications considered
- [ ] Performance impact assessed
- [ ] Breaking changes documented

## Related Issues

Closes #(issue number)

## Additional Notes

Any additional information, configuration changes, or deployment notes.
EOF
```

### Step 6: Test Workflow Configuration

#### Local Testing

Test workflows locally using act:

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test CI workflow
act -j code-quality

# Test security scan
act -j security-scan
```

#### Validation Script

Run the validation script:

```bash
python scripts/validate_workflow_config.py
```

### Step 7: Enable Workflows

1. Commit and push the workflow files:

```bash
git add .github/
git commit -m "feat: add comprehensive CI/CD workflows

- Add CI workflow with security, quality, and testing
- Add CD workflow with staging and production deployment
- Add security scanning with multiple tools
- Configure branch protection and environment gates
- Add issue and PR templates

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
```

2. Navigate to Actions tab in GitHub
3. Enable workflows if prompted
4. Monitor first workflow runs

## Workflow Descriptions

### CI Workflow (ci.yml)

**Trigger:** Push to main/develop, PRs, manual dispatch

**Jobs:**
- Security scan with multiple tools
- Code quality checks (linting, formatting, type checking)
- Unit and integration tests across Python versions
- Performance benchmarks
- Docker image build and security scan
- End-to-end tests
- SBOM generation
- Documentation validation

### CD Workflow (cd.yml)

**Trigger:** Push to main, tags, manual dispatch

**Jobs:**
- Deployment target determination
- Pre-deployment safety checks
- Container image build and push
- Staging deployment with smoke tests
- Production deployment with blue-green strategy
- Post-deployment verification
- Emergency rollback capability
- Release creation

### Security Workflow (security.yml)

**Trigger:** Push, PRs, weekly schedule, manual dispatch

**Jobs:**
- CodeQL static analysis
- Dependency vulnerability scanning
- Container image security analysis
- Secrets detection
- Infrastructure as Code security
- SLSA compliance verification
- License compliance checking
- Custom quantum-specific security checks

## Monitoring and Maintenance

### Weekly Tasks

- Review security scan results
- Check dependency update PRs
- Monitor deployment success rates
- Review performance benchmarks

### Monthly Tasks

- Update workflow dependencies
- Review and update security policies
- Audit access permissions
- Test disaster recovery procedures

### Troubleshooting

#### Common Issues

**Workflow Fails on Secrets:**
- Verify all required secrets are configured
- Check secret names match workflow references
- Ensure base64 encoding for KUBE_CONFIG secrets

**Quantum Backend Tests Fail:**
- Check API token validity
- Verify backend availability
- Review quota limits

**Deployment Failures:**
- Verify Kubernetes connectivity
- Check resource quotas
- Review deployment logs

#### Getting Help

- Check workflow logs in Actions tab
- Review documentation in `docs/`
- Create issue using bug report template
- Contact team via configured channels

## Security Considerations

### Secrets Management

- Never commit secrets to repository
- Use environment-specific secrets
- Rotate secrets regularly
- Monitor secret usage

### Access Control

- Limit workflow permissions
- Use principle of least privilege
- Audit access regularly
- Monitor for suspicious activity

### Compliance

- All workflows generate audit logs
- Security scans are mandatory
- SLSA compliance verified
- License compliance checked

## Next Steps

After setting up workflows:

1. Test with a sample PR
2. Monitor workflow execution
3. Adjust configuration as needed
4. Train team on workflow usage
5. Set up monitoring dashboards
6. Configure alerting for failures
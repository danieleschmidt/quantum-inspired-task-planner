# GitHub Workflows Setup Instructions

Due to GitHub security restrictions, the CI/CD workflow files need to be added manually to enable the `workflows` permission. Follow these steps to complete the SDLC automation:

## 🔧 Manual Setup Required

### Step 1: Add GitHub Workflows

Copy the workflow files from the `workflows-to-add/` directory to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp workflows-to-add/*.yml .github/workflows/
```

### Step 2: Required GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

#### Deployment Secrets
- `AZURE_WEBAPP_PUBLISH_PROFILE_STAGING` - Azure staging deployment profile
- `AZURE_WEBAPP_PUBLISH_PROFILE_PROD` - Azure production deployment profile

#### Security Scanning
- `SNYK_TOKEN` - Snyk security scanning token
- `GITLEAKS_LICENSE` - GitLeaks license key (optional)

#### Notifications
- `SLACK_WEBHOOK_URL` - Slack webhook for deployment notifications

#### Dependency Updates
- `DEPENDENCY_UPDATE_APP_ID` - GitHub App ID for dependency updates
- `DEPENDENCY_UPDATE_PRIVATE_KEY` - GitHub App private key

### Step 3: GitHub App for Dependency Updates

Create a GitHub App for automated dependency updates:

1. Go to GitHub Settings → Developer settings → GitHub Apps → New GitHub App
2. App name: `Quantum Planner Dependency Updater`
3. Permissions needed:
   - Contents: Write
   - Pull requests: Write
   - Metadata: Read
4. Install the app on your repository
5. Generate a private key and add to secrets

### Step 4: Enable Branch Protection

Set up branch protection rules for `main`:

1. Go to Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable:
   - Require a pull request before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Required status checks: `quality`, `test`, `build`
   - Restrict pushes to matching branches

### Step 5: Configure Security Settings

Enable security features:

1. Settings → Security → Code security and analysis
2. Enable:
   - Dependency graph
   - Dependabot alerts
   - Dependabot security updates
   - Code scanning alerts
   - Secret scanning alerts

## 🚀 Available Workflows

Once added, you'll have these automated workflows:

### 🔄 **ci.yml** - Continuous Integration
- **Triggers**: Push to main/develop, PRs, daily schedule
- **Features**:
  - Code quality checks (Black, Ruff, MyPy)
  - Multi-platform testing (Linux, macOS, Windows)
  - Python version matrix (3.9-3.12)
  - Test coverage reporting
  - Package building and validation

### 🚀 **cd.yml** - Continuous Deployment
- **Triggers**: Push to main, tags, manual dispatch
- **Features**:
  - Blue-green deployment strategy
  - Staging and production environments
  - Container building and publishing
  - Security scanning before deployment
  - Automated rollback on failure
  - GitHub releases for tags

### 🔒 **security.yml** - Security Scanning
- **Triggers**: Push, PRs, daily schedule
- **Features**:
  - Dependency vulnerability scanning (Safety, pip-audit)
  - Code security analysis (Bandit, Semgrep, CodeQL)
  - Secret scanning (TruffleHog, GitLeaks)
  - Container security (Trivy, Snyk, Docker Scout)
  - License compliance checking
  - Security score calculation

### 🔄 **dependency-update.yml** - Dependency Management
- **Triggers**: Weekly schedule, manual dispatch
- **Features**:
  - Automated dependency updates
  - Test validation of updates
  - Automated PR creation
  - Security-focused updates

## 🎯 Workflow Features

### Quality Gates
- **Test Coverage**: 80% minimum threshold
- **Security Score**: Automated calculation and reporting
- **Performance**: Benchmark regression detection
- **Code Quality**: Multi-tool validation

### Security Integration
- **Multiple Scanners**: CodeQL, Bandit, Trivy, Snyk, Semgrep
- **Fail-Fast**: Critical vulnerabilities block deployment
- **Compliance**: License and security policy enforcement
- **Secrets Protection**: Multiple layers of secret detection

### Performance Monitoring
- **Benchmark Tracking**: Automated performance regression detection
- **Resource Monitoring**: Memory and CPU usage validation
- **Load Testing**: Integration with performance testing tools

### Deployment Safety
- **Blue-Green Strategy**: Zero-downtime deployments
- **Health Checks**: Automated validation after deployment
- **Rollback**: Automatic rollback on failure detection
- **Smoke Tests**: Post-deployment validation

## 📊 Monitoring and Observability

### Metrics Collection
- GitHub Actions execution metrics
- Test coverage trends
- Security score tracking
- Performance benchmark history

### Alerting
- Slack notifications for deployment status
- Security vulnerability alerts
- Performance regression notifications
- Dependency update notifications

### Dashboards
- Security score dashboard
- Test coverage trends
- Deployment success rates
- Performance metrics

## 🔧 Customization

### Environment-Specific Configuration
- Staging vs Production settings
- Security scanning thresholds
- Performance benchmarks
- Notification preferences

### Framework Integration
- Quantum backend testing
- Agent framework compatibility
- Custom validation rules
- Domain-specific checks

## 🚨 Troubleshooting

### Common Issues
1. **Workflow Permission Denied**: Ensure repository has `workflows` write permission
2. **Secret Not Found**: Verify all required secrets are configured
3. **Test Failures**: Check quantum backend credentials and dependencies
4. **Deployment Failures**: Validate Azure connection and permissions

### Debug Steps
1. Check workflow run logs in GitHub Actions tab
2. Validate secret configuration
3. Test locally with same environment
4. Review security scan results
5. Check deployment target connectivity

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Azure Deployment Guide](docs/deployment/azure.md)
- [Security Scanning Setup](docs/security/scanning.md)
- [Performance Testing Guide](docs/testing/performance.md)

---

Once these workflows are in place, you'll have a **world-class CI/CD pipeline** with comprehensive security, testing, and deployment automation! 🌟
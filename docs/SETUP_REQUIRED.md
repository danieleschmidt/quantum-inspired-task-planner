# Manual Setup Requirements

## GitHub Repository Settings

### Branch Protection Rules
- **Target**: main branch  
- **Requirements**: PR required, status checks must pass
- **Status Checks**: CI tests, linting, type checking

### Repository Secrets
```
PYPI_API_TOKEN          # For package publishing
CODECOV_TOKEN          # For coverage reporting  
SLACK_WEBHOOK          # For notifications (optional)
```

### GitHub Pages
- **Source**: GitHub Actions
- **Custom Domain**: docs.your-org.com (optional)
- **Enforce HTTPS**: Enabled

## External Integrations

### Code Quality
- **CodeCov**: Coverage reporting
- **Snyk**: Dependency vulnerability scanning
- **Sonatype**: License compliance

### Monitoring
- **Sentry**: Error tracking and performance monitoring
- **DataDog**: Application performance monitoring
- **GitHub Advanced Security**: Secret scanning, code scanning

## CI/CD Pipeline Setup

### Required GitHub Actions
1. Copy workflows from `workflows-to-add/` to `.github/workflows/`
2. Enable Actions in repository settings
3. Configure environment protection rules
4. Test workflows with test releases

### Documentation Deployment
1. Enable GitHub Pages in repository settings
2. Configure custom domain (if applicable)  
3. Update documentation URLs in package metadata

## Security Checklist

- [ ] Enable secret scanning
- [ ] Configure dependency review
- [ ] Setup security advisories
- [ ] Enable 2FA for all maintainers
- [ ] Review access permissions regularly
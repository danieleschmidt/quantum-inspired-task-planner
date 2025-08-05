# 🚀 Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Quantum-Inspired Task Planner in production environments with enterprise-grade reliability, security, and compliance.

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] **Python 3.9+** installed and configured
- [ ] **Memory**: Minimum 2GB RAM, recommended 8GB+ for large problems
- [ ] **CPU**: Multi-core processor recommended for concurrent processing
- [ ] **Disk**: 1GB+ available space for caching and logs
- [ ] **Network**: Internet access for quantum backend connectivity (optional)

### Security Requirements
- [ ] **Firewall**: Configure appropriate ports and access controls
- [ ] **SSL/TLS**: Enable encryption for all external communications
- [ ] **Access Control**: Implement authentication and authorization
- [ ] **Audit Logging**: Enable comprehensive activity logging
- [ ] **Data Encryption**: Configure encryption at rest and in transit

### Compliance Requirements
- [ ] **GDPR Compliance**: For EU operations
- [ ] **CCPA Compliance**: For California operations
- [ ] **PDPA Compliance**: For Singapore operations
- [ ] **Data Residency**: Configure regional data storage requirements
- [ ] **Retention Policies**: Set appropriate data retention periods

## 🔧 Installation Instructions

### Step 1: Environment Setup

```bash
# Create production virtual environment
python3.9 -m venv quantum-planner-prod
source quantum-planner-prod/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel
```

### Step 2: Core Installation

```bash
# Install from source (recommended for production)
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner

# Install with production dependencies
pip install -e ".[all]"

# Verify core installation
python -c "from quantum_planner import QuantumTaskPlanner; print('✅ Core installation verified')"
```

### Step 3: Optional Quantum Backends

```bash
# D-Wave Ocean SDK (for D-Wave quantum computers)
pip install dwave-ocean-sdk
export DWAVE_API_TOKEN="your-dwave-token"

# Azure Quantum (for Microsoft quantum services)
pip install azure-quantum
export AZURE_QUANTUM_RESOURCE_ID="your-resource-id"

# IBM Quantum (for IBM quantum computers)
pip install qiskit qiskit-ibm-runtime
export IBMQ_TOKEN="your-ibm-token"
```

### Step 4: Production Dependencies

```bash
# Monitoring and observability
pip install prometheus-client grafana-api

# Database connectivity (if needed)
pip install psycopg2-binary redis

# Web framework (if building APIs)
pip install fastapi uvicorn

# Security enhancements
pip install cryptography bcrypt
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for production configuration:

```bash
# Core Configuration
QUANTUM_PLANNER_ENV=production
QUANTUM_PLANNER_LOG_LEVEL=INFO
QUANTUM_PLANNER_CACHE_SIZE=1000
QUANTUM_PLANNER_CACHE_TTL=3600

# Quantum Backends
DWAVE_API_TOKEN=your-dwave-token
AZURE_QUANTUM_RESOURCE_ID=your-azure-resource-id
IBMQ_TOKEN=your-ibm-token

# Regional Configuration
DEFAULT_REGION=us-east-1
DATA_RESIDENCY_REQUIRED=false
ENCRYPTION_ENABLED=true

# Compliance
GDPR_ENABLED=true
CCPA_ENABLED=true
AUDIT_LOGGING_ENABLED=true

# Performance
MAX_CONCURRENT_JOBS=10
CACHE_MEMORY_LIMIT_MB=500
PERFORMANCE_MONITORING=true

# Security
ENABLE_INPUT_VALIDATION=true
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=100
```

### Configuration File

Create `config/production.yaml`:

```yaml
# Production Configuration
quantum_planner:
  environment: production
  
  # Logging
  logging:
    level: INFO
    format: json
    file: /var/log/quantum-planner/app.log
    max_size_mb: 100
    backup_count: 5
  
  # Performance
  performance:
    cache:
      solutions:
        max_size: 1000
        ttl_seconds: 3600
        max_memory_mb: 200
      problem_analysis:
        max_size: 500
        ttl_seconds: 7200
        max_memory_mb: 100
    
    concurrent:
      max_workers: 4
      queue_size: 100
      timeout_seconds: 300
  
  # Reliability
  reliability:
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 60
    retry:
      max_attempts: 3
      base_delay: 1.0
      max_delay: 30.0
  
  # Monitoring
  monitoring:
    enabled: true
    export_interval: 60
    alert_rules:
      - name: high_error_rate
        threshold: 0.1
        severity: critical
      - name: high_memory_usage
        threshold: 0.8
        severity: warning
  
  # Global Configuration
  globalization:
    default_language: en
    default_region: us-east-1
    
    regions:
      us-east-1:
        data_residency_required: false
        encryption_required: true
        retention_days: 730
        compliance: [ccpa, soc2]
      
      eu-west-1:
        data_residency_required: true
        encryption_required: true
        retention_days: 365
        compliance: [gdpr, iso27001]
      
      ap-southeast-1:
        data_residency_required: true
        encryption_required: true
        retention_days: 365
        compliance: [pdpa, iso27001]
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 quantum && chown -R quantum:quantum /app
USER quantum

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from quantum_planner import QuantumTaskPlanner; QuantumTaskPlanner()" || exit 1

# Start command
CMD ["python", "-m", "quantum_planner.server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  quantum-planner:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QUANTUM_PLANNER_ENV=production
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    depends_on:
      - redis
      - prometheus
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

## ☸️ Kubernetes Deployment

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-planner
  labels:
    name: quantum-planner
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-planner-config
  namespace: quantum-planner
data:
  config.yaml: |
    quantum_planner:
      environment: production
      logging:
        level: INFO
        format: json
      performance:
        cache:
          solutions:
            max_size: 1000
            ttl_seconds: 3600
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: quantum-planner-secrets
  namespace: quantum-planner
type: Opaque
data:
  dwave-token: <base64-encoded-token>
  azure-resource-id: <base64-encoded-resource-id>
  ibm-token: <base64-encoded-token>
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-planner
  namespace: quantum-planner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-planner
  template:
    metadata:
      labels:
        app: quantum-planner
    spec:
      containers:
      - name: quantum-planner
        image: quantum-planner:latest
        ports:
        - containerPort: 8000
        env:
        - name: QUANTUM_PLANNER_ENV
          value: "production"
        - name: DWAVE_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: quantum-planner-secrets
              key: dwave-token
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: quantum-planner-config
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: quantum-planner-service
  namespace: quantum-planner
spec:
  selector:
    app: quantum-planner
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantum-planner-ingress
  namespace: quantum-planner
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - quantum-planner.yourdomain.com
    secretName: quantum-planner-tls
  rules:
  - host: quantum-planner.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: quantum-planner-service
            port:
              number: 80
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'quantum-planner'
    static_configs:
      - targets: ['quantum-planner:8000']
    scrape_interval: 30s
    metrics_path: /metrics

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: quantum-planner
    rules:
      - alert: HighErrorRate
        expr: quantum_planner_error_rate > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: HighMemoryUsage
        expr: quantum_planner_memory_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
      
      - alert: ServiceDown
        expr: up{job="quantum-planner"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Quantum Planner service is down"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Quantum Task Planner",
    "panels": [
      {
        "title": "Assignment Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(quantum_planner_assignments_total[5m])",
            "legendFormat": "Assignments/sec"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(quantum_planner_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "quantum_planner_cache_hit_rate",
            "legendFormat": "Hit Rate"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "quantum_planner_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
```

## 🔐 Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificate (production should use proper CA)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure HTTPS in application
export SSL_CERT_PATH=/path/to/cert.pem
export SSL_KEY_PATH=/path/to/key.pem
```

### Authentication Setup

```python
# config/auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### Rate Limiting

```python
# config/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# Apply to endpoints
@app.post("/assign")
@limiter.limit("10/minute")
async def assign_tasks(request: Request, ...):
    # Implementation
    pass
```

## 🗄️ Database Integration

### PostgreSQL Setup

```sql
-- Create database and user
CREATE DATABASE quantum_planner;
CREATE USER qp_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE quantum_planner TO qp_user;

-- Create tables
CREATE TABLE assignments (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    problem_hash VARCHAR(255) NOT NULL,
    solution JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_problem_hash (problem_hash)
);

CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    operation VARCHAR(255) NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_operation (operation)
);
```

### Redis Configuration

```bash
# Redis configuration for caching
redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## 🚀 Deployment Process

### Step 1: Pre-deployment Testing

```bash
# Run all test suites
python test_generation1_basic.py
python test_generation2_robust.py
python test_generation3_optimized.py
python test_quality_gates.py
python test_global_implementation.py

# Verify configuration
python -c "
from quantum_planner import QuantumTaskPlanner
from quantum_planner.globalization import globalization
print('✅ All systems ready for deployment')
"
```

### Step 2: Staged Deployment

```bash
# Deploy to staging environment
docker-compose -f staging.yml up -d

# Run integration tests
python integration_tests.py --environment=staging

# Deploy to production
docker-compose -f production.yml up -d

# Health check
curl -f http://localhost:8000/health || exit 1
```

### Step 3: Post-deployment Verification

```bash
# Verify all services are running
docker-compose ps

# Check logs
docker-compose logs quantum-planner

# Test functionality
python production_smoke_test.py

# Verify monitoring
curl http://localhost:9090/metrics
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check memory usage
   python -c "from quantum_planner.performance import performance; print(performance.get_performance_stats())"
   
   # Clear caches
   python -c "from quantum_planner.performance import performance; performance.clear_all_caches()"
   ```

2. **Backend Connectivity**
   ```bash
   # Test quantum backend connectivity
   python -c "
   from quantum_planner import QuantumTaskPlanner
   planner = QuantumTaskPlanner(backend='dwave')
   print(planner.get_device_properties())
   "
   ```

3. **Compliance Issues**
   ```bash
   # Check compliance status
   python -c "
   from quantum_planner.globalization import globalization
   report = globalization.compliance.generate_compliance_report()
   print(f'Compliance rate: {report[\"compliance_rate\"]:.2%}')
   "
   ```

### Log Analysis

```bash
# View application logs
tail -f /var/log/quantum-planner/app.log

# Search for errors
grep "ERROR" /var/log/quantum-planner/app.log | tail -20

# Monitor performance
grep "performance" /var/log/quantum-planner/app.log | tail -10
```

## 📞 Support and Maintenance

### Health Monitoring

```python
# Health check endpoint
@app.get("/health")
async def health_check():
    from quantum_planner import QuantumTaskPlanner
    
    planner = QuantumTaskPlanner()
    health = planner.get_health_status()
    
    if health['overall_status'] == 'healthy':
        return {"status": "healthy", "timestamp": time.time()}
    else:
        raise HTTPException(status_code=503, detail="Service unhealthy")
```

### Backup Procedures

```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/

# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/

# Database backup (if using PostgreSQL)
pg_dump quantum_planner > backup-$(date +%Y%m%d).sql
```

### Update Procedures

```bash
# Update to new version
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations (if any)
python scripts/migrate.py

# Restart services
docker-compose restart quantum-planner
```

## 📋 Production Checklist

### Before Go-Live
- [ ] All test suites passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Monitoring configured
- [ ] Backup procedures tested
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Incident response plan ready

### Go-Live
- [ ] Deploy to production
- [ ] Verify health checks
- [ ] Test core functionality
- [ ] Monitor for 24 hours
- [ ] Confirm monitoring alerts
- [ ] Validate compliance reports

### Post Go-Live
- [ ] Performance monitoring
- [ ] Error rate tracking
- [ ] User feedback collection
- [ ] Compliance auditing
- [ ] Capacity planning
- [ ] Regular maintenance

---

**This deployment guide ensures enterprise-grade production deployment with comprehensive monitoring, security, and compliance capabilities.**
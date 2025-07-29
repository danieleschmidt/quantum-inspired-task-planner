# Deployment Guide

## Overview

This guide covers deploying the Quantum-Inspired Task Planner across different environments, from development to production-scale quantum computing infrastructure.

## Quick Start

### Local Development
```bash
# Clone and setup
git clone https://github.com/your-org/quantum-inspired-task-planner
cd quantum-inspired-task-planner
make dev

# Run with classical simulation
python examples/basic_assignment.py

# Start with Docker
docker-compose up -d
```

### Cloud Deployment
```bash
# Deploy to Azure with quantum backends
./scripts/deploy-azure.sh production

# Deploy to AWS with classical fallbacks
./scripts/deploy-aws.sh staging

# Deploy to GCP with hybrid optimization
./scripts/deploy-gcp.sh production
```

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │ -> │  Application     │ -> │ Quantum         │
│   (nginx/ALB)   │    │  Servers         │    │ Backends        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              v                         v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │  Monitoring      │    │ Classical       │
│   Job Queue     │    │  (Prometheus)    │    │ Solvers         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Environment-Specific Deployments

### Development Environment

**Purpose**: Local development and testing
**Scale**: Single instance, simulated backends
**Monitoring**: Basic logging

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  quantum-planner:
    build: .
    environment:
      - ENV=development
      - DEBUG=true
      - MOCK_QUANTUM_BACKENDS=true
    ports:
      - "8080:8080"
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
      
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

### Staging Environment

**Purpose**: Pre-production testing with real quantum backends
**Scale**: 2-3 instances, limited quantum credits
**Monitoring**: Full observability stack

**Infrastructure as Code (Terraform)**:
```hcl
# terraform/staging/main.tf
module "quantum_planner_staging" {
  source = "../modules/quantum-planner"
  
  environment = "staging"
  instance_count = 2
  instance_type = "t3.medium"
  
  # Quantum Backend Configuration
  enable_dwave = true
  enable_ibm_quantum = false  # Disable expensive backends
  dwave_solver = "DW_2000Q_6"  # Smaller, cheaper solver
  
  # Resource Limits
  max_problem_size = 50
  quantum_credits_limit = 1000
  
  # Monitoring
  enable_prometheus = true
  enable_grafana = true
  alert_slack_webhook = var.staging_slack_webhook
  
  tags = {
    Environment = "staging"
    Project = "quantum-task-planner"
    CostCenter = "research-development"
  }
}
```

### Production Environment

**Purpose**: High-availability quantum optimization service
**Scale**: Auto-scaling 2-20 instances
**Monitoring**: Comprehensive SLA monitoring and alerting

**Kubernetes Deployment**:
```yaml
# k8s/production/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-planner
  namespace: quantum-systems
spec:
  replicas: 5
  selector:
    matchLabels:
      app: quantum-planner
  template:
    metadata:
      labels:
        app: quantum-planner
        version: v1.0.0
    spec:
      containers:
      - name: quantum-planner
        image: quantum-planner:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: ENV
          value: "production"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: quantum-secrets
              key: redis-url
        - name: DWAVE_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: quantum-secrets
              key: dwave-token
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      imagePullSecrets:
      - name: docker-registry-secret
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-planner-service
  namespace: quantum-systems
spec:
  selector:
    app: quantum-planner
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

**Horizontal Pod Autoscaler**:
```yaml
# k8s/production/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-planner-hpa
  namespace: quantum-systems
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantum-planner
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: quantum_job_queue_size
      target:
        type: AverageValue
        averageValue: "10"
```

## Quantum Backend Configuration

### D-Wave Quantum Annealing

**Production Setup**:
```bash
# Environment variables
export DWAVE_API_TOKEN="your-production-token"
export DWAVE_SOLVER="Advantage_system6.4"  # Latest hardware
export DWAVE_NUM_READS=1000
export DWAVE_CHAIN_STRENGTH=2.0
export DWAVE_ANNEALING_TIME=20

# Solver selection strategy
export DWAVE_AUTO_SCALE=true
export DWAVE_FALLBACK_SOLVER="hybrid_binary_quadratic_model_version2"
```

**Monitoring Configuration**:
```python
# monitoring/dwave_monitor.py
import dwave.inspector
from prometheus_client import Gauge, Counter

# Metrics
dwave_qpu_access_time = Gauge('dwave_qpu_access_time_seconds', 'Time to access QPU')
dwave_queue_position = Gauge('dwave_queue_position', 'Position in quantum queue')
dwave_solve_success_rate = Gauge('dwave_solve_success_rate', 'Success rate of quantum solves')
dwave_chain_breaks = Counter('dwave_chain_breaks_total', 'Number of chain breaks')
```

### IBM Quantum

**Production Setup**:
```bash
# IBM Quantum Network configuration
export IBM_QUANTUM_TOKEN="your-production-token"
export IBM_QUANTUM_HUB="your-hub"
export IBM_QUANTUM_GROUP="your-group" 
export IBM_QUANTUM_PROJECT="your-project"

# Backend selection
export IBM_QUANTUM_BACKEND="ibm_lagos"  # 7-qubit processor
export IBM_QUANTUM_FALLBACK="ibmq_qasm_simulator"
export IBM_QUANTUM_OPTIMIZATION_LEVEL=3
export IBM_QUANTUM_SHOTS=8192
```

### Azure Quantum

**Production Setup**:
```bash
# Azure configuration
export AZURE_QUANTUM_SUBSCRIPTION_ID="your-subscription"
export AZURE_QUANTUM_RESOURCE_GROUP="quantum-resources"
export AZURE_QUANTUM_WORKSPACE="quantum-workspace-prod"
export AZURE_QUANTUM_LOCATION="westus"

# Provider selection
export AZURE_QUANTUM_PROVIDER="microsoft.simulatedannealing"
export AZURE_QUANTUM_TARGET="microsoft.simulatedannealing-parameterfree.cpu"
```

## Security Configuration

### Secrets Management

**Using AWS Secrets Manager**:
```python
# security/secrets.py
import boto3
from botocore.exceptions import ClientError

def get_quantum_credentials():
    """Retrieve quantum backend credentials from AWS Secrets Manager."""
    secret_name = "quantum-planner/prod/credentials"
    region_name = "us-west-2"
    
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        logger.error(f"Failed to retrieve credentials: {e}")
        raise
```

**Using HashiCorp Vault**:
```bash
# vault/policies/quantum-planner.hcl
path "secret/data/quantum/*" {
  capabilities = ["read"]
}

path "secret/data/monitoring/*" {
  capabilities = ["read"]
}

# Retrieve credentials
vault kv get -field=dwave_token secret/quantum/prod
vault kv get -field=ibm_token secret/quantum/prod
```

### Network Security

**Production Network Configuration**:
```yaml
# k8s/network-policies/quantum-planner-network-policy.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quantum-planner-network-policy
  namespace: quantum-systems
spec:
  podSelector:
    matchLabels:
      app: quantum-planner
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []  # Allow all outbound (for quantum backends)
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 80   # HTTP
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Monitoring and Alerting

### Prometheus Configuration

**Production Metrics Collection**:
```yaml
# monitoring/prometheus-production.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'quantum-prod'
    region: 'us-west-2'

scrape_configs:
  - job_name: 'quantum-planner'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - quantum-systems
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: quantum-planner
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### Grafana Dashboards

**Quantum Operations Dashboard**:
```json
{
  "dashboard": {
    "title": "Quantum Task Planner - Operations",
    "panels": [
      {
        "title": "Quantum Solve Success Rate",
        "type": "stat",
        "targets": [{
          "expr": "rate(quantum_solve_success_total[5m]) / rate(quantum_solve_attempts_total[5m])"
        }]
      },
      {
        "title": "Average Solve Time by Backend",
        "type": "graph", 
        "targets": [{
          "expr": "histogram_quantile(0.5, quantum_solve_duration_seconds_bucket) by (backend)"
        }]
      },
      {
        "title": "Queue Sizes",
        "type": "graph",
        "targets": [{
          "expr": "quantum_job_queue_size by (backend)"
        }]
      }
    ]
  }
}
```

## Deployment Scripts

### Automated Deployment

**Production Deployment Script**:
```bash
#!/bin/bash
# scripts/deploy-production.sh

set -euo pipefail

echo "🚀 Starting production deployment..."

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
./scripts/check-dependencies.sh
./scripts/validate-config.sh production
./scripts/run-smoke-tests.sh

# Build and push container
echo "🏗️  Building production container..."
docker build -t quantum-planner:${BUILD_VERSION} .
docker tag quantum-planner:${BUILD_VERSION} ${REGISTRY}/quantum-planner:${BUILD_VERSION}
docker push ${REGISTRY}/quantum-planner:${BUILD_VERSION}

# Deploy to Kubernetes
echo "🚁 Deploying to Kubernetes..."
kubectl set image deployment/quantum-planner \
  quantum-planner=${REGISTRY}/quantum-planner:${BUILD_VERSION} \
  -n quantum-systems

# Wait for rollout
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/quantum-planner -n quantum-systems --timeout=600s

# Post-deployment validation
echo "✅ Running post-deployment tests..."
./scripts/health-check.sh production
./scripts/integration-tests.sh production

# Update monitoring
echo "📊 Updating monitoring dashboards..."
./scripts/update-grafana-dashboards.sh

echo "🎉 Production deployment completed successfully!"
```

### Rollback Procedure

**Automated Rollback**:
```bash
#!/bin/bash
# scripts/rollback-production.sh

set -euo pipefail

PREVIOUS_VERSION=${1:-$(kubectl rollout history deployment/quantum-planner -n quantum-systems | tail -2 | head -1 | awk '{print $1}')}

echo "🔄 Rolling back to revision ${PREVIOUS_VERSION}..."

# Rollback deployment
kubectl rollout undo deployment/quantum-planner \
  --to-revision=${PREVIOUS_VERSION} \
  -n quantum-systems

# Wait for rollback
kubectl rollout status deployment/quantum-planner -n quantum-systems --timeout=300s

# Validate rollback
./scripts/health-check.sh production

echo "✅ Rollback completed successfully!"
```

## Performance Optimization

### Resource Scaling

**Auto-scaling Configuration**:
```yaml
# k8s/cluster-autoscaler.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.max: "50"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
  skip-nodes-with-local-storage: "false"
  skip-nodes-with-system-pods: "false"
```

### Database Optimization

**Redis Cluster for High Availability**:
```yaml
# k8s/redis-cluster.yml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: quantum-systems
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        - containerPort: 16379
        command:
        - redis-server
        - /etc/redis/redis.conf
        - --cluster-enabled
        - "yes"
        - --cluster-config-file
        - nodes.conf
        - --cluster-node-timeout
        - "5000"
        volumeMounts:
        - name: redis-data
          mountPath: /data
        - name: redis-config
          mountPath: /etc/redis
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

## Troubleshooting

### Common Issues

**1. Quantum Backend Connection Issues**
```bash
# Check backend connectivity
kubectl exec -it deployment/quantum-planner -n quantum-systems -- \
  python -c "from quantum_planner.backends import DWaveBackend; DWaveBackend().test_connection()"

# Check credentials
kubectl get secret quantum-secrets -n quantum-systems -o yaml
```

**2. High Memory Usage**
```bash
# Check memory usage
kubectl top pods -n quantum-systems

# Scale up memory limits
kubectl patch deployment quantum-planner -n quantum-systems -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"quantum-planner","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

**3. Queue Backlog**
```bash
# Check queue status
kubectl exec -it deployment/quantum-planner -n quantum-systems -- \
  redis-cli -h redis-cluster LLEN quantum_job_queue

# Scale up replicas
kubectl scale deployment quantum-planner --replicas=10 -n quantum-systems
```

### Log Analysis

**Centralized Logging with ELK Stack**:
```yaml
# logging/elasticsearch.yml
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: quantum-logs
  namespace: logging
spec:
  version: 8.6.0
  nodeSets:
  - name: default
    count: 3
    config:
      node.store.allow_mmap: false
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          resources:
            requests:
              memory: 4Gi
              cpu: 1000m
            limits:
              memory: 4Gi
              cpu: 2000m
```

## Cost Optimization

### Quantum Credits Management

**Budget Alerts**:
```python
# monitoring/cost_monitoring.py
def monitor_quantum_costs():
    """Monitor and alert on quantum backend costs."""
    dwave_credits = get_dwave_credit_usage()
    ibm_credits = get_ibm_quantum_units()
    azure_costs = get_azure_quantum_costs()
    
    total_monthly_cost = dwave_credits * 0.001 + ibm_credits * 0.01 + azure_costs
    
    if total_monthly_cost > MONTHLY_BUDGET * 0.8:
        send_cost_alert(f"Quantum costs at {total_monthly_cost:.2f} USD")
    
    # Auto-scale down if budget exceeded
    if total_monthly_cost > MONTHLY_BUDGET:
        enable_classical_only_mode()
```

### Resource Right-sizing

**Cost-optimized Instance Selection**:
```hcl
# terraform/cost-optimization.tf
resource "aws_instance" "quantum_planner" {
  count = var.environment == "production" ? 3 : 1
  
  # Use spot instances for cost savings
  instance_market_options {
    market_type = "spot"
    spot_options {
      instance_interruption_behavior = "terminate"
      max_price = "0.10"  # 50% of on-demand price
    }
  }
  
  instance_type = var.environment == "production" ? "c5.2xlarge" : "t3.medium"
  
  tags = {
    Name = "quantum-planner-${var.environment}"
    AutoShutdown = var.environment != "production" ? "true" : "false"
  }
}
```

This deployment guide provides comprehensive coverage for deploying the Quantum-Inspired Task Planner across all environments with proper security, monitoring, and cost optimization practices.
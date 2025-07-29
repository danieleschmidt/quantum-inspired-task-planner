# Deployment Guide

This document provides comprehensive deployment instructions for the Quantum-Inspired Task Planner across different environments.

## Overview

The Quantum-Inspired Task Planner supports multiple deployment models:
- **Local Development**: Single-node setup for development and testing
- **Docker Containers**: Containerized deployment for consistency
- **Kubernetes**: Scalable cloud-native deployment
- **Serverless**: Function-based deployment for specific use cases

## Prerequisites

- Python 3.9+
- Docker (for containerized deployments)
- Kubernetes cluster (for K8s deployments)
- Valid quantum backend credentials (optional)

## Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/quantum-inspired-task-planner
cd quantum-inspired-task-planner

# Install with Poetry
poetry install --with dev

# Run basic setup
make dev

# Run tests to verify installation
make test
```

### Environment Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Configure environment variables:

```env
# Application settings
QUANTUM_PLANNER_ENV=development
QUANTUM_PLANNER_LOG_LEVEL=DEBUG
QUANTUM_PLANNER_MAX_WORKERS=4

# Quantum backend configurations (optional)
DWAVE_TOKEN=your_dwave_token_here
AZURE_QUANTUM_SUBSCRIPTION_ID=your_subscription_id
AZURE_QUANTUM_RESOURCE_GROUP=your_resource_group
AZURE_QUANTUM_WORKSPACE=your_workspace

# Performance settings
QUANTUM_PLANNER_CACHE_SIZE=1000
QUANTUM_PLANNER_TIMEOUT=300
```

## Docker Deployment

### Single Container

Build and run the application:

```bash
# Build image
docker build -t quantum-planner:latest .

# Run container
docker run -d \
  --name quantum-planner \
  -p 8000:8000 \
  -e QUANTUM_PLANNER_ENV=production \
  --env-file .env \
  quantum-planner:latest
```

### Docker Compose

Create `docker-compose.prod.yml`:

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
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - quantum-planner
    restart: unless-stopped

volumes:
  redis_data:
```

Deploy with Docker Compose:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Kubernetes Deployment

### Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-planner
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-planner-config
  namespace: quantum-planner
data:
  QUANTUM_PLANNER_ENV: "production"
  QUANTUM_PLANNER_LOG_LEVEL: "INFO"
  QUANTUM_PLANNER_MAX_WORKERS: "8"
  REDIS_URL: "redis://redis-service:6379"
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: quantum-planner-secrets
  namespace: quantum-planner
type: Opaque
data:
  dwave-token: <base64-encoded-token>
  azure-subscription-id: <base64-encoded-id>
```

### Deployment

```yaml
# deployment.yaml
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
        image: quantum-planner:1.0.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: quantum-planner-config
        - secretRef:
            name: quantum-planner-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: quantum-planner-service
  namespace: quantum-planner
spec:
  selector:
    app: quantum-planner
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantum-planner-ingress
  namespace: quantum-planner
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - quantum-planner.your-domain.com
    secretName: quantum-planner-tls
  rules:
  - host: quantum-planner.your-domain.com
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

### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Verify deployment
kubectl get pods -n quantum-planner
kubectl get services -n quantum-planner
kubectl logs -n quantum-planner -l app=quantum-planner
```

## Cloud Deployments

### AWS ECS

```json
{
  "family": "quantum-planner",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/quantum-planner-task-role",
  "containerDefinitions": [
    {
      "name": "quantum-planner",
      "image": "your-account.dkr.ecr.region.amazonaws.com/quantum-planner:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "QUANTUM_PLANNER_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DWAVE_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:quantum-planner/dwave-token"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/quantum-planner",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: quantum-planner
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/your-project/quantum-planner:latest
        ports:
        - containerPort: 8000
        env:
        - name: QUANTUM_PLANNER_ENV
          value: "production"
        - name: DWAVE_TOKEN
          valueFrom:
            secretKeyRef:
              name: quantum-secrets
              key: dwave-token
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
```

### Azure Container Instances

```json
{
  "apiVersion": "2021-09-01",
  "type": "Microsoft.ContainerInstance/containerGroups",
  "name": "quantum-planner",
  "location": "West US 2",
  "properties": {
    "containers": [
      {
        "name": "quantum-planner",
        "properties": {
          "image": "youracr.azurecr.io/quantum-planner:latest",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ],
          "environmentVariables": [
            {
              "name": "QUANTUM_PLANNER_ENV",
              "value": "production"
            }
          ],
          "resources": {
            "requests": {
              "cpu": 1,
              "memoryInGB": 2
            }
          }
        }
      }
    ],
    "osType": "Linux",
    "ipAddress": {
      "type": "Public",
      "ports": [
        {
          "port": 8000,
          "protocol": "TCP"
        }
      ]
    }
  }
}
```

## Serverless Deployment

### AWS Lambda

```python
# lambda_handler.py
import json
from quantum_planner import QuantumTaskPlanner

def lambda_handler(event, context):
    """AWS Lambda handler for quantum optimization."""
    
    try:
        # Parse input
        agents = event['agents']
        tasks = event['tasks']
        
        # Initialize planner
        planner = QuantumTaskPlanner(backend="auto")
        
        # Solve optimization
        solution = planner.assign(
            agents=agents,
            tasks=tasks,
            objective="minimize_makespan"
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'assignments': solution.assignments,
                'makespan': solution.makespan,
                'cost': solution.cost
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

## Production Configuration

### Environment Variables

```env
# Production settings
QUANTUM_PLANNER_ENV=production
QUANTUM_PLANNER_LOG_LEVEL=INFO
QUANTUM_PLANNER_DEBUG=false

# Performance tuning
QUANTUM_PLANNER_MAX_WORKERS=16
QUANTUM_PLANNER_WORKER_TIMEOUT=600
QUANTUM_PLANNER_CACHE_SIZE=10000

# Security
QUANTUM_PLANNER_SECRET_KEY=your-secret-key-here
QUANTUM_PLANNER_ALLOWED_HOSTS=quantum-planner.your-domain.com

# Database
DATABASE_URL=postgresql://user:pass@db-host:5432/quantum_planner

# Redis
REDIS_URL=redis://redis-host:6379/0

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENDPOINT=http://prometheus:9090
```

### Nginx Configuration

```nginx
# nginx.conf
upstream quantum_planner {
    server quantum-planner:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name quantum-planner.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name quantum-planner.your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://quantum_planner;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://quantum_planner/health;
        access_log off;
    }
}
```

## Monitoring and Observability

### Health Checks

```python
# health.py
from fastapi import FastAPI
from quantum_planner.health import HealthChecker

app = FastAPI()
health = HealthChecker()

@app.get("/health")
def health_check():
    """Application health check endpoint."""
    return health.check_all()

@app.get("/ready")
def readiness_check():
    """Kubernetes readiness probe."""
    return {"status": "ready"}

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return health.get_metrics()
```

### Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "json",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/quantum_planner.log",
            "maxBytes": 10485760,
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Security Considerations

### Secrets Management

- Use environment variables for configuration
- Store sensitive data in secret management systems
- Rotate credentials regularly
- Implement least-privilege access

### Network Security

- Use HTTPS/TLS for all communications
- Implement proper firewall rules
- Use VPCs/private networks where possible
- Enable DDoS protection

### Container Security

- Use minimal base images
- Run as non-root user
- Scan images for vulnerabilities
- Keep dependencies updated

## Troubleshooting

### Common Issues

1. **Container fails to start**:
   - Check environment variables
   - Verify image build
   - Review logs: `docker logs <container-id>`

2. **High memory usage**:
   - Check problem sizes
   - Review caching configuration
   - Monitor for memory leaks

3. **Slow performance**:
   - Verify backend connectivity
   - Check resource limits
   - Review optimization parameters

4. **Backend connection errors**:
   - Validate credentials
   - Check network connectivity
   - Review rate limits

### Debug Commands

```bash
# Check container status
docker ps -a

# View logs
docker logs -f quantum-planner

# Execute into container
docker exec -it quantum-planner /bin/bash

# Check Kubernetes pods
kubectl get pods -n quantum-planner
kubectl describe pod <pod-name> -n quantum-planner
kubectl logs <pod-name> -n quantum-planner

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Backup and Recovery

### Data Backup

```bash
# Database backup
pg_dump quantum_planner > backup_$(date +%Y%m%d).sql

# Redis backup
redis-cli --rdb backup_redis_$(date +%Y%m%d).rdb

# Configuration backup
kubectl get configmap quantum-planner-config -o yaml > config_backup.yaml
```

### Disaster Recovery

1. **Database Recovery**:
   ```bash
   psql quantum_planner < backup_20240129.sql
   ```

2. **Configuration Recovery**:
   ```bash
   kubectl apply -f config_backup.yaml
   ```

3. **Rolling Deployment Rollback**:
   ```bash
   kubectl rollout undo deployment/quantum-planner -n quantum-planner
   ```

## Performance Tuning

### Application Tuning

- Optimize worker count based on CPU cores
- Tune cache sizes for your workload
- Configure appropriate timeouts
- Enable connection pooling

### Infrastructure Tuning

- Right-size containers/VMs
- Use appropriate storage types
- Configure auto-scaling
- Implement load balancing

### Quantum Backend Optimization

- Choose appropriate backends for problem sizes
- Implement intelligent backend selection
- Use connection pooling for backends
- Monitor backend quotas and limits
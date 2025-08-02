# Deployment Procedures

## Standard Deployment Process

### Pre-deployment Checklist

- [ ] All CI/CD checks passing
- [ ] Security scans completed
- [ ] Performance benchmarks validated
- [ ] Quantum backend connectivity tested
- [ ] Staging environment validated
- [ ] Rollback plan prepared
- [ ] Monitoring alerts configured

### Deployment Steps

1. **Preparation Phase**
   ```bash
   # Verify current system status
   kubectl get pods -n quantum-planner
   docker system df
   
   # Check resource availability
   kubectl top nodes
   kubectl top pods -n quantum-planner
   ```

2. **Pre-deployment Health Check**
   ```bash
   # Test quantum backend connectivity
   curl -f http://localhost:8080/health/quantum
   
   # Verify database connections
   curl -f http://localhost:8080/health/database
   
   # Check API responsiveness
   curl -f http://localhost:8080/health/ready
   ```

3. **Deployment Execution**
   ```bash
   # Rolling update deployment
   kubectl set image deployment/quantum-planner \
     quantum-planner=quantum-planner:${NEW_VERSION}
   
   # Monitor rollout
   kubectl rollout status deployment/quantum-planner
   ```

4. **Post-deployment Validation**
   ```bash
   # Verify pods are healthy
   kubectl get pods -l app=quantum-planner
   
   # Test critical paths
   curl -f http://localhost:8080/api/v1/optimize/test
   
   # Check metrics
   curl -s http://localhost:8080/metrics | grep quantum_solve_duration
   ```

## Rollback Procedures

### Automatic Rollback Triggers

- Pod restart count > 3 in 5 minutes
- Error rate > 5% for 2 minutes
- Response time > 10 seconds for 1 minute
- Quantum backend failures > 50%

### Manual Rollback Steps

1. **Immediate Rollback**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/quantum-planner
   
   # Verify rollback
   kubectl rollout status deployment/quantum-planner
   ```

2. **Health Verification**
   ```bash
   # Check pod status
   kubectl get pods -l app=quantum-planner
   
   # Verify functionality
   curl -f http://localhost:8080/health/ready
   ```

3. **Incident Documentation**
   - Log rollback reason in incident tracker
   - Update deployment status in monitoring
   - Notify stakeholders of rollback completion

## Blue-Green Deployment

### Setup Phase
```bash
# Deploy to green environment
kubectl apply -f k8s/green-deployment.yaml

# Verify green environment
kubectl get pods -l env=green
```

### Traffic Switch
```bash
# Update service selector to green
kubectl patch service quantum-planner-service \
  -p '{"spec":{"selector":{"env":"green"}}}'

# Monitor traffic shift
kubectl logs -f service/quantum-planner-service
```

### Cleanup
```bash
# After successful validation, cleanup blue
kubectl delete deployment quantum-planner-blue
```

## Database Migrations

### Pre-migration Checklist
- [ ] Database backup completed
- [ ] Migration tested in staging
- [ ] Rollback scripts prepared
- [ ] Downtime window scheduled

### Migration Process
```bash
# Create backup
pg_dump quantum_planner > backup_$(date +%Y%m%d_%H%M%S).sql

# Apply migrations
python manage.py migrate

# Verify data integrity
python manage.py check_data_integrity
```

## Emergency Procedures

### System Outage Response
1. Activate incident response team
2. Switch to maintenance mode
3. Investigate root cause
4. Apply emergency fixes
5. Communicate status to users
6. Conduct post-incident review

### Data Corruption Recovery
1. Stop all write operations
2. Assess corruption scope
3. Restore from latest backup
4. Replay transaction logs
5. Validate data integrity
6. Resume normal operations

## Monitoring During Deployment

### Key Metrics to Watch
- Pod restart count
- Memory and CPU usage
- Error rates and response times
- Quantum backend connectivity
- Database connection pool

### Alert Thresholds
- Error rate > 1% for 5 minutes
- Response time > 5 seconds for 2 minutes
- Memory usage > 80% for 10 minutes
- Failed quantum calls > 10% for 5 minutes
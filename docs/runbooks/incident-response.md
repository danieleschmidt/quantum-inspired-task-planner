# Incident Response Runbook

## Incident Classification

### Severity Levels

**P0 - Critical**
- Complete system outage
- Data corruption or loss
- Security breach
- Response time: 15 minutes

**P1 - High**
- Major feature unavailable
- Performance degradation >50%
- Quantum backend failures
- Response time: 1 hour

**P2 - Medium**
- Minor feature issues
- Performance degradation <50%
- Non-critical errors
- Response time: 4 hours

**P3 - Low**
- Documentation issues
- Minor UI problems
- Enhancement requests
- Response time: 1 business day

## Incident Response Process

### 1. Detection and Alert
```bash
# Check system status
curl -f http://localhost:8080/health
kubectl get pods -n quantum-planner

# Verify monitoring systems
curl -s http://prometheus:9090/api/v1/query?query=up
```

### 2. Initial Assessment
- Determine incident severity
- Identify affected components
- Estimate user impact
- Activate appropriate response team

### 3. Communication
```bash
# Update status page
curl -X POST https://status.company.com/api/incidents \
  -H "Authorization: Bearer $STATUS_TOKEN" \
  -d '{"title":"Quantum Planner Service Degradation"}'
```

### 4. Investigation
```bash
# Check recent deployments
kubectl rollout history deployment/quantum-planner

# Review logs
kubectl logs -l app=quantum-planner --tail=1000

# Check resource usage
kubectl top pods -n quantum-planner
```

### 5. Resolution
- Apply immediate fixes
- Implement workarounds
- Coordinate with external vendors
- Monitor recovery progress

### 6. Post-Incident
- Conduct blameless postmortem
- Document lessons learned
- Update runbooks and procedures
- Implement preventive measures

## Common Incident Scenarios

### Complete System Outage

**Symptoms:**
- All health checks failing
- No response from API endpoints
- Users unable to access system

**Investigation Steps:**
1. Check pod status: `kubectl get pods -n quantum-planner`
2. Review recent changes: `kubectl rollout history deployment/quantum-planner`
3. Check resource availability: `kubectl describe nodes`
4. Examine logs: `kubectl logs -l app=quantum-planner --tail=500`

**Resolution:**
```bash
# If pods are down, check for resource constraints
kubectl describe pod <pod-name>

# If deployment issue, rollback
kubectl rollout undo deployment/quantum-planner

# If infrastructure issue, check cluster health
kubectl get nodes
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Quantum Backend Failures

**Symptoms:**
- Increased solve times
- Authentication errors
- Backend timeout exceptions

**Investigation Steps:**
1. Check backend connectivity: `curl -f https://cloud.dwavesys.com/api/status`
2. Verify credentials: Check environment variables and secrets
3. Review quota usage: Check API limits and usage
4. Test fallback systems: Verify classical solvers working

**Resolution:**
```bash
# Switch to fallback backend
kubectl set env deployment/quantum-planner QUANTUM_BACKEND=classical

# Update backend configuration
kubectl patch configmap quantum-config \
  -p '{"data":{"backend.yaml":"backend: classical\ntimeout: 30s"}}'
```

### Performance Degradation

**Symptoms:**
- Increased response times
- Higher memory usage
- CPU throttling

**Investigation Steps:**
1. Check metrics: `curl -s http://localhost:8080/metrics | grep duration`
2. Analyze resource usage: `kubectl top pods`
3. Review problem complexity: Check recent job sizes
4. Examine database performance: Check query execution times

**Resolution:**
```bash
# Scale up replicas
kubectl scale deployment quantum-planner --replicas=5

# Increase resource limits
kubectl patch deployment quantum-planner \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"quantum-planner","resources":{"limits":{"memory":"4Gi","cpu":"2"}}}]}}}}'
```

### Database Connection Issues

**Symptoms:**
- Connection timeout errors
- Pool exhaustion warnings
- Transaction rollback failures

**Investigation Steps:**
1. Check database status: `pg_isready -h $DB_HOST`
2. Review connection pool: Check active connections
3. Examine slow queries: Check database logs
4. Verify credentials: Test connection manually

**Resolution:**
```bash
# Restart database connections
kubectl rollout restart deployment/quantum-planner

# Increase connection pool size
kubectl set env deployment/quantum-planner DB_POOL_SIZE=20

# Check database locks
psql -h $DB_HOST -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

## Emergency Contacts

### Primary On-call
- **Role:** Primary Incident Commander
- **Contact:** Pager duty rotation
- **Escalation:** After 30 minutes if no response

### Secondary On-call
- **Role:** Technical Lead
- **Contact:** Phone and email
- **Escalation:** For P0/P1 incidents

### Vendor Contacts
- **D-Wave Support:** support@dwavesys.com
- **Azure Quantum:** quantum-support@microsoft.com
- **AWS Support:** Technical support case

## Tools and Resources

### Monitoring Dashboards
- Grafana: http://grafana.company.com/quantum-planner
- Prometheus: http://prometheus.company.com:9090
- Logs: http://kibana.company.com

### Communication Channels
- Slack: #quantum-planner-incidents
- Status Page: https://status.company.com
- Conference Bridge: Automated dial-in

### Documentation
- Deployment Guide: docs/DEPLOYMENT.md
- Architecture: ARCHITECTURE.md
- API Reference: docs/api/

## Post-Incident Checklist

- [ ] Incident timeline documented
- [ ] Root cause identified
- [ ] Fix verified in production
- [ ] Status page updated
- [ ] Stakeholders notified
- [ ] Postmortem scheduled
- [ ] Action items created
- [ ] Runbooks updated
# Incident Response Playbook

This document provides procedures for responding to incidents in the Quantum-Inspired Task Planner system.

## Incident Classification

### Severity Levels

- **P0 (Critical)**: Complete system outage affecting all users
- **P1 (High)**: Major functionality impaired affecting most users
- **P2 (Medium)**: Some functionality degraded affecting subset of users
- **P3 (Low)**: Minor issues with workarounds available

### Common Incident Types

1. **Backend Outages**: Quantum or classical backend unavailable
2. **Performance Degradation**: Slow response times or timeouts
3. **High Error Rates**: Increased failure rates in optimizations
4. **Security Incidents**: Unauthorized access or data breaches
5. **Data Corruption**: Invalid solutions or corrupted problem data

## Response Procedures

### P0 - Critical Incidents

**Response Time**: 15 minutes
**Resolution Target**: 4 hours

#### Immediate Actions (0-15 minutes)
1. **Incident Commander Assignment**
   - Senior engineer takes incident command
   - Create incident channel: `#incident-YYYYMMDD-001`
   - Post in `#general`: "P0 incident declared for quantum-planner"

2. **Initial Assessment**
   ```bash
   # Check system status
   kubectl get pods -n quantum-planner
   curl -f https://quantum-planner.domain.com/health
   
   # Check monitoring dashboards
   # - Grafana: System overview
   # - Prometheus: Alert status
   # - ELK: Recent error logs
   ```

3. **Communications**
   - Update status page: "Investigating issue"
   - Notify stakeholders via email/Slack
   - Page on-call engineer if after hours

#### Investigation Phase (15-60 minutes)
1. **Gather Information**
   ```bash
   # Recent deployments
   kubectl rollout history deployment/quantum-planner -n quantum-planner
   
   # System resources
   kubectl top pods -n quantum-planner
   kubectl describe pods -n quantum-planner
   
   # Application logs
   kubectl logs -n quantum-planner -l app=quantum-planner --tail=1000
   ```

2. **Backend Health Check**
   ```python
   # Check all quantum backends
   from quantum_planner.health import check_all_backends
   backend_status = check_all_backends()
   for backend, status in backend_status.items():
       print(f"{backend}: {status}")
   ```

3. **Performance Analysis**
   ```bash
   # Check response times
   curl -w "@curl-format.txt" -s -o /dev/null https://quantum-planner.domain.com/api/optimize
   
   # Database performance
   psql -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
   ```

#### Resolution Actions
1. **Common Fixes**
   - **Backend Failover**: Switch to backup quantum backend
   - **Pod Restart**: `kubectl rollout restart deployment/quantum-planner -n quantum-planner`
   - **Rollback**: `kubectl rollout undo deployment/quantum-planner -n quantum-planner`
   - **Scale Up**: `kubectl scale deployment quantum-planner --replicas=5 -n quantum-planner`

2. **Emergency Patches**
   ```bash
   # Hot-fix deployment
   kubectl patch deployment quantum-planner -n quantum-planner -p '{"spec":{"template":{"spec":{"containers":[{"name":"quantum-planner","image":"quantum-planner:hotfix-v1.0.1"}]}}}}'
   ```

### P1 - High Severity Incidents

**Response Time**: 30 minutes
**Resolution Target**: 8 hours

#### Backend-Specific Issues

**D-Wave Backend Down**
```python
# Emergency failover script
from quantum_planner.backends import switch_backend

# Disable D-Wave backend
switch_backend("dwave", enabled=False)

# Enable alternative backends
switch_backend("azure_quantum", enabled=True)
switch_backend("simulated_annealing", enabled=True)

# Notify operations team
send_alert("D-Wave backend disabled - using fallback")
```

**High Error Rate**
```bash
# Check error patterns
grep -E "(ERROR|EXCEPTION)" /var/log/quantum-planner.log | tail -100

# Monitor error rate
curl "http://prometheus:9090/api/v1/query?query=rate(errors_total[5m])"

# If error rate > 10%, enable circuit breaker
kubectl patch configmap quantum-planner-config -p '{"data":{"CIRCUIT_BREAKER_ENABLED":"true"}}'
```

### P2 - Medium Severity Incidents

**Response Time**: 2 hours
**Resolution Target**: 24 hours

Common P2 scenarios and responses:

1. **Performance Degradation**
   - Monitor resource usage
   - Check for memory leaks
   - Analyze slow queries
   - Consider scaling

2. **Partial Feature Outage**
   - Identify affected components
   - Implement workarounds
   - Plan proper fix

## Runbooks

### Backend Connection Issues

**Symptoms**: Backend timeout errors, connection refused
**Diagnosis**:
```bash
# Test backend connectivity
curl -v https://cloud.dwavesys.com/sapi/
curl -v https://quantum.azure.com/api/

# Check credentials
python -c "from quantum_planner.backends import test_credentials; test_credentials()"
```

**Resolution**:
1. Verify credentials are current and not expired
2. Check backend service status pages
3. Test with minimal problem to isolate issue
4. Switch to alternative backend if needed

### Memory Issues

**Symptoms**: OOMKilled pods, high memory usage alerts
**Diagnosis**:
```bash
# Check memory usage
kubectl top pods -n quantum-planner
kubectl describe pod <pod-name> -n quantum-planner | grep -A 10 "State:"

# Analyze memory leaks
python -c "
import tracemalloc
tracemalloc.start()
# Run optimization
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"
```

**Resolution**:
1. Increase memory limits temporarily
2. Identify memory leak sources
3. Implement memory optimization
4. Deploy fix and monitor

### Database Performance Issues

**Symptoms**: Slow queries, database connection timeouts
**Diagnosis**:
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Identify slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check locks
SELECT * FROM pg_locks WHERE NOT granted;
```

**Resolution**:
1. Kill long-running queries if necessary
2. Add database indexes if needed
3. Optimize query patterns
4. Scale database if required

## Communication Templates

### Initial Incident Report
```
🚨 INCIDENT DECLARED - P1

System: Quantum-Inspired Task Planner
Impact: High error rate (>15%) affecting optimization requests
Start Time: 2024-01-29 14:30 UTC
Incident Commander: @engineer

Current Status: Investigating
Next Update: 15 minutes

Status Page: https://status.quantum-planner.com
Incident Channel: #incident-20240129-001
```

### Status Updates
```
📊 INCIDENT UPDATE - P1

Time: 2024-01-29 14:45 UTC (+15 min)
Status: Root cause identified - D-Wave backend timeout

Actions Taken:
✅ Switched to Azure Quantum backup
✅ Confirmed system stability
🔄 Monitoring error rates

Current Metrics:
- Error Rate: 2.1% (improved from 15.8%)
- Response Time: 850ms (target: <2s)
- Active Users: 89% of normal

Next Update: 30 minutes
```

### Resolution Notification
```
✅ INCIDENT RESOLVED - P1

System: Quantum-Inspired Task Planner
Resolution Time: 2024-01-29 16:15 UTC
Total Duration: 1h 45min

Root Cause: D-Wave backend experiencing intermittent timeouts

Actions Taken:
✅ Implemented automatic failover to Azure Quantum
✅ Increased timeout thresholds for quantum backends
✅ Added circuit breaker to prevent cascade failures

Preventive Measures:
🔄 Enhanced monitoring for backend health
🔄 Improved failover automation
📅 Scheduled backend redundancy review

Post-Incident Review: Scheduled for tomorrow 10:00 UTC
```

## Post-Incident Review

### Review Template

**Incident Summary**
- Date/Time: 
- Duration:
- Impact:
- Root Cause:

**Timeline**
- Detection:
- Response:
- Mitigation:
- Resolution:

**What Went Well**
- Fast detection through monitoring
- Effective communication
- Quick failover to backup systems

**What Could Be Improved**
- Automated failover took too long
- Documentation was outdated
- Monitoring could be more granular

**Action Items**
- [ ] Improve automated failover (Owner: @dev-team, Due: 2024-02-15)
- [ ] Update runbooks (Owner: @ops-team, Due: 2024-02-01)
- [ ] Add backend health dashboards (Owner: @sre-team, Due: 2024-02-10)

## Emergency Contacts

### On-Call Rotation
- **Primary**: Current on-call engineer (PagerDuty)
- **Secondary**: Backup on-call engineer
- **Escalation**: Engineering Manager

### External Contacts
- **D-Wave Support**: support@dwavesys.com, +1-XXX-XXX-XXXX
- **Azure Quantum**: quantum-support@microsoft.com
- **Cloud Provider**: See respective support channels

### Internal Escalation
1. **Level 1**: On-call engineer
2. **Level 2**: Senior engineer + Engineering manager
3. **Level 3**: VP Engineering + CTO
4. **Level 4**: CEO (for business-critical issues)

## Tools and Resources

### Monitoring Dashboards
- **System Overview**: https://grafana.company.com/d/quantum-overview
- **Performance Metrics**: https://grafana.company.com/d/quantum-performance
- **Error Tracking**: https://sentry.io/quantum-planner

### Documentation
- **Architecture**: `docs/ARCHITECTURE.md`
- **Deployment**: `docs/DEPLOYMENT.md`
- **Monitoring**: `docs/MONITORING.md`

### Emergency Scripts
```bash
# Failover script
./scripts/emergency-failover.sh

# System health check
./scripts/health-check.sh

# Performance diagnostics
./scripts/performance-diagnostic.sh
```

Remember: Stay calm, communicate clearly, and focus on resolution. Every incident is a learning opportunity to improve system reliability.
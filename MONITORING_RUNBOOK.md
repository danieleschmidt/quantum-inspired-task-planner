# Quantum Planner Monitoring Runbook

## Key Metrics to Monitor

### Application Metrics
- Request rate (target: < 1000 req/s)
- Response time (target: < 200ms p95)
- Error rate (target: < 1%)
- Task assignment success rate (target: > 95%)

### Infrastructure Metrics
- CPU utilization (target: < 70%)
- Memory usage (target: < 80%)
- Disk usage (target: < 85%)
- Network latency (target: < 50ms)

### Business Metrics
- Active users
- Task completion rate
- Quantum backend utilization
- Cache hit rate (target: > 80%)

## Alert Thresholds

### Critical Alerts
- Service down (response code 5xx > 10%)
- Database connection failures
- Memory usage > 95%
- Certificate expiration < 7 days

### Warning Alerts
- High response time (p95 > 500ms)
- Error rate > 2%
- CPU usage > 80%
- Low cache hit rate (< 60%)

## Dashboard Access
- Grafana: https://monitoring.quantum-planner.com
- Kibana: https://logs.quantum-planner.com
- Prometheus: https://metrics.quantum-planner.com

## Common Queries

### Prometheus Queries
```promql
# Request rate by endpoint
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time percentiles
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Cache hit rate
rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])
```

### Log Searches (Kibana)
```
# Application errors
level:ERROR AND service:quantum-planner

# Slow queries
message:"slow query" AND duration:>1000ms

# Authentication failures
message:"authentication failed"
```

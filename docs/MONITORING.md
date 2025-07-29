# Monitoring and Observability

This document outlines monitoring, logging, and observability practices for the Quantum-Inspired Task Planner.

## Overview

The system provides comprehensive monitoring capabilities including:
- Performance metrics collection
- Error tracking and alerting
- Resource utilization monitoring
- Quantum backend performance tracking
- Business metrics and KPIs

## Logging Framework

### Configuration

```python
from loguru import logger
import sys

# Configure structured logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="INFO",
    enqueue=True
)

# Add file logging for production
logger.add(
    "logs/quantum_planner.log",
    rotation="100 MB",
    retention="30 days",
    compression="gzip",
    level="DEBUG"
)
```

### Structured Logging Example

```python
from quantum_planner.monitoring import get_logger

logger = get_logger(__name__)

# Performance logging
logger.info(
    "Quantum optimization completed",
    extra={
        "backend": "dwave",
        "problem_size": 50,
        "solve_time": 2.34,
        "solution_quality": 0.95,
        "num_variables": 250
    }
)

# Error logging
logger.error(
    "Backend connection failed",
    extra={
        "backend": "azure_quantum",
        "error_code": "AUTH_FAILED",
        "retry_count": 3
    }
)
```

## Metrics Collection

### Custom Metrics

```python
from quantum_planner.metrics import MetricsCollector

metrics = MetricsCollector()

# Counter metrics
metrics.increment("tasks_scheduled_total", tags={"backend": "dwave"})
metrics.increment("optimization_errors_total", tags={"error_type": "timeout"})

# Gauge metrics
metrics.gauge("active_agents", 15)
metrics.gauge("queue_depth", 42)

# Histogram metrics
metrics.histogram("solve_time_seconds", 2.34, tags={"backend": "dwave"})
metrics.histogram("solution_quality", 0.95)

# Timer context manager
with metrics.timer("full_optimization_duration"):
    solution = planner.solve(agents, tasks)
```

### Key Performance Indicators

Monitor these critical metrics:

- **Throughput Metrics**:
  - `tasks_scheduled_per_second`
  - `solutions_generated_per_hour`
  - `agent_utilization_rate`

- **Quality Metrics**:
  - `solution_optimality_gap`
  - `constraint_violation_rate`
  - `solver_convergence_rate`

- **Performance Metrics**:
  - `avg_solve_time_seconds`
  - `p95_solve_time_seconds`
  - `backend_response_time_seconds`

- **Error Metrics**:
  - `optimization_error_rate`
  - `backend_failure_rate`
  - `timeout_rate`

## Health Checks

### Application Health

```python
from quantum_planner.health import HealthChecker

health = HealthChecker()

# Register health checks
health.register("database", check_database_connection)
health.register("quantum_backend", check_quantum_backend)
health.register("memory_usage", check_memory_usage)

# Endpoint for health checks
@app.get("/health")
def health_check():
    return health.check_all()
```

### Quantum Backend Health

```python
def check_quantum_backends():
    """Check health of all configured quantum backends."""
    backends = ["dwave", "azure_quantum", "ibm_quantum"]
    results = {}
    
    for backend_name in backends:
        try:
            backend = get_backend(backend_name)
            # Simple problem to test connectivity
            test_problem = np.array([[1, -1], [-1, 1]])
            start_time = time.time()
            result = backend.solve_qubo(test_problem)
            response_time = time.time() - start_time
            
            results[backend_name] = {
                "status": "healthy",
                "response_time": response_time,
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            results[backend_name] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    return results
```

## Alerting

### Alert Configuration

```yaml
# alerts.yml
alerts:
  - name: high_error_rate
    condition: error_rate > 0.05
    duration: 5m
    channels: [slack, email]
    message: "Error rate exceeded 5% for 5 minutes"

  - name: slow_optimization
    condition: p95_solve_time > 30s
    duration: 10m
    channels: [slack]
    message: "95th percentile solve time exceeded 30 seconds"

  - name: backend_down
    condition: backend_health_check == false
    duration: 1m
    channels: [pagerduty, slack]
    message: "Quantum backend health check failed"

  - name: low_solution_quality
    condition: avg_solution_quality < 0.8
    duration: 15m
    channels: [email]
    message: "Average solution quality below 80% for 15 minutes"
```

### Alert Handlers

```python
from quantum_planner.alerts import AlertManager

alerts = AlertManager()

@alerts.register("high_error_rate")
def handle_high_error_rate(alert_data):
    """Handle high error rate alerts."""
    error_rate = alert_data["error_rate"]
    
    # Auto-remediation: Switch to fallback solver
    if error_rate > 0.10:
        planner.switch_to_fallback()
        logger.warning(f"Switched to fallback solver due to {error_rate:.1%} error rate")
    
    # Notify team
    slack.send_message(
        channel="#quantum-alerts",
        message=f"🚨 Error rate: {error_rate:.1%} - Fallback activated"
    )

@alerts.register("backend_down")
def handle_backend_down(alert_data):
    """Handle backend downtime alerts."""
    backend = alert_data["backend"]
    
    # Remove backend from rotation
    planner.disable_backend(backend)
    
    # Create incident
    pagerduty.create_incident(
        title=f"Quantum backend {backend} is down",
        service="quantum-planner-production"
    )
```

## Performance Monitoring

### Resource Monitoring

```python
import psutil
from quantum_planner.monitoring import ResourceMonitor

monitor = ResourceMonitor()

# CPU monitoring
cpu_percent = psutil.cpu_percent(interval=1)
monitor.gauge("cpu_usage_percent", cpu_percent)

# Memory monitoring
memory = psutil.virtual_memory()
monitor.gauge("memory_usage_percent", memory.percent)
monitor.gauge("memory_available_mb", memory.available / 1024 / 1024)

# Quantum resource monitoring
monitor.gauge("quantum_queue_depth", backend.get_queue_depth())
monitor.gauge("quantum_credits_remaining", backend.get_credits())
```

### Performance Profiling

```python
import cProfile
import pstats
from quantum_planner.profiling import profile_optimization

@profile_optimization
def solve_large_problem():
    """Profile performance of large optimization problems."""
    planner = QuantumTaskPlanner()
    agents = generate_agents(100)
    tasks = generate_tasks(200)
    
    with cProfile.Profile() as pr:
        solution = planner.solve(agents, tasks)
        
        # Generate performance report
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        
        return solution
```

## Dashboards

### Grafana Dashboard Configuration

Create dashboards to visualize:

1. **Operations Dashboard**:
   - Tasks scheduled per minute
   - Active agents count
   - Queue depth over time
   - Solution quality distribution

2. **Performance Dashboard**:
   - Solve time percentiles
   - Backend response times
   - Resource utilization
   - Throughput metrics

3. **Error Dashboard**:
   - Error rate by type
   - Failed optimizations
   - Backend availability
   - Alert summary

### Example Dashboard Query

```sql
-- Prometheus queries for Grafana
rate(tasks_scheduled_total[5m])
histogram_quantile(0.95, solve_time_seconds_bucket[5m])
avg(solution_quality) by (backend)
sum(rate(optimization_errors_total[5m])) by (error_type)
```

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Trace optimization requests
@tracer.start_as_current_span("solve_optimization")
def solve_with_tracing(agents, tasks):
    """Solve optimization with distributed tracing."""
    span = trace.get_current_span()
    
    span.set_attribute("num_agents", len(agents))
    span.set_attribute("num_tasks", len(tasks))
    
    with tracer.start_as_current_span("build_qubo"):
        Q = build_qubo_matrix(agents, tasks)
        span.set_attribute("qubo_size", Q.shape[0])
    
    with tracer.start_as_current_span("quantum_solve"):
        solution = backend.solve_qubo(Q)
        span.set_attribute("solution_quality", solution.quality)
    
    return solution
```

## Incident Response

### Runbook

1. **High Error Rate**:
   - Check backend status
   - Review recent deployments
   - Enable fallback solvers
   - Scale up if needed

2. **Performance Degradation**:
   - Check resource utilization
   - Analyze slow queries
   - Review problem complexity
   - Consider backend switching

3. **Backend Outage**:
   - Confirm outage with provider
   - Switch to alternative backends
   - Notify users of service impact
   - Monitor recovery

### Automated Recovery

```python
from quantum_planner.recovery import AutoRecovery

recovery = AutoRecovery()

@recovery.register("backend_timeout")
def handle_backend_timeout(context):
    """Automatic recovery for backend timeouts."""
    backend_name = context["backend"]
    
    # Switch to fallback
    planner.switch_backend("fallback")
    
    # Retry failed operations
    for operation in context["failed_operations"]:
        try:
            operation.retry()
        except Exception as e:
            logger.error(f"Retry failed: {e}")
    
    # Schedule backend health check
    recovery.schedule_health_check(backend_name, delay=300)
```

## Best Practices

1. **Logging**:
   - Use structured logging with consistent fields
   - Include correlation IDs for request tracing
   - Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
   - Avoid logging sensitive information

2. **Metrics**:
   - Use consistent naming conventions
   - Include relevant tags/labels
   - Monitor both technical and business metrics
   - Set up proper aggregation and retention

3. **Alerts**:
   - Set meaningful thresholds
   - Avoid alert fatigue
   - Include actionable information
   - Test alert channels regularly

4. **Dashboards**:
   - Focus on key metrics
   - Use appropriate visualization types
   - Include context and annotations
   - Make dashboards accessible to all stakeholders

5. **Performance**:
   - Profile regularly
   - Monitor resource usage
   - Set up capacity planning
   - Optimize hot paths
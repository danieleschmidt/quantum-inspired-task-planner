"""Advanced monitoring and observability for quantum task planner."""

import time
import logging
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import os


class MetricType(Enum):
    """Types of metrics we can collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    condition: Callable[[float], bool]
    message: str
    severity: str = "warning"
    cooldown_seconds: float = 300.0
    last_triggered: Optional[float] = None


class MonitoringSystem:
    """Comprehensive monitoring system for the quantum task planner."""
    
    def __init__(self, export_interval: int = 60):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: List[AlertRule] = []
        self.export_interval = export_interval
        self.running = False
        self._export_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Built-in metrics
        self._setup_builtin_metrics()
        
        # Built-in alerts
        self._setup_builtin_alerts()
    
    def _setup_builtin_metrics(self):
        """Setup built-in system metrics."""
        self.record_gauge("system.started", 1.0, {"version": "1.0.0"})
    
    def _setup_builtin_alerts(self):
        """Setup built-in alert rules."""
        self.add_alert_rule(
            name="high_error_rate",
            condition=lambda x: x > 0.1,  # More than 0.1 errors per second
            message="High error rate detected",
            severity="critical"
        )
        
        self.add_alert_rule(
            name="slow_optimization",
            condition=lambda x: x > 30.0,  # Optimization takes more than 30s
            message="Optimization performance degraded",
            severity="warning"
        )
        
        self.add_alert_rule(
            name="memory_usage_high",
            condition=lambda x: x > 0.8,  # More than 80% memory usage
            message="High memory usage detected",
            severity="warning"
        )
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (cumulative)."""
        self._record_metric(name, value, MetricType.COUNTER, labels or {})
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (snapshot value)."""
        self._record_metric(name, value, MetricType.GAUGE, labels or {})
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric (duration in seconds)."""
        self._record_metric(name, duration, MetricType.TIMER, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels or {})
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str]):
        """Internal method to record a metric."""
        with self._lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels,
                metric_type=metric_type
            )
            
            self.metrics[name].append(metric_point)
            
            # Check alert rules for this metric
            self._check_alerts(name, value)
    
    def add_alert_rule(self, name: str, condition: Callable[[float], bool], message: str, 
                      severity: str = "warning", cooldown_seconds: float = 300.0):
        """Add an alert rule."""
        alert_rule = AlertRule(
            name=name,
            condition=condition,
            message=message,
            severity=severity,
            cooldown_seconds=cooldown_seconds
        )
        self.alert_rules.append(alert_rule)
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if any alert rules are triggered."""
        current_time = time.time()
        
        for alert in self.alert_rules:
            # Check if this alert applies to this metric
            if not self._alert_applies_to_metric(alert, metric_name):
                continue
            
            # Check cooldown
            if (alert.last_triggered and 
                current_time - alert.last_triggered < alert.cooldown_seconds):
                continue
            
            # Check condition
            if alert.condition(value):
                self._trigger_alert(alert, metric_name, value)
                alert.last_triggered = current_time
    
    def _alert_applies_to_metric(self, alert: AlertRule, metric_name: str) -> bool:
        """Check if an alert rule applies to a specific metric."""
        # Simple name-based matching for now
        # Could be extended with more sophisticated rules
        if "error" in alert.name and "error" in metric_name:
            return True
        if "optimization" in alert.name and "optimization" in metric_name:
            return True
        if "memory" in alert.name and "memory" in metric_name:
            return True
        return False
    
    def _trigger_alert(self, alert: AlertRule, metric_name: str, value: float):
        """Trigger an alert."""
        alert_data = {
            "alert_name": alert.name,
            "metric_name": metric_name,
            "value": value,
            "message": alert.message,
            "severity": alert.severity,
            "timestamp": time.time()
        }
        
        # Log the alert
        if alert.severity == "critical":
            self.logger.critical(f"ALERT: {alert.message} (metric: {metric_name}, value: {value})")
        else:
            self.logger.warning(f"ALERT: {alert.message} (metric: {metric_name}, value: {value})")
        
        # Could extend to send to external alerting systems
        self._export_alert(alert_data)
    
    def _export_alert(self, alert_data: Dict[str, Any]):
        """Export alert to external systems."""
        # Write to file for now, could be extended to send to Slack, PagerDuty, etc.
        alerts_file = "alerts.jsonl"
        try:
            with open(alerts_file, "a") as f:
                f.write(json.dumps(alert_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to export alert: {e}")
    
    def get_metric_summary(self, name: str, time_window: float = 300.0) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time window."""
        if name not in self.metrics:
            return {"error": "Metric not found"}
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            recent_points = [
                point for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
        
        if not recent_points:
            return {"error": "No recent data points"}
        
        values = [point.value for point in recent_points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "time_window": time_window
        }
    
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics data."""
        with self._lock:
            result = {}
            for name, points in self.metrics.items():
                result[name] = [
                    {
                        "value": point.value,
                        "timestamp": point.timestamp,
                        "labels": point.labels,
                        "type": point.metric_type.value
                    }
                    for point in points
                ]
            return result
    
    def start_export_thread(self):
        """Start the metrics export thread."""
        if not self.running:
            self.running = True
            self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
            self._export_thread.start()
            self.logger.info("Monitoring system started")
    
    def stop_export_thread(self):
        """Stop the metrics export thread."""
        self.running = False
        if self._export_thread:
            self._export_thread.join(timeout=5.0)
        self.logger.info("Monitoring system stopped")
    
    def _export_loop(self):
        """Main export loop running in background thread."""
        while self.running:
            try:
                self._export_metrics()
                time.sleep(self.export_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics export loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _export_metrics(self):
        """Export metrics to files/external systems."""
        timestamp = time.time()
        
        # Export to JSON file
        metrics_file = f"metrics_{int(timestamp)}.json"
        
        try:
            with open(metrics_file, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "metrics": self.get_all_metrics()
                }, f, indent=2)
            
            # Keep only recent files (cleanup)
            self._cleanup_old_files("metrics_*.json", max_files=10)
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def _cleanup_old_files(self, pattern: str, max_files: int):
        """Clean up old metric files."""
        import glob
        
        files = glob.glob(pattern)
        if len(files) > max_files:
            # Sort by modification time and remove oldest
            files.sort(key=os.path.getmtime)
            for old_file in files[:-max_files]:
                try:
                    os.remove(old_file)
                except OSError:
                    pass
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard."""
        dashboard_data = {
            "timestamp": time.time(),
            "system_health": self._get_system_health(),
            "key_metrics": {},
            "recent_alerts": self._get_recent_alerts()
        }
        
        # Get key metrics summaries
        key_metrics = [
            "optimization.duration",
            "optimization.success_rate", 
            "backend.availability",
            "error.rate",
            "memory.usage"
        ]
        
        for metric in key_metrics:
            summary = self.get_metric_summary(metric)
            if "error" not in summary:
                dashboard_data["key_metrics"][metric] = summary
        
        return dashboard_data
    
    def _get_system_health(self) -> str:
        """Get overall system health status."""
        # Simple health check based on recent error rates
        error_summary = self.get_metric_summary("error.rate", time_window=300.0)
        
        if "error" in error_summary:
            return "unknown"
        
        error_rate = error_summary.get("avg", 0)
        
        if error_rate > 0.1:
            return "critical"
        elif error_rate > 0.05:
            return "warning"
        else:
            return "healthy"
    
    def _get_recent_alerts(self, time_window: float = 3600.0) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        alerts_file = "alerts.jsonl"
        recent_alerts = []
        
        if not os.path.exists(alerts_file):
            return recent_alerts
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        try:
            with open(alerts_file, "r") as f:
                for line in f:
                    alert_data = json.loads(line.strip())
                    if alert_data.get("timestamp", 0) >= cutoff_time:
                        recent_alerts.append(alert_data)
        except Exception as e:
            self.logger.error(f"Error reading alerts file: {e}")
        
        return recent_alerts


# Context manager for timing operations
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, monitoring_system: MonitoringSystem, metric_name: str, 
                 labels: Optional[Dict[str, str]] = None):
        self.monitoring_system = monitoring_system
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitoring_system.record_timer(self.metric_name, duration, self.labels)


# Global monitoring instance
monitoring = MonitoringSystem()


# Decorators for easy monitoring
def monitor_performance(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(monitoring, metric_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_errors(metric_name: str = "errors"):
    """Decorator to monitor function errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                monitoring.record_counter(f"{metric_name}.success")
                return result
            except Exception as e:
                monitoring.record_counter(f"{metric_name}.failure", labels={
                    "error_type": type(e).__name__,
                    "function": func.__name__
                })
                raise
        return wrapper
    return decorator
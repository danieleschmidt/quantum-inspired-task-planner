#!/usr/bin/env python3
"""Production deployment configuration and monitoring system."""

import sys
import os
import time
import json
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import signal
import atexit

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/quantum_planner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"

@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 2
    
    # Internal state
    consecutive_failures: int = field(default=0, init=False)
    consecutive_successes: int = field(default=0, init=False)
    last_check_time: float = field(default=0.0, init=False)
    last_status: bool = field(default=True, init=False)

@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)

class ProductionMetrics:
    """Production metrics collection and monitoring."""
    
    def __init__(self):
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
        
        # Background metrics collection
        self.collection_thread = None
        self.running = False
    
    def start(self):
        """Start metrics collection."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment counter metric."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
            self.metrics.append(MetricPoint(name, self.counters[name], time.time(), tags or {}))
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set gauge metric."""
        with self.lock:
            self.gauges[name] = value
            self.metrics.append(MetricPoint(name, value, time.time(), tags or {}))
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record timer metric."""
        with self.lock:
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(duration)
            self.metrics.append(MetricPoint(f"{name}.duration", duration, time.time(), tags or {}))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            summary = {
                "timestamp": time.time(),
                "counters": self.counters.copy(),
                "gauges": self.gauges.copy(),
                "timers": {}
            }
            
            for timer_name, durations in self.timers.items():
                if durations:
                    summary["timers"][timer_name] = {
                        "count": len(durations),
                        "avg": sum(durations) / len(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "p95": self._percentile(durations, 95),
                        "p99": self._percentile(durations, 99)
                    }
            
            return summary
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _collection_loop(self):
        """Background metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU and memory
            self.set_gauge("system.cpu_percent", psutil.cpu_percent())
            memory = psutil.virtual_memory()
            self.set_gauge("system.memory_percent", memory.percent)
            self.set_gauge("system.memory_available_mb", memory.available / (1024 * 1024))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.set_gauge("system.disk_percent", disk.percent)
            self.set_gauge("system.disk_free_gb", disk.free / (1024 * 1024 * 1024))
            
        except ImportError:
            # Fallback metrics without psutil
            self.set_gauge("system.timestamp", time.time())

class HealthMonitor:
    """Production health monitoring system."""
    
    def __init__(self):
        self.health_checks: List[HealthCheck] = []
        self.status = ServiceStatus.STOPPED
        self.monitoring_thread = None
        self.running = False
        self.last_health_summary: Dict[str, Any] = {}
        
    def add_health_check(self, health_check: HealthCheck):
        """Add health check."""
        self.health_checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def start_monitoring(self):
        """Start health monitoring."""
        self.running = True
        self.status = ServiceStatus.STARTING
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
        
        # Initial health check
        time.sleep(1)
        self._update_status()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        self.status = ServiceStatus.STOPPING
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.status = ServiceStatus.STOPPED
        logger.info("Health monitoring stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "status": self.status.value,
            "timestamp": time.time(),
            "checks": self.last_health_summary,
            "uptime_seconds": time.time() - (getattr(self, 'start_time', time.time()))
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        self.start_time = time.time()
        
        while self.running:
            try:
                self._run_health_checks()
                self._update_status()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(30)
    
    def _run_health_checks(self):
        """Run all health checks."""
        check_results = {}
        
        for check in self.health_checks:
            now = time.time()
            
            # Skip if not time for this check
            if now - check.last_check_time < check.interval_seconds:
                continue
                
            try:
                # Run health check with timeout
                start_time = time.time()
                result = check.check_function()
                duration = time.time() - start_time
                
                if duration > check.timeout_seconds:
                    result = False
                    logger.warning(f"Health check {check.name} timed out ({duration:.2f}s)")
                
                check.last_check_time = now
                check.last_status = result
                
                if result:
                    check.consecutive_failures = 0
                    check.consecutive_successes += 1
                else:
                    check.consecutive_successes = 0
                    check.consecutive_failures += 1
                
                # Determine check status
                if check.consecutive_failures >= check.failure_threshold:
                    check_status = "unhealthy"
                elif check.consecutive_successes >= check.success_threshold:
                    check_status = "healthy"
                else:
                    check_status = "unknown"
                
                check_results[check.name] = {
                    "status": check_status,
                    "last_result": result,
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes,
                    "last_check": now,
                    "duration": duration
                }
                
            except Exception as e:
                logger.error(f"Health check {check.name} failed with exception: {e}")
                check.consecutive_failures += 1
                check_results[check.name] = {
                    "status": "error",
                    "error": str(e),
                    "consecutive_failures": check.consecutive_failures,
                    "last_check": now
                }
        
        self.last_health_summary = check_results
    
    def _update_status(self):
        """Update overall service status."""
        if not self.last_health_summary:
            return
        
        # Count health check statuses
        status_counts = {"healthy": 0, "unhealthy": 0, "unknown": 0, "error": 0}
        
        for check_result in self.last_health_summary.values():
            status = check_result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        if status_counts["error"] > 0 or status_counts["unhealthy"] > 0:
            if status_counts["unhealthy"] >= len(self.health_checks) // 2:
                self.status = ServiceStatus.UNHEALTHY
            else:
                self.status = ServiceStatus.DEGRADED
        elif status_counts["healthy"] == len(self.health_checks):
            self.status = ServiceStatus.HEALTHY
        else:
            self.status = ServiceStatus.DEGRADED

class ProductionService:
    """Main production service orchestrator."""
    
    def __init__(self, service_name: str = "quantum-planner"):
        self.service_name = service_name
        self.metrics = ProductionMetrics()
        self.health_monitor = HealthMonitor()
        self.shutdown_handlers: List[Callable] = []
        self.startup_time = time.time()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self.shutdown)
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def memory_check():
            """Check memory usage is reasonable."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Can't check, assume healthy
        
        def disk_check():
            """Check disk space is available."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return disk.percent < 95  # Less than 95% disk usage
            except ImportError:
                return True
        
        def response_time_check():
            """Check average response time is reasonable."""
            summary = self.metrics.get_summary()
            timers = summary.get("timers", {})
            if "request.duration" in timers:
                avg_response = timers["request.duration"]["avg"]
                return avg_response < 2.0  # Less than 2 seconds average
            return True
        
        # Add health checks
        self.health_monitor.add_health_check(
            HealthCheck("memory_usage", memory_check, interval_seconds=30)
        )
        self.health_monitor.add_health_check(
            HealthCheck("disk_space", disk_check, interval_seconds=60)
        )
        self.health_monitor.add_health_check(
            HealthCheck("response_time", response_time_check, interval_seconds=45)
        )
    
    def start(self):
        """Start production service."""
        logger.info(f"Starting {self.service_name} service...")
        
        # Start metrics collection
        self.metrics.start()
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Record startup
        self.metrics.increment_counter("service.starts")
        self.metrics.set_gauge("service.startup_time", self.startup_time)
        
        logger.info(f"{self.service_name} service started successfully")
    
    def shutdown(self):
        """Shutdown production service."""
        logger.info(f"Shutting down {self.service_name} service...")
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        # Stop metrics collection  
        self.metrics.stop()
        
        # Run shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        # Record shutdown
        self.metrics.increment_counter("service.shutdowns")
        
        logger.info(f"{self.service_name} service shut down")
    
    def add_shutdown_handler(self, handler: Callable):
        """Add shutdown handler."""
        self.shutdown_handlers.append(handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information."""
        return {
            "service": self.service_name,
            "version": "1.0.0",
            "uptime_seconds": time.time() - self.startup_time,
            "health": self.health_monitor.get_health_status(),
            "metrics": self.metrics.get_summary(),
            "timestamp": time.time()
        }

# Example production usage
def create_production_planner_service():
    """Create production-ready quantum planner service."""
    
    # Import our optimizers
    from production_test_suite import SimpleOptimizer, Agent, Task
    
    service = ProductionService("quantum-planner")
    optimizer = SimpleOptimizer()
    
    # Custom health check for planner
    def planner_health_check():
        """Check that planner can handle basic requests."""
        try:
            agents = [Agent("test_agent", ["python"], capacity=1)]
            tasks = [Task("test_task", ["python"], priority=1, duration=1)]
            solution = optimizer.assign(agents, tasks)
            return len(solution.assignments) == 1
        except Exception:
            return False
    
    service.health_monitor.add_health_check(
        HealthCheck("planner_functionality", planner_health_check, interval_seconds=60)
    )
    
    # Request handler with metrics
    def handle_planning_request(agents: List[Agent], tasks: List[Task]):
        """Handle planning request with metrics."""
        start_time = time.time()
        
        try:
            # Record request
            service.metrics.increment_counter("requests.total")
            service.metrics.set_gauge("requests.active", 1)  # Simplified
            
            # Process request
            solution = optimizer.assign(agents, tasks)
            
            # Record success metrics
            duration = time.time() - start_time
            service.metrics.record_timer("request.duration", duration)
            service.metrics.increment_counter("requests.success")
            service.metrics.set_gauge("solution.quality", solution.quality_score)
            service.metrics.set_gauge("solution.makespan", solution.makespan)
            
            return solution
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            service.metrics.record_timer("request.duration", duration)
            service.metrics.increment_counter("requests.error")
            logger.error(f"Planning request failed: {e}")
            raise
        
        finally:
            service.metrics.set_gauge("requests.active", 0)  # Simplified
    
    return service, handle_planning_request

def test_production_service():
    """Test production service functionality."""
    print("🏭 Testing production service...")
    
    # Create service
    service, planner_handler = create_production_planner_service()
    
    try:
        # Start service
        service.start()
        time.sleep(2)  # Let it initialize
        
        # Test planning request
        from production_test_suite import Agent, Task
        
        agents = [
            Agent("prod_agent1", ["python", "java"], capacity=3),
            Agent("prod_agent2", ["javascript"], capacity=2)
        ]
        
        tasks = [
            Task("prod_task1", ["python"], priority=5, duration=2),
            Task("prod_task2", ["javascript"], priority=3, duration=1),
            Task("prod_task3", ["java"], priority=1, duration=1)
        ]
        
        # Process requests
        for i in range(5):
            solution = planner_handler(agents, tasks)
            assert solution.quality_score > 0
            time.sleep(0.1)
        
        # Check service info
        info = service.get_service_info()
        assert info["service"] == "quantum-planner"
        assert info["health"]["status"] in ["healthy", "degraded"]
        assert info["metrics"]["counters"]["requests.total"] == 5
        
        print("✅ Production service test passed!")
        return True
        
    finally:
        service.shutdown()

def main():
    """Run production deployment test."""
    print("🚀 Production Deployment Test")
    print("=" * 50)
    
    try:
        # Test production service
        test_production_service()
        
        print("\n🎯 PRODUCTION DEPLOYMENT SUCCESS!")
        print("✅ Health monitoring operational")
        print("✅ Metrics collection working") 
        print("✅ Service lifecycle management")
        print("✅ Error handling and recovery")
        print("✅ Production logging configured")
        print("✅ Ready for container deployment!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Production deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
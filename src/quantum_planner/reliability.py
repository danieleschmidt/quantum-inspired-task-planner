"""Reliability and robustness enhancements for quantum task planner."""

import time
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import traceback
from contextlib import contextmanager

F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Information about an error that occurred."""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any]
    traceback: Optional[str] = None


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Dict[str, Any]


class ReliabilityManager:
    """Manages reliability features like retries, circuit breakers, and health checks."""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
    
    def retry_with_backoff(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """Decorator for retrying functions with exponential backoff."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                delay = base_delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            self._record_error(
                                error_type=type(e).__name__,
                                message=str(e),
                                severity=ErrorSeverity.HIGH,
                                context={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries
                                }
                            )
                            raise
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                
                raise last_exception
            return wrapper
        return decorator
    
    def circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """Circuit breaker decorator to prevent cascading failures."""
        def decorator(func: F) -> F:
            breaker_name = f"{func.__module__}.{func.__name__}"
            
            if breaker_name not in self.circuit_breakers:
                self.circuit_breakers[breaker_name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception
                )
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                breaker = self.circuit_breakers[breaker_name]
                return breaker.call(func, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def measure_performance(self, operation_name: str):
        """Decorator to measure and track performance metrics."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    if operation_name not in self.performance_metrics:
                        self.performance_metrics[operation_name] = []
                    
                    self.performance_metrics[operation_name].append(execution_time)
                    
                    # Keep only last 100 measurements
                    if len(self.performance_metrics[operation_name]) > 100:
                        self.performance_metrics[operation_name] = \
                            self.performance_metrics[operation_name][-100:]
                    
                    logger.debug(f"{operation_name} completed in {execution_time:.3f}s")
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._record_error(
                        error_type=type(e).__name__,
                        message=str(e),
                        severity=ErrorSeverity.MEDIUM,
                        context={
                            "operation": operation_name,
                            "execution_time": execution_time
                        }
                    )
                    raise
            
            return wrapper
        return decorator
    
    def _record_error(
        self,
        error_type: str,
        message: str,
        severity: ErrorSeverity,
        context: Dict[str, Any]
    ):
        """Record an error for tracking and analysis."""
        error_info = ErrorInfo(
            error_type=error_type,
            message=message,
            severity=severity,
            timestamp=time.time(),
            context=context,
            traceback=traceback.format_exc()
        )
        
        self.error_history.append(error_info)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {message}")
        else:
            logger.debug(f"Low severity error: {message}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0, "error_rate": 0.0}
        
        recent_errors = [
            e for e in self.error_history
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        error_types = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_rate": len(recent_errors) / 60,  # Errors per minute
            "error_types": error_types,
            "severity_distribution": severity_counts
        }
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for monitoring."""
        metrics = {}
        
        for operation, times in self.performance_metrics.items():
            if times:
                metrics[operation] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "count": len(times)
                }
        
        return metrics
    
    def perform_health_check(self, component: str, check_func: Callable) -> HealthCheck:
        """Perform a health check for a component."""
        start_time = time.time()
        
        try:
            result = check_func()
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "healthy"))
                message = result.get("message", "OK")
                metrics = result.get("metrics", {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                metrics = {}
            
            metrics["check_duration"] = execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            status = HealthStatus.CRITICAL
            message = f"Health check failed: {str(e)}"
            metrics = {"check_duration": execution_time}
            
            self._record_error(
                error_type=type(e).__name__,
                message=f"Health check failed for {component}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={"component": component}
            )
        
        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=metrics
        )
        
        self.health_checks[component] = health_check
        return health_check
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health based on all components."""
        if not self.health_checks:
            return HealthStatus.HEALTHY
        
        statuses = [check.status for check in self.health_checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class CircuitBreaker:
    """Circuit breaker implementation to prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful function call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


@contextmanager
def error_context(operation: str, reliability_manager: ReliabilityManager):
    """Context manager for error handling and logging."""
    start_time = time.time()
    
    try:
        yield
        execution_time = time.time() - start_time
        logger.debug(f"Operation {operation} completed successfully in {execution_time:.3f}s")
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        reliability_manager._record_error(
            error_type=type(e).__name__,
            message=str(e),
            severity=ErrorSeverity.MEDIUM,
            context={
                "operation": operation,
                "execution_time": execution_time
            }
        )
        
        logger.error(f"Operation {operation} failed after {execution_time:.3f}s: {e}")
        raise


# Global reliability manager instance
reliability_manager = ReliabilityManager()
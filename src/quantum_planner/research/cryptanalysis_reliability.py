"""Reliability and Fault Tolerance Module for Cryptanalysis.

Provides circuit breakers, retry mechanisms, fallback strategies,
and health monitoring for robust cryptanalysis operations.
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import torch
import numpy as np
from loguru import logger
from contextlib import contextmanager
import pickle
import json
import os
from pathlib import Path


class OperationStatus(Enum):
    """Status of cryptanalysis operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_exceptions: List[type] = field(default_factory=lambda: [
        RuntimeError, ConnectionError, TimeoutError
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    success_threshold: int = 2  # For half-open state


@dataclass
class HealthCheckConfig:
    """Configuration for health monitoring."""
    check_interval: float = 30.0
    timeout: float = 10.0
    max_memory_usage: float = 0.8  # 80% of available memory
    max_cpu_usage: float = 0.9     # 90% CPU
    enable_auto_recovery: bool = True
    health_history_size: int = 100


class OperationResult:
    """Result container for cryptanalysis operations."""
    
    def __init__(
        self,
        operation_id: str,
        status: OperationStatus,
        result: Optional[Any] = None,
        error: Optional[Exception] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation_id = operation_id
        self.status = status
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def is_success(self) -> bool:
        return self.status == OperationStatus.COMPLETED
    
    def is_failure(self) -> bool:
        return self.status in [OperationStatus.FAILED, OperationStatus.TIMEOUT]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "has_result": self.result is not None,
            "has_error": self.error is not None,
            "error_type": type(self.error).__name__ if self.error else None
        }


class RetryMechanism:
    """Implements retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logger.bind(component="retry_mechanism")
    
    def execute_with_retry(self, operation: Callable, operation_id: str, *args, **kwargs) -> OperationResult:
        """Execute operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                start_time = time.time()
                result = operation(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.logger.info(f"Operation {operation_id} succeeded on attempt {attempt + 1}")
                
                return OperationResult(
                    operation_id=operation_id,
                    status=OperationStatus.COMPLETED,
                    result=result,
                    execution_time=execution_time,
                    metadata={"attempts": attempt + 1}
                )
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    self.logger.error(f"Operation {operation_id} failed permanently: {e}")
                    break
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Operation {operation_id} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.2f}s: {e}"
                )
                time.sleep(delay)
        
        return OperationResult(
            operation_id=operation_id,
            status=OperationStatus.FAILED,
            error=last_exception,
            metadata={"attempts": self.config.max_attempts}
        )
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts - 1:
            return False
        
        return any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        if self.config.exponential_backoff:
            delay = min(self.config.base_delay * (2 ** attempt), self.config.max_delay)
        else:
            delay = self.config.base_delay
        
        if self.config.jitter:
            delay *= (0.5 + 0.5 * np.random.random())
        
        return delay


class CircuitBreaker:
    """Circuit breaker for failing operations."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()
        self.logger = logger.bind(component="circuit_breaker")
    
    @contextmanager
    def protect(self, operation_id: str):
        """Context manager for circuit breaker protection."""
        if not self._allow_request():
            raise RuntimeError(f"Circuit breaker open for operation {operation_id}")
        
        try:
            yield
            self._record_success()
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self.lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info("Circuit breaker transitioning to half-open")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def _record_success(self):
        """Record successful operation."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.logger.info("Circuit breaker closed - service recovered")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def _record_failure(self, exception: Exception):
        """Record failed operation."""
        with self.lock:
            if isinstance(exception, self.config.expected_exception):
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time
            }


class FallbackStrategy:
    """Implements fallback strategies for failed operations."""
    
    def __init__(self):
        self.fallback_handlers = {}
        self.logger = logger.bind(component="fallback_strategy")
    
    def register_fallback(self, operation_type: str, fallback_function: Callable):
        """Register fallback function for operation type."""
        self.fallback_handlers[operation_type] = fallback_function
        self.logger.info(f"Registered fallback for {operation_type}")
    
    def execute_fallback(
        self,
        operation_type: str,
        original_args: tuple,
        original_kwargs: dict,
        original_error: Exception
    ) -> OperationResult:
        """Execute fallback for failed operation."""
        if operation_type not in self.fallback_handlers:
            return OperationResult(
                operation_id="fallback",
                status=OperationStatus.FAILED,
                error=RuntimeError(f"No fallback available for {operation_type}")
            )
        
        try:
            fallback_function = self.fallback_handlers[operation_type]
            result = fallback_function(*original_args, **original_kwargs)
            
            self.logger.info(f"Fallback successful for {operation_type}")
            
            return OperationResult(
                operation_id="fallback",
                status=OperationStatus.COMPLETED,
                result=result,
                metadata={
                    "fallback_used": True,
                    "original_error": str(original_error)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Fallback failed for {operation_type}: {e}")
            return OperationResult(
                operation_id="fallback",
                status=OperationStatus.FAILED,
                error=e,
                metadata={"fallback_failed": True}
            )


class HealthMonitor:
    """Monitors system health and component status."""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.health_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        self.logger = logger.bind(component="health_monitor")
        self.health_checks = {}
    
    def register_health_check(self, name: str, check_function: Callable[[], bool]):
        """Register a custom health check."""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                health_status = self.check_health()
                
                with self.lock:
                    self.health_history.append({
                        "timestamp": time.time(),
                        "status": health_status
                    })
                    
                    # Limit history size
                    if len(self.health_history) > self.config.health_history_size:
                        self.health_history = self.health_history[-self.config.health_history_size//2:]
                
                if not health_status["overall_healthy"] and self.config.enable_auto_recovery:
                    self._attempt_auto_recovery(health_status)
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.check_interval)
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "timestamp": time.time(),
            "checks": {},
            "overall_healthy": True
        }
        
        # System resource checks
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            memory_healthy = memory.percent / 100 < self.config.max_memory_usage
            cpu_healthy = cpu_percent / 100 < self.config.max_cpu_usage
            
            health_status["checks"]["memory"] = {
                "healthy": memory_healthy,
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3)
            }
            
            health_status["checks"]["cpu"] = {
                "healthy": cpu_healthy,
                "usage_percent": cpu_percent
            }
            
            if not memory_healthy or not cpu_healthy:
                health_status["overall_healthy"] = False
                
        except ImportError:
            health_status["checks"]["system"] = {
                "healthy": False,
                "error": "psutil not available"
            }
        
        # GPU checks if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_healthy = gpu_memory < 0.9
                
                health_status["checks"]["gpu"] = {
                    "healthy": gpu_healthy,
                    "memory_usage": gpu_memory,
                    "device_count": torch.cuda.device_count()
                }
                
                if not gpu_healthy:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                health_status["checks"]["gpu"] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # Custom health checks
        for name, check_function in self.health_checks.items():
            try:
                is_healthy = check_function()
                health_status["checks"][name] = {"healthy": is_healthy}
                
                if not is_healthy:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                health_status["checks"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        
        return health_status
    
    def _attempt_auto_recovery(self, health_status: Dict[str, Any]):
        """Attempt automatic recovery from health issues."""
        self.logger.warning("Attempting auto-recovery from health issues")
        
        # GPU memory recovery
        if "gpu" in health_status["checks"] and not health_status["checks"]["gpu"]["healthy"]:
            try:
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared for recovery")
            except Exception as e:
                self.logger.error(f"GPU recovery failed: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.logger.info("Auto-recovery attempt completed")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health status."""
        with self.lock:
            if not self.health_history:
                return {"status": "no_data"}
            
            recent_checks = self.health_history[-10:]  # Last 10 checks
            healthy_count = sum(1 for check in recent_checks if check["status"]["overall_healthy"])
            
            return {
                "recent_healthy_ratio": healthy_count / len(recent_checks),
                "last_check": self.health_history[-1],
                "total_checks": len(self.health_history),
                "monitoring_active": self.is_monitoring
            }


class OperationCache:
    """Caches results of expensive cryptanalysis operations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
        self.logger = logger.bind(component="operation_cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Cache result."""
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_ratio": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }


class ReliabilityManager:
    """Manages all reliability components for cryptanalysis operations."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        health_check_config: Optional[HealthCheckConfig] = None,
        enable_caching: bool = True
    ):
        self.retry_mechanism = RetryMechanism(retry_config or RetryConfig())
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig())
        self.fallback_strategy = FallbackStrategy()
        self.health_monitor = HealthMonitor(health_check_config or HealthCheckConfig())
        self.operation_cache = OperationCache() if enable_caching else None
        
        self.logger = logger.bind(component="reliability_manager")
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
    
    def execute_reliable_operation(
        self,
        operation: Callable,
        operation_id: str,
        operation_type: str,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute operation with full reliability stack."""
        
        # Check cache first
        if use_cache and self.operation_cache and cache_key:
            cached_result = self.operation_cache.get(cache_key)
            if cached_result is not None:
                self.logger.info(f"Cache hit for operation {operation_id}")
                return OperationResult(
                    operation_id=operation_id,
                    status=OperationStatus.COMPLETED,
                    result=cached_result,
                    metadata={"from_cache": True}
                )
        
        # Execute with circuit breaker protection
        try:
            with self.circuit_breaker.protect(operation_id):
                result = self.retry_mechanism.execute_with_retry(
                    operation, operation_id, *args, **kwargs
                )
                
                # Cache successful results
                if result.is_success() and use_cache and self.operation_cache and cache_key:
                    self.operation_cache.put(cache_key, result.result)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Operation {operation_id} failed with circuit breaker: {e}")
            
            # Try fallback
            fallback_result = self.fallback_strategy.execute_fallback(
                operation_type, args, kwargs, e
            )
            
            if fallback_result.is_success():
                return fallback_result
            
            return OperationResult(
                operation_id=operation_id,
                status=OperationStatus.FAILED,
                error=e
            )
    
    def get_reliability_status(self) -> Dict[str, Any]:
        """Get overall reliability status."""
        return {
            "circuit_breaker": self.circuit_breaker.get_state(),
            "health_monitor": self.health_monitor.get_health_summary(),
            "cache_stats": self.operation_cache.get_stats() if self.operation_cache else None,
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """Shutdown reliability manager."""
        self.health_monitor.stop_monitoring()
        if self.operation_cache:
            self.operation_cache.clear()
        self.logger.info("Reliability manager shutdown")


def create_reliable_cryptanalysis_environment() -> ReliabilityManager:
    """Create a complete reliable environment for cryptanalysis."""
    return ReliabilityManager(
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
        health_check_config=HealthCheckConfig(check_interval=30.0),
        enable_caching=True
    )

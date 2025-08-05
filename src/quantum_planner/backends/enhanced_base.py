"""Enhanced base backend with comprehensive error handling and monitoring."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import hashlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BackendStatus(Enum):
    """Backend availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class BackendMetrics:
    """Metrics for backend performance monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_solve_time: float = 0.0
    total_solve_time: float = 0.0
    last_request_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


@dataclass 
class HealthCheck:
    """Health check result for a backend."""
    status: BackendStatus
    latency: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendCapabilities:
    """Detailed backend capabilities."""
    max_variables: int
    supports_constraints: bool = True
    supports_embedding: bool = False
    supports_async: bool = False
    supports_batching: bool = False
    max_batch_size: int = 1
    supported_objectives: List[str] = field(default_factory=lambda: ["minimize"])
    constraint_types: List[str] = field(default_factory=list)
    
    def is_compatible_with_problem(self, num_variables: int, constraints: List[str]) -> bool:
        """Check if backend can handle the given problem."""
        if num_variables > self.max_variables:
            return False
        
        if constraints and not self.supports_constraints:
            return False
            
        if constraints:
            unsupported = set(constraints) - set(self.constraint_types)
            if unsupported:
                logger.warning(f"Unsupported constraints: {unsupported}")
                return False
        
        return True


class EnhancedQuantumBackend(ABC):
    """Enhanced base class for quantum and classical backends with robust error handling."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize backend with configuration."""
        self.name = name
        self.config = config or {}
        self.metrics = BackendMetrics()
        self._last_health_check: Optional[HealthCheck] = None
        self._solution_cache: Dict[str, Any] = {}
        self.max_cache_size = self.config.get("max_cache_size", 100)
        
        # Initialize backend-specific settings
        self._initialize_backend()
    
    @abstractmethod
    def _initialize_backend(self) -> None:
        """Initialize backend-specific components."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        pass
    
    @abstractmethod
    def _solve_qubo(self, Q: Any, **kwargs) -> Dict[int, int]:
        """Solve QUBO problem (backend-specific implementation)."""
        pass
    
    def solve_qubo(self, Q: Any, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """Solve QUBO with comprehensive error handling and monitoring."""
        start_time = time.time()
        self.metrics.total_requests += 1
        self.metrics.last_request_time = start_time
        
        try:
            # Check cache first
            if use_cache:
                cache_key = self._generate_cache_key(Q, kwargs)
                cached_result = self._get_cached_solution(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for {self.name}")
                    return cached_result
            
            # Validate problem
            self._validate_problem(Q, **kwargs)
            
            # Pre-processing
            Q_processed = self._preprocess_problem(Q, **kwargs)
            
            # Solve
            with self._monitor_execution():
                solution = self._solve_qubo(Q_processed, **kwargs)
            
            # Post-processing
            result = self._postprocess_solution(solution, Q, **kwargs)
            
            # Update metrics
            solve_time = time.time() - start_time
            self._update_success_metrics(solve_time)
            
            # Cache result
            if use_cache:
                self._cache_solution(cache_key, result)
            
            logger.info(f"Successfully solved problem with {self.name} in {solve_time:.3f}s")
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            self._update_failure_metrics(solve_time, str(e))
            
            logger.error(f"Failed to solve with {self.name}: {e}")
            
            # Return error result
            return {
                "solution": {},
                "energy": float('inf'),
                "success": False,
                "error_message": str(e),
                "solve_time": solve_time,
                "backend": self.name
            }
    
    def health_check(self) -> HealthCheck:
        """Perform comprehensive health check."""
        start_time = time.time()
        
        try:
            # Basic connectivity check
            status = self._check_connectivity()
            
            # Performance check with small problem
            if status == BackendStatus.AVAILABLE:
                status = self._check_performance()
            
            latency = time.time() - start_time
            
            self._last_health_check = HealthCheck(
                status=status,
                latency=latency,
                timestamp=start_time
            )
            
            return self._last_health_check
            
        except Exception as e:
            latency = time.time() - start_time
            
            self._last_health_check = HealthCheck(
                status=BackendStatus.ERROR,
                latency=latency,
                error_message=str(e),
                timestamp=start_time
            )
            
            return self._last_health_check
    
    def is_available(self) -> bool:
        """Check if backend is currently available."""
        # Use cached health check if recent
        if (self._last_health_check and 
            time.time() - self._last_health_check.timestamp < 300):  # 5 minutes
            return self._last_health_check.status == BackendStatus.AVAILABLE
        
        # Perform new health check
        health = self.health_check()
        return health.status == BackendStatus.AVAILABLE
    
    def estimate_solve_time(self, num_variables: int) -> float:
        """Estimate solve time based on problem size and historical data."""
        base_estimate = self._base_time_estimation(num_variables)
        
        # Adjust based on recent performance
        if self.metrics.avg_solve_time > 0:
            recent_factor = min(2.0, self.metrics.avg_solve_time / base_estimate)
            return base_estimate * recent_factor
        
        return base_estimate
    
    def get_metrics(self) -> BackendMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = BackendMetrics()
    
    # Private methods
    
    def _validate_problem(self, Q: Any, **kwargs) -> None:
        """Validate that the problem can be solved by this backend."""
        capabilities = self.get_capabilities()
        
        # Basic validation - subclasses can override
        if hasattr(Q, 'shape'):
            num_vars = Q.shape[0]
            if num_vars > capabilities.max_variables:
                raise ValueError(f"Problem too large: {num_vars} > {capabilities.max_variables}")
    
    def _preprocess_problem(self, Q: Any, **kwargs) -> Any:
        """Preprocess problem before solving. Override in subclasses."""
        return Q
    
    def _postprocess_solution(self, solution: Dict[int, int], Q: Any, **kwargs) -> Dict[str, Any]:
        """Postprocess solution after solving."""
        return {
            "solution": solution,
            "energy": self._calculate_energy(solution, Q),
            "success": True,
            "backend": self.name,
            "num_variables": len(solution)
        }
    
    def _calculate_energy(self, solution: Dict[int, int], Q: Any) -> float:
        """Calculate energy of solution. Override in subclasses."""
        try:
            if hasattr(Q, 'shape'):
                # Numpy matrix
                x = [solution.get(i, 0) for i in range(Q.shape[0])]
                return float(np.dot(x, np.dot(Q, x)))
            else:
                # Dictionary form
                energy = 0.0
                for (i, j), coeff in Q.items():
                    energy += coeff * solution.get(i, 0) * solution.get(j, 0)
                return energy
        except Exception:
            return 0.0
    
    @contextmanager
    def _monitor_execution(self):
        """Context manager for monitoring execution."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Execution time for {self.name}: {execution_time:.3f}s")
    
    def _update_success_metrics(self, solve_time: float) -> None:
        """Update metrics after successful solve."""
        self.metrics.successful_requests += 1
        self.metrics.total_solve_time += solve_time
        self.metrics.avg_solve_time = (self.metrics.total_solve_time / 
                                     self.metrics.successful_requests)
        self.metrics.last_success_time = time.time()
        self.metrics.consecutive_failures = 0
    
    def _update_failure_metrics(self, solve_time: float, error_msg: str) -> None:
        """Update metrics after failed solve."""
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        logger.warning(f"Backend {self.name} failure #{self.metrics.consecutive_failures}: {error_msg}")
    
    def _generate_cache_key(self, Q: Any, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for problem."""
        try:
            # Create deterministic key from Q matrix and parameters
            if hasattr(Q, 'tobytes'):
                q_bytes = Q.tobytes()
            else:
                q_bytes = str(sorted(Q.items())).encode()
            
            kwargs_str = str(sorted(kwargs.items()))
            combined = q_bytes + kwargs_str.encode()
            
            return hashlib.md5(combined).hexdigest()
        except Exception:
            return str(hash((str(Q), str(kwargs))))
    
    def _get_cached_solution(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached solution if available."""
        return self._solution_cache.get(cache_key)
    
    def _cache_solution(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache solution result."""
        if len(self._solution_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._solution_cache))
            del self._solution_cache[oldest_key]
        
        self._solution_cache[cache_key] = result
    
    @abstractmethod
    def _check_connectivity(self) -> BackendStatus:
        """Check basic connectivity to backend."""
        pass
    
    def _check_performance(self) -> BackendStatus:
        """Check performance with small test problem."""
        try:
            # Create small test QUBO
            test_Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
            
            start_time = time.time()
            result = self._solve_qubo(test_Q, num_reads=10)
            solve_time = time.time() - start_time
            
            if solve_time > 30:  # 30 second timeout
                return BackendStatus.DEGRADED
            
            return BackendStatus.AVAILABLE
            
        except Exception as e:
            logger.warning(f"Performance check failed for {self.name}: {e}")
            return BackendStatus.DEGRADED
    
    @abstractmethod
    def _base_time_estimation(self, num_variables: int) -> float:
        """Estimate base solve time for given problem size."""
        pass
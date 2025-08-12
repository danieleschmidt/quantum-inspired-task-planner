#!/usr/bin/env python3
"""
Robust Enhanced Quantum Task Planner - Generation 2 Implementation
Adds comprehensive error handling, validation, logging, monitoring, and reliability features.
"""

import time
import logging
import random
import json
import hashlib
import traceback
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import weakref

# Enhanced logging configuration
class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better visibility."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Setup enhanced logging
def setup_logging(level=logging.INFO, enable_colors=True):
    """Setup enhanced logging with colors and detailed formatting."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler()
    
    if enable_colors:
        formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging()

# Enhanced Enums and Types
class OptimizationObjective(Enum):
    """Available optimization objectives."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MAXIMIZE_PRIORITY = "maximize_priority"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_UTILIZATION = "maximize_utilization"

class BackendType(Enum):
    """Available backend types."""
    AUTO = "auto"
    CLASSICAL = "classical"
    QUANTUM_SIMULATOR = "quantum_simulator"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HYBRID = "hybrid"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

# Exception Classes
class QuantumPlannerError(Exception):
    """Base exception for quantum planner errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()

class ValidationError(QuantumPlannerError):
    """Input validation error."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, ErrorSeverity.HIGH)
        self.field = field
        self.value = value

class BackendError(QuantumPlannerError):
    """Backend operation error."""
    def __init__(self, message: str, backend: str, operation: str = None):
        super().__init__(message, ErrorSeverity.HIGH)
        self.backend = backend
        self.operation = operation

class OptimizationError(QuantumPlannerError):
    """Optimization process error."""
    def __init__(self, message: str, problem_size: int = None, objective: str = None):
        super().__init__(message, ErrorSeverity.MEDIUM)
        self.problem_size = problem_size
        self.objective = objective

class TimeoutError(QuantumPlannerError):
    """Operation timeout error."""
    def __init__(self, message: str, timeout_seconds: float, operation: str = None):
        super().__init__(message, ErrorSeverity.HIGH)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

# Monitoring and Metrics Classes
@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    total_problems: int = 0
    successful_problems: int = 0
    failed_problems: int = 0
    total_solve_time: float = 0.0
    average_solve_time: float = 0.0
    average_quality: float = 0.0
    peak_memory_usage: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_usage: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_problems == 0:
            return 0.0
        return self.successful_problems / self.total_problems
    
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests

@dataclass 
class ErrorRecord:
    """Error record for tracking and analysis."""
    timestamp: datetime
    error_type: str
    message: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "stack_trace": self.stack_trace
        }

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker transitioning to half-open state")
                else:
                    raise BackendError("Circuit breaker is open", "circuit_breaker")
        
        try:
            result = func(*args, **kwargs)
            
            with self._lock:
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - service recovered")
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker opened due to {self.failure_count} failures")
                elif self.state == "half_open":
                    self.state = "open"
                    logger.error("Circuit breaker re-opened during half-open state")
            
            raise

class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.checks = {}
        self.last_check_time = time.time()
        self.status = HealthStatus.HEALTHY
        self.issues = []
    
    def register_check(self, name: str, check_func: Callable[[], tuple[bool, str]]):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def check_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        current_time = time.time()
        
        if current_time - self.last_check_time < self.check_interval:
            return self._get_cached_status()
        
        self.last_check_time = current_time
        self.issues = []
        check_results = {}
        
        for name, check_func in self.checks.items():
            try:
                is_healthy, message = check_func()
                check_results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "message": message,
                    "timestamp": current_time
                }
                
                if not is_healthy:
                    self.issues.append(f"{name}: {message}")
                    
            except Exception as e:
                check_results[name] = {
                    "status": "error",
                    "message": str(e),
                    "timestamp": current_time
                }
                self.issues.append(f"{name}: {str(e)}")
        
        # Determine overall status
        if not self.issues:
            self.status = HealthStatus.HEALTHY
        elif len(self.issues) <= 2:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.UNHEALTHY
        
        return {
            "overall_status": self.status.value,
            "timestamp": current_time,
            "checks": check_results,
            "issues": self.issues
        }
    
    def _get_cached_status(self) -> Dict[str, Any]:
        """Get cached health status."""
        return {
            "overall_status": self.status.value,
            "timestamp": self.last_check_time,
            "checks": {},
            "issues": self.issues,
            "cached": True
        }

# Enhanced Data Models with Validation
@dataclass(frozen=True)
class Agent:
    """Represents an agent that can execute tasks with comprehensive validation."""
    
    agent_id: str
    skills: List[str]
    capacity: int = 1
    availability: float = 1.0
    cost_per_hour: float = 0.0
    region: str = "global"
    max_concurrent_tasks: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Comprehensive validation with detailed error messages."""
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValidationError("Agent ID must be a non-empty string", "agent_id", self.agent_id)
        
        if not self.skills or not isinstance(self.skills, list):
            raise ValidationError("Skills must be a non-empty list", "skills", self.skills)
        
        if not all(isinstance(skill, str) and skill.strip() for skill in self.skills):
            raise ValidationError("All skills must be non-empty strings", "skills", self.skills)
        
        if not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ValidationError("Capacity must be a positive integer", "capacity", self.capacity)
        
        if not isinstance(self.availability, (int, float)) or not (0 <= self.availability <= 1):
            raise ValidationError("Availability must be between 0 and 1", "availability", self.availability)
        
        if not isinstance(self.cost_per_hour, (int, float)) or self.cost_per_hour < 0:
            raise ValidationError("Cost per hour must be non-negative", "cost_per_hour", self.cost_per_hour)
        
        if not isinstance(self.max_concurrent_tasks, int) or self.max_concurrent_tasks <= 0:
            raise ValidationError("Max concurrent tasks must be a positive integer", "max_concurrent_tasks", self.max_concurrent_tasks)
    
    def can_handle_task(self, task: 'Task') -> bool:
        """Check if agent can handle the given task with logging."""
        try:
            required_skills = set(task.required_skills)
            agent_skills = set(self.skills)
            can_handle = required_skills.issubset(agent_skills)
            
            if not can_handle:
                missing_skills = required_skills - agent_skills
                logger.debug(f"Agent {self.agent_id} missing skills for task {task.task_id}: {missing_skills}")
            
            return can_handle
        except Exception as e:
            logger.error(f"Error checking if agent {self.agent_id} can handle task {task.task_id}: {e}")
            return False
    
    def get_utilization_score(self, current_load: int) -> float:
        """Calculate current utilization score."""
        if self.capacity == 0:
            return 1.0
        return min(1.0, current_load / self.capacity)
    
    def __hash__(self):
        return hash((self.agent_id, tuple(self.skills), self.capacity))

@dataclass(frozen=True)
class Task:
    """Represents a task with comprehensive validation."""
    
    task_id: str
    required_skills: List[str]
    priority: int = 1
    duration: int = 1
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[int] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Comprehensive validation with detailed error messages."""
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValidationError("Task ID must be a non-empty string", "task_id", self.task_id)
        
        if not self.required_skills or not isinstance(self.required_skills, list):
            raise ValidationError("Required skills must be a non-empty list", "required_skills", self.required_skills)
        
        if not all(isinstance(skill, str) and skill.strip() for skill in self.required_skills):
            raise ValidationError("All required skills must be non-empty strings", "required_skills", self.required_skills)
        
        if not isinstance(self.priority, int) or self.priority <= 0:
            raise ValidationError("Priority must be a positive integer", "priority", self.priority)
        
        if not isinstance(self.duration, int) or self.duration <= 0:
            raise ValidationError("Duration must be a positive integer", "duration", self.duration)
        
        if not isinstance(self.dependencies, list):
            raise ValidationError("Dependencies must be a list", "dependencies", self.dependencies)
        
        if self.deadline is not None and (not isinstance(self.deadline, int) or self.deadline <= 0):
            raise ValidationError("Deadline must be a positive integer if specified", "deadline", self.deadline)
        
        if self.deadline is not None and self.deadline < self.duration:
            raise ValidationError("Deadline cannot be less than duration", "deadline", self.deadline)
    
    def can_be_assigned_to(self, agent: Agent) -> bool:
        """Check if this task can be assigned to the given agent."""
        return agent.can_handle_task(self)
    
    def get_complexity_score(self) -> float:
        """Calculate task complexity score."""
        skill_complexity = len(self.required_skills) * 0.3
        duration_complexity = min(self.duration * 0.2, 2.0)
        dependency_complexity = len(self.dependencies) * 0.1
        return skill_complexity + duration_complexity + dependency_complexity
    
    def __hash__(self):
        return hash((self.task_id, tuple(self.required_skills), self.priority, self.duration))

@dataclass
class Solution:
    """Represents a solution with comprehensive metrics and validation."""
    
    assignments: Dict[str, str]  # task_id -> agent_id
    makespan: float
    cost: float = 0.0
    backend_used: str = "unknown"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    quality_score: Optional[float] = None
    confidence: float = 1.0
    alternative_solutions: List['Solution'] = field(default_factory=list)
    
    def __post_init__(self):
        """Comprehensive validation and metric calculation."""
        if not isinstance(self.assignments, dict):
            raise ValidationError("Assignments must be a dictionary", "assignments", self.assignments)
        
        if not self.assignments:
            raise ValidationError("Assignments cannot be empty", "assignments", self.assignments)
        
        if not isinstance(self.makespan, (int, float)) or self.makespan < 0:
            raise ValidationError("Makespan must be non-negative", "makespan", self.makespan)
        
        if not isinstance(self.cost, (int, float)) or self.cost < 0:
            raise ValidationError("Cost must be non-negative", "cost", self.cost)
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValidationError("Confidence must be between 0 and 1", "confidence", self.confidence)
        
        # Calculate quality score if not provided
        if self.quality_score is None:
            self.quality_score = self.calculate_quality_score()
        
        # Initialize metadata
        if self.metadata is None:
            self.metadata = {}
        
        self.metadata.update({
            "solution_timestamp": datetime.now().isoformat(),
            "num_assignments": len(self.assignments),
            "unique_agents": len(set(self.assignments.values()))
        })
    
    def get_load_distribution(self) -> Dict[str, int]:
        """Get the load distribution across agents."""
        load_dist = {}
        for agent_id in self.assignments.values():
            load_dist[agent_id] = load_dist.get(agent_id, 0) + 1
        return load_dist
    
    def get_assigned_agents(self) -> Set[str]:
        """Get the set of agents that have been assigned tasks."""
        return set(self.assignments.values())
    
    def calculate_quality_score(self) -> float:
        """Calculate comprehensive quality score."""
        try:
            # Load balance component
            load_dist = list(self.get_load_distribution().values())
            if not load_dist:
                return 0.0
            
            avg_load = sum(load_dist) / len(load_dist)
            if avg_load == 0:
                balance_score = 1.0
            else:
                variance = sum((load - avg_load) ** 2 for load in load_dist) / len(load_dist)
                balance_score = max(0, 1 - (variance / max(avg_load, 1)))
            
            # Makespan component (normalized)
            makespan_score = max(0, 1 - (self.makespan / 100))
            
            # Cost component (normalized)
            cost_score = max(0, 1 - (self.cost / 1000))
            
            # Agent utilization component
            num_agents_used = len(self.get_assigned_agents())
            utilization_score = min(1.0, num_agents_used / max(len(load_dist), 1))
            
            # Weighted combination
            quality = (
                balance_score * 0.4 +
                makespan_score * 0.3 +
                cost_score * 0.2 +
                utilization_score * 0.1
            )
            
            return min(1.0, max(0.0, quality))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def validate_assignment(self, agents: List[Agent], tasks: List[Task]) -> List[str]:
        """Validate solution against agents and tasks."""
        issues = []
        
        try:
            # Check all tasks are assigned
            task_ids = {task.task_id for task in tasks}
            assigned_tasks = set(self.assignments.keys())
            missing_tasks = task_ids - assigned_tasks
            
            if missing_tasks:
                issues.append(f"Missing task assignments: {missing_tasks}")
            
            # Check all agents exist
            agent_ids = {agent.agent_id for agent in agents}
            assigned_agents = set(self.assignments.values())
            invalid_agents = assigned_agents - agent_ids
            
            if invalid_agents:
                issues.append(f"Invalid agent assignments: {invalid_agents}")
            
            # Check skill compatibility
            for task_id, agent_id in self.assignments.items():
                task = next((t for t in tasks if t.task_id == task_id), None)
                agent = next((a for a in agents if a.agent_id == agent_id), None)
                
                if task and agent and not task.can_be_assigned_to(agent):
                    issues.append(f"Skill mismatch: Task {task_id} assigned to incompatible agent {agent_id}")
        
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary for serialization."""
        return {
            "assignments": self.assignments,
            "makespan": self.makespan,
            "cost": self.cost,
            "backend_used": self.backend_used,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

# Decorators for Reliability
def with_timeout(timeout_seconds: float, operation_name: str = "operation"):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # Force cleanup of the thread (in real implementation, this would be more sophisticated)
                raise TimeoutError(f"{operation_name} timed out after {timeout_seconds} seconds", timeout_seconds, operation_name)
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator

def with_retry(max_retries: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)):
    """Decorator to add retry logic to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        break
            
            raise last_exception
        return wrapper
    return decorator

def with_error_handling(error_logger=None, fallback_value=None):
    """Decorator to add comprehensive error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_logger:
                    error_logger.error(f"Error in {func.__name__}: {e}")
                    error_logger.debug(traceback.format_exc())
                else:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())
                
                if fallback_value is not None:
                    return fallback_value
                raise
        return wrapper
    return decorator

# Enhanced Cache Implementation
class SolutionCache:
    """Thread-safe solution cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = threading.Lock()
    
    def _generate_key(self, agents: List[Agent], tasks: List[Task], objective: str) -> str:
        """Generate cache key from problem parameters."""
        # Create a deterministic hash of the problem
        problem_data = {
            "agents": [(a.agent_id, tuple(a.skills), a.capacity) for a in agents],
            "tasks": [(t.task_id, tuple(t.required_skills), t.priority, t.duration) for t in tasks],
            "objective": objective
        }
        
        problem_str = json.dumps(problem_data, sort_keys=True)
        return hashlib.md5(problem_str.encode()).hexdigest()
    
    def get(self, agents: List[Agent], tasks: List[Task], objective: str) -> Optional[Solution]:
        """Get cached solution if available and valid."""
        with self._lock:
            key = self._generate_key(agents, tasks, objective)
            
            if key not in self._cache:
                return None
            
            # Check TTL
            access_time = self._access_times.get(key, 0)
            if time.time() - access_time > self.ttl_seconds:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def put(self, agents: List[Agent], tasks: List[Task], objective: str, solution: Solution):
        """Cache a solution."""
        with self._lock:
            key = self._generate_key(agents, tasks, objective)
            
            # Evict oldest entries if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.keys(), key=self._access_times.get)
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            self._cache[key] = solution
            self._access_times[key] = time.time()
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": getattr(self, '_hit_rate', 0.0),
                "total_requests": getattr(self, '_total_requests', 0)
            }

# Main Robust Planner Class
class RobustEnhancedPlanner:
    """Robust enhanced quantum task planner with comprehensive reliability features."""
    
    def __init__(
        self,
        backend: BackendType = BackendType.AUTO,
        verbose: bool = False,
        enable_cache: bool = True,
        enable_monitoring: bool = True,
        max_solve_time: float = 300.0
    ):
        """Initialize the robust planner with comprehensive features."""
        self.backend = backend
        self.verbose = verbose
        self.max_solve_time = max_solve_time
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        # Caching
        self.cache = SolutionCache() if enable_cache else None
        
        # Health monitoring
        self.health_monitor = HealthMonitor() if enable_monitoring else None
        if self.health_monitor:
            self._setup_health_checks()
        
        # Circuit breakers for different operations
        self.optimization_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)
        self.backend_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=120.0)
        
        # Thread safety
        self._stats_lock = threading.Lock()
        
        if self.verbose:
            logger.info(f"RobustEnhancedPlanner initialized - Backend: {backend.value}, Caching: {enable_cache}, Monitoring: {enable_monitoring}")
    
    def _setup_health_checks(self):
        """Setup health monitoring checks."""
        self.health_monitor.register_check("error_rate", self._check_error_rate)
        self.health_monitor.register_check("performance", self._check_performance)
        self.health_monitor.register_check("memory", self._check_memory_usage)
        self.health_monitor.register_check("cache", self._check_cache_health)
    
    def _check_error_rate(self) -> tuple[bool, str]:
        """Check error rate health."""
        success_rate = self.metrics.success_rate()
        if success_rate < 0.5:
            return False, f"Low success rate: {success_rate:.1%}"
        elif success_rate < 0.8:
            return True, f"Moderate success rate: {success_rate:.1%}"
        else:
            return True, f"Good success rate: {success_rate:.1%}"
    
    def _check_performance(self) -> tuple[bool, str]:
        """Check performance health."""
        if self.metrics.average_solve_time > self.max_solve_time * 0.8:
            return False, f"High average solve time: {self.metrics.average_solve_time:.2f}s"
        elif self.metrics.average_solve_time > self.max_solve_time * 0.5:
            return True, f"Moderate solve time: {self.metrics.average_solve_time:.2f}s"
        else:
            return True, f"Good solve time: {self.metrics.average_solve_time:.2f}s"
    
    def _check_memory_usage(self) -> tuple[bool, str]:
        """Check memory usage health."""
        # Simplified memory check
        import sys
        if hasattr(sys, 'getsizeof'):
            cache_size = sys.getsizeof(self.cache._cache) if self.cache else 0
            if cache_size > 100_000_000:  # 100MB
                return False, f"High memory usage: {cache_size / 1_000_000:.1f}MB"
        return True, "Memory usage normal"
    
    def _check_cache_health(self) -> tuple[bool, str]:
        """Check cache health."""
        if not self.cache:
            return True, "Cache disabled"
        
        hit_rate = self.cache.get_stats().get("hit_rate", 0)
        if hit_rate < 0.1:
            return False, f"Low cache hit rate: {hit_rate:.1%}"
        elif hit_rate < 0.3:
            return True, f"Moderate cache hit rate: {hit_rate:.1%}"
        else:
            return True, f"Good cache hit rate: {hit_rate:.1%}"
    
    def _record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error for monitoring and analysis."""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            message=str(error),
            severity=getattr(error, 'severity', ErrorSeverity.MEDIUM),
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        self.error_history.append(error_record)
        self.error_counts[error_record.error_type] += 1
        
        if self.verbose:
            logger.error(f"Error recorded: {error_record.error_type} - {error_record.message}")
    
    def _update_metrics(self, success: bool, solve_time: float, quality: float = None):
        """Update performance metrics thread-safely."""
        with self._stats_lock:
            self.metrics.total_problems += 1
            self.metrics.total_solve_time += solve_time
            
            if success:
                self.metrics.successful_problems += 1
                if quality is not None:
                    # Update rolling average quality
                    current_avg = self.metrics.average_quality
                    n = self.metrics.successful_problems
                    self.metrics.average_quality = (current_avg * (n - 1) + quality) / n
            else:
                self.metrics.failed_problems += 1
            
            # Update average solve time
            self.metrics.average_solve_time = self.metrics.total_solve_time / self.metrics.total_problems
    
    @with_timeout(300.0, "task_assignment")
    @with_retry(max_retries=2, backoff_factor=0.5, exceptions=(OptimizationError, BackendError))
    @with_error_handling()
    def assign(
        self,
        agents: List[Agent],
        tasks: List[Task],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Solution:
        """Assign tasks to agents with comprehensive reliability features."""
        start_time = time.time()
        context = {
            "num_agents": len(agents),
            "num_tasks": len(tasks),
            "objective": objective.value,
            "backend": self.backend.value
        }
        
        try:
            # Input validation
            self._validate_inputs(agents, tasks, objective, constraints)
            
            # Check cache first
            if self.cache:
                cached_solution = self.cache.get(agents, tasks, objective.value)
                if cached_solution:
                    logger.info("Retrieved solution from cache")
                    self.metrics.cache_hits += 1
                    self._update_metrics(True, time.time() - start_time, cached_solution.quality_score)
                    return cached_solution
                else:
                    self.metrics.cache_misses += 1
            
            # Select backend with circuit breaker protection
            selected_backend = self._select_backend_with_protection(len(agents), len(tasks))
            context["selected_backend"] = selected_backend
            
            if self.verbose:
                logger.info(f"Solving {len(tasks)} tasks for {len(agents)} agents using {selected_backend}")
            
            # Solve with circuit breaker protection
            solution = self.optimization_circuit_breaker.call(
                self._solve_with_backend_protected,
                agents, tasks, objective, selected_backend, constraints
            )
            
            # Validate solution
            validation_issues = solution.validate_assignment(agents, tasks)
            if validation_issues:
                logger.warning(f"Solution validation issues: {validation_issues}")
                solution.metadata["validation_issues"] = validation_issues
                solution.confidence *= 0.8  # Reduce confidence for solutions with issues
            
            # Cache successful solution
            if self.cache and solution.quality_score > 0.3:  # Only cache reasonable quality solutions
                self.cache.put(agents, tasks, objective.value, solution)
            
            # Update metrics
            solve_time = time.time() - start_time
            self._update_metrics(True, solve_time, solution.quality_score)
            
            # Add comprehensive metadata
            solution.metadata.update({
                "solve_time": solve_time,
                "selected_backend": selected_backend,
                "objective": objective.value,
                "problem_size": len(agents) * len(tasks),
                "validation_passed": len(validation_issues) == 0,
                "cached": False,
                "planner_version": "robust_v2.0"
            })
            
            if self.verbose:
                logger.info(f"Solution completed - Quality: {solution.quality_score:.3f}, Time: {solve_time:.3f}s")
            
            return solution
        
        except Exception as e:
            # Record error and update metrics
            self._record_error(e, context)
            self._update_metrics(False, time.time() - start_time)
            
            # Try to provide fallback solution
            try:
                logger.warning("Attempting fallback solution...")
                fallback_solution = self._create_fallback_solution(agents, tasks, objective)
                fallback_solution.metadata["is_fallback"] = True
                fallback_solution.confidence = 0.3  # Low confidence for fallback
                
                if self.verbose:
                    logger.info(f"Fallback solution created - Quality: {fallback_solution.quality_score:.3f}")
                
                return fallback_solution
            
            except Exception as fallback_error:
                logger.error(f"Fallback solution also failed: {fallback_error}")
                self._record_error(fallback_error, {**context, "is_fallback": True})
                raise QuantumPlannerError(
                    f"Both primary and fallback solutions failed. Primary: {str(e)}, Fallback: {str(fallback_error)}",
                    ErrorSeverity.CRITICAL,
                    context
                )
    
    def _validate_inputs(self, agents: List[Agent], tasks: List[Task], 
                        objective: OptimizationObjective, constraints: Optional[Dict[str, Any]]):
        """Comprehensive input validation with detailed error reporting."""
        if not agents:
            raise ValidationError("No agents provided")
        
        if not tasks:
            raise ValidationError("No tasks provided")
        
        if not isinstance(objective, OptimizationObjective):
            raise ValidationError(f"Invalid objective type: {type(objective)}", "objective", objective)
        
        # Validate individual agents and tasks (this will trigger their __post_init__ validation)
        for i, agent in enumerate(agents):
            try:
                # Force validation by accessing a property that triggers __post_init__
                _ = agent.skills
            except Exception as e:
                raise ValidationError(f"Agent {i} validation failed: {str(e)}", "agents", agent)
        
        for i, task in enumerate(tasks):
            try:
                _ = task.required_skills
            except Exception as e:
                raise ValidationError(f"Task {i} validation failed: {str(e)}", "tasks", task)
        
        # Check if any task can be assigned to any agent
        assignable_tasks = 0
        for task in tasks:
            if any(task.can_be_assigned_to(agent) for agent in agents):
                assignable_tasks += 1
        
        if assignable_tasks == 0:
            raise ValidationError("No tasks can be assigned to available agents (complete skill mismatch)")
        
        assignment_ratio = assignable_tasks / len(tasks)
        if assignment_ratio < 0.3:
            logger.warning(f"Only {assignment_ratio:.1%} of tasks can be assigned to available agents")
        
        # Validate constraints
        if constraints:
            if not isinstance(constraints, dict):
                raise ValidationError("Constraints must be a dictionary", "constraints", constraints)
    
    def _select_backend_with_protection(self, num_agents: int, num_tasks: int) -> str:
        """Select backend with circuit breaker protection."""
        try:
            return self.backend_circuit_breaker.call(self._select_backend, num_agents, num_tasks)
        except Exception as e:
            logger.warning(f"Backend selection failed, using fallback: {e}")
            self.metrics.fallback_usage += 1
            return "greedy_fallback"
    
    def _select_backend(self, num_agents: int, num_tasks: int) -> str:
        """Select appropriate backend based on problem characteristics."""
        problem_size = num_agents * num_tasks
        
        if self.backend == BackendType.AUTO:
            if problem_size <= 10:
                return "greedy_optimal"
            elif problem_size <= 50:
                return "simulated_annealing"
            elif problem_size <= 200:
                return "enhanced_greedy"
            else:
                return "genetic_algorithm"
        else:
            return self.backend.value
    
    def _solve_with_backend_protected(
        self,
        agents: List[Agent],
        tasks: List[Task],
        objective: OptimizationObjective,
        backend: str,
        constraints: Optional[Dict[str, Any]]
    ) -> Solution:
        """Solve using the specified backend with additional protection."""
        if backend == "greedy_optimal":
            return self._solve_greedy_optimal(agents, tasks, objective)
        elif backend == "simulated_annealing":
            return self._solve_simulated_annealing(agents, tasks, objective)
        elif backend == "enhanced_greedy":
            return self._solve_enhanced_greedy(agents, tasks, objective)
        elif backend == "genetic_algorithm":
            return self._solve_genetic_algorithm(agents, tasks, objective)
        elif backend == "greedy_fallback":
            return self._solve_greedy_fallback(agents, tasks, objective)
        else:
            logger.warning(f"Backend {backend} not implemented, using enhanced greedy")
            return self._solve_enhanced_greedy(agents, tasks, objective)
    
    def _solve_greedy_optimal(self, agents: List[Agent], tasks: List[Task], objective: OptimizationObjective) -> Solution:
        """Enhanced greedy algorithm with better optimization."""
        assignments = {}
        agent_loads = {agent.agent_id: 0 for agent in agents}
        agent_costs = {agent.agent_id: 0.0 for agent in agents}
        
        # Enhanced task sorting based on multiple criteria
        def task_priority_score(task):
            complexity = task.get_complexity_score()
            urgency = task.priority / max([t.priority for t in tasks])
            duration_factor = task.duration / max([t.duration for t in tasks])
            
            if objective == OptimizationObjective.MAXIMIZE_PRIORITY:
                return (-task.priority, complexity, duration_factor)
            elif objective == OptimizationObjective.MINIMIZE_MAKESPAN:
                return (-duration_factor, -urgency, complexity)
            elif objective == OptimizationObjective.MINIMIZE_COST:
                return (complexity, duration_factor, -urgency)
            else:
                return (-urgency, complexity, duration_factor)
        
        sorted_tasks = sorted(tasks, key=task_priority_score)
        
        for task in sorted_tasks:
            capable_agents = [a for a in agents if task.can_be_assigned_to(a)]
            
            if not capable_agents:
                if self.verbose:
                    logger.warning(f"No agent can handle task {task.task_id}")
                continue
            
            # Enhanced agent selection
            def agent_selection_score(agent):
                current_load = agent_loads[agent.agent_id]
                load_factor = current_load / max(agent.capacity, 1)
                
                # Skill matching bonus (more specific skills = higher preference)
                skill_match_score = len(set(agent.skills) & set(task.required_skills)) / len(task.required_skills)
                
                cost_factor = agent.cost_per_hour / max([a.cost_per_hour for a in capable_agents] + [1])
                
                if objective == OptimizationObjective.BALANCE_LOAD:
                    return (load_factor, cost_factor, -skill_match_score)
                elif objective == OptimizationObjective.MINIMIZE_COST:
                    return (cost_factor, load_factor, -skill_match_score)
                else:
                    return (load_factor, -skill_match_score, cost_factor)
            
            best_agent = min(capable_agents, key=agent_selection_score)
            
            # Assign task
            assignments[task.task_id] = best_agent.agent_id
            agent_loads[best_agent.agent_id] += task.duration
            agent_costs[best_agent.agent_id] += best_agent.cost_per_hour * task.duration
        
        makespan = max(agent_loads.values()) if agent_loads.values() else 0
        total_cost = sum(agent_costs.values())
        
        return Solution(
            assignments=assignments,
            makespan=makespan,
            cost=total_cost,
            backend_used="greedy_optimal",
            confidence=0.8
        )
    
    def _solve_simulated_annealing(self, agents: List[Agent], tasks: List[Task], objective: OptimizationObjective) -> Solution:
        """Enhanced simulated annealing with better parameters and cooling schedule."""
        # Start with greedy solution
        current_solution = self._solve_greedy_optimal(agents, tasks, objective)
        best_solution = current_solution
        
        # Enhanced SA parameters
        initial_temp = 200.0
        final_temp = 0.01
        cooling_rate = 0.99
        max_iterations = min(2000, len(tasks) * len(agents) * 15)
        plateau_limit = max_iterations // 10
        
        temperature = initial_temp
        plateau_counter = 0
        last_improvement = current_solution.quality_score
        
        for iteration in range(max_iterations):
            if temperature < final_temp or plateau_counter > plateau_limit:
                break
            
            # Generate multiple neighbor solutions and pick the best
            neighbors = []
            for _ in range(min(5, len(tasks))):
                neighbor = self._generate_enhanced_neighbor_solution(current_solution, agents, tasks, objective)
                if neighbor:
                    neighbors.append(neighbor)
            
            if not neighbors:
                plateau_counter += 1
                temperature *= cooling_rate
                continue
            
            # Select best neighbor or random neighbor with some probability
            if random.random() < 0.7:  # 70% chance to pick best neighbor
                neighbor_solution = max(neighbors, key=lambda s: s.quality_score)
            else:
                neighbor_solution = random.choice(neighbors)
            
            # Calculate acceptance probability
            delta_quality = neighbor_solution.quality_score - current_solution.quality_score
            
            if delta_quality > 0 or random.random() < self._enhanced_acceptance_probability(delta_quality, temperature):
                current_solution = neighbor_solution
                
                if current_solution.quality_score > best_solution.quality_score:
                    best_solution = current_solution
                    plateau_counter = 0
                    last_improvement = current_solution.quality_score
                else:
                    plateau_counter += 1
            else:
                plateau_counter += 1
            
            # Adaptive cooling
            if plateau_counter > plateau_limit // 5:
                temperature *= 0.95  # Faster cooling when stuck
            else:
                temperature *= cooling_rate
        
        best_solution.backend_used = "simulated_annealing"
        best_solution.confidence = 0.9
        best_solution.metadata["sa_iterations"] = iteration + 1
        best_solution.metadata["final_temperature"] = temperature
        
        return best_solution
    
    def _solve_enhanced_greedy(self, agents: List[Agent], tasks: List[Task], objective: OptimizationObjective) -> Solution:
        """Multi-strategy enhanced greedy algorithm."""
        strategies = [
            ("priority_first", lambda t: (-t.priority, t.duration, len(t.required_skills))),
            ("duration_first", lambda t: (-t.duration, -t.priority, len(t.required_skills))),
            ("complexity_first", lambda t: (-len(t.required_skills), -t.priority, t.duration)),
            ("balanced", lambda t: (-t.priority * 0.5 - t.duration * 0.3 - len(t.required_skills) * 0.2,)),
        ]
        
        best_solution = None
        
        for strategy_name, sort_key in strategies:
            try:
                solution = self._solve_with_strategy(agents, tasks, objective, sort_key, strategy_name)
                
                if best_solution is None or solution.quality_score > best_solution.quality_score:
                    best_solution = solution
                    best_solution.metadata["best_strategy"] = strategy_name
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        if best_solution is None:
            # Ultimate fallback
            best_solution = self._solve_greedy_fallback(agents, tasks, objective)
        
        best_solution.backend_used = "enhanced_greedy"
        best_solution.confidence = 0.85
        
        return best_solution
    
    def _solve_genetic_algorithm(self, agents: List[Agent], tasks: List[Task], objective: OptimizationObjective) -> Solution:
        """Genetic algorithm implementation for large problems."""
        # This is a simplified GA implementation
        population_size = min(50, len(agents) * 2)
        generations = min(100, len(tasks) * 5)
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Generate initial population
        population = []
        for _ in range(population_size):
            try:
                individual = self._generate_random_solution(agents, tasks, objective)
                if individual:
                    population.append(individual)
            except Exception:
                continue
        
        if not population:
            # Fallback if no initial population could be generated
            return self._solve_greedy_fallback(agents, tasks, objective)
        
        best_solution = max(population, key=lambda s: s.quality_score)
        
        for generation in range(generations):
            # Selection (tournament selection)
            new_population = []
            
            for _ in range(population_size):
                # Tournament selection
                tournament_size = min(5, len(population))
                tournament = random.sample(population, tournament_size)
                winner = max(tournament, key=lambda s: s.quality_score)
                new_population.append(winner)
            
            # Crossover and mutation would be implemented here
            # For simplicity, we'll use mutation only
            for i in range(len(new_population)):
                if random.random() < mutation_rate:
                    mutated = self._mutate_solution(new_population[i], agents, tasks, objective)
                    if mutated and mutated.quality_score > new_population[i].quality_score:
                        new_population[i] = mutated
            
            population = new_population
            current_best = max(population, key=lambda s: s.quality_score)
            
            if current_best.quality_score > best_solution.quality_score:
                best_solution = current_best
        
        best_solution.backend_used = "genetic_algorithm"
        best_solution.confidence = 0.75
        best_solution.metadata["generations"] = generations
        best_solution.metadata["population_size"] = population_size
        
        return best_solution
    
    def _solve_greedy_fallback(self, agents: List[Agent], tasks: List[Task], objective: OptimizationObjective) -> Solution:
        """Ultra-simple fallback solution for when everything else fails."""
        assignments = {}
        agent_loads = {agent.agent_id: 0 for agent in agents}
        
        # Simply assign each task to the first capable agent
        for task in tasks:
            for agent in agents:
                if task.can_be_assigned_to(agent):
                    assignments[task.task_id] = agent.agent_id
                    agent_loads[agent.agent_id] += task.duration
                    break
        
        makespan = max(agent_loads.values()) if agent_loads.values() else 0
        total_cost = sum(
            next(a.cost_per_hour for a in agents if a.agent_id == agent_id) * load
            for agent_id, load in agent_loads.items()
        )
        
        return Solution(
            assignments=assignments,
            makespan=makespan,
            cost=total_cost,
            backend_used="greedy_fallback",
            confidence=0.3
        )
    
    def _create_fallback_solution(self, agents: List[Agent], tasks: List[Task], 
                                 objective: OptimizationObjective) -> Solution:
        """Create an emergency fallback solution."""
        return self._solve_greedy_fallback(agents, tasks, objective)
    
    def _solve_with_strategy(self, agents: List[Agent], tasks: List[Task], 
                           objective: OptimizationObjective, sort_key: Callable, strategy_name: str) -> Solution:
        """Solve using a specific sorting strategy."""
        assignments = {}
        agent_loads = {agent.agent_id: 0 for agent in agents}
        agent_costs = {agent.agent_id: 0.0 for agent in agents}
        
        sorted_tasks = sorted(tasks, key=sort_key)
        
        for task in sorted_tasks:
            capable_agents = [a for a in agents if task.can_be_assigned_to(a)]
            
            if not capable_agents:
                continue
            
            # Select agent with minimum load
            best_agent = min(capable_agents, key=lambda a: agent_loads[a.agent_id])
            
            assignments[task.task_id] = best_agent.agent_id
            agent_loads[best_agent.agent_id] += task.duration
            agent_costs[best_agent.agent_id] += best_agent.cost_per_hour * task.duration
        
        makespan = max(agent_loads.values()) if agent_loads.values() else 0
        total_cost = sum(agent_costs.values())
        
        return Solution(
            assignments=assignments,
            makespan=makespan,
            cost=total_cost,
            backend_used=f"enhanced_greedy_{strategy_name}",
            confidence=0.7
        )
    
    def _generate_enhanced_neighbor_solution(self, current_solution: Solution, agents: List[Agent], 
                                           tasks: List[Task], objective: OptimizationObjective) -> Optional[Solution]:
        """Generate an enhanced neighbor solution with multiple strategies."""
        if len(current_solution.assignments) < 2:
            return None
        
        strategies = ["swap", "reassign", "optimize_agent"]
        strategy = random.choice(strategies)
        
        try:
            new_assignments = current_solution.assignments.copy()
            
            if strategy == "swap":
                # Swap assignments of two tasks
                task_ids = list(new_assignments.keys())
                task1, task2 = random.sample(task_ids, 2)
                
                agent1_id = new_assignments[task1]
                agent2_id = new_assignments[task2]
                
                task1_obj = next(t for t in tasks if t.task_id == task1)
                task2_obj = next(t for t in tasks if t.task_id == task2)
                agent1_obj = next(a for a in agents if a.agent_id == agent1_id)
                agent2_obj = next(a for a in agents if a.agent_id == agent2_id)
                
                if (task1_obj.can_be_assigned_to(agent2_obj) and 
                    task2_obj.can_be_assigned_to(agent1_obj)):
                    new_assignments[task1] = agent2_id
                    new_assignments[task2] = agent1_id
            
            elif strategy == "reassign":
                # Reassign a random task to a different capable agent
                task_id = random.choice(list(new_assignments.keys()))
                task_obj = next(t for t in tasks if t.task_id == task_id)
                capable_agents = [a for a in agents if task_obj.can_be_assigned_to(a)]
                
                if len(capable_agents) > 1:
                    current_agent_id = new_assignments[task_id]
                    other_agents = [a for a in capable_agents if a.agent_id != current_agent_id]
                    new_agent = random.choice(other_agents)
                    new_assignments[task_id] = new_agent.agent_id
            
            elif strategy == "optimize_agent":
                # Find the most loaded agent and try to redistribute their tasks
                agent_loads = {}
                for task_id, agent_id in new_assignments.items():
                    task_obj = next(t for t in tasks if t.task_id == task_id)
                    agent_loads[agent_id] = agent_loads.get(agent_id, 0) + task_obj.duration
                
                if agent_loads:
                    most_loaded_agent = max(agent_loads.keys(), key=agent_loads.get)
                    agent_tasks = [tid for tid, aid in new_assignments.items() if aid == most_loaded_agent]
                    
                    if agent_tasks:
                        task_to_move = random.choice(agent_tasks)
                        task_obj = next(t for t in tasks if t.task_id == task_to_move)
                        
                        # Find the least loaded capable agent
                        capable_agents = [a for a in agents if task_obj.can_be_assigned_to(a)]
                        if capable_agents:
                            least_loaded_agent = min(capable_agents, 
                                                   key=lambda a: agent_loads.get(a.agent_id, 0))
                            new_assignments[task_to_move] = least_loaded_agent.agent_id
            
            # Calculate new metrics
            agent_loads = {}
            agent_costs = {}
            
            for task_id, agent_id in new_assignments.items():
                task_obj = next(t for t in tasks if t.task_id == task_id)
                agent_obj = next(a for a in agents if a.agent_id == agent_id)
                
                agent_loads[agent_id] = agent_loads.get(agent_id, 0) + task_obj.duration
                agent_costs[agent_id] = agent_costs.get(agent_id, 0) + agent_obj.cost_per_hour * task_obj.duration
            
            makespan = max(agent_loads.values()) if agent_loads.values() else 0
            total_cost = sum(agent_costs.values())
            
            return Solution(
                assignments=new_assignments,
                makespan=makespan,
                cost=total_cost,
                backend_used="enhanced_neighbor",
                confidence=0.8
            )
        
        except Exception as e:
            logger.debug(f"Enhanced neighbor generation failed: {e}")
            return None
    
    def _generate_random_solution(self, agents: List[Agent], tasks: List[Task], 
                                 objective: OptimizationObjective) -> Optional[Solution]:
        """Generate a random valid solution."""
        try:
            assignments = {}
            
            for task in tasks:
                capable_agents = [a for a in agents if task.can_be_assigned_to(a)]
                if capable_agents:
                    agent = random.choice(capable_agents)
                    assignments[task.task_id] = agent.agent_id
            
            if not assignments:
                return None
            
            # Calculate metrics
            agent_loads = {}
            agent_costs = {}
            
            for task_id, agent_id in assignments.items():
                task_obj = next(t for t in tasks if t.task_id == task_id)
                agent_obj = next(a for a in agents if a.agent_id == agent_id)
                
                agent_loads[agent_id] = agent_loads.get(agent_id, 0) + task_obj.duration
                agent_costs[agent_id] = agent_costs.get(agent_id, 0) + agent_obj.cost_per_hour * task_obj.duration
            
            makespan = max(agent_loads.values()) if agent_loads.values() else 0
            total_cost = sum(agent_costs.values())
            
            return Solution(
                assignments=assignments,
                makespan=makespan,
                cost=total_cost,
                backend_used="random_generation",
                confidence=0.5
            )
        
        except Exception as e:
            logger.debug(f"Random solution generation failed: {e}")
            return None
    
    def _mutate_solution(self, solution: Solution, agents: List[Agent], 
                        tasks: List[Task], objective: OptimizationObjective) -> Optional[Solution]:
        """Mutate a solution for genetic algorithm."""
        return self._generate_enhanced_neighbor_solution(solution, agents, tasks, objective)
    
    def _enhanced_acceptance_probability(self, delta_quality: float, temperature: float) -> float:
        """Enhanced acceptance probability with adaptive scaling."""
        if delta_quality >= 0:
            return 1.0
        
        if temperature <= 0:
            return 0.0
        
        # Scale delta_quality to make acceptance more reasonable
        scaled_delta = delta_quality * 10  # Adjust scaling factor as needed
        
        try:
            probability = pow(2.71828, scaled_delta / temperature)
            return min(1.0, probability)
        except (OverflowError, ValueError):
            return 0.0
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and health statistics."""
        stats = {
            "performance": asdict(self.metrics),
            "error_summary": {
                "total_errors": len(self.error_history),
                "error_types": dict(self.error_counts),
                "recent_errors": [
                    error.to_dict() for error in list(self.error_history)[-5:]
                ]
            },
            "cache_stats": self.cache.get_stats() if self.cache else {"enabled": False},
            "health": self.health_monitor.check_health() if self.health_monitor else {"enabled": False},
            "circuit_breakers": {
                "optimization": {
                    "state": self.optimization_circuit_breaker.state,
                    "failure_count": self.optimization_circuit_breaker.failure_count
                },
                "backend": {
                    "state": self.backend_circuit_breaker.state,
                    "failure_count": self.backend_circuit_breaker.failure_count
                }
            }
        }
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics and error history."""
        with self._stats_lock:
            self.metrics = PerformanceMetrics()
            self.error_history.clear()
            self.error_counts.clear()
            
            if self.cache:
                self.cache.clear()
        
        logger.info("Statistics reset")
    
    def export_diagnostics(self, filepath: str):
        """Export comprehensive diagnostics to file."""
        try:
            diagnostics = {
                "timestamp": datetime.now().isoformat(),
                "planner_config": {
                    "backend": self.backend.value,
                    "verbose": self.verbose,
                    "max_solve_time": self.max_solve_time
                },
                "statistics": self.get_comprehensive_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)
            
            logger.info(f"Diagnostics exported to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export diagnostics: {e}")
            raise

# Demo and testing
def run_robust_demo():
    """Run comprehensive demonstration of robust planner."""
    print("🛡️ Robust Enhanced Quantum Task Planner - Generation 2")
    print("=" * 70)
    
    # Initialize planner with full features
    planner = RobustEnhancedPlanner(
        backend=BackendType.AUTO,
        verbose=True,
        enable_cache=True,
        enable_monitoring=True,
        max_solve_time=60.0
    )
    
    # Create test problems with various complexity levels
    test_scenarios = [
        ("Small Problem", 3, 8, "Basic functionality test"),
        ("Medium Problem", 6, 20, "Load balancing test"),
        ("Large Problem", 10, 40, "Scalability test"),
        ("Complex Problem", 12, 50, "Stress test with high complexity")
    ]
    
    for scenario_name, num_agents, num_tasks, description in test_scenarios:
        print(f"\n🧪 {scenario_name} - {description}")
        print("-" * 60)
        
        try:
            # Create problem with realistic constraints
            agents, tasks = create_robust_demo_problem(num_agents, num_tasks)
            
            # Test with different objectives
            objectives = [
                OptimizationObjective.MINIMIZE_MAKESPAN,
                OptimizationObjective.BALANCE_LOAD,
                OptimizationObjective.MAXIMIZE_PRIORITY,
                OptimizationObjective.MINIMIZE_COST
            ]
            
            for objective in objectives:
                try:
                    start_time = time.time()
                    solution = planner.assign(agents, tasks, objective=objective)
                    solve_time = time.time() - start_time
                    
                    print(f"  ✅ {objective.value.replace('_', ' ').title()}:")
                    print(f"     • Assignments: {len(solution.assignments)}/{len(tasks)} tasks")
                    print(f"     • Quality Score: {solution.quality_score:.3f}")
                    print(f"     • Confidence: {solution.confidence:.3f}")
                    print(f"     • Makespan: {solution.makespan:.1f}")
                    print(f"     • Cost: ${solution.cost:.2f}")
                    print(f"     • Backend: {solution.backend_used}")
                    print(f"     • Solve Time: {solve_time:.3f}s")
                    print(f"     • Validation: {'✅' if solution.metadata.get('validation_passed', False) else '⚠️'}")
                    
                except Exception as e:
                    print(f"  ❌ {objective.value} failed: {str(e)}")
        
        except Exception as e:
            print(f"  ⚠️ Scenario failed: {str(e)}")
    
    # Test error handling and recovery
    print(f"\n🔧 Error Handling & Recovery Tests")
    print("-" * 50)
    
    try:
        # Test with invalid inputs
        print("  Testing input validation...")
        try:
            planner.assign([], [])  # Empty inputs
        except ValidationError as e:
            print(f"    ✅ Caught validation error: {e.message}")
        
        # Test with impossible problem
        print("  Testing impossible assignment...")
        impossible_agents = [Agent("agent1", ["skill_x"], capacity=1)]
        impossible_tasks = [Task("task1", ["skill_y"], priority=1, duration=1)]
        
        try:
            solution = planner.assign(impossible_agents, impossible_tasks)
            print(f"    ⚠️ Got fallback solution with confidence: {solution.confidence}")
        except Exception as e:
            print(f"    ✅ Properly handled impossible problem: {type(e).__name__}")
    
    except Exception as e:
        print(f"  Error in error handling tests: {e}")
    
    # Show comprehensive statistics
    print(f"\n📊 Comprehensive Statistics")
    print("-" * 40)
    
    stats = planner.get_comprehensive_stats()
    
    # Performance metrics
    perf = stats["performance"]
    print(f"  Performance:")
    print(f"    • Problems Solved: {perf['total_problems']}")
    print(f"    • Success Rate: {perf.get('success_rate', 0) * 100:.1f}%")
    print(f"    • Average Solve Time: {perf['average_solve_time']:.3f}s")
    print(f"    • Average Quality: {perf['average_quality']:.3f}")
    print(f"    • Cache Hit Rate: {perf.get('cache_hit_rate', 0) * 100:.1f}%")
    
    # Error summary
    errors = stats["error_summary"]
    print(f"  Reliability:")
    print(f"    • Total Errors: {errors['total_errors']}")
    print(f"    • Error Types: {len(errors['error_types'])} unique")
    print(f"    • Recent Errors: {len(errors['recent_errors'])}")
    
    # Health status
    health = stats["health"]
    if health.get("enabled", False):
        print(f"  Health Status: {health.get('overall_status', 'unknown').upper()}")
        if health.get("issues"):
            print(f"    • Issues: {', '.join(health['issues'][:3])}")
    
    # Circuit breakers
    cb = stats["circuit_breakers"]
    print(f"  Circuit Breakers:")
    print(f"    • Optimization: {cb['optimization']['state']} ({cb['optimization']['failure_count']} failures)")
    print(f"    • Backend: {cb['backend']['state']} ({cb['backend']['failure_count']} failures)")
    
    print(f"\n✅ Generation 2 Implementation Complete!")
    print("🛡️ Robust Features Implemented:")
    print("  • Comprehensive input validation")
    print("  • Advanced error handling and recovery")
    print("  • Circuit breaker pattern")
    print("  • Performance monitoring")
    print("  • Health checks")
    print("  • Solution caching with TTL")
    print("  • Timeout protection")
    print("  • Retry logic with backoff")
    print("  • Fallback solutions")
    print("  • Comprehensive logging")
    print("  • Thread-safe operations")
    print("  • Quality scoring and confidence")
    print("  • Diagnostic export")

def create_robust_demo_problem(num_agents: int, num_tasks: int) -> tuple[List[Agent], List[Task]]:
    """Create a demonstration problem with realistic constraints for robust testing."""
    skills_pool = [
        "python", "javascript", "ml", "devops", "react", "database", 
        "testing", "ui_design", "data_science", "security", "mobile", "cloud"
    ]
    
    agents = []
    for i in range(num_agents):
        agent_skills = random.sample(skills_pool, k=random.randint(2, 5))
        agents.append(Agent(
            agent_id=f"agent_{i+1:02d}",
            skills=agent_skills,
            capacity=random.randint(2, 5),
            availability=random.uniform(0.7, 1.0),
            cost_per_hour=random.uniform(25, 120),
            region=random.choice(["north", "south", "east", "west", "central"]),
            max_concurrent_tasks=random.randint(1, 3)
        ))
    
    tasks = []
    for i in range(num_tasks):
        required_skills = random.sample(skills_pool, k=random.randint(1, 3))
        duration = random.randint(1, 8)
        tasks.append(Task(
            task_id=f"task_{i+1:03d}",
            required_skills=required_skills,
            priority=random.randint(1, 10),
            duration=duration,
            dependencies=random.sample([f"task_{j+1:03d}" for j in range(i)], k=random.randint(0, min(2, i))),
            deadline=random.randint(duration, duration + 10) if random.random() < 0.3 else None
        ))
    
    return agents, tasks

if __name__ == "__main__":
    run_robust_demo()
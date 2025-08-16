"""
Scalable Quantum Performance Optimizer - Generation 3: MAKE IT SCALE

This module implements high-performance, scalable quantum optimization with:
- Concurrent processing and resource pooling
- Intelligent caching and memoization
- Load balancing and auto-scaling
- Performance monitoring and optimization
- Resource-aware execution planning
- Distributed computation capabilities

Author: Terragon Labs Quantum Research Team
Version: 3.0.0 (Production Scale)
"""

import asyncio
import logging
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import queue
import weakref
import gc
import psutil
import numpy as np
from pathlib import Path
import pickle
import json
import hashlib
from collections import defaultdict, deque
import warnings

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Using in-memory caching only.")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray not available. Distributed execution disabled.")

from ..validation import InputValidator, ValidationSeverity, ValidationResult
from ..security import SecurityManager, SecurityLevel, secure_operation, require_permission
from ..monitoring import PerformanceMonitor, MetricType

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for workload management."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    QUANTUM_BACKEND = "quantum_backend"
    NETWORK = "network"
    STORAGE = "storage"


class CacheStrategy(Enum):
    """Caching strategies for optimization."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"


@dataclass
class ResourceProfile:
    """Profile of available computational resources."""
    
    cpu_cores: int
    memory_gb: float
    quantum_backends: List[str]
    network_bandwidth_mbps: float
    storage_gb: float
    
    # Current utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    quantum_queue_depth: int = 0
    
    # Performance characteristics
    cpu_benchmark_score: float = 1.0
    memory_bandwidth_gbps: float = 10.0
    quantum_shot_rate: float = 100.0  # shots per second
    
    # Cost information
    cpu_cost_per_hour: float = 0.1
    memory_cost_per_gb_hour: float = 0.01
    quantum_cost_per_shot: float = 0.001
    
    def get_available_capacity(self, resource_type: ResourceType) -> float:
        """Get available capacity for a resource type."""
        if resource_type == ResourceType.CPU:
            return max(0, 1.0 - self.cpu_utilization)
        elif resource_type == ResourceType.MEMORY:
            return max(0, 1.0 - self.memory_utilization)
        elif resource_type == ResourceType.QUANTUM_BACKEND:
            return max(0, 1.0 - (self.quantum_queue_depth / 100.0))
        else:
            return 1.0


@dataclass
class OptimizationTask:
    """Optimization task with resource requirements and metadata."""
    
    task_id: str
    problem_matrix: np.ndarray
    constraints: Dict[str, Any]
    priority: int = 5
    
    # Resource requirements
    estimated_cpu_hours: float = 1.0
    estimated_memory_gb: float = 1.0
    estimated_quantum_shots: int = 1000
    
    # Execution preferences
    preferred_backends: List[str] = field(default_factory=list)
    max_execution_time: float = 3600.0
    quality_threshold: float = 0.8
    
    # State tracking
    status: str = "pending"
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Caching
    cache_key: str = field(init=False)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.cache_key = self._compute_cache_key()
    
    def _compute_cache_key(self) -> str:
        """Compute cache key for the optimization task."""
        key_data = f"{self.problem_matrix.tobytes()}_{str(self.constraints)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    
    # Current load
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Resource utilization
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    avg_quantum_utilization: float = 0.0
    
    # Performance metrics
    avg_task_duration: float = 0.0
    throughput_tasks_per_hour: float = 0.0
    success_rate: float = 1.0
    
    # Predictive metrics
    predicted_load_1h: float = 0.0
    predicted_load_24h: float = 0.0
    
    # Cost metrics
    cost_per_task: float = 0.0
    total_cost_per_hour: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'active_tasks': self.active_tasks,
            'queued_tasks': self.queued_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'avg_cpu_utilization': self.avg_cpu_utilization,
            'avg_memory_utilization': self.avg_memory_utilization,
            'avg_quantum_utilization': self.avg_quantum_utilization,
            'avg_task_duration': self.avg_task_duration,
            'throughput_tasks_per_hour': self.throughput_tasks_per_hour,
            'success_rate': self.success_rate,
            'predicted_load_1h': self.predicted_load_1h,
            'predicted_load_24h': self.predicted_load_24h,
            'cost_per_task': self.cost_per_task,
            'total_cost_per_hour': self.total_cost_per_hour
        }


class DistributedCache:
    """High-performance distributed cache for optimization results."""
    
    def __init__(self, 
                 cache_strategy: CacheStrategy = CacheStrategy.LRU,
                 max_memory_mb: int = 1024,
                 redis_url: Optional[str] = None):
        
        self.cache_strategy = cache_strategy
        self.max_memory_mb = max_memory_mb
        self.redis_url = redis_url
        
        # Local cache
        self.local_cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_access_counts: Dict[str, int] = defaultdict(int)
        self.cache_lock = threading.RLock()
        
        # Redis cache
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        
        logger.info(f"Distributed cache initialized: strategy={cache_strategy.value}")
    
    @contextmanager
    def performance_tracking(self, operation: str):
        """Track cache operation performance."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.debug(f"Cache {operation} took {duration:.3f}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with multi-level lookup."""
        with self.performance_tracking("get"):
            # Try local cache first
            with self.cache_lock:
                if key in self.local_cache:
                    self._update_access_stats(key)
                    self.cache_hits += 1
                    return self.local_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                try:
                    value_bytes = self.redis_client.get(key)
                    if value_bytes:
                        value = pickle.loads(value_bytes)
                        # Store in local cache for faster access
                        self._store_local(key, value)
                        self.cache_hits += 1
                        return value
                except Exception as e:
                    logger.warning(f"Redis get failed: {e}")
            
            self.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with multi-level storage."""
        with self.performance_tracking("set"):
            # Store in local cache
            self._store_local(key, value)
            
            # Store in Redis cache
            if self.redis_client:
                try:
                    value_bytes = pickle.dumps(value)
                    if ttl:
                        self.redis_client.setex(key, ttl, value_bytes)
                    else:
                        self.redis_client.set(key, value_bytes)
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
    
    def _store_local(self, key: str, value: Any) -> None:
        """Store value in local cache with eviction."""
        with self.cache_lock:
            # Check memory usage and evict if necessary
            self._evict_if_needed()
            
            self.local_cache[key] = value
            self._update_access_stats(key)
    
    def _update_access_stats(self, key: str) -> None:
        """Update access statistics for cache key."""
        current_time = time.time()
        self.cache_access_times[key] = current_time
        self.cache_access_counts[key] += 1
    
    def _evict_if_needed(self) -> None:
        """Evict cache entries based on strategy and memory usage."""
        # Estimate memory usage (simplified)
        estimated_memory_mb = len(self.local_cache) * 0.1  # Rough estimate
        
        if estimated_memory_mb > self.max_memory_mb:
            num_to_evict = max(1, len(self.local_cache) // 10)  # Evict 10%
            
            if self.cache_strategy == CacheStrategy.LRU:
                keys_to_evict = sorted(
                    self.cache_access_times.keys(),
                    key=lambda k: self.cache_access_times[k]
                )[:num_to_evict]
            
            elif self.cache_strategy == CacheStrategy.LFU:
                keys_to_evict = sorted(
                    self.cache_access_counts.keys(),
                    key=lambda k: self.cache_access_counts[k]
                )[:num_to_evict]
            
            else:  # FIFO
                keys_to_evict = list(self.local_cache.keys())[:num_to_evict]
            
            for key in keys_to_evict:
                self.local_cache.pop(key, None)
                self.cache_access_times.pop(key, None)
                self.cache_access_counts.pop(key, None)
                self.cache_evictions += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_evictions': self.cache_evictions,
            'hit_rate': hit_rate,
            'local_cache_size': len(self.local_cache),
            'redis_connected': self.redis_client is not None,
            'cache_strategy': self.cache_strategy.value
        }


class ResourcePool:
    """Intelligent resource pool for quantum optimization workloads."""
    
    def __init__(self, 
                 max_workers: int = None,
                 max_processes: int = None,
                 quantum_backends: List[str] = None):
        
        # CPU resources
        self.max_workers = max_workers or mp.cpu_count()
        self.max_processes = max_processes or max(2, mp.cpu_count() // 2)
        
        # Quantum backends
        self.quantum_backends = quantum_backends or ["simulator"]
        
        # Resource pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Quantum backend queue management
        self.quantum_queues = {
            backend: queue.Queue(maxsize=100)
            for backend in self.quantum_backends
        }
        
        # Resource tracking
        self.active_threads = 0
        self.active_processes = 0
        self.quantum_active_jobs = defaultdict(int)
        
        # Performance monitoring
        self.task_completion_times = deque(maxlen=1000)
        self.resource_utilization_history = deque(maxlen=100)
        
        # Thread safety
        self.resource_lock = threading.RLock()
        
        logger.info(f"Resource pool initialized: {self.max_workers} threads, {self.max_processes} processes")
    
    @contextmanager
    def acquire_thread_resource(self):
        """Context manager for thread resource acquisition."""
        with self.resource_lock:
            self.active_threads += 1
        
        try:
            yield
        finally:
            with self.resource_lock:
                self.active_threads -= 1
    
    @contextmanager
    def acquire_process_resource(self):
        """Context manager for process resource acquisition."""
        with self.resource_lock:
            self.active_processes += 1
        
        try:
            yield
        finally:
            with self.resource_lock:
                self.active_processes -= 1
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU-bound task to thread pool."""
        def wrapped_func(*args, **kwargs):
            with self.acquire_thread_resource():
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.task_completion_times.append(duration)
                    return result
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    raise
        
        return self.thread_pool.submit(wrapped_func, *args, **kwargs)
    
    def submit_compute_task(self, func: Callable, *args, **kwargs):
        """Submit compute-intensive task to process pool."""
        def wrapped_func(*args, **kwargs):
            with self.acquire_process_resource():
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.task_completion_times.append(duration)
                    return result
                except Exception as e:
                    logger.error(f"Compute task execution failed: {e}")
                    raise
        
        return self.process_pool.submit(wrapped_func, *args, **kwargs)
    
    def get_optimal_backend(self, task: OptimizationTask) -> str:
        """Select optimal quantum backend for task."""
        # Prefer user-specified backends
        if task.preferred_backends:
            for backend in task.preferred_backends:
                if backend in self.quantum_backends:
                    queue_depth = self.quantum_active_jobs[backend]
                    if queue_depth < 10:  # Reasonable queue limit
                        return backend
        
        # Find backend with lowest queue depth
        backend_loads = {
            backend: self.quantum_active_jobs[backend]
            for backend in self.quantum_backends
        }
        
        return min(backend_loads.keys(), key=lambda b: backend_loads[b])
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        with self.resource_lock:
            thread_utilization = self.active_threads / self.max_workers
            process_utilization = self.active_processes / self.max_processes
            
            quantum_utilization = sum(self.quantum_active_jobs.values()) / len(self.quantum_backends)
            
            # System-level metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            utilization = {
                'thread_utilization': thread_utilization,
                'process_utilization': process_utilization,
                'quantum_utilization': quantum_utilization,
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory_percent
            }
            
            # Store for history
            self.resource_utilization_history.append({
                'timestamp': time.time(),
                **utilization
            })
            
            return utilization
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if self.task_completion_times:
            avg_completion_time = np.mean(self.task_completion_times)
            median_completion_time = np.median(self.task_completion_times)
            p95_completion_time = np.percentile(self.task_completion_times, 95)
        else:
            avg_completion_time = median_completion_time = p95_completion_time = 0.0
        
        utilization = self.get_resource_utilization()
        
        return {
            'avg_completion_time': avg_completion_time,
            'median_completion_time': median_completion_time,
            'p95_completion_time': p95_completion_time,
            'total_tasks_completed': len(self.task_completion_times),
            'current_utilization': utilization,
            'active_threads': self.active_threads,
            'active_processes': self.active_processes,
            'quantum_active_jobs': dict(self.quantum_active_jobs)
        }
    
    def shutdown(self):
        """Gracefully shutdown resource pools."""
        logger.info("Shutting down resource pools...")
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("Resource pools shutdown complete")


class AutoScaler:
    """Intelligent auto-scaling for quantum optimization workloads."""
    
    def __init__(self,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
                 min_workers: int = 2,
                 max_workers: int = 100,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 cooldown_period: int = 300):
        
        self.scaling_strategy = scaling_strategy
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        # State tracking
        self.current_workers = min_workers
        self.last_scaling_time = 0
        self.scaling_history = deque(maxlen=100)
        
        # Prediction models (simplified)
        self.load_predictor = LoadPredictor()
        
        logger.info(f"Auto-scaler initialized: strategy={scaling_strategy.value}")
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling up is needed."""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_scaling_time < self.cooldown_period:
            return False
        
        # Already at maximum
        if self.current_workers >= self.max_workers:
            return False
        
        # Check scaling conditions based on strategy
        if self.scaling_strategy == ScalingStrategy.STATIC:
            return False  # No auto-scaling
        
        elif self.scaling_strategy == ScalingStrategy.DYNAMIC:
            return (metrics.avg_cpu_utilization > self.scale_up_threshold or
                    metrics.queued_tasks > self.current_workers * 2)
        
        elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
            return (metrics.predicted_load_1h > self.scale_up_threshold * 1.2 or
                    metrics.avg_cpu_utilization > self.scale_up_threshold)
        
        elif self.scaling_strategy == ScalingStrategy.ADAPTIVE:
            # Combine multiple factors
            utilization_pressure = (
                metrics.avg_cpu_utilization > self.scale_up_threshold or
                metrics.avg_memory_utilization > self.scale_up_threshold
            )
            
            queue_pressure = metrics.queued_tasks > self.current_workers
            performance_degradation = metrics.avg_task_duration > 3600  # 1 hour
            
            return utilization_pressure and (queue_pressure or performance_degradation)
        
        return False
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling down is needed."""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_scaling_time < self.cooldown_period:
            return False
        
        # Already at minimum
        if self.current_workers <= self.min_workers:
            return False
        
        # Check scaling conditions
        if self.scaling_strategy == ScalingStrategy.STATIC:
            return False
        
        elif self.scaling_strategy == ScalingStrategy.DYNAMIC:
            return (metrics.avg_cpu_utilization < self.scale_down_threshold and
                    metrics.queued_tasks == 0)
        
        elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
            return (metrics.predicted_load_1h < self.scale_down_threshold and
                    metrics.avg_cpu_utilization < self.scale_down_threshold)
        
        elif self.scaling_strategy == ScalingStrategy.ADAPTIVE:
            low_utilization = (
                metrics.avg_cpu_utilization < self.scale_down_threshold and
                metrics.avg_memory_utilization < self.scale_down_threshold
            )
            
            no_queue_pressure = metrics.queued_tasks == 0
            stable_performance = metrics.success_rate > 0.95
            
            return low_utilization and no_queue_pressure and stable_performance
        
        return False
    
    def get_scaling_recommendation(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Get scaling recommendation based on current metrics."""
        
        recommendation = {
            'action': 'none',
            'target_workers': self.current_workers,
            'reasoning': [],
            'confidence': 0.5
        }
        
        if self.should_scale_up(metrics):
            # Calculate scale-up amount
            if metrics.queued_tasks > self.current_workers * 5:
                scale_factor = 2  # Aggressive scaling
            elif metrics.avg_cpu_utilization > 0.9:
                scale_factor = 1.5  # Moderate scaling
            else:
                scale_factor = 1.2  # Conservative scaling
            
            new_workers = min(self.max_workers, int(self.current_workers * scale_factor))
            
            recommendation.update({
                'action': 'scale_up',
                'target_workers': new_workers,
                'reasoning': [
                    f"CPU utilization: {metrics.avg_cpu_utilization:.1%}",
                    f"Queued tasks: {metrics.queued_tasks}",
                    f"Current workers: {self.current_workers}"
                ],
                'confidence': 0.8
            })
        
        elif self.should_scale_down(metrics):
            new_workers = max(self.min_workers, int(self.current_workers * 0.8))
            
            recommendation.update({
                'action': 'scale_down',
                'target_workers': new_workers,
                'reasoning': [
                    f"Low CPU utilization: {metrics.avg_cpu_utilization:.1%}",
                    f"No queued tasks",
                    f"Stable performance: {metrics.success_rate:.1%}"
                ],
                'confidence': 0.7
            })
        
        return recommendation
    
    def record_scaling_action(self, action: str, old_workers: int, new_workers: int):
        """Record scaling action for analysis."""
        self.current_workers = new_workers
        self.last_scaling_time = time.time()
        
        self.scaling_history.append({
            'timestamp': time.time(),
            'action': action,
            'old_workers': old_workers,
            'new_workers': new_workers
        })
        
        logger.info(f"Scaling action: {action} from {old_workers} to {new_workers} workers")


class LoadPredictor:
    """Simple load prediction for auto-scaling."""
    
    def __init__(self):
        self.load_history = deque(maxlen=1440)  # 24 hours of minutes
    
    def add_load_sample(self, load: float):
        """Add load sample to history."""
        self.load_history.append({
            'timestamp': time.time(),
            'load': load
        })
    
    def predict_load(self, horizon_minutes: int = 60) -> float:
        """Predict load for given horizon."""
        if len(self.load_history) < 10:
            return 0.5  # Default prediction
        
        # Simple moving average prediction
        recent_loads = [sample['load'] for sample in list(self.load_history)[-10:]]
        return np.mean(recent_loads)


class ScalableQuantumOptimizer:
    """
    Production-scale quantum optimization engine with:
    - Intelligent caching and memoization
    - Concurrent processing and resource pooling
    - Auto-scaling and load balancing
    - Performance monitoring and optimization
    """
    
    def __init__(self,
                 max_workers: int = None,
                 cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
                 redis_url: Optional[str] = None,
                 enable_distributed: bool = False):
        
        # Core configuration
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.cache_strategy = cache_strategy
        self.scaling_strategy = scaling_strategy
        self.enable_distributed = enable_distributed and RAY_AVAILABLE
        
        # Security and validation
        self.security_manager = SecurityManager()
        self.validator = InputValidator(strict_mode=True)
        
        # Initialize components
        self.cache = DistributedCache(
            cache_strategy=cache_strategy,
            max_memory_mb=2048,
            redis_url=redis_url
        )
        
        self.resource_pool = ResourcePool(
            max_workers=self.max_workers,
            quantum_backends=["simulator", "ibm_quantum", "dwave"]
        )
        
        self.auto_scaler = AutoScaler(
            scaling_strategy=scaling_strategy,
            min_workers=2,
            max_workers=self.max_workers * 2
        )
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.completed_tasks: Dict[str, OptimizationTask] = {}
        
        # Performance monitoring
        self.metrics = ScalingMetrics()
        self.performance_monitor = PerformanceMonitor()
        
        # Threading
        self.scheduler_thread = None
        self.metrics_thread = None
        self.running = False
        
        # Initialize distributed computing if available
        if self.enable_distributed:
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Ray distributed computing initialized")
            except Exception as e:
                logger.warning(f"Ray initialization failed: {e}")
                self.enable_distributed = False
        
        logger.info(f"Scalable quantum optimizer initialized: {self.max_workers} workers")
    
    def start(self):
        """Start the optimization engine."""
        self.running = True
        
        # Start background threads
        self.scheduler_thread = threading.Thread(target=self._task_scheduler, daemon=True)
        self.metrics_thread = threading.Thread(target=self._metrics_collector, daemon=True)
        
        self.scheduler_thread.start()
        self.metrics_thread.start()
        
        logger.info("Scalable quantum optimizer started")
    
    def stop(self):
        """Stop the optimization engine."""
        self.running = False
        
        # Wait for threads to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        
        # Shutdown resource pools
        self.resource_pool.shutdown()
        
        # Shutdown distributed computing
        if self.enable_distributed:
            try:
                ray.shutdown()
            except Exception:
                pass
        
        logger.info("Scalable quantum optimizer stopped")
    
    @secure_operation(SecurityLevel.HIGH)
    def submit_optimization(self,
                           problem_matrix: np.ndarray,
                           constraints: Dict[str, Any],
                           priority: int = 5,
                           **kwargs) -> str:
        """Submit optimization task to the scalable engine."""
        
        # Create task
        task = OptimizationTask(
            task_id=f"opt_{int(time.time() * 1000000)}",
            problem_matrix=problem_matrix,
            constraints=constraints,
            priority=priority,
            **kwargs
        )
        
        # Validate task
        self._validate_task(task)
        
        # Check cache first
        cached_result = self.cache.get(task.cache_key)
        if cached_result:
            logger.info(f"Cache hit for task {task.task_id}")
            task.result = cached_result
            task.status = "completed"
            task.completion_time = time.time()
            self.completed_tasks[task.task_id] = task
            return task.task_id
        
        # Add to queue
        self.task_queue.put((-priority, time.time(), task))  # Negative for max priority
        
        logger.info(f"Task {task.task_id} submitted with priority {priority}")
        
        # Update metrics
        self.metrics.queued_tasks += 1
        
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of optimization task."""
        
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status,
                'start_time': task.start_time,
                'estimated_completion': self._estimate_completion_time(task),
                'progress': self._estimate_progress(task)
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status,
                'start_time': task.start_time,
                'completion_time': task.completion_time,
                'result': task.result,
                'error': task.error
            }
        
        return {'error': f'Task {task_id} not found'}
    
    def get_optimization_result(self, task_id: str) -> Optional[Any]:
        """Get optimization result for completed task."""
        
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            if task.status == "completed" and task.result:
                return task.result
        
        return None
    
    def _validate_task(self, task: OptimizationTask):
        """Validate optimization task."""
        
        # Basic validation
        if task.problem_matrix.size == 0:
            raise ValueError("Problem matrix cannot be empty")
        
        if task.problem_matrix.shape[0] != task.problem_matrix.shape[1]:
            raise ValueError("Problem matrix must be square")
        
        if task.problem_matrix.shape[0] > 10000:
            raise ValueError("Problem matrix too large (max 10000x10000)")
        
        # Resource validation
        if task.estimated_cpu_hours > 24:
            raise ValueError("Estimated CPU hours too high (max 24)")
        
        if task.estimated_memory_gb > 64:
            raise ValueError("Estimated memory too high (max 64GB)")
        
        # Security validation
        self.security_manager.validate_problem_size(
            task.problem_matrix.shape[0], task.problem_matrix.shape[0]
        )
    
    def _task_scheduler(self):
        """Background task scheduler."""
        
        while self.running:
            try:
                # Get next task
                try:
                    priority, submit_time, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if we have capacity
                utilization = self.resource_pool.get_resource_utilization()
                if utilization['thread_utilization'] > 0.9:
                    # Put task back and wait
                    self.task_queue.put((priority, submit_time, task))
                    time.sleep(5)
                    continue
                
                # Execute task
                self._execute_task(task)
                
                # Update metrics
                self.metrics.queued_tasks -= 1
                
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: OptimizationTask):
        """Execute optimization task."""
        
        task.status = "running"
        task.start_time = time.time()
        self.active_tasks[task.task_id] = task
        self.metrics.active_tasks += 1
        
        # Select execution strategy
        if self.enable_distributed and task.problem_matrix.shape[0] > 100:
            future = self._execute_distributed(task)
        elif task.estimated_cpu_hours > 2:
            future = self.resource_pool.submit_compute_task(self._solve_optimization, task)
        else:
            future = self.resource_pool.submit_cpu_task(self._solve_optimization, task)
        
        # Handle completion
        def on_completion(fut):
            try:
                result = fut.result()
                task.result = result
                task.status = "completed"
                task.completion_time = time.time()
                
                # Cache result
                self.cache.set(task.cache_key, result, ttl=3600)
                
                # Update metrics
                self.metrics.active_tasks -= 1
                self.metrics.completed_tasks += 1
                
                # Move to completed tasks
                self.completed_tasks[task.task_id] = task
                del self.active_tasks[task.task_id]
                
                logger.info(f"Task {task.task_id} completed successfully")
                
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                task.completion_time = time.time()
                
                # Update metrics
                self.metrics.active_tasks -= 1
                self.metrics.failed_tasks += 1
                
                # Move to completed tasks
                self.completed_tasks[task.task_id] = task
                del self.active_tasks[task.task_id]
                
                logger.error(f"Task {task.task_id} failed: {e}")
        
        future.add_done_callback(on_completion)
    
    def _solve_optimization(self, task: OptimizationTask) -> Any:
        """Solve optimization problem."""
        
        # This is a placeholder for the actual optimization logic
        # In practice, this would integrate with the quantum planning algorithms
        
        start_time = time.time()
        
        # Simulate optimization work
        problem_size = task.problem_matrix.shape[0]
        
        # Simple heuristic solution for demonstration
        np.random.seed(hash(task.task_id) % 2**32)
        solution = np.random.choice([0, 1], size=problem_size)
        
        # Calculate objective value
        objective_value = solution.T @ task.problem_matrix @ solution
        
        # Simulate computation time based on problem complexity
        computation_time = min(task.max_execution_time, problem_size * 0.01)
        time.sleep(computation_time)
        
        result = {
            'solution': solution.tolist(),
            'objective_value': float(objective_value),
            'computation_time': time.time() - start_time,
            'algorithm_used': 'scalable_heuristic',
            'problem_size': problem_size
        }
        
        return result
    
    def _execute_distributed(self, task: OptimizationTask):
        """Execute task using distributed computing."""
        
        if not self.enable_distributed:
            return self.resource_pool.submit_compute_task(self._solve_optimization, task)
        
        # Use Ray for distributed execution
        @ray.remote
        def distributed_solve(task_data):
            # Reconstruct task (Ray requires serializable objects)
            task = OptimizationTask(
                task_id=task_data['task_id'],
                problem_matrix=np.array(task_data['problem_matrix']),
                constraints=task_data['constraints'],
                priority=task_data['priority']
            )
            return self._solve_optimization(task)
        
        # Prepare task data for serialization
        task_data = {
            'task_id': task.task_id,
            'problem_matrix': task.problem_matrix.tolist(),
            'constraints': task.constraints,
            'priority': task.priority
        }
        
        # Submit to Ray
        return distributed_solve.remote(task_data)
    
    def _metrics_collector(self):
        """Background metrics collection."""
        
        while self.running:
            try:
                # Collect resource utilization
                utilization = self.resource_pool.get_resource_utilization()
                
                # Update scaling metrics
                self.metrics.avg_cpu_utilization = utilization['system_cpu_percent'] / 100.0
                self.metrics.avg_memory_utilization = utilization['system_memory_percent'] / 100.0
                self.metrics.avg_quantum_utilization = utilization['quantum_utilization']
                
                # Calculate throughput
                completed_last_hour = len([
                    task for task in self.completed_tasks.values()
                    if task.completion_time and time.time() - task.completion_time < 3600
                ])
                self.metrics.throughput_tasks_per_hour = completed_last_hour
                
                # Calculate success rate
                total_recent = self.metrics.completed_tasks + self.metrics.failed_tasks
                if total_recent > 0:
                    self.metrics.success_rate = self.metrics.completed_tasks / total_recent
                
                # Check auto-scaling
                self._check_auto_scaling()
                
                # Sleep before next collection
                time.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(10)
    
    def _check_auto_scaling(self):
        """Check and execute auto-scaling decisions."""
        
        recommendation = self.auto_scaler.get_scaling_recommendation(self.metrics)
        
        if recommendation['action'] != 'none':
            logger.info(f"Auto-scaling recommendation: {recommendation}")
            
            # In production, this would trigger actual resource scaling
            # For now, just record the decision
            self.auto_scaler.record_scaling_action(
                recommendation['action'],
                self.auto_scaler.current_workers,
                recommendation['target_workers']
            )
    
    def _estimate_completion_time(self, task: OptimizationTask) -> Optional[float]:
        """Estimate task completion time."""
        
        if not task.start_time:
            return None
        
        elapsed = time.time() - task.start_time
        
        # Simple estimation based on problem size
        estimated_total = task.problem_matrix.shape[0] * 0.1
        
        if elapsed > 0:
            progress = min(0.9, elapsed / estimated_total)
            remaining = (estimated_total - elapsed) / max(progress, 0.1)
            return task.start_time + elapsed + remaining
        
        return None
    
    def _estimate_progress(self, task: OptimizationTask) -> float:
        """Estimate task progress (0-1)."""
        
        if not task.start_time:
            return 0.0
        
        elapsed = time.time() - task.start_time
        estimated_total = task.problem_matrix.shape[0] * 0.1
        
        return min(0.95, elapsed / estimated_total)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            'system': {
                'running': self.running,
                'uptime': time.time() - (self.metrics.completed_tasks + self.metrics.failed_tasks) * 60,
                'version': '3.0.0'
            },
            'tasks': {
                'active': len(self.active_tasks),
                'queued': self.task_queue.qsize(),
                'completed': len(self.completed_tasks),
                'total_processed': self.metrics.completed_tasks + self.metrics.failed_tasks
            },
            'resources': self.resource_pool.get_performance_stats(),
            'cache': self.cache.get_statistics(),
            'metrics': self.metrics.to_dict(),
            'auto_scaling': {
                'strategy': self.auto_scaler.scaling_strategy.value,
                'current_workers': self.auto_scaler.current_workers,
                'scaling_history': list(self.auto_scaler.scaling_history)[-10:]
            }
        }


# Example usage and demonstration
def demonstrate_scalable_optimization():
    """Demonstrate scalable quantum optimization capabilities."""
    
    print("Scalable Quantum Optimization Demonstration")
    print("=" * 55)
    
    # Initialize optimizer
    optimizer = ScalableQuantumOptimizer(
        max_workers=4,
        cache_strategy=CacheStrategy.ADAPTIVE,
        scaling_strategy=ScalingStrategy.ADAPTIVE
    )
    
    # Start the engine
    optimizer.start()
    
    try:
        # Submit multiple optimization tasks
        task_ids = []
        
        for i in range(10):
            # Create random problem
            size = np.random.randint(20, 100)
            problem_matrix = np.random.randn(size, size)
            problem_matrix = (problem_matrix + problem_matrix.T) / 2  # Make symmetric
            
            constraints = {
                'type': 'quadratic',
                'variables': size,
                'max_iterations': 1000
            }
            
            task_id = optimizer.submit_optimization(
                problem_matrix=problem_matrix,
                constraints=constraints,
                priority=np.random.randint(1, 10)
            )
            
            task_ids.append(task_id)
            print(f"Submitted task {i+1}: {task_id}")
        
        # Monitor progress
        print("\\nMonitoring optimization progress...")
        completed_count = 0
        
        while completed_count < len(task_ids):
            time.sleep(2)
            
            # Check status of all tasks
            new_completed = 0
            for task_id in task_ids:
                status = optimizer.get_task_status(task_id)
                if status.get('status') == 'completed':
                    new_completed += 1
            
            if new_completed > completed_count:
                completed_count = new_completed
                print(f"Completed: {completed_count}/{len(task_ids)} tasks")
        
        # Get system status
        system_status = optimizer.get_system_status()
        
        print(f"\\nOptimization Summary:")
        print(f"{'='*30}")
        print(f"Tasks processed: {system_status['tasks']['total_processed']}")
        print(f"Cache hit rate: {system_status['cache']['hit_rate']:.1%}")
        print(f"Average completion time: {system_status['resources']['avg_completion_time']:.2f}s")
        print(f"CPU utilization: {system_status['metrics']['avg_cpu_utilization']:.1%}")
        print(f"Throughput: {system_status['metrics']['throughput_tasks_per_hour']:.1f} tasks/hour")
        
        # Show sample results
        print(f"\\nSample Results:")
        for i, task_id in enumerate(task_ids[:3]):
            result = optimizer.get_optimization_result(task_id)
            if result:
                print(f"Task {i+1}: objective={result['objective_value']:.2f}, "
                      f"time={result['computation_time']:.2f}s")
        
    finally:
        # Stop the optimizer
        optimizer.stop()
        print("\\nScalable optimization demonstration complete!")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_scalable_optimization()
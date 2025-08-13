#!/usr/bin/env python3
"""
Scalable Quantum Task Planner - Generation 3: Make It Scale
High-performance implementation with concurrency, caching, load balancing, and auto-scaling.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
import sys
import os
from typing import List, Dict, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import hashlib
from contextlib import asynccontextmanager
import queue
import statistics
from collections import defaultdict, deque

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_planner import QuantumTaskPlanner, Agent, Task, Solution


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking."""
    requests_per_second: float = 0.0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput_mbps: float = 0.0
    cache_hit_rate: float = 0.0
    worker_utilization: float = 0.0
    queue_depth: int = 0
    active_connections: int = 0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0


@dataclass
class WorkerPool:
    """Worker pool for parallel processing."""
    workers: List[QuantumTaskPlanner]
    utilization: List[float]
    performance_history: List[float]
    last_used: List[float]
    
    def __post_init__(self):
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()
        self.result_cache = {}
        
    def get_least_loaded_worker(self) -> Tuple[int, QuantumTaskPlanner]:
        """Get the worker with lowest current utilization."""
        with self.lock:
            min_util_idx = min(range(len(self.utilization)), key=self.utilization.__getitem__)
            return min_util_idx, self.workers[min_util_idx]
    
    def get_best_performing_worker(self) -> Tuple[int, QuantumTaskPlanner]:
        """Get the worker with best historical performance."""
        with self.lock:
            if all(perf == 0 for perf in self.performance_history):
                return self.get_least_loaded_worker()
            
            max_perf_idx = max(range(len(self.performance_history)), 
                              key=self.performance_history.__getitem__)
            return max_perf_idx, self.workers[max_perf_idx]
    
    def update_worker_metrics(self, worker_idx: int, latency: float, success: bool):
        """Update worker performance metrics."""
        with self.lock:
            # Update utilization (exponential moving average)
            current_util = 1.0 if not success else 0.5  # Higher util for failures
            self.utilization[worker_idx] = (
                0.7 * self.utilization[worker_idx] + 0.3 * current_util
            )
            
            # Update performance (inverse latency for better = higher)
            performance_score = 1.0 / max(latency, 0.001) if success else 0.01
            self.performance_history[worker_idx] = (
                0.8 * self.performance_history[worker_idx] + 0.2 * performance_score
            )
            
            self.last_used[worker_idx] = time.time()


class AdvancedCache:
    """High-performance caching with LRU eviction and compression."""
    
    def __init__(self, max_size: int = 10000, compress: bool = True):
        self.max_size = max_size
        self.compress = compress
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_key(self, agents: List[Dict], tasks: List[Dict], params: Dict) -> str:
        """Generate a deterministic cache key."""
        # Create normalized representation
        agents_normalized = tuple(sorted([
            (a['id'], tuple(sorted(a['skills'])), a.get('capacity', 1))
            for a in agents
        ]))
        
        tasks_normalized = tuple(sorted([
            (t['id'], tuple(sorted(t['skills'])), t.get('priority', 1), t.get('duration', 1))
            for t in tasks
        ]))
        
        params_normalized = tuple(sorted(params.items()))
        
        # Hash the normalized data
        cache_input = (agents_normalized, tasks_normalized, params_normalized)
        return hashlib.sha256(str(cache_input).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get item from cache with access tracking."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hits += 1
                
                # Decompress if needed
                result = self.cache[key]
                if self.compress and isinstance(result, bytes):
                    result = pickle.loads(result)
                    
                return result.copy() if isinstance(result, dict) else result
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Dict) -> None:
        """Store item in cache with LRU eviction."""
        with self.lock:
            # Compress if enabled
            if self.compress:
                value_to_store = pickle.dumps(value)
            else:
                value_to_store = value.copy() if isinstance(value, dict) else value
            
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value_to_store
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
            
        # Find LRU item
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Remove from all structures
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.access_counts[lru_key]
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self) -> None:
        """Clear all cache data."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()


class ScalableQuantumPlanner:
    """
    High-performance scalable quantum task planner with auto-scaling,
    load balancing, advanced caching, and concurrent processing.
    """
    
    def __init__(
        self,
        worker_pool_size: int = None,
        scaling_strategy: ScalingStrategy = ScalingStrategy.BALANCED,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        cache_size: int = 10000,
        enable_compression: bool = True,
        enable_async: bool = True,
        performance_monitoring: bool = True
    ):
        """
        Initialize scalable planner with performance optimizations.
        
        Args:
            worker_pool_size: Number of worker planners (None = auto-detect)
            scaling_strategy: How aggressively to scale resources
            load_balancing: Load balancing algorithm
            cache_size: Maximum cache entries
            enable_compression: Use compression for cached data
            enable_async: Enable asynchronous operations
            performance_monitoring: Collect detailed performance metrics
        """
        self.scaling_strategy = scaling_strategy
        self.load_balancing = load_balancing
        self.enable_async = enable_async
        self.performance_monitoring = performance_monitoring
        
        # Auto-detect optimal worker pool size
        if worker_pool_size is None:
            cpu_count = multiprocessing.cpu_count()
            worker_pool_size = max(2, min(cpu_count, 8))  # Between 2-8 workers
        
        self.worker_pool_size = worker_pool_size
        
        # Initialize worker pool
        self._initialize_worker_pool()
        
        # Initialize advanced caching
        self.cache = AdvancedCache(max_size=cache_size, compress=enable_compression)
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.request_times = deque(maxlen=1000)  # Last 1000 requests
        self.throughput_history = deque(maxlen=100)  # Last 100 measurements
        
        # Load balancing state
        self.round_robin_counter = 0
        self.request_count = 0
        self.start_time = time.time()
        
        # Threading for background tasks
        self.metrics_thread = threading.Thread(target=self._metrics_updater, daemon=True)
        self.metrics_thread.start()
        
        # Auto-scaling
        self.scaling_lock = threading.Lock()
        self.last_scale_time = time.time()
        self.scale_cooldown = 30  # seconds
        
        print(f"🚀 ScalableQuantumPlanner initialized with {worker_pool_size} workers")
        print(f"   Scaling: {scaling_strategy.value}, Load Balancing: {load_balancing.value}")
        print(f"   Cache: {cache_size} entries, Compression: {enable_compression}")
    
    def _initialize_worker_pool(self) -> None:
        """Initialize the worker pool with quantum planners."""
        workers = []
        for i in range(self.worker_pool_size):
            try:
                worker = QuantumTaskPlanner(
                    backend="auto",
                    fallback="simulated_annealing"
                )
                workers.append(worker)
            except Exception as e:
                print(f"Warning: Failed to initialize worker {i}: {e}")
                # Create simpler fallback worker
                worker = QuantumTaskPlanner(
                    backend="simulated_annealing",
                    fallback=None
                )
                workers.append(worker)
        
        self.worker_pool = WorkerPool(
            workers=workers,
            utilization=[0.0] * len(workers),
            performance_history=[1.0] * len(workers),  # Start with neutral performance
            last_used=[time.time()] * len(workers)
        )
    
    def _select_worker(self) -> Tuple[int, QuantumTaskPlanner]:
        """Select worker based on load balancing strategy."""
        if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            idx = self.round_robin_counter % len(self.worker_pool.workers)
            self.round_robin_counter += 1
            return idx, self.worker_pool.workers[idx]
        
        elif self.load_balancing == LoadBalancingStrategy.LEAST_LOADED:
            return self.worker_pool.get_least_loaded_worker()
        
        elif self.load_balancing == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self.worker_pool.get_best_performing_worker()
        
        elif self.load_balancing == LoadBalancingStrategy.ADAPTIVE:
            # Use performance-based if we have enough data, otherwise least loaded
            avg_requests = self.request_count / len(self.worker_pool.workers)
            if avg_requests > 10:  # Enough data for performance-based
                return self.worker_pool.get_best_performing_worker()
            else:
                return self.worker_pool.get_least_loaded_worker()
        
        else:
            return 0, self.worker_pool.workers[0]
    
    def _check_auto_scaling(self) -> None:
        """Check if auto-scaling is needed and adjust worker pool."""
        with self.scaling_lock:
            current_time = time.time()
            
            # Respect cooldown period
            if current_time - self.last_scale_time < self.scale_cooldown:
                return
            
            # Calculate utilization metrics
            avg_utilization = statistics.mean(self.worker_pool.utilization) if self.worker_pool.utilization else 0
            max_utilization = max(self.worker_pool.utilization) if self.worker_pool.utilization else 0
            queue_size = self.worker_pool.task_queue.qsize()
            
            scale_up_needed = False
            scale_down_needed = False
            
            # Scaling thresholds based on strategy
            if self.scaling_strategy == ScalingStrategy.CONSERVATIVE:
                scale_up_threshold = 0.9
                scale_down_threshold = 0.2
            elif self.scaling_strategy == ScalingStrategy.BALANCED:
                scale_up_threshold = 0.75
                scale_down_threshold = 0.3
            else:  # AGGRESSIVE
                scale_up_threshold = 0.6
                scale_down_threshold = 0.4
            
            # Decide on scaling
            if (avg_utilization > scale_up_threshold or 
                max_utilization > 0.95 or 
                queue_size > len(self.worker_pool.workers) * 2):
                scale_up_needed = True
            
            elif (avg_utilization < scale_down_threshold and 
                  max_utilization < 0.5 and 
                  len(self.worker_pool.workers) > 2):
                scale_down_needed = True
            
            # Execute scaling
            if scale_up_needed and len(self.worker_pool.workers) < 16:  # Max 16 workers
                self._scale_up()
                self.last_scale_time = current_time
                
            elif scale_down_needed:
                self._scale_down()
                self.last_scale_time = current_time
    
    def _scale_up(self) -> None:
        """Add a new worker to the pool."""
        try:
            new_worker = QuantumTaskPlanner(
                backend="auto",
                fallback="simulated_annealing"
            )
            
            self.worker_pool.workers.append(new_worker)
            self.worker_pool.utilization.append(0.0)
            self.worker_pool.performance_history.append(1.0)
            self.worker_pool.last_used.append(time.time())
            
            print(f"📈 Scaled up: {len(self.worker_pool.workers)} workers")
            
        except Exception as e:
            print(f"⚠️ Scale up failed: {e}")
    
    def _scale_down(self) -> None:
        """Remove the least used worker from the pool."""
        if len(self.worker_pool.workers) <= 2:
            return
            
        # Find least recently used worker
        oldest_idx = min(range(len(self.worker_pool.last_used)), 
                        key=self.worker_pool.last_used.__getitem__)
        
        # Remove worker and its metrics
        self.worker_pool.workers.pop(oldest_idx)
        self.worker_pool.utilization.pop(oldest_idx)
        self.worker_pool.performance_history.pop(oldest_idx)
        self.worker_pool.last_used.pop(oldest_idx)
        
        print(f"📉 Scaled down: {len(self.worker_pool.workers)} workers")
    
    def _metrics_updater(self) -> None:
        """Background thread to update performance metrics."""
        while True:
            try:
                time.sleep(1)  # Update every second
                
                if self.performance_monitoring:
                    self._update_performance_metrics()
                    self._check_auto_scaling()
                    
            except Exception as e:
                print(f"Metrics updater error: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update comprehensive performance metrics."""
        current_time = time.time()
        
        # Calculate requests per second
        if self.request_times:
            recent_requests = [t for t in self.request_times if current_time - t < 60]
            self.metrics.requests_per_second = len(recent_requests) / 60.0
        
        # Calculate latency percentiles
        if self.request_times:
            recent_latencies = list(self.request_times)[-100:]  # Last 100 requests
            if recent_latencies:
                sorted_latencies = sorted(recent_latencies)
                self.metrics.average_latency = statistics.mean(sorted_latencies)
                
                if len(sorted_latencies) >= 20:  # Enough data for percentiles
                    p95_idx = int(0.95 * len(sorted_latencies))
                    p99_idx = int(0.99 * len(sorted_latencies))
                    self.metrics.p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
                    self.metrics.p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        
        # Update cache hit rate
        cache_stats = self.cache.get_stats()
        self.metrics.cache_hit_rate = cache_stats['hit_rate']
        
        # Update worker utilization
        if self.worker_pool.utilization:
            self.metrics.worker_utilization = statistics.mean(self.worker_pool.utilization)
        
        # Update queue depth
        self.metrics.queue_depth = self.worker_pool.task_queue.qsize()
        
        # Update active connections (approximation)
        self.metrics.active_connections = len(self.worker_pool.workers)
    
    def assign_tasks_scalable(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        minimize: str = "time",
        timeout: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        High-performance task assignment with scaling and optimization.
        
        Args:
            agents: List of agent dictionaries
            tasks: List of task dictionaries
            minimize: Optimization objective ('time' or 'cost')
            timeout: Maximum execution time (None = use default)
            use_cache: Whether to use result caching
            
        Returns:
            Comprehensive result with performance metrics and assignments
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Step 1: Check cache first
            if use_cache:
                cache_key = self.cache._generate_key(agents, tasks, {'minimize': minimize})
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    # Update cache hit metrics
                    cached_result['cache_hit'] = True
                    cached_result['cache_retrieval_time'] = time.time() - start_time
                    cached_result['worker_used'] = 'cached'
                    
                    self.request_times.append(time.time() - start_time)
                    
                    return cached_result
            
            # Step 2: Select optimal worker
            worker_idx, selected_worker = self._select_worker()
            
            # Step 3: Execute assignment with selected worker
            assignment_start = time.time()
            
            # Convert to internal models
            agent_objects = [Agent(
                agent_id=a['id'],
                skills=a['skills'],
                capacity=a.get('capacity', 1),
                availability=a.get('availability', 1.0),
                cost_per_hour=a.get('cost_per_hour', 0.0)
            ) for a in agents]
            
            task_objects = [Task(
                task_id=t['id'],
                required_skills=t['skills'],
                priority=t.get('priority', 1),
                duration=t.get('duration', 1),
                dependencies=t.get('dependencies', [])
            ) for t in tasks]
            
            # Execute assignment
            objective_map = {
                'time': 'minimize_makespan',
                'cost': 'minimize_cost'
            }
            objective = objective_map.get(minimize, 'minimize_makespan')
            
            solution = selected_worker.assign(
                agents=agent_objects,
                tasks=task_objects,
                objective=objective,
                constraints={
                    'skill_match': True,
                    'capacity_limit': True
                }
            )
            
            assignment_time = time.time() - assignment_start
            total_time = time.time() - start_time
            
            # Step 4: Update worker performance metrics
            self.worker_pool.update_worker_metrics(worker_idx, assignment_time, True)
            
            # Step 5: Build comprehensive result
            result = {
                'success': True,
                'assignments': solution.assignments,
                'completion_time': solution.makespan,
                'total_cost': solution.cost,
                'backend_used': solution.backend_used,
                'message': 'Scalable assignment completed successfully',
                
                # Performance metrics
                'performance': {
                    'total_time': total_time,
                    'assignment_time': assignment_time,
                    'cache_hit': False,
                    'worker_used': f'worker_{worker_idx}',
                    'worker_pool_size': len(self.worker_pool.workers),
                    'queue_depth': self.worker_pool.task_queue.qsize(),
                    'requests_per_second': self.metrics.requests_per_second,
                    'cache_hit_rate': self.metrics.cache_hit_rate
                },
                
                # Scaling information
                'scaling': {
                    'worker_utilization': statistics.mean(self.worker_pool.utilization),
                    'scaling_strategy': self.scaling_strategy.value,
                    'load_balancing': self.load_balancing.value,
                    'auto_scaling_enabled': True
                },
                
                # Quality metrics
                'quality': {
                    'quality_score': solution.calculate_quality_score(),
                    'load_distribution': solution.get_load_distribution(),
                    'assigned_agents': list(solution.get_assigned_agents())
                }
            }
            
            # Step 6: Cache result if enabled
            if use_cache:
                cache_result = result.copy()
                cache_result.pop('cache_hit', None)  # Don't cache the cache_hit flag
                self.cache.put(cache_key, cache_result)
                result['cache_stored'] = True
            
            # Step 7: Update performance tracking
            self.request_times.append(total_time)
            
            return result
            
        except Exception as e:
            # Handle errors with performance tracking
            error_time = time.time() - start_time
            self.request_times.append(error_time)
            
            # Update worker metrics for failure
            if 'worker_idx' in locals():
                self.worker_pool.update_worker_metrics(worker_idx, error_time, False)
            
            return {
                'success': False,
                'assignments': {},
                'completion_time': 0,
                'total_cost': 0,
                'backend_used': 'none',
                'error': str(e),
                'message': f'Scalable assignment failed: {e}',
                'performance': {
                    'total_time': error_time,
                    'assignment_time': 0,
                    'cache_hit': False,
                    'worker_used': 'error',
                    'error_type': type(e).__name__
                }
            }
    
    async def assign_tasks_async(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        minimize: str = "time",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous task assignment for high concurrency.
        
        Args:
            agents: List of agent dictionaries
            tasks: List of task dictionaries  
            minimize: Optimization objective
            timeout: Maximum execution time
            
        Returns:
            Assignment result dictionary
        """
        if not self.enable_async:
            # Fallback to synchronous
            return self.assign_tasks_scalable(agents, tasks, minimize, timeout)
        
        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_pool_size) as executor:
            result = await loop.run_in_executor(
                executor,
                self.assign_tasks_scalable,
                agents,
                tasks,
                minimize,
                timeout
            )
        
        return result
    
    def batch_assign_tasks(
        self,
        batch_requests: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple assignment requests concurrently.
        
        Args:
            batch_requests: List of request dictionaries with 'agents', 'tasks', 'minimize'
            max_concurrency: Maximum concurrent assignments (None = use worker pool size)
            
        Returns:
            List of assignment results in order
        """
        if max_concurrency is None:
            max_concurrency = len(self.worker_pool.workers)
        
        results = [None] * len(batch_requests)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Submit all tasks
            future_to_index = {}
            
            for i, request in enumerate(batch_requests):
                future = executor.submit(
                    self.assign_tasks_scalable,
                    request.get('agents', []),
                    request.get('tasks', []),
                    request.get('minimize', 'time'),
                    request.get('timeout')
                )
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = {
                        'success': False,
                        'error': str(e),
                        'message': f'Batch assignment {index} failed'
                    }
        
        return results
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get detailed system status and performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Worker pool status
        worker_status = []
        for i, worker in enumerate(self.worker_pool.workers):
            worker_status.append({
                'worker_id': i,
                'utilization': self.worker_pool.utilization[i],
                'performance_score': self.worker_pool.performance_history[i],
                'last_used': current_time - self.worker_pool.last_used[i],
                'status': 'active'
            })
        
        # Cache statistics
        cache_stats = self.cache.get_stats()
        
        # Performance summary
        performance_summary = {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'requests_per_second': self.metrics.requests_per_second,
            'average_latency': self.metrics.average_latency,
            'p95_latency': self.metrics.p95_latency,
            'p99_latency': self.metrics.p99_latency,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'worker_utilization': self.metrics.worker_utilization
        }
        
        # System configuration
        system_config = {
            'worker_pool_size': len(self.worker_pool.workers),
            'scaling_strategy': self.scaling_strategy.value,
            'load_balancing': self.load_balancing.value,
            'cache_size': cache_stats['size'],
            'cache_max_size': cache_stats['max_size'],
            'async_enabled': self.enable_async,
            'performance_monitoring': self.performance_monitoring
        }
        
        return {
            'timestamp': current_time,
            'overall_health': self._calculate_health_score(),
            'performance': performance_summary,
            'workers': worker_status,
            'cache': cache_stats,
            'configuration': system_config,
            'scaling': {
                'last_scale_time': self.last_scale_time,
                'scale_cooldown': self.scale_cooldown,
                'can_scale_up': len(self.worker_pool.workers) < 16,
                'can_scale_down': len(self.worker_pool.workers) > 2
            }
        }
    
    def _calculate_health_score(self) -> str:
        """Calculate overall system health."""
        # Check various health indicators
        worker_health = statistics.mean(self.worker_pool.utilization) < 0.9
        cache_health = self.metrics.cache_hit_rate > 0.1
        latency_health = self.metrics.average_latency < 5.0
        queue_health = self.metrics.queue_depth < len(self.worker_pool.workers) * 5
        
        health_indicators = [worker_health, cache_health, latency_health, queue_health]
        healthy_count = sum(health_indicators)
        
        if healthy_count >= 3:
            return "excellent"
        elif healthy_count >= 2:
            return "good"
        elif healthy_count >= 1:
            return "fair"
        else:
            return "poor"
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Auto-tune configuration based on observed performance."""
        recommendations = {}
        
        # Analyze performance patterns
        avg_utilization = statistics.mean(self.worker_pool.utilization)
        cache_hit_rate = self.metrics.cache_hit_rate
        avg_latency = self.metrics.average_latency
        
        # Worker pool size recommendations
        if avg_utilization > 0.8:
            recommendations['scale_up'] = f"Consider scaling up (utilization: {avg_utilization:.1%})"
        elif avg_utilization < 0.3:
            recommendations['scale_down'] = f"Consider scaling down (utilization: {avg_utilization:.1%})"
        
        # Cache size recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats['evictions'] > cache_stats['hits'] * 0.1:
            recommendations['increase_cache'] = "Consider increasing cache size (high eviction rate)"
        
        # Load balancing recommendations
        utilization_variance = statistics.variance(self.worker_pool.utilization)
        if utilization_variance > 0.2:
            recommendations['load_balancing'] = "Consider switching to performance-based load balancing"
        
        # Scaling strategy recommendations
        if self.scaling_strategy == ScalingStrategy.CONSERVATIVE and avg_latency > 2.0:
            recommendations['scaling_strategy'] = "Consider more aggressive scaling strategy"
        
        return {
            'timestamp': time.time(),
            'current_config': {
                'workers': len(self.worker_pool.workers),
                'cache_size': cache_stats['max_size'],
                'scaling': self.scaling_strategy.value,
                'load_balancing': self.load_balancing.value
            },
            'performance_analysis': {
                'avg_utilization': avg_utilization,
                'cache_hit_rate': cache_hit_rate,
                'avg_latency': avg_latency,
                'utilization_variance': utilization_variance
            },
            'recommendations': recommendations
        }


def demo_scalable_usage():
    """Demonstrate scalable planner capabilities with performance testing."""
    print("🚀 Scalable Quantum Task Planner Demo")
    print("=" * 70)
    
    # Initialize with different configurations
    configs = [
        {
            'name': 'Conservative Scaling',
            'config': {
                'worker_pool_size': 2,
                'scaling_strategy': ScalingStrategy.CONSERVATIVE,
                'load_balancing': LoadBalancingStrategy.ROUND_ROBIN
            }
        },
        {
            'name': 'Aggressive Scaling',
            'config': {
                'worker_pool_size': 4,
                'scaling_strategy': ScalingStrategy.AGGRESSIVE,
                'load_balancing': LoadBalancingStrategy.ADAPTIVE
            }
        }
    ]
    
    for config_info in configs:
        print(f"\\n🔧 Testing {config_info['name']}")
        print("-" * 50)
        
        planner = ScalableQuantumPlanner(**config_info['config'])
        
        # Test single assignment
        agents = [
            {'id': 'alice', 'skills': ['python', 'ml'], 'capacity': 3},
            {'id': 'bob', 'skills': ['javascript', 'react'], 'capacity': 2},
            {'id': 'charlie', 'skills': ['python', 'devops'], 'capacity': 2}
        ]
        
        tasks = [
            {'id': 'backend_api', 'skills': ['python'], 'priority': 5, 'duration': 2},
            {'id': 'react_frontend', 'skills': ['javascript', 'react'], 'priority': 3, 'duration': 3},
            {'id': 'ml_model', 'skills': ['python', 'ml'], 'priority': 8, 'duration': 4},
            {'id': 'deployment', 'skills': ['devops'], 'priority': 6, 'duration': 1}
        ]
        
        # Single assignment test
        result = planner.assign_tasks_scalable(agents, tasks, minimize="time")
        
        if result['success']:
            print(f"✅ Single assignment: {result['performance']['total_time']:.3f}s")
            print(f"   Worker used: {result['performance']['worker_used']}")
            print(f"   Cache hit rate: {result['performance']['cache_hit_rate']:.1%}")
            
            # Test cache hit
            result2 = planner.assign_tasks_scalable(agents, tasks, minimize="time")
            if result2.get('cache_hit'):
                print(f"✅ Cache hit: {result2['cache_retrieval_time']:.3f}s")
        
        # Batch assignment test
        print("\\n📦 Testing batch assignment...")
        
        batch_requests = []
        for i in range(5):
            # Create slightly different problems
            modified_tasks = tasks.copy()
            modified_tasks[0]['priority'] = 5 + i  # Vary priority
            
            batch_requests.append({
                'agents': agents,
                'tasks': modified_tasks,
                'minimize': 'time'
            })
        
        batch_start = time.time()
        batch_results = planner.batch_assign_tasks(batch_requests)
        batch_time = time.time() - batch_start
        
        successful_batches = sum(1 for r in batch_results if r.get('success', False))
        print(f"✅ Batch processing: {successful_batches}/5 successful in {batch_time:.3f}s")
        print(f"   Average per request: {batch_time/len(batch_requests):.3f}s")
        
        # Performance stress test  
        print("\\n⚡ Performance stress test...")
        
        stress_times = []
        for i in range(10):
            start = time.time()
            result = planner.assign_tasks_scalable(agents, tasks[:2], minimize="time")  # Smaller problem
            if result['success']:
                stress_times.append(time.time() - start)
        
        if stress_times:
            avg_time = statistics.mean(stress_times)
            min_time = min(stress_times)
            max_time = max(stress_times)
            print(f"   10 assignments - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        # System status
        status = planner.get_comprehensive_status()
        print(f"\\n📊 System Status:")
        print(f"   Overall Health: {status['overall_health']}")
        print(f"   Workers: {status['configuration']['worker_pool_size']}")
        print(f"   Total Requests: {status['performance']['total_requests']}")
        print(f"   Cache Hit Rate: {status['performance']['cache_hit_rate']:.1%}")
        
        # Configuration recommendations
        recommendations = planner.optimize_configuration()
        if recommendations['recommendations']:
            print(f"\\n💡 Recommendations:")
            for key, rec in recommendations['recommendations'].items():
                print(f"   {key}: {rec}")
        
        time.sleep(1)  # Let metrics update
    
    print("\\n🎯 Scalable Quantum Task Planner Demo Complete!")


if __name__ == "__main__":
    demo_scalable_usage()
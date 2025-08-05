"""Performance optimization features for quantum task planner."""

import time
import threading
import functools
import hashlib
import json
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import weakref
import pickle

from .models import Agent, Task, Solution

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    result: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0


class LRUCache:
    """Thread-safe LRU cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        self._hits = 0
        self._misses = 0
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            return len(str(obj)) * 2
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        # Create a deterministic key from args and kwargs
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.result
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return
            
            # Remove old entry if exists
            if key in self._cache:
                self._remove_entry(key)
            
            # Make room if needed
            while (len(self._cache) >= self.max_size or 
                   self._current_memory + size_bytes > self.max_memory_bytes):
                if not self._cache:
                    break
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                result=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._current_memory += size_bytes
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            del self._cache[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'memory_utilization': self._current_memory / self.max_memory_bytes
            }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
        self.thread_pools: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self.optimization_enabled = True
        self._lock = threading.Lock()
        
        # Default caches
        self.get_cache('solutions', max_size=500, ttl_seconds=1800)  # 30 min for solutions
        self.get_cache('problem_analysis', max_size=1000, ttl_seconds=3600)  # 1 hour for analysis
        self.get_cache('backend_selection', max_size=100, ttl_seconds=300)  # 5 min for backend decisions
    
    def get_cache(self, name: str, **kwargs) -> LRUCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self.caches:
                self.caches[name] = LRUCache(**kwargs)
            return self.caches[name]
    
    def get_thread_pool(self, name: str, max_workers: int = 4) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create a named thread pool."""
        with self._lock:
            if name not in self.thread_pools:
                self.thread_pools[name] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=f"qtp-{name}"
                )
            return self.thread_pools[name]
    
    def cached(self, cache_name: str = 'default', ttl_seconds: float = 3600.0):
        """Decorator to cache function results."""
        def decorator(func: Callable) -> Callable:
            cache = self.get_cache(cache_name, ttl_seconds=ttl_seconds)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return func(*args, **kwargs)
                
                # Create cache key
                key = cache._make_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = cache.get(key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Store in cache
                cache.put(key, result)
                logger.debug(f"Cache miss for {func.__name__}, result cached")
                
                return result
            
            return wrapper
        return decorator
    
    def parallel(self, pool_name: str = 'default', max_workers: int = 4):
        """Decorator to run function in thread pool."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return func(*args, **kwargs)
                
                pool = self.get_thread_pool(pool_name, max_workers)
                future = pool.submit(func, *args, **kwargs)
                return future.result()  # Block and return result
            
            return wrapper
        return decorator
    
    def memoize_problem_analysis(self, agents: List[Agent], tasks: List[Task]) -> Dict[str, Any]:
        """Memoized problem analysis for optimization decisions."""
        @self.cached('problem_analysis', ttl_seconds=3600)
        def _analyze_problem(agent_hash: str, task_hash: str, num_agents: int, num_tasks: int):
            """Analyze problem characteristics for optimization."""
            analysis = {
                'problem_size': num_agents * num_tasks,
                'num_agents': num_agents,
                'num_tasks': num_tasks,
                'complexity_score': self._calculate_complexity(agents, tasks),
                'skill_diversity': self._calculate_skill_diversity(agents, tasks),
                'load_balance_potential': self._calculate_load_balance_potential(agents, tasks),
                'optimal_backend': self._suggest_optimal_backend(num_agents, num_tasks),
                'timestamp': time.time()
            }
            return analysis
        
        # Create stable hashes for agents and tasks
        agent_hash = self._hash_agents(agents)
        task_hash = self._hash_tasks(tasks)
        
        return _analyze_problem(agent_hash, task_hash, len(agents), len(tasks))
    
    def _hash_agents(self, agents: List[Agent]) -> str:
        """Create stable hash for list of agents."""
        agent_data = []
        for agent in agents:
            agent_data.append({
                'id': agent.agent_id,
                'skills': sorted(agent.skills),
                'capacity': agent.capacity
            })
        return hashlib.md5(json.dumps(agent_data, sort_keys=True).encode()).hexdigest()
    
    def _hash_tasks(self, tasks: List[Task]) -> str:
        """Create stable hash for list of tasks."""
        task_data = []
        for task in tasks:
            task_data.append({
                'id': task.task_id,
                'skills': sorted(task.required_skills),
                'priority': task.priority,
                'duration': task.duration
            })
        return hashlib.md5(json.dumps(task_data, sort_keys=True).encode()).hexdigest()
    
    def _calculate_complexity(self, agents: List[Agent], tasks: List[Task]) -> float:
        """Calculate problem complexity score."""
        # Factors that increase complexity
        skill_mismatch_penalty = 0
        total_combinations = 0
        
        for task in tasks:
            compatible_agents = sum(1 for agent in agents if task.can_be_assigned_to(agent))
            total_combinations += compatible_agents
            if compatible_agents == 0:
                skill_mismatch_penalty += 10
            elif compatible_agents == 1:
                skill_mismatch_penalty += 5
        
        avg_compatibility = total_combinations / len(tasks) if tasks else 1
        complexity = len(agents) * len(tasks) / avg_compatibility + skill_mismatch_penalty
        
        return min(complexity, 100.0)  # Normalize to 0-100
    
    def _calculate_skill_diversity(self, agents: List[Agent], tasks: List[Task]) -> float:
        """Calculate skill diversity score."""
        all_agent_skills = set()
        all_task_skills = set()
        
        for agent in agents:
            all_agent_skills.update(agent.skills)
        
        for task in tasks:
            all_task_skills.update(task.required_skills)
        
        if not all_task_skills:
            return 1.0
        
        coverage = len(all_agent_skills & all_task_skills) / len(all_task_skills)
        return coverage
    
    def _calculate_load_balance_potential(self, agents: List[Agent], tasks: List[Task]) -> float:
        """Calculate potential for load balancing."""
        if not agents or not tasks:
            return 1.0
        
        total_capacity = sum(agent.capacity for agent in agents)
        total_work = sum(task.duration for task in tasks)
        
        if total_capacity == 0:
            return 0.0
        
        utilization = total_work / total_capacity
        
        # Best load balance when utilization is high but not over capacity
        if utilization > 1.0:
            return max(0.0, 2.0 - utilization)  # Penalty for over-utilization
        else:
            return utilization  # Higher is better up to 1.0
    
    def _suggest_optimal_backend(self, num_agents: int, num_tasks: int) -> str:
        """Suggest optimal backend based on problem size."""
        problem_size = num_agents * num_tasks
        
        if problem_size < 10:
            return "classical"
        elif problem_size < 50:
            return "quantum"
        elif problem_size < 200:
            return "hybrid"
        else:
            return "decomposition"
    
    def optimize_solution_search(self, 
                                agents: List[Agent], 
                                tasks: List[Task],
                                objective: str = "minimize_makespan") -> Optional[Solution]:
        """Try to find cached solution or similar solution that can be adapted."""
        
        # Check for exact match first
        cache = self.get_cache('solutions')
        exact_key = cache._make_key('solution', 
                                  self._hash_agents(agents), 
                                  self._hash_tasks(tasks), 
                                  objective)
        
        exact_solution = cache.get(exact_key)
        if exact_solution:
            logger.debug("Found exact cached solution")
            return exact_solution
        
        # Look for similar solutions (same problem size, similar skill requirements)
        similar_solution = self._find_similar_solution(agents, tasks, objective)
        if similar_solution:
            logger.debug("Found similar cached solution")
            return similar_solution
        
        return None
    
    def _find_similar_solution(self, 
                              agents: List[Agent], 
                              tasks: List[Task],
                              objective: str) -> Optional[Solution]:
        """Find similar cached solution that might be adaptable."""
        # This is a simplified version - in practice, would implement
        # more sophisticated similarity matching
        cache = self.get_cache('solutions')
        
        # For now, just check if we have solutions for same problem size
        problem_size = len(agents) * len(tasks)
        
        with cache._lock:
            for key, entry in cache._cache.items():
                if 'solution' in key and hasattr(entry.result, 'metadata'):
                    cached_size = entry.result.metadata.get('problem_size', 0)
                    if cached_size == problem_size:
                        # Could implement more sophisticated matching here
                        logger.debug(f"Found similar solution with same problem size: {problem_size}")
                        return entry.result
        
        return None
    
    def cache_solution(self, 
                      agents: List[Agent], 
                      tasks: List[Task],
                      objective: str,
                      solution: Solution) -> None:
        """Cache a solution for future use."""
        cache = self.get_cache('solutions')
        key = cache._make_key('solution', 
                            self._hash_agents(agents), 
                            self._hash_tasks(tasks), 
                            objective)
        
        # Add metadata to solution
        if not solution.metadata:
            solution.metadata = {}
        solution.metadata['cached_at'] = time.time()
        solution.metadata['problem_size'] = len(agents) * len(tasks)
        
        cache.put(key, solution)
        logger.debug(f"Cached solution for problem size {len(agents)}x{len(tasks)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'optimization_enabled': self.optimization_enabled,
            'caches': {},
            'thread_pools': {},
            'memory_usage': self._get_memory_usage()
        }
        
        # Cache statistics
        for name, cache in self.caches.items():
            stats['caches'][name] = cache.get_stats()
        
        # Thread pool statistics
        for name, pool in self.thread_pools.items():
            stats['thread_pools'][name] = {
                'max_workers': pool._max_workers,
                'active_threads': len(pool._threads) if hasattr(pool, '_threads') else 0
            }
        
        return stats
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("All caches cleared")
    
    def shutdown(self) -> None:
        """Shutdown thread pools and cleanup resources."""
        for pool in self.thread_pools.values():
            pool.shutdown(wait=True)
        self.thread_pools.clear()
        logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
performance = PerformanceOptimizer()


def optimize_performance(func: Callable) -> Callable:
    """Decorator to add performance optimizations to functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log slow operations
            if execution_time > 5.0:
                logger.warning(f"Slow operation detected: {func.__name__} took {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Operation {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper
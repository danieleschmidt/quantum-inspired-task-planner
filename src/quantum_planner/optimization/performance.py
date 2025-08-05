"""Performance optimization and scaling utilities for quantum task planning."""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import heapq
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Caching settings
    enable_caching: bool = True
    max_cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # Parallelization settings
    enable_parallel: bool = True
    max_workers: int = 4
    parallel_threshold: int = 100  # problem size threshold
    
    # Problem decomposition
    enable_decomposition: bool = True
    max_subproblem_size: int = 50
    decomposition_overlap: float = 0.1
    
    # Load balancing
    enable_load_balancing: bool = True
    backend_weights: Dict[str, float] = field(default_factory=dict)
    
    # Memory management
    max_memory_mb: int = 1024
    gc_threshold: int = 100  # number of operations before GC


@dataclass
class ProblemStats:
    """Statistics about a problem for optimization decisions."""
    
    num_variables: int
    num_constraints: int
    density: float  # constraint density
    complexity_score: float
    estimated_solve_time: float
    memory_requirement: int  # in MB


class IntelligentCache:
    """Intelligent caching system with TTL and adaptive eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, access_count)
        self._access_times = deque()  # (key, timestamp) pairs
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp, access_count = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            
            # Update access count and time
            self._cache[key] = (value, timestamp, access_count + 1)
            self._access_times.append((key, time.time()))
            
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            # Remove expired items
            self._cleanup_expired()
            
            # Evict if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = (value, current_time, 0)
            self._access_times.append((key, current_time))
    
    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp, _) in self._cache.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Find LRU item based on access times
        while self._access_times:
            old_key, _ = self._access_times.popleft()
            if old_key in self._cache:
                del self._cache[old_key]
                break
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": self._calculate_hit_rate(),
                "memory_estimate": self._estimate_memory_usage()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Simplified calculation based on access counts
        if not self._cache:
            return 0.0
        
        total_accesses = sum(access_count for _, _, access_count in self._cache.values())
        return min(1.0, total_accesses / max(1, len(self._cache)))
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate
        return len(self._cache) * 1024  # 1KB per entry estimate


class ProblemDecomposer:
    """Decompose large problems into smaller subproblems."""
    
    def __init__(self, max_subproblem_size: int = 50, overlap: float = 0.1):
        self.max_subproblem_size = max_subproblem_size
        self.overlap = overlap
    
    def should_decompose(self, problem_stats: ProblemStats) -> bool:
        """Check if problem should be decomposed."""
        return (problem_stats.num_variables > self.max_subproblem_size or
                problem_stats.complexity_score > 100.0)
    
    def decompose(self, Q: Dict[Tuple[int, int], float], 
                  agents: List[Any], tasks: List[Any]) -> List[Dict[str, Any]]:
        """Decompose problem into subproblems."""
        
        # Get all variables
        variables = set()
        for i, j in Q.keys():
            variables.update([i, j])
        
        if len(variables) <= self.max_subproblem_size:
            return [{"Q": Q, "agents": agents, "tasks": tasks, "variables": list(variables)}]
        
        # Simple partitioning by variable groups
        var_list = sorted(variables)
        subproblems = []
        
        start_idx = 0
        while start_idx < len(var_list):
            end_idx = min(start_idx + self.max_subproblem_size, len(var_list))
            
            # Add overlap
            if end_idx < len(var_list):
                overlap_size = int(self.max_subproblem_size * self.overlap)
                end_idx = min(end_idx + overlap_size, len(var_list))
            
            subproblem_vars = set(var_list[start_idx:end_idx])
            
            # Extract relevant Q entries
            sub_Q = {
                (i, j): coeff for (i, j), coeff in Q.items()
                if i in subproblem_vars and j in subproblem_vars
            }
            
            # Simple agent/task distribution (could be more sophisticated)
            num_agents = len(agents)
            num_tasks = len(tasks)
            agent_slice = slice(
                start_idx * num_agents // len(var_list),
                end_idx * num_agents // len(var_list)
            )
            task_slice = slice(
                start_idx * num_tasks // len(var_list),
                end_idx * num_tasks // len(var_list)
            )
            
            subproblems.append({
                "Q": sub_Q,
                "agents": agents[agent_slice],
                "tasks": tasks[task_slice],
                "variables": list(subproblem_vars),
                "var_range": (start_idx, end_idx)
            })
            
            start_idx = end_idx - int(self.max_subproblem_size * self.overlap)
        
        logger.info(f"Decomposed problem into {len(subproblems)} subproblems")
        return subproblems
    
    def merge_solutions(self, subproblem_solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge solutions from subproblems."""
        
        merged_assignments = {}
        total_energy = 0.0
        max_makespan = 0.0
        
        for solution in subproblem_solutions:
            if solution.get("success", False):
                assignments = solution.get("assignments", {})
                merged_assignments.update(assignments)
                
                total_energy += solution.get("energy", 0.0)
                max_makespan = max(max_makespan, solution.get("makespan", 0.0))
        
        return {
            "assignments": merged_assignments,
            "energy": total_energy,
            "makespan": max_makespan,
            "success": len(merged_assignments) > 0,
            "method": "decomposed"
        }


class LoadBalancer:
    """Intelligent load balancer for multiple backends."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self._backend_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "success_rate": 1.0,
            "avg_time": 1.0,
            "queue_length": 0.0,
            "last_update": time.time()
        })
        self._lock = threading.RLock()
    
    def select_backend(self, available_backends: List[str], 
                      problem_stats: ProblemStats) -> str:
        """Select optimal backend for given problem."""
        
        if len(available_backends) == 1:
            return available_backends[0]
        
        with self._lock:
            scores = {}
            
            for backend in available_backends:
                score = self._calculate_backend_score(backend, problem_stats)
                scores[backend] = score
            
            # Select backend with highest score
            best_backend = max(scores.keys(), key=lambda b: scores[b])
            
            logger.debug(f"Backend selection scores: {scores}")
            logger.debug(f"Selected backend: {best_backend}")
            
            return best_backend
    
    def update_backend_performance(self, backend: str, solve_time: float, 
                                 success: bool, queue_length: int = 0) -> None:
        """Update backend performance statistics."""
        
        with self._lock:
            stats = self._backend_stats[backend]
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            stats["success_rate"] = (alpha * current_success + 
                                   (1 - alpha) * stats["success_rate"])
            
            # Update average time
            if success:
                stats["avg_time"] = (alpha * solve_time + 
                                   (1 - alpha) * stats["avg_time"])
            
            stats["queue_length"] = queue_length
            stats["last_update"] = time.time()
    
    def _calculate_backend_score(self, backend: str, 
                               problem_stats: ProblemStats) -> float:
        """Calculate score for backend selection."""
        
        stats = self._backend_stats[backend]
        
        # Base score from success rate
        score = stats["success_rate"] * 100
        
        # Penalize slow backends
        if stats["avg_time"] > 0:
            time_penalty = min(50, 10 / stats["avg_time"])  # Faster is better
            score += time_penalty
        
        # Penalize busy backends
        queue_penalty = max(0, 20 - stats["queue_length"] * 2)
        score += queue_penalty
        
        # Apply configured weights
        backend_weight = self.config.backend_weights.get(backend, 1.0)
        score *= backend_weight
        
        # Problem-specific adjustments
        if problem_stats.complexity_score > 50:
            # Prefer quantum backends for complex problems
            if "quantum" in backend.lower() or "dwave" in backend.lower():
                score *= 1.2
        else:
            # Prefer classical backends for simple problems
            if "classical" in backend.lower() or "simulated" in backend.lower():
                score *= 1.1
        
        return score
    
    def get_backend_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current backend statistics."""
        with self._lock:
            return dict(self._backend_stats)


class ParallelSolver:
    """Parallel problem solver with work distribution."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.cache = IntelligentCache(
            max_size=config.max_cache_size if config else 1000,
            ttl=config.cache_ttl if config else 3600
        )
        self.decomposer = ProblemDecomposer(
            max_subproblem_size=config.max_subproblem_size if config else 50,
            overlap=config.decomposition_overlap if config else 0.1
        )
        self.load_balancer = LoadBalancer(config)
    
    def solve_parallel(self, problems: List[Dict[str, Any]], 
                      solve_func: Callable, 
                      available_backends: List[str]) -> List[Dict[str, Any]]:
        """Solve multiple problems in parallel."""
        
        if not self.config.enable_parallel or len(problems) == 1:
            # Solve sequentially
            return [self._solve_single(problem, solve_func, available_backends) 
                   for problem in problems]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all problems
            future_to_problem = {
                executor.submit(
                    self._solve_single, problem, solve_func, available_backends
                ): problem 
                for problem in problems
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_problem):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel solve failed: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "assignments": {}
                    })
        
        return results
    
    def solve_with_optimization(self, Q: Dict[Tuple[int, int], float], 
                              agents: List[Any], tasks: List[Any],
                              solve_func: Callable,
                              available_backends: List[str]) -> Dict[str, Any]:
        """Solve problem with full optimization pipeline."""
        
        # Generate cache key
        cache_key = self._generate_cache_key(Q, agents, tasks)
        
        # Check cache first
        if self.config.enable_caching:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Cache hit for problem")
                return cached_result
        
        # Analyze problem
        problem_stats = self._analyze_problem(Q, agents, tasks)
        
        # Decide on approach
        if (self.config.enable_decomposition and 
            self.decomposer.should_decompose(problem_stats)):
            
            logger.info("Using problem decomposition")
            result = self._solve_with_decomposition(
                Q, agents, tasks, solve_func, available_backends
            )
        else:
            logger.info("Using direct solving")
            backend = self.load_balancer.select_backend(available_backends, problem_stats)
            result = self._solve_direct(Q, agents, tasks, solve_func, backend)
        
        # Cache result
        if self.config.enable_caching and result.get("success", False):
            self.cache.put(cache_key, result)
        
        return result
    
    def _solve_single(self, problem: Dict[str, Any], solve_func: Callable,
                     available_backends: List[str]) -> Dict[str, Any]:
        """Solve a single problem."""
        
        Q = problem["Q"]
        agents = problem.get("agents", [])
        tasks = problem.get("tasks", [])
        
        problem_stats = self._analyze_problem(Q, agents, tasks)
        backend = self.load_balancer.select_backend(available_backends, problem_stats)
        
        return self._solve_direct(Q, agents, tasks, solve_func, backend)
    
    def _solve_with_decomposition(self, Q: Dict[Tuple[int, int], float],
                                agents: List[Any], tasks: List[Any],
                                solve_func: Callable,
                                available_backends: List[str]) -> Dict[str, Any]:
        """Solve using problem decomposition."""
        
        # Decompose problem
        subproblems = self.decomposer.decompose(Q, agents, tasks)
        
        # Solve subproblems in parallel
        subproblem_results = self.solve_parallel(
            subproblems, solve_func, available_backends
        )
        
        # Merge results
        final_result = self.decomposer.merge_solutions(subproblem_results)
        
        return final_result
    
    def _solve_direct(self, Q: Dict[Tuple[int, int], float], agents: List[Any],
                     tasks: List[Any], solve_func: Callable, backend: str) -> Dict[str, Any]:
        """Solve problem directly."""
        
        start_time = time.time()
        
        try:
            result = solve_func(Q, agents, tasks, backend)
            solve_time = time.time() - start_time
            
            # Update load balancer
            self.load_balancer.update_backend_performance(
                backend, solve_time, result.get("success", False)
            )
            
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            
            # Update load balancer with failure
            self.load_balancer.update_backend_performance(
                backend, solve_time, False
            )
            
            return {
                "success": False,
                "error": str(e),
                "assignments": {},
                "solve_time": solve_time,
                "backend": backend
            }
    
    def _analyze_problem(self, Q: Dict[Tuple[int, int], float], 
                        agents: List[Any], tasks: List[Any]) -> ProblemStats:
        """Analyze problem characteristics."""
        
        variables = set()
        for i, j in Q.keys():
            variables.update([i, j])
        
        num_variables = len(variables)
        num_constraints = len(Q)
        
        # Calculate density
        max_possible_constraints = num_variables * num_variables
        density = num_constraints / max(1, max_possible_constraints)
        
        # Simple complexity score
        complexity_score = num_variables * density * len(tasks)
        
        # Rough time estimation
        estimated_time = 0.1 * (num_variables ** 1.2)
        
        # Memory estimate (rough)
        memory_mb = max(1, num_variables * num_constraints // 1000)
        
        return ProblemStats(
            num_variables=num_variables,
            num_constraints=num_constraints,
            density=density,
            complexity_score=complexity_score,
            estimated_solve_time=estimated_time,
            memory_requirement=memory_mb
        )
    
    def _generate_cache_key(self, Q: Dict[Tuple[int, int], float],
                          agents: List[Any], tasks: List[Any]) -> str:
        """Generate cache key for problem."""
        
        # Create deterministic representation
        q_str = str(sorted(Q.items()))
        agents_str = str([(getattr(a, 'id', str(a)), getattr(a, 'skills', [])) for a in agents])
        tasks_str = str([(getattr(t, 'id', str(t)), getattr(t, 'required_skills', [])) for t in tasks])
        
        combined = q_str + agents_str + tasks_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        return {
            "cache": self.cache.stats(),
            "backends": self.load_balancer.get_backend_stats(),
            "config": {
                "parallel_enabled": self.config.enable_parallel,
                "caching_enabled": self.config.enable_caching,
                "decomposition_enabled": self.config.enable_decomposition,
                "max_workers": self.config.max_workers
            }
        }
"""Optimized quantum task planner with performance and scaling features."""

from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass, field

from .models import Agent, Task, TimeWindowTask, Solution
from .planner import QuantumTaskPlanner, PlannerConfig
from .optimization.performance import (
    PerformanceConfig, 
    ParallelSolver,
    ProblemStats
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizedPlannerConfig(PlannerConfig):
    """Extended configuration for optimized planner."""
    
    # Performance optimization settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Auto-scaling settings
    auto_scale_backends: bool = True
    scale_threshold_agents: int = 20
    scale_threshold_tasks: int = 50
    
    # Quality vs speed trade-offs
    quality_mode: str = "balanced"  # "speed", "balanced", "quality"
    max_solve_time: int = 300  # seconds
    
    # Monitoring and logging
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True


class OptimizedQuantumTaskPlanner(QuantumTaskPlanner):
    """Optimized quantum task planner with advanced performance features.
    
    This planner includes:
    - Intelligent caching with TTL
    - Problem decomposition for large problems
    - Parallel solving capabilities  
    - Load balancing across backends
    - Performance monitoring and optimization
    - Auto-scaling backend selection
    
    Examples:
        High-performance setup:
        >>> config = OptimizedPlannerConfig(
        ...     backend="auto",
        ...     performance=PerformanceConfig(
        ...         enable_parallel=True,
        ...         max_workers=8,
        ...         enable_decomposition=True
        ...     ),
        ...     quality_mode="speed"
        ... )
        >>> planner = OptimizedQuantumTaskPlanner(config=config)
        
        Quality-focused setup:
        >>> config = OptimizedPlannerConfig(
        ...     backend="dwave",
        ...     quality_mode="quality",
        ...     performance=PerformanceConfig(
        ...         enable_caching=True,
        ...         max_cache_size=2000
        ...     )
        ... )
        >>> planner = OptimizedQuantumTaskPlanner(config=config)
    """
    
    def __init__(self, config: Optional[OptimizedPlannerConfig] = None, **kwargs):
        """Initialize optimized planner."""
        
        self.optimized_config = config or OptimizedPlannerConfig()
        
        # Initialize base planner
        super().__init__(
            backend=self.optimized_config.backend,
            fallback=self.optimized_config.fallback,
            config=self.optimized_config,
            **kwargs
        )
        
        # Initialize performance optimization
        self.parallel_solver = ParallelSolver(self.optimized_config.performance)
        
        # Performance tracking
        self._solve_history: List[Dict[str, Any]] = []
        self._performance_stats = {
            "total_solves": 0,
            "cache_hits": 0,
            "parallel_solves": 0,
            "decomposed_solves": 0,
            "avg_solve_time": 0.0
        }
    
    def assign(self, agents: List[Agent], tasks: List[Task], 
               objective: str = "minimize_makespan",
               constraints: Optional[Dict[str, Any]] = None,
               optimization_mode: str = None,
               **kwargs) -> Solution:
        """Optimized task assignment with performance features.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to assign
            objective: Optimization objective
            constraints: Dictionary of constraints
            optimization_mode: Override quality mode ("speed", "balanced", "quality")
            **kwargs: Additional optimization parameters
            
        Returns:
            Solution with optimal assignments and performance metadata
        """
        
        import time
        start_time = time.time()
        
        # Determine optimization mode
        mode = optimization_mode or self.optimized_config.quality_mode
        
        # Adapt configuration based on mode
        self._adapt_config_for_mode(mode)
        
        # Check if we should use optimized solving
        problem_size = len(agents) * len(tasks)
        use_optimization = self._should_use_optimization(problem_size, mode)
        
        if use_optimization:
            logger.info(f"Using optimized solving for {len(agents)} agents, {len(tasks)} tasks")
            solution = self._solve_optimized(agents, tasks, objective, constraints, **kwargs)
        else:
            logger.info("Using standard solving approach")
            solution = super().assign(agents, tasks, objective, constraints, **kwargs)
        
        # Add performance metadata
        solve_time = time.time() - start_time
        self._update_performance_stats(solution, solve_time, use_optimization)
        
        return solution
    
    def assign_batch(self, batch_problems: List[Dict[str, Any]], 
                    optimization_mode: str = None) -> List[Solution]:
        """Assign multiple problems in batch with optimization.
        
        Args:
            batch_problems: List of problem dictionaries with 'agents', 'tasks', etc.
            optimization_mode: Override quality mode
            
        Returns:
            List of solutions for each problem
        """
        
        if not batch_problems:
            return []
        
        import time
        start_time = time.time()
        
        mode = optimization_mode or self.optimized_config.quality_mode
        self._adapt_config_for_mode(mode)
        
        logger.info(f"Solving batch of {len(batch_problems)} problems")
        
        # Convert to internal format for parallel solver
        internal_problems = []
        for problem in batch_problems:
            internal_problems.append({
                "Q": self._build_qubo_matrix(
                    problem["agents"], 
                    problem["tasks"], 
                    problem.get("constraints", {})
                ),
                "agents": problem["agents"],
                "tasks": problem["tasks"],
                "objective": problem.get("objective", "minimize_makespan"),
                "constraints": problem.get("constraints", {})
            })
        
        # Get available backends
        available_backends = self._get_available_backends()
        
        # Solve in parallel
        def solve_func(Q, agents, tasks, backend):
            return self._solve_with_backend(Q, agents, tasks, backend)
        
        results = self.parallel_solver.solve_parallel(
            internal_problems, solve_func, available_backends
        )
        
        # Convert back to Solution objects
        solutions = []
        for result in results:
            solution = self._result_to_solution(result)
            solutions.append(solution)
        
        batch_time = time.time() - start_time
        logger.info(f"Batch solving completed in {batch_time:.3f}s")
        
        return solutions
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Analyze and optimize current performance settings.
        
        Returns:
            Performance analysis and recommendations
        """
        
        # Get current performance stats
        stats = self.parallel_solver.get_performance_stats()
        
        # Analyze solve history
        history_analysis = self._analyze_solve_history()
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(stats, history_analysis)
        
        return {
            "current_stats": stats,
            "history_analysis": history_analysis,
            "recommendations": recommendations,
            "planner_stats": self._performance_stats
        }
    
    def benchmark_backends(self, test_problems: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Benchmark available backends with test problems.
        
        Args:
            test_problems: Optional custom test problems
            
        Returns:
            Benchmark results for all backends
        """
        
        if test_problems is None:
            test_problems = self._generate_test_problems()
        
        available_backends = self._get_available_backends()
        results = {}
        
        for backend_name in available_backends:
            logger.info(f"Benchmarking backend: {backend_name}")
            
            backend_results = {
                "solve_times": [],
                "success_rate": 0.0,
                "avg_energy": 0.0,
                "total_time": 0.0
            }
            
            successful_solves = 0
            total_energy = 0.0
            
            import time
            start_time = time.time()
            
            for problem in test_problems:
                try:
                    Q = self._build_qubo_matrix(
                        problem["agents"], 
                        problem["tasks"], 
                        problem.get("constraints", {})
                    )
                    
                    solve_start = time.time()
                    result = self._solve_with_backend(
                        Q, problem["agents"], problem["tasks"], backend_name
                    )
                    solve_time = time.time() - solve_start
                    
                    backend_results["solve_times"].append(solve_time)
                    
                    if result.get("success", False):
                        successful_solves += 1
                        total_energy += result.get("energy", 0.0)
                    
                except Exception as e:
                    logger.warning(f"Backend {backend_name} failed on test problem: {e}")
                    continue
            
            backend_results["total_time"] = time.time() - start_time
            backend_results["success_rate"] = successful_solves / len(test_problems)
            
            if successful_solves > 0:
                backend_results["avg_energy"] = total_energy / successful_solves
                backend_results["avg_solve_time"] = sum(backend_results["solve_times"]) / len(backend_results["solve_times"])
            
            results[backend_name] = backend_results
        
        return results
    
    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report.
        
        Returns:
            Formatted performance report string
        """
        
        stats = self.optimize_performance()
        
        report = []
        report.append("🚀 Optimized Quantum Task Planner - Performance Report")
        report.append("=" * 60)
        
        # Planner statistics
        planner_stats = stats["planner_stats"]
        report.append(f"\n📊 Planner Statistics:")
        report.append(f"  Total solves: {planner_stats['total_solves']}")
        report.append(f"  Cache hits: {planner_stats['cache_hits']}")
        report.append(f"  Parallel solves: {planner_stats['parallel_solves']}")
        report.append(f"  Decomposed solves: {planner_stats['decomposed_solves']}")
        report.append(f"  Average solve time: {planner_stats['avg_solve_time']:.3f}s")
        
        # Cache performance
        cache_stats = stats["current_stats"]["cache"]
        report.append(f"\n💾 Cache Performance:")
        report.append(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
        report.append(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        report.append(f"  Memory estimate: {cache_stats['memory_estimate']} bytes")
        
        # Backend performance
        backend_stats = stats["current_stats"]["backends"]
        report.append(f"\n🔧 Backend Performance:")
        for backend, backend_data in backend_stats.items():
            report.append(f"  {backend}:")
            report.append(f"    Success rate: {backend_data['success_rate']:.1%}")
            report.append(f"    Avg time: {backend_data['avg_time']:.3f}s")
            report.append(f"    Queue length: {backend_data['queue_length']}")
        
        # Recommendations
        recommendations = stats["recommendations"]
        if recommendations:
            report.append(f"\n💡 Recommendations:")
            for rec in recommendations:
                report.append(f"  • {rec}")
        
        return "\n".join(report)
    
    # Private methods
    
    def _should_use_optimization(self, problem_size: int, mode: str) -> bool:
        """Determine if optimized solving should be used."""
        
        # Always use optimization for large problems
        if problem_size > self.optimized_config.scale_threshold_agents * self.optimized_config.scale_threshold_tasks:
            return True
        
        # Use optimization based on mode
        if mode == "speed":
            return True  # Always optimize for speed
        elif mode == "quality":
            return problem_size > 10  # Use for medium+ problems
        else:  # balanced
            return problem_size > 5  # Use for small+ problems
    
    def _adapt_config_for_mode(self, mode: str) -> None:
        """Adapt configuration based on optimization mode."""
        
        if mode == "speed":
            # Optimize for speed
            self.optimized_config.performance.enable_parallel = True
            self.optimized_config.performance.enable_decomposition = True
            self.optimized_config.performance.max_subproblem_size = 30
            self.optimized_config.performance.enable_caching = True
            
        elif mode == "quality":
            # Optimize for quality
            self.optimized_config.performance.enable_parallel = False
            self.optimized_config.performance.enable_decomposition = False
            self.optimized_config.performance.enable_caching = True
            self.optimized_config.performance.max_cache_size = 2000
            
        else:  # balanced
            # Default balanced settings
            pass
    
    def _solve_optimized(self, agents: List[Agent], tasks: List[Task],
                        objective: str, constraints: Optional[Dict[str, Any]],
                        **kwargs) -> Solution:
        """Solve using optimized approach."""
        
        # Build QUBO matrix
        Q = self._build_qubo_matrix(agents, tasks, constraints or {})
        
        # Get available backends
        available_backends = self._get_available_backends()
        
        # Solve using parallel solver
        def solve_func(Q_inner, agents_inner, tasks_inner, backend):
            return self._solve_with_backend(Q_inner, agents_inner, tasks_inner, backend)
        
        result = self.parallel_solver.solve_with_optimization(
            Q, agents, tasks, solve_func, available_backends
        )
        
        # Convert to Solution object
        return self._result_to_solution(result)
    
    def _build_qubo_matrix(self, agents: List[Agent], tasks: List[Task], 
                          constraints: Dict[str, Any]) -> Dict[tuple, float]:
        """Build QUBO matrix for the problem."""
        
        # Simplified QUBO construction for testing
        # In real implementation, this would use the QUBOBuilder
        
        Q = {}
        num_agents = len(agents)
        num_tasks = len(tasks)
        
        # Simple assignment variables: x[i,j] = 1 if task i assigned to agent j
        for i in range(num_tasks):
            for j in range(num_agents):
                var_idx = i * num_agents + j
                
                # Objective: minimize makespan (simplified)
                task_duration = getattr(tasks[i], 'duration', 1)
                Q[(var_idx, var_idx)] = task_duration
                
                # Constraint: each task assigned to exactly one agent
                for k in range(j + 1, num_agents):
                    other_var = i * num_agents + k
                    Q[(var_idx, other_var)] = 100  # Large penalty
        
        return Q
    
    def _solve_with_backend(self, Q: Dict[tuple, float], agents: List[Agent],
                          tasks: List[Task], backend_name: str) -> Dict[str, Any]:
        """Solve problem with specific backend."""
        
        # This would interface with the actual backend
        # For now, return a mock result
        
        import random
        import time
        
        time.sleep(0.1)  # Simulate solve time
        
        # Generate mock solution
        assignments = {}
        for i, task in enumerate(tasks):
            agent_idx = random.randint(0, len(agents) - 1)
            assignments[task.id] = agents[agent_idx].id
        
        return {
            "success": True,
            "assignments": assignments,
            "energy": random.uniform(0, 100),
            "makespan": random.uniform(1, 10),
            "backend": backend_name,
            "solve_time": 0.1
        }
    
    def _get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        # This would check actual backend availability
        return ["enhanced_simulated_annealing", "enhanced_dwave", "enhanced_azure_quantum"]
    
    def _result_to_solution(self, result: Dict[str, Any]) -> Solution:
        """Convert internal result to Solution object."""
        
        return Solution(
            assignments=result.get("assignments", {}),
            makespan=result.get("makespan", 0.0),
            cost=result.get("energy", 0.0),
            backend_used=result.get("backend", "unknown"),
            metadata=result.get("metadata", {}),
            total_cost=result.get("total_cost")
        )
    
    def _update_performance_stats(self, solution: Solution, solve_time: float, 
                                was_optimized: bool) -> None:
        """Update performance statistics."""
        
        self._performance_stats["total_solves"] += 1
        
        # Update average solve time
        total = self._performance_stats["total_solves"]
        current_avg = self._performance_stats["avg_solve_time"]
        self._performance_stats["avg_solve_time"] = (
            (current_avg * (total - 1) + solve_time) / total
        )
        
        if was_optimized:
            if solution.metadata and solution.metadata.get("cache_hit"):
                self._performance_stats["cache_hits"] += 1
            if solution.metadata and solution.metadata.get("parallel"):
                self._performance_stats["parallel_solves"] += 1
            if solution.metadata and solution.metadata.get("decomposed"):
                self._performance_stats["decomposed_solves"] += 1
        
        # Store solve history (keep last 100)
        self._solve_history.append({
            "solve_time": solve_time,
            "was_optimized": was_optimized,
            "success": bool(solution.assignments),
            "problem_size": len(solution.assignments),
            "backend": solution.backend_used
        })
        
        if len(self._solve_history) > 100:
            self._solve_history.pop(0)
    
    def _analyze_solve_history(self) -> Dict[str, Any]:
        """Analyze solve history for patterns."""
        
        if not self._solve_history:
            return {"message": "No solve history available"}
        
        total_solves = len(self._solve_history)
        successful_solves = sum(1 for h in self._solve_history if h["success"])
        optimized_solves = sum(1 for h in self._solve_history if h["was_optimized"])
        
        avg_time_all = sum(h["solve_time"] for h in self._solve_history) / total_solves
        avg_time_optimized = 0.0
        avg_time_standard = 0.0
        
        if optimized_solves > 0:
            avg_time_optimized = sum(
                h["solve_time"] for h in self._solve_history if h["was_optimized"]
            ) / optimized_solves
        
        standard_solves = total_solves - optimized_solves
        if standard_solves > 0:
            avg_time_standard = sum(
                h["solve_time"] for h in self._solve_history if not h["was_optimized"]
            ) / standard_solves
        
        return {
            "total_solves": total_solves,
            "success_rate": successful_solves / total_solves,
            "optimization_rate": optimized_solves / total_solves,
            "avg_time_all": avg_time_all,
            "avg_time_optimized": avg_time_optimized,
            "avg_time_standard": avg_time_standard,
            "speedup_factor": avg_time_standard / avg_time_optimized if avg_time_optimized > 0 else 1.0
        }
    
    def _generate_performance_recommendations(self, stats: Dict[str, Any], 
                                           history: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # Cache recommendations
        cache_stats = stats["current_stats"]["cache"]
        if cache_stats["hit_rate"] < 0.3:
            recommendations.append("Consider increasing cache size for better hit rate")
        
        # Parallel processing recommendations
        config_stats = stats["current_stats"]["config"]
        if history.get("avg_time_all", 0) > 5.0 and not config_stats["parallel_enabled"]:
            recommendations.append("Enable parallel processing for faster solving")
        
        # Backend recommendations
        backend_stats = stats["current_stats"]["backends"]
        slow_backends = [
            name for name, data in backend_stats.items() 
            if data["avg_time"] > 10.0 and data["success_rate"] < 0.8
        ]
        
        if slow_backends:
            recommendations.append(f"Consider avoiding slow/unreliable backends: {', '.join(slow_backends)}")
        
        # Decomposition recommendations
        if (history.get("avg_time_all", 0) > 10.0 and 
            not config_stats["decomposition_enabled"]):
            recommendations.append("Enable problem decomposition for large problems")
        
        return recommendations
    
    def _generate_test_problems(self) -> List[Dict[str, Any]]:
        """Generate test problems for benchmarking."""
        
        test_problems = []
        
        # Small problem
        agents_small = [Agent(f"agent_{i}", skills=["python"], capacity=1) for i in range(2)]
        tasks_small = [Task(f"task_{i}", required_skills=["python"], duration=1) for i in range(3)]
        
        test_problems.append({
            "agents": agents_small,
            "tasks": tasks_small,
            "constraints": {"skill_match": True}
        })
        
        # Medium problem
        agents_medium = [Agent(f"agent_{i}", skills=["python", "data"], capacity=2) for i in range(5)]
        tasks_medium = [Task(f"task_{i}", required_skills=["python"], duration=2) for i in range(8)]
        
        test_problems.append({
            "agents": agents_medium,
            "tasks": tasks_medium,
            "constraints": {"skill_match": True, "capacity_limit": True}
        })
        
        return test_problems
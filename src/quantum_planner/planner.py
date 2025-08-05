"""High-level quantum task planner interface."""

from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import time

from .models import Agent, Task, TimeWindowTask, Solution
from .optimizer import OptimizationBackend, create_optimizer, optimize_tasks
from .backends.enhanced_base import EnhancedQuantumBackend
from .backends.enhanced_classical import EnhancedSimulatedAnnealingBackend
from .backends.enhanced_quantum import EnhancedDWaveBackend, EnhancedAzureQuantumBackend
from .reliability import reliability_manager, error_context
from .monitoring import monitoring, Timer, monitor_performance, monitor_errors
from .performance import performance, optimize_performance
from .globalization import globalization, with_globalization


logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    """Configuration for the quantum task planner."""
    
    backend: str = "auto"
    fallback: Optional[str] = "simulated_annealing" 
    fallback_threshold: int = 20
    max_solve_time: int = 300
    num_reads: int = 1000
    optimize_embedding: bool = True
    verbose: bool = False


class QuantumTaskPlanner:
    """High-level interface for quantum-inspired task scheduling.
    
    This class provides the main user interface for solving task assignment
    problems using quantum annealing, gate-based quantum computing, or 
    classical optimization algorithms.
    
    Examples:
        Basic usage:
        >>> planner = QuantumTaskPlanner(backend="auto")
        >>> solution = planner.assign(agents, tasks, objective="minimize_makespan")
        
        With specific backend:
        >>> planner = QuantumTaskPlanner(
        ...     backend="dwave", 
        ...     fallback="simulated_annealing"
        ... )
        >>> solution = planner.assign(agents, tasks)
    """
    
    def __init__(
        self,
        backend: str = "auto",
        fallback: Optional[str] = "simulated_annealing",
        config: Optional[PlannerConfig] = None,
        **kwargs
    ):
        """Initialize the quantum task planner.
        
        Args:
            backend: Backend to use ("auto", "dwave", "azure", "ibm", "simulator")
            fallback: Fallback backend for when quantum is unavailable
            config: Optional configuration object
            **kwargs: Additional backend-specific parameters
        """
        self.config = config or PlannerConfig(backend=backend, fallback=fallback)
        self.backend_kwargs = kwargs
        self._backend: Optional[EnhancedQuantumBackend] = None
        self._fallback_backend: Optional[EnhancedQuantumBackend] = None
        
        self._initialize_backends()
    
    def _initialize_backends(self) -> None:
        """Initialize primary and fallback backends."""
        try:
            if self.config.backend == "auto":
                self._backend = self._auto_select_backend()
            else:
                self._backend = self._create_backend(self.config.backend)
                
            if self.config.verbose:
                logger.info(f"Primary backend: {self.config.backend}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize backend {self.config.backend}: {e}")
            self._backend = None
            
        # Initialize fallback
        if self.config.fallback:
            try:
                self._fallback_backend = self._create_backend(self.config.fallback)
                if self.config.verbose:
                    logger.info(f"Fallback backend: {self.config.fallback}")
            except Exception as e:
                logger.warning(f"Failed to initialize fallback {self.config.fallback}: {e}")
                # Default to simulated annealing as last resort
                self._fallback_backend = EnhancedSimulatedAnnealingBackend()
    
    def _auto_select_backend(self) -> EnhancedQuantumBackend:
        """Automatically select the best available backend."""
        # Try quantum backends in order of preference
        backends_to_try = ["dwave", "azure", "ibm", "simulator"]
        
        for backend_name in backends_to_try:
            try:
                backend = self._create_backend(backend_name)
                if backend.is_available():
                    if self.config.verbose:
                        logger.info(f"Auto-selected backend: {backend_name}")
                    return backend
            except Exception as e:
                if self.config.verbose:
                    logger.debug(f"Backend {backend_name} not available: {e}")
                continue
                
        # Fall back to classical
        logger.info("No quantum backends available, using simulated annealing")
        return EnhancedSimulatedAnnealingBackend()
    
    def _create_backend(self, backend_name: str) -> EnhancedQuantumBackend:
        """Create a backend instance by name."""
        if backend_name == "simulated_annealing":
            return EnhancedSimulatedAnnealingBackend(self.backend_kwargs)
        elif backend_name == "dwave":
            return EnhancedDWaveBackend(self.backend_kwargs)
        elif backend_name == "azure":
            return EnhancedAzureQuantumBackend(self.backend_kwargs)
        elif backend_name == "ibm":
            # For now, fall back to simulated annealing for IBM
            logger.warning("IBM backend not yet implemented, using simulated annealing")
            return EnhancedSimulatedAnnealingBackend(self.backend_kwargs)
        elif backend_name == "simulator":
            # For now, fall back to simulated annealing for simulator
            logger.warning("Simulator backend not yet implemented, using simulated annealing") 
            return EnhancedSimulatedAnnealingBackend(self.backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    @with_globalization
    @optimize_performance
    @reliability_manager.retry_with_backoff(max_retries=1, base_delay=0.1)
    @reliability_manager.circuit_breaker(failure_threshold=10, recovery_timeout=5.0)
    @monitor_performance("optimization.duration")
    @monitor_errors("optimization.errors")
    def assign(
        self,
        agents: List[Agent],
        tasks: List[Task],
        objective: str = "minimize_makespan",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Solution:
        """Assign tasks to agents optimally.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to assign
            objective: Optimization objective ("minimize_makespan", "maximize_priority", 
                      "balance_load", "minimize_cost")
            constraints: Dictionary of constraints to apply
            **kwargs: Additional optimization parameters
            
        Returns:
            Solution object with optimal assignments
            
        Examples:
            >>> solution = planner.assign(
            ...     agents=agents,
            ...     tasks=tasks, 
            ...     objective="minimize_makespan",
            ...     constraints={
            ...         "skill_match": True,
            ...         "capacity_limit": True,
            ...         "precedence": {"task2": ["task1"]}
            ...     }
            ... )
        """
        # Input validation with detailed error messages
        if not agents:
            raise ValueError("No agents provided for task assignment")
        if not tasks:
            raise ValueError("No tasks provided for assignment")
        
        # Validate agents
        for i, agent in enumerate(agents):
            if not agent.skills:
                raise ValueError(f"Agent {i} ({agent.agent_id}) has no skills defined")
            if agent.capacity <= 0:
                raise ValueError(f"Agent {i} ({agent.agent_id}) has invalid capacity: {agent.capacity}")
        
        # Validate tasks
        for i, task in enumerate(tasks):
            if not task.required_skills:
                raise ValueError(f"Task {i} ({task.task_id}) has no required skills defined")
            if task.duration <= 0:
                raise ValueError(f"Task {i} ({task.task_id}) has invalid duration: {task.duration}")
        
        # Check if any task can be assigned to any agent
        assignable_tasks = []
        for task in tasks:
            can_assign = any(task.can_be_assigned_to(agent) for agent in agents)
            if can_assign:
                assignable_tasks.append(task)
            else:
                logger.warning(f"Task {task.task_id} cannot be assigned to any available agent")
        
        if not assignable_tasks:
            raise ValueError("No tasks can be assigned to available agents (skill mismatch)")
        
        constraints = constraints or {}
        
        # Record problem metrics
        monitoring.record_gauge("problem.agents", len(agents))
        monitoring.record_gauge("problem.tasks", len(tasks))
        monitoring.record_gauge("problem.assignable_tasks", len(assignable_tasks))
        
        # Performance optimizations: Check for cached solutions
        cached_solution = performance.optimize_solution_search(agents, assignable_tasks, objective)
        if cached_solution:
            logger.info("Using cached solution for similar problem")
            monitoring.record_counter("optimization.cache_hit")
            
            # Update metadata
            cached_solution.metadata = cached_solution.metadata or {}
            cached_solution.metadata["cache_hit"] = True
            cached_solution.metadata["optimization_time"] = time.time()
            
            return cached_solution
        
        # Analyze problem for optimization decisions
        problem_analysis = performance.memoize_problem_analysis(agents, assignable_tasks)
        monitoring.record_gauge("problem.complexity", problem_analysis["complexity_score"])
        monitoring.record_gauge("problem.skill_diversity", problem_analysis["skill_diversity"])
        
        with error_context("task_assignment", reliability_manager):
            # Determine which backend to use
            backend_to_use = self._select_backend_for_problem(len(agents), len(tasks))
            
            try:
                with Timer(monitoring, "backend.optimization_time", 
                          {"backend": self._get_backend_name(backend_to_use)}):
                    
                    # Use the optimize_tasks function from optimizer module
                    solution = optimize_tasks(
                        agents=agents,
                        tasks=assignable_tasks,  # Use only assignable tasks
                        backend=OptimizationBackend.QUANTUM if backend_to_use != self._fallback_backend else OptimizationBackend.CLASSICAL,
                        objective=objective,
                        constraints=constraints,
                        **kwargs
                    )
                
                # Add metadata about which backend was used
                solution.metadata = solution.metadata or {}
                solution.metadata["backend_used"] = self._get_backend_name(backend_to_use)
                solution.metadata["fallback_used"] = backend_to_use == self._fallback_backend
                solution.metadata["problem_size"] = len(agents) * len(tasks)
                solution.metadata["optimization_time"] = time.time()
                
                # Record success metrics
                monitoring.record_counter("optimization.success")
                monitoring.record_gauge("solution.makespan", solution.makespan)
                monitoring.record_gauge("solution.cost", solution.cost)
                monitoring.record_gauge("solution.assignments", len(solution.assignments))
                
                # Validate solution
                self._validate_solution(solution, agents, assignable_tasks)
                
                # Cache the solution for future use
                performance.cache_solution(agents, assignable_tasks, objective, solution)
                monitoring.record_counter("optimization.cache_store")
                
                return solution
                
            except Exception as e:
                logger.error(f"Assignment failed with primary backend: {e}")
                monitoring.record_counter("backend.failure", labels={
                    "backend": self._get_backend_name(backend_to_use),
                    "error_type": type(e).__name__
                })
                
                # Try fallback if available and not already using it
                if self._fallback_backend and backend_to_use != self._fallback_backend:
                    logger.info("Trying fallback backend...")
                    try:
                        with Timer(monitoring, "backend.optimization_time", 
                                  {"backend": self._get_backend_name(self._fallback_backend)}):
                            
                            solution = optimize_tasks(
                                agents=agents,
                                tasks=assignable_tasks,
                                backend=OptimizationBackend.CLASSICAL,
                                objective=objective,
                                constraints=constraints,
                                **kwargs
                            )
                        
                        solution.metadata = solution.metadata or {}
                        solution.metadata["backend_used"] = self._get_backend_name(self._fallback_backend)
                        solution.metadata["fallback_used"] = True
                        solution.metadata["primary_backend_error"] = str(e)
                        solution.metadata["problem_size"] = len(agents) * len(tasks)
                        
                        # Record fallback success
                        monitoring.record_counter("optimization.fallback_success")
                        monitoring.record_gauge("solution.makespan", solution.makespan)
                        monitoring.record_gauge("solution.cost", solution.cost)
                        
                        # Validate solution
                        self._validate_solution(solution, agents, assignable_tasks)
                        
                        # Cache the fallback solution too
                        performance.cache_solution(agents, assignable_tasks, objective, solution)
                        monitoring.record_counter("optimization.cache_store")
                        
                        return solution
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        monitoring.record_counter("optimization.total_failure")
                        raise RuntimeError(f"Both primary and fallback backends failed. Primary: {e}, Fallback: {fallback_error}")
                else:
                    raise
    
    def assign_with_time(
        self,
        agents: List[Agent],
        tasks: List[TimeWindowTask],
        time_horizon: int,
        objective: str = "minimize_makespan",
        **kwargs
    ) -> Solution:
        """Assign tasks with temporal constraints.
        
        Args:
            agents: List of available agents
            tasks: List of time-windowed tasks
            time_horizon: Maximum time horizon for scheduling
            objective: Optimization objective
            **kwargs: Additional parameters
            
        Returns:
            Solution with schedule information
        """
        # Add time horizon to constraints
        constraints = kwargs.get("constraints", {})
        constraints["time_windows"] = True
        constraints["time_horizon"] = time_horizon
        kwargs["constraints"] = constraints
        
        return self.assign(agents, tasks, objective, **kwargs)
    
    def submit_async(
        self,
        agents: List[Agent],
        tasks: List[Task],
        job_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Submit an optimization job asynchronously.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to assign
            job_name: Optional name for the job
            **kwargs: Additional parameters
            
        Returns:
            Job object for tracking status
            
        Note:
            This is a placeholder for async functionality.
            Real implementation would depend on the specific backend.
        """
        # This would need to be implemented based on specific backend capabilities
        raise NotImplementedError("Async submission not yet implemented")
    
    def _select_backend_for_problem(self, num_agents: int, num_tasks: int) -> EnhancedQuantumBackend:
        """Select appropriate backend based on problem size."""
        problem_size = num_agents * num_tasks
        
        # Use fallback for small problems or if primary backend unavailable
        if (problem_size < self.config.fallback_threshold or 
            not self._backend or 
            not self._backend.is_available()):
            
            if self._fallback_backend:
                return self._fallback_backend
            else:
                # Create emergency fallback
                return EnhancedSimulatedAnnealingBackend()
        
        return self._backend
    
    def _get_backend_name(self, backend: EnhancedQuantumBackend) -> str:
        """Get the name of a backend instance."""
        return backend.__class__.__name__.replace("Backend", "").lower()
    
    def get_device_properties(self) -> Dict[str, Any]:
        """Get properties of the current quantum device/simulator."""
        if self._backend:
            try:
                return self._backend.get_device_properties()
            except Exception as e:
                logger.warning(f"Could not get device properties: {e}")
                return {"error": str(e)}
        return {"status": "no_backend"}
    
    def _validate_solution(self, solution: Solution, agents: List[Agent], tasks: List[Task]):
        """Validate that the solution is correct and feasible."""
        # Check all tasks are assigned
        assigned_tasks = set(solution.assignments.keys())
        expected_tasks = {task.task_id for task in tasks}
        
        missing_tasks = expected_tasks - assigned_tasks
        if missing_tasks:
            logger.warning(f"Tasks not assigned: {missing_tasks}")
        
        # Check all assigned agents exist
        assigned_agents = set(solution.assignments.values())
        available_agents = {agent.agent_id for agent in agents}
        invalid_agents = assigned_agents - available_agents
        
        if invalid_agents:
            raise ValueError(f"Solution contains invalid agent assignments: {invalid_agents}")
        
        # Check skill compatibility
        skill_violations = []
        for task_id, agent_id in solution.assignments.items():
            task = next((t for t in tasks if t.task_id == task_id), None)
            agent = next((a for a in agents if a.agent_id == agent_id), None)
            
            if task and agent and not task.can_be_assigned_to(agent):
                skill_violations.append(f"Task {task_id} assigned to incompatible agent {agent_id}")
        
        if skill_violations:
            logger.error(f"Skill compatibility violations: {skill_violations}")
            # Don't raise exception for skill violations in fallback scenarios
            monitoring.record_counter("solution.skill_violations", len(skill_violations))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the planner."""
        health_data = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
            "metrics": {},
            "recent_errors": []
        }
        
        # Check backend availability
        if self._backend:
            try:
                backend_available = self._backend.is_available()
                health_data["components"]["primary_backend"] = {
                    "status": "healthy" if backend_available else "unhealthy",
                    "name": self._get_backend_name(self._backend),
                    "available": backend_available
                }
            except Exception as e:
                health_data["components"]["primary_backend"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Check fallback backend
        if self._fallback_backend:
            try:
                fallback_available = self._fallback_backend.is_available()
                health_data["components"]["fallback_backend"] = {
                    "status": "healthy" if fallback_available else "unhealthy",
                    "name": self._get_backend_name(self._fallback_backend),
                    "available": fallback_available
                }
            except Exception as e:
                health_data["components"]["fallback_backend"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Get reliability metrics
        error_stats = reliability_manager.get_error_statistics()
        performance_metrics = reliability_manager.get_performance_metrics()
        
        health_data["metrics"]["error_rate"] = error_stats.get("error_rate", 0)
        health_data["metrics"]["total_errors"] = error_stats.get("total_errors", 0)
        health_data["metrics"]["performance"] = performance_metrics
        
        # Get recent errors
        recent_errors = reliability_manager.error_history[-5:] if reliability_manager.error_history else []
        health_data["recent_errors"] = [
            {
                "type": error.error_type,
                "message": error.message,
                "severity": error.severity.value,
                "timestamp": error.timestamp
            }
            for error in recent_errors
        ]
        
        # Determine overall status
        if any(comp.get("status") == "error" for comp in health_data["components"].values()):
            health_data["overall_status"] = "critical"
        elif error_stats.get("error_rate", 0) > 0.1:
            health_data["overall_status"] = "degraded"
        elif any(comp.get("status") == "unhealthy" for comp in health_data["components"].values()):
            health_data["overall_status"] = "degraded"
        
        return health_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        performance_stats = performance.get_performance_stats()
        
        # Add planner-specific stats
        performance_stats["planner"] = {
            "backend_name": self._get_backend_name(self._backend) if self._backend else "none",
            "fallback_name": self._get_backend_name(self._fallback_backend) if self._fallback_backend else "none",
            "config": {
                "backend": self.config.backend,
                "fallback": self.config.fallback,
                "fallback_threshold": self.config.fallback_threshold
            }
        }
        
        return performance_stats
    
    def clear_performance_caches(self) -> None:
        """Clear all performance caches."""
        performance.clear_all_caches()
        logger.info("Performance caches cleared")
    
    def estimate_solve_time(self, num_agents: int, num_tasks: int) -> float:
        """Estimate solve time for a problem of given size."""
        backend = self._select_backend_for_problem(num_agents, num_tasks)
        try:
            return backend.estimate_solve_time(num_agents * num_tasks)
        except Exception:
            # Rough estimate based on problem size
            return 0.1 * (num_agents * num_tasks) ** 1.5
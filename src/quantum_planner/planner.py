"""High-level quantum task planner interface."""

from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass, field

from .models import Agent, Task, TimeWindowTask, Solution
from .optimizer import OptimizationBackend, create_optimizer, optimize_tasks
from .backends.enhanced_base import EnhancedQuantumBackend
from .backends.enhanced_classical import EnhancedSimulatedAnnealingBackend
from .backends.enhanced_quantum import EnhancedDWaveBackend, EnhancedAzureQuantumBackend


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
        constraints = constraints or {}
        
        # Determine which backend to use
        backend_to_use = self._select_backend_for_problem(len(agents), len(tasks))
        
        try:
            # Use the optimize_tasks function from optimizer module
            solution = optimize_tasks(
                agents=agents,
                tasks=tasks,
                backend=OptimizationBackend.QUANTUM if backend_to_use != self._fallback_backend else OptimizationBackend.CLASSICAL,
                objective=objective,
                constraints=constraints,
                **kwargs
            )
            
            # Add metadata about which backend was used
            solution.metadata = solution.metadata or {}
            solution.metadata["backend_used"] = self._get_backend_name(backend_to_use)
            solution.metadata["fallback_used"] = backend_to_use == self._fallback_backend
            
            return solution
            
        except Exception as e:
            logger.error(f"Assignment failed with primary backend: {e}")
            
            # Try fallback if available and not already using it
            if self._fallback_backend and backend_to_use != self._fallback_backend:
                logger.info("Trying fallback backend...")
                try:
                    solution = optimize_tasks(
                        agents=agents,
                        tasks=tasks,
                        backend=OptimizationBackend.CLASSICAL,
                        objective=objective,
                        constraints=constraints,
                        **kwargs
                    )
                    
                    solution.metadata = solution.metadata or {}
                    solution.metadata["backend_used"] = self._get_backend_name(self._fallback_backend)
                    solution.metadata["fallback_used"] = True
                    solution.metadata["primary_backend_error"] = str(e)
                    
                    return solution
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
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
    
    def estimate_solve_time(self, num_agents: int, num_tasks: int) -> float:
        """Estimate solve time for a problem of given size."""
        backend = self._select_backend_for_problem(num_agents, num_tasks)
        try:
            return backend.estimate_solve_time(num_agents * num_tasks)
        except Exception:
            # Rough estimate based on problem size
            return 0.1 * (num_agents * num_tasks) ** 1.5
"""Quantum optimization backend for task scheduling."""

from typing import Dict, List, Optional, Any, Protocol
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

from quantum_planner.models import Agent, Task, Solution

logger = logging.getLogger(__name__)


class OptimizationBackend(Enum):
    """Available optimization backends."""
    CLASSICAL = "classical"
    DWAVE = "dwave"
    IBMQ = "ibmq"
    AZURE = "azure"
    SIMULATOR = "simulator"


@dataclass
class OptimizationParams:
    """Parameters for quantum optimization."""
    backend: OptimizationBackend = OptimizationBackend.SIMULATOR
    num_reads: int = 1000
    chain_strength: float = 1.0
    annealing_time: int = 20
    timeout: float = 300.0
    max_retries: int = 3


class QuantumOptimizer(Protocol):
    """Protocol for quantum optimization backends."""
    
    def optimize(
        self,
        agents: List[Agent],
        tasks: List[Task],
        params: OptimizationParams
    ) -> Solution:
        """Optimize task assignment using quantum computing."""
        ...


class BaseOptimizer(ABC):
    """Base class for optimization backends."""
    
    def __init__(self, backend: OptimizationBackend):
        self.backend = backend
        self.logger = logging.getLogger(f"{__name__}.{backend.value}")
    
    @abstractmethod
    def _formulate_qubo(
        self,
        agents: List[Agent],
        tasks: List[Task]
    ) -> Dict[tuple, float]:
        """Formulate the problem as a QUBO matrix."""
        pass
    
    @abstractmethod
    def _solve_qubo(
        self,
        qubo: Dict[tuple, float],
        params: OptimizationParams
    ) -> Dict[str, Any]:
        """Solve the QUBO problem."""
        pass
    
    @abstractmethod
    def _parse_solution(
        self,
        raw_solution: Dict[str, Any],
        agents: List[Agent],
        tasks: List[Task]
    ) -> Solution:
        """Parse raw solution into Solution object."""
        pass
    
    def optimize(
        self,
        agents: List[Agent],
        tasks: List[Task],
        params: OptimizationParams
    ) -> Solution:
        """Main optimization workflow."""
        self.logger.info(f"Starting optimization with {self.backend.value} backend")
        
        # Input validation
        if not agents:
            raise ValueError("No agents provided")
        if not tasks:
            raise ValueError("No tasks provided")
        
        try:
            # Step 1: Formulate QUBO
            self.logger.debug("Formulating QUBO matrix")
            qubo = self._formulate_qubo(agents, tasks)
            
            # Step 2: Solve optimization problem
            self.logger.debug(f"Solving with {len(qubo)} variables")
            raw_solution = self._solve_qubo(qubo, params)
            
            # Step 3: Parse solution
            self.logger.debug("Parsing solution")
            solution = self._parse_solution(raw_solution, agents, tasks)
            
            self.logger.info(
                f"Optimization completed: makespan={solution.makespan:.2f}, "
                f"cost={solution.cost:.2f}, backend={solution.backend_used}"
            )
            
            return solution
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise


class SimulatorOptimizer(BaseOptimizer):
    """Classical simulator for quantum optimization."""
    
    def __init__(self):
        super().__init__(OptimizationBackend.SIMULATOR)
    
    def _formulate_qubo(
        self,
        agents: List[Agent],
        tasks: List[Task]
    ) -> Dict[tuple, float]:
        """Formulate task assignment as QUBO."""
        qubo = {}
        num_agents = len(agents)
        num_tasks = len(tasks)
        
        # Binary variables: x[i,j] = 1 if task i assigned to agent j
        for i in range(num_tasks):
            for j in range(num_agents):
                # Objective: minimize makespan (simplified)
                cost = tasks[i].duration / agents[j].capacity
                qubo[(i * num_agents + j, i * num_agents + j)] = cost
                
                # Constraint: each task assigned to exactly one agent
                for k in range(j + 1, num_agents):
                    penalty = 100.0  # Large penalty for constraint violation
                    qubo[(i * num_agents + j, i * num_agents + k)] = penalty
        
        return qubo
    
    def _solve_qubo(
        self,
        qubo: Dict[tuple, float],
        params: OptimizationParams
    ) -> Dict[str, Any]:
        """Solve QUBO using simulated annealing."""
        import random
        
        # Simple greedy solution for simulation
        variables = set()
        for (i, j) in qubo.keys():
            variables.add(i)
            variables.add(j)
        
        # Random binary assignment
        solution = {var: random.choice([0, 1]) for var in variables}
        
        # Calculate energy
        energy = 0.0
        for (i, j), coeff in qubo.items():
            if i == j:
                energy += coeff * solution[i]
            else:
                energy += coeff * solution[i] * solution[j]
        
        return {
            "solution": solution,
            "energy": energy,
            "num_reads": params.num_reads,
            "chain_breaks": 0
        }
    
    def _parse_solution(
        self,
        raw_solution: Dict[str, Any],
        agents: List[Agent],
        tasks: List[Task]
    ) -> Solution:
        """Parse simulator solution."""
        solution_vars = raw_solution["solution"]
        num_agents = len(agents)
        
        # Extract assignments
        assignments = {}
        for i, task in enumerate(tasks):
            for j, agent in enumerate(agents):
                var_idx = i * num_agents + j
                if solution_vars.get(var_idx, 0) == 1:
                    # Simple skill check
                    if task.can_be_assigned_to(agent):
                        assignments[task.task_id] = agent.agent_id
                        break
            
            # Fallback assignment if no valid assignment found
            if task.task_id not in assignments:
                # Assign to first compatible agent
                for agent in agents:
                    if task.can_be_assigned_to(agent):
                        assignments[task.task_id] = agent.agent_id
                        break
        
        # Calculate metrics
        makespan = self._calculate_makespan(assignments, tasks, agents)
        cost = len(assignments) * 1.0  # Simple cost model
        
        return Solution(
            assignments=assignments,
            makespan=makespan,
            cost=cost,
            backend_used=self.backend.value
        )
    
    def _calculate_makespan(
        self,
        assignments: Dict[str, str],
        tasks: List[Task],
        agents: List[Agent]
    ) -> float:
        """Calculate makespan for given assignments."""
        agent_loads = {agent.agent_id: 0.0 for agent in agents}
        
        for task in tasks:
            if task.task_id in assignments:
                agent_id = assignments[task.task_id]
                if agent_id in agent_loads:
                    agent_loads[agent_id] += task.duration
        
        return max(agent_loads.values()) if agent_loads else 0.0


class OptimizerFactory:
    """Factory for creating optimization backends."""
    
    _optimizers = {
        OptimizationBackend.SIMULATOR: SimulatorOptimizer
    }
    
    @classmethod
    def create(
        self,
        backend: OptimizationBackend,
        **kwargs
    ) -> BaseOptimizer:
        """Create optimizer instance."""
        if backend not in self._optimizers:
            raise ValueError(f"Unsupported backend: {backend}")
        
        optimizer_class = self._optimizers[backend]
        return optimizer_class(**kwargs)
    
    @classmethod
    def register(
        cls,
        backend: OptimizationBackend,
        optimizer_class: type
    ):
        """Register new optimizer backend."""
        cls._optimizers[backend] = optimizer_class
    
    @classmethod
    def available_backends(cls) -> List[OptimizationBackend]:
        """Get list of available backends."""
        return list(cls._optimizers.keys())


# Utility functions
def create_optimizer(backend: str = "simulator") -> BaseOptimizer:
    """Convenience function to create optimizer."""
    backend_enum = OptimizationBackend(backend)
    return OptimizerFactory.create(backend_enum)


def optimize_tasks(
    agents: List[Agent],
    tasks: List[Task],
    backend: str = "simulator",
    **params
) -> Solution:
    """High-level task optimization function."""
    optimizer = create_optimizer(backend)
    optimization_params = OptimizationParams(**params)
    return optimizer.optimize(agents, tasks, optimization_params)
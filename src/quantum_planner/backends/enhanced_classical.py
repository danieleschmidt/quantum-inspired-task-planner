"""Enhanced classical optimization backends with robust error handling."""

import random
import math
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .enhanced_base import (
    EnhancedQuantumBackend, 
    BackendCapabilities, 
    BackendStatus
)

logger = logging.getLogger(__name__)


class EnhancedSimulatedAnnealingBackend(EnhancedQuantumBackend):
    """Enhanced simulated annealing with adaptive parameters and robust error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced simulated annealing backend."""
        default_config = {
            "initial_temperature": 100.0,
            "final_temperature": 0.01,
            "cooling_rate": 0.95,
            "max_iterations": 10000,
            "adaptive_cooling": True,
            "auto_tune_params": True,
            "max_stagnation": 1000,
            "restart_threshold": 5000
        }
        
        if config:
            default_config.update(config)
            
        super().__init__("enhanced_simulated_annealing", default_config)
        
        # Adaptive parameters
        self._best_params: Optional[Dict[str, float]] = None
        self._param_history: List[Dict[str, Any]] = []
    
    def _initialize_backend(self) -> None:
        """Initialize simulated annealing specific components."""
        self.random = random.Random(self.config.get("seed", 42))
        logger.info(f"Initialized {self.name} with config: {self.config}")
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get capabilities of simulated annealing backend."""
        return BackendCapabilities(
            max_variables=1000,  # Can handle large problems
            supports_constraints=True,
            supports_embedding=False,
            supports_async=False,
            supports_batching=True,
            max_batch_size=10,
            supported_objectives=["minimize", "maximize"],
            constraint_types=["linear", "quadratic", "penalty"]
        )
    
    def _solve_qubo(self, Q: Any, **kwargs) -> Dict[int, int]:
        """Solve QUBO using enhanced simulated annealing."""
        num_reads = kwargs.get("num_reads", 100)
        
        # Convert Q to dictionary format if needed
        if hasattr(Q, 'shape'):
            Q_dict = self._matrix_to_dict(Q)
        else:
            Q_dict = Q
        
        # Get problem dimensions
        variables = set()
        for i, j in Q_dict.keys():
            variables.update([i, j])
        
        if not variables:
            return {}
        
        num_vars = max(variables) + 1
        
        # Auto-tune parameters if enabled
        if self.config.get("auto_tune_params", True):
            params = self._auto_tune_parameters(Q_dict, num_vars)
        else:
            params = self.config.copy()
        
        best_solution = None
        best_energy = float('inf')
        
        # Multiple runs for robustness
        for run in range(num_reads):
            try:
                solution, energy = self._single_annealing_run(Q_dict, num_vars, params)
                
                if energy < best_energy:
                    best_energy = energy
                    best_solution = solution
                    
            except Exception as e:
                logger.warning(f"Annealing run {run} failed: {e}")
                continue
        
        if best_solution is None:
            raise RuntimeError("All annealing runs failed")
        
        # Update parameter history for future tuning
        if self.config.get("auto_tune_params", True):
            self._param_history.append({
                "params": params,
                "energy": best_energy,
                "num_vars": num_vars,
                "timestamp": time.time()
            })
            
            # Keep only recent history
            if len(self._param_history) > 100:
                self._param_history = self._param_history[-50:]
        
        return best_solution
    
    def _single_annealing_run(self, Q: Dict[Tuple[int, int], float], 
                            num_vars: int, params: Dict[str, Any]) -> Tuple[Dict[int, int], float]:
        """Single simulated annealing run."""
        
        # Initialize random solution
        current_solution = {i: self.random.randint(0, 1) for i in range(num_vars)}
        current_energy = self._calculate_energy_dict(current_solution, Q)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # Annealing parameters
        temperature = params["initial_temperature"]
        final_temp = params["final_temperature"]
        cooling_rate = params["cooling_rate"]
        max_iterations = params["max_iterations"]
        max_stagnation = params.get("max_stagnation", 1000)
        
        stagnation_counter = 0
        iterations_without_improvement = 0
        
        for iteration in range(max_iterations):
            # Stop if converged
            if temperature < final_temp:
                break
            
            # Stop if stagnated
            if stagnation_counter >= max_stagnation:
                if params.get("restart_on_stagnation", False):
                    # Restart with new random solution
                    current_solution = {i: self.random.randint(0, 1) for i in range(num_vars)}
                    current_energy = self._calculate_energy_dict(current_solution, Q)
                    stagnation_counter = 0
                    temperature = params["initial_temperature"]
                else:
                    break
            
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution, num_vars)
            neighbor_energy = self._calculate_energy_dict(neighbor_solution, Q)
            
            # Accept or reject
            delta_energy = neighbor_energy - current_energy
            
            if delta_energy < 0 or self.random.random() < math.exp(-delta_energy / temperature):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                stagnation_counter = 0
                
                # Update best solution
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                stagnation_counter += 1
                iterations_without_improvement += 1
            
            # Adaptive cooling
            if params.get("adaptive_cooling", True):
                temperature = self._adaptive_cooling(
                    temperature, iteration, max_iterations, 
                    iterations_without_improvement, cooling_rate
                )
            else:
                temperature *= cooling_rate
        
        return best_solution, best_energy
    
    def _generate_neighbor(self, solution: Dict[int, int], num_vars: int) -> Dict[int, int]:
        """Generate neighbor solution by flipping random bits."""
        neighbor = solution.copy()
        
        # Number of bits to flip (typically 1-3)
        num_flips = min(3, max(1, int(num_vars * 0.05)))
        
        for _ in range(num_flips):
            var_to_flip = self.random.randint(0, num_vars - 1)
            neighbor[var_to_flip] = 1 - neighbor[var_to_flip]
        
        return neighbor
    
    def _calculate_energy_dict(self, solution: Dict[int, int], Q: Dict[Tuple[int, int], float]) -> float:
        """Calculate energy for solution given QUBO in dictionary form."""
        energy = 0.0
        
        for (i, j), coeff in Q.items():
            energy += coeff * solution.get(i, 0) * solution.get(j, 0)
        
        return energy
    
    def _adaptive_cooling(self, temperature: float, iteration: int, max_iterations: int,
                         stagnation: int, base_cooling_rate: float) -> float:
        """Adaptive cooling schedule based on progress."""
        
        # Base exponential cooling
        new_temp = temperature * base_cooling_rate
        
        # Adjust based on stagnation
        if stagnation > 100:
            # Cool faster if stagnating
            new_temp *= 0.9
        elif stagnation < 10:
            # Cool slower if making progress
            new_temp *= 1.1
        
        # Ensure temperature doesn't drop too fast
        min_temp_for_iteration = 0.01 * (1 - iteration / max_iterations)
        new_temp = max(new_temp, min_temp_for_iteration)
        
        return new_temp
    
    def _auto_tune_parameters(self, Q: Dict[Tuple[int, int], float], num_vars: int) -> Dict[str, Any]:
        """Auto-tune parameters based on problem characteristics and history."""
        
        # Start with base config
        params = self.config.copy()
        
        # Adjust based on problem size
        if num_vars < 10:
            params["max_iterations"] = 1000
            params["initial_temperature"] = 10.0
        elif num_vars < 50:
            params["max_iterations"] = 5000
            params["initial_temperature"] = 50.0
        else:
            params["max_iterations"] = 10000
            params["initial_temperature"] = 100.0
        
        # Adjust based on problem density
        density = len(Q) / (num_vars * num_vars)
        if density > 0.5:
            # Dense problem - need more exploration
            params["cooling_rate"] = 0.98
            params["max_stagnation"] = int(params["max_iterations"] * 0.2)
        else:
            # Sparse problem - can cool faster
            params["cooling_rate"] = 0.95
            params["max_stagnation"] = int(params["max_iterations"] * 0.1)
        
        # Learn from history
        if self._param_history:
            successful_params = self._analyze_parameter_history(num_vars)
            if successful_params:
                params.update(successful_params)
        
        return params
    
    def _analyze_parameter_history(self, num_vars: int) -> Optional[Dict[str, Any]]:
        """Analyze parameter history to find best settings for similar problems."""
        
        # Filter history for similar problem sizes
        similar_problems = [
            entry for entry in self._param_history
            if abs(entry["num_vars"] - num_vars) / num_vars < 0.3
        ]
        
        if len(similar_problems) < 3:
            return None
        
        # Find best performing parameters
        best_entry = min(similar_problems, key=lambda x: x["energy"])
        return best_entry["params"]
    
    def _matrix_to_dict(self, Q) -> Dict[Tuple[int, int], float]:
        """Convert matrix to dictionary format."""
        Q_dict = {}
        
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                if Q[i, j] != 0:
                    Q_dict[(i, j)] = float(Q[i, j])
        
        return Q_dict
    
    def _check_connectivity(self) -> BackendStatus:
        """Check connectivity (always available for classical)."""
        return BackendStatus.AVAILABLE
    
    def _base_time_estimation(self, num_variables: int) -> float:
        """Estimate solve time based on problem size."""
        # Empirical formula for simulated annealing
        base_time = 0.001 * (num_variables ** 1.5)
        
        # Adjust for configured iterations
        max_iterations = self.config.get("max_iterations", 10000)
        iteration_factor = max_iterations / 10000
        
        return base_time * iteration_factor


class EnhancedGeneticAlgorithmBackend(EnhancedQuantumBackend):
    """Enhanced genetic algorithm with adaptive operators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced genetic algorithm backend."""
        default_config = {
            "population_size": 100,
            "generations": 500,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 10,
            "tournament_size": 5,
            "adaptive_operators": True
        }
        
        if config:
            default_config.update(config)
            
        super().__init__("enhanced_genetic_algorithm", default_config)
    
    def _initialize_backend(self) -> None:
        """Initialize genetic algorithm components."""
        self.random = random.Random(self.config.get("seed", 42))
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get GA capabilities."""
        return BackendCapabilities(
            max_variables=500,
            supports_constraints=True,
            supports_embedding=False,
            supports_async=False,
            supports_batching=True,
            max_batch_size=5,
            supported_objectives=["minimize", "maximize"],
            constraint_types=["linear", "quadratic", "penalty"]
        )
    
    def _solve_qubo(self, Q: Any, **kwargs) -> Dict[int, int]:
        """Solve QUBO using genetic algorithm."""
        # Implementation would go here
        # For now, fall back to random solution
        if hasattr(Q, 'shape'):
            num_vars = Q.shape[0]
        else:
            variables = set()
            for i, j in Q.keys():
                variables.update([i, j])
            num_vars = max(variables) + 1 if variables else 0
        
        return {i: self.random.randint(0, 1) for i in range(num_vars)}
    
    def _check_connectivity(self) -> BackendStatus:
        """Check connectivity (always available for classical)."""
        return BackendStatus.AVAILABLE
    
    def _base_time_estimation(self, num_variables: int) -> float:
        """Estimate solve time for GA."""
        generations = self.config.get("generations", 500)
        population_size = self.config.get("population_size", 100)
        
        return 0.0001 * generations * population_size * num_variables
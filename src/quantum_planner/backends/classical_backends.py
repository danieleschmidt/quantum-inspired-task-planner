"""Classical optimization backends for quantum-inspired algorithms."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import logging
import random
from abc import ABC

from .base import BaseBackend, BackendType, BackendInfo, OptimizationResult

logger = logging.getLogger(__name__)


class SimulatedAnnealingBackend(BaseBackend):
    """High-performance simulated annealing backend."""
    
    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 10000
    ):
        super().__init__("simulated_annealing", BackendType.CLASSICAL_HEURISTIC)
        
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
    
    def get_backend_info(self) -> BackendInfo:
        """Get simulated annealing backend information."""
        return BackendInfo(
            name=self.name,
            backend_type=self.backend_type,
            max_variables=10000,  # Very scalable
            connectivity="software",
            availability=True,
            cost_per_sample=0.0,
            typical_solve_time=1.0,
            supports_embedding=False,
            supports_constraints=True
        )
    
    def solve_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Solve QUBO using simulated annealing."""
        
        start_time = time.time()
        
        try:
            # Validate problem
            is_valid, error_msg = self.validate_problem(Q)
            if not is_valid:
                return OptimizationResult(
                    solution={},
                    energy=float('inf'),
                    num_samples=0,
                    timing={'total': time.time() - start_time},
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )
            
            n = Q.shape[0]
            best_solution = None
            best_energy = float('inf')
            
            # Run multiple independent runs
            num_runs = min(num_reads // 100, 10)  # Limit concurrent runs
            iterations_per_run = self.max_iterations // max(1, num_runs)
            
            for run in range(num_runs):
                solution, energy = self._single_sa_run(Q, iterations_per_run)
                
                if energy < best_energy:
                    best_energy = energy
                    best_solution = solution
            
            # Final solution format
            solution_dict = {i: int(best_solution[i]) for i in range(n)}
            
            solve_time = time.time() - start_time
            
            # Timing breakdown
            timing = {
                'total': solve_time,
                'solve': solve_time * 0.9,
                'preprocessing': solve_time * 0.05,
                'postprocessing': solve_time * 0.05
            }
            
            # Metadata
            metadata = {
                'algorithm': 'simulated_annealing',
                'initial_temperature': self.initial_temperature,
                'cooling_rate': self.cooling_rate,
                'iterations': self.max_iterations,
                'num_runs': num_runs,
                'num_variables': n
            }
            
            self.logger.info(
                f"Simulated annealing completed: energy={best_energy:.4f}, "
                f"runs={num_runs}, time={solve_time:.2f}s"
            )
            
            return OptimizationResult(
                solution=solution_dict,
                energy=best_energy,
                num_samples=num_runs,
                timing=timing,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Simulated annealing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return OptimizationResult(
                solution={},
                energy=float('inf'),
                num_samples=0,
                timing={'total': time.time() - start_time},
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )
    
    def _single_sa_run(self, Q: np.ndarray, max_iter: int) -> Tuple[np.ndarray, float]:
        """Single simulated annealing run."""
        n = Q.shape[0]
        
        # Initialize random solution
        current_solution = np.random.choice([0, 1], size=n)
        current_energy = self.calculate_energy({i: current_solution[i] for i in range(n)}, Q)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temperature = self.initial_temperature
        
        for iteration in range(max_iter):
            # Generate neighbor by flipping random bit
            neighbor = current_solution.copy()
            flip_index = random.randint(0, n - 1)
            neighbor[flip_index] = 1 - neighbor[flip_index]
            
            # Calculate energy difference
            delta_energy = self._calculate_energy_diff(Q, current_solution, neighbor, flip_index)
            neighbor_energy = current_energy + delta_energy
            
            # Accept or reject move
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            
            # Cool down
            temperature *= self.cooling_rate
            if temperature < self.min_temperature:
                temperature = self.min_temperature
        
        return best_solution, best_energy
    
    def _calculate_energy_diff(
        self,
        Q: np.ndarray,
        current: np.ndarray,
        neighbor: np.ndarray,
        flip_index: int
    ) -> float:
        """Calculate energy difference for single bit flip efficiently."""
        n = Q.shape[0]
        delta = 0.0
        
        # Only calculate difference for the flipped bit
        old_val = current[flip_index]
        new_val = neighbor[flip_index]
        
        # Linear term
        delta += Q[flip_index, flip_index] * (new_val - old_val)
        
        # Quadratic terms
        for j in range(n):
            if j != flip_index:
                # Terms involving flipped variable
                delta += Q[flip_index, j] * current[j] * (new_val - old_val)
                delta += Q[j, flip_index] * current[j] * (new_val - old_val)
        
        return delta
    
    def estimate_solve_time(self, problem_size: int) -> float:
        """Estimate simulated annealing solve time."""
        # Linear scaling with problem size and iterations
        base_time = 0.1
        scaling_factor = (problem_size * self.max_iterations) / 100000
        return base_time + scaling_factor


class GeneticAlgorithmBackend(BaseBackend):
    """Genetic algorithm backend for QUBO optimization."""
    
    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        generations: int = 500,
        elitism_size: int = 10
    ):
        super().__init__("genetic_algorithm", BackendType.CLASSICAL_HEURISTIC)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.elitism_size = elitism_size
    
    def get_backend_info(self) -> BackendInfo:
        """Get genetic algorithm backend information."""
        return BackendInfo(
            name=self.name,
            backend_type=self.backend_type,
            max_variables=5000,
            connectivity="software",
            availability=True,
            cost_per_sample=0.0,
            typical_solve_time=2.0,
            supports_embedding=False,
            supports_constraints=True
        )
    
    def solve_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Solve QUBO using genetic algorithm."""
        
        start_time = time.time()
        
        try:
            # Validate problem
            is_valid, error_msg = self.validate_problem(Q)
            if not is_valid:
                return OptimizationResult(
                    solution={},
                    energy=float('inf'),
                    num_samples=0,
                    timing={'total': time.time() - start_time},
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )
            
            n = Q.shape[0]
            
            # Initialize population
            population = [np.random.choice([0, 1], size=n) for _ in range(self.population_size)]
            
            best_solution = None
            best_energy = float('inf')
            
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    energy = self.calculate_energy({i: individual[i] for i in range(n)}, Q)
                    fitness_scores.append(-energy)  # Negative for maximization
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_solution = individual.copy()
                
                # Selection, crossover, mutation
                population = self._evolve_population(population, fitness_scores)
            
            # Final solution
            solution_dict = {i: int(best_solution[i]) for i in range(n)}
            
            solve_time = time.time() - start_time
            
            # Timing breakdown
            timing = {
                'total': solve_time,
                'solve': solve_time * 0.9,
                'preprocessing': solve_time * 0.05,
                'postprocessing': solve_time * 0.05
            }
            
            # Metadata
            metadata = {
                'algorithm': 'genetic_algorithm',
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'num_variables': n
            }
            
            self.logger.info(
                f"Genetic algorithm completed: energy={best_energy:.4f}, "
                f"generations={self.generations}, time={solve_time:.2f}s"
            )
            
            return OptimizationResult(
                solution=solution_dict,
                energy=best_energy,
                num_samples=self.population_size * self.generations,
                timing=timing,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Genetic algorithm failed: {str(e)}"
            self.logger.error(error_msg)
            
            return OptimizationResult(
                solution={},
                energy=float('inf'),
                num_samples=0,
                timing={'total': time.time() - start_time},
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )
    
    def _evolve_population(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """Evolve population through selection, crossover, and mutation."""
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        new_population = []
        
        # Elitism - keep best individuals
        for i in range(min(self.elitism_size, len(sorted_population))):
            new_population.append(sorted_population[i].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _tournament_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> np.ndarray:
        """Tournament selection for parent selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        n = len(parent1)
        crossover_point = random.randint(1, n - 1)
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Bit-flip mutation."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        return mutated
    
    def estimate_solve_time(self, problem_size: int) -> float:
        """Estimate genetic algorithm solve time."""
        base_time = 0.5
        scaling_factor = (problem_size * self.population_size * self.generations) / 100000
        return base_time + scaling_factor


class TabuSearchBackend(BaseBackend):
    """Tabu search backend for QUBO optimization."""
    
    def __init__(
        self,
        tabu_tenure: int = 20,
        max_iterations: int = 1000,
        neighborhood_size: int = 50,
        aspiration_criterion: bool = True
    ):
        super().__init__("tabu_search", BackendType.CLASSICAL_HEURISTIC)
        
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        self.aspiration_criterion = aspiration_criterion
    
    def get_backend_info(self) -> BackendInfo:
        """Get tabu search backend information."""
        return BackendInfo(
            name=self.name,
            backend_type=self.backend_type,
            max_variables=2000,
            connectivity="software",
            availability=True,
            cost_per_sample=0.0,
            typical_solve_time=3.0,
            supports_embedding=False,
            supports_constraints=True
        )
    
    def solve_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Solve QUBO using tabu search."""
        
        start_time = time.time()
        
        try:
            # Validate problem
            is_valid, error_msg = self.validate_problem(Q)
            if not is_valid:
                return OptimizationResult(
                    solution={},
                    energy=float('inf'),
                    num_samples=0,
                    timing={'total': time.time() - start_time},
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )
            
            n = Q.shape[0]
            
            # Initialize solution
            current_solution = np.random.choice([0, 1], size=n)
            current_energy = self.calculate_energy({i: current_solution[i] for i in range(n)}, Q)
            
            best_solution = current_solution.copy()
            best_energy = current_energy
            
            # Tabu list (stores moves that are forbidden)
            tabu_list = []
            
            for iteration in range(self.max_iterations):
                # Generate neighborhood
                neighborhood = self._generate_neighborhood(current_solution)
                
                best_neighbor = None
                best_neighbor_energy = float('inf')
                best_move = None
                
                for neighbor, move in neighborhood:
                    neighbor_energy = self.calculate_energy({i: neighbor[i] for i in range(n)}, Q)
                    
                    # Check if move is not tabu or satisfies aspiration criterion
                    is_tabu = move in tabu_list
                    aspires = self.aspiration_criterion and neighbor_energy < best_energy
                    
                    if not is_tabu or aspires:
                        if neighbor_energy < best_neighbor_energy:
                            best_neighbor = neighbor
                            best_neighbor_energy = neighbor_energy
                            best_move = move
                
                if best_neighbor is not None:
                    current_solution = best_neighbor
                    current_energy = best_neighbor_energy
                    
                    # Update tabu list
                    tabu_list.append(best_move)
                    if len(tabu_list) > self.tabu_tenure:
                        tabu_list.pop(0)
                    
                    # Update best solution
                    if current_energy < best_energy:
                        best_solution = current_solution.copy()
                        best_energy = current_energy
            
            # Final solution
            solution_dict = {i: int(best_solution[i]) for i in range(n)}
            
            solve_time = time.time() - start_time
            
            # Timing breakdown
            timing = {
                'total': solve_time,
                'solve': solve_time * 0.9,
                'preprocessing': solve_time * 0.05,
                'postprocessing': solve_time * 0.05
            }
            
            # Metadata
            metadata = {
                'algorithm': 'tabu_search',
                'tabu_tenure': self.tabu_tenure,
                'max_iterations': self.max_iterations,
                'neighborhood_size': self.neighborhood_size,
                'num_variables': n
            }
            
            self.logger.info(
                f"Tabu search completed: energy={best_energy:.4f}, "
                f"iterations={self.max_iterations}, time={solve_time:.2f}s"
            )
            
            return OptimizationResult(
                solution=solution_dict,
                energy=best_energy,
                num_samples=self.max_iterations * self.neighborhood_size,
                timing=timing,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Tabu search failed: {str(e)}"
            self.logger.error(error_msg)
            
            return OptimizationResult(
                solution={},
                energy=float('inf'),
                num_samples=0,
                timing={'total': time.time() - start_time},
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )
    
    def _generate_neighborhood(self, solution: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """Generate neighborhood by flipping bits."""
        neighborhood = []
        n = len(solution)
        
        # Limit neighborhood size for performance
        indices = list(range(n))
        if len(indices) > self.neighborhood_size:
            indices = random.sample(indices, self.neighborhood_size)
        
        for i in indices:
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighborhood.append((neighbor, i))  # (neighbor_solution, flipped_bit_index)
        
        return neighborhood
    
    def estimate_solve_time(self, problem_size: int) -> float:
        """Estimate tabu search solve time."""
        base_time = 0.3
        scaling_factor = (problem_size * self.max_iterations * self.neighborhood_size) / 500000
        return base_time + scaling_factor
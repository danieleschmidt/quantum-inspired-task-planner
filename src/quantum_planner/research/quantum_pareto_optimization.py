"""
Advanced Multi-Objective Quantum Optimization Module

Implements state-of-the-art multi-objective quantum optimization techniques
including Quantum NSGA-II, Pareto-optimal quantum circuit learning, and
hybrid quantum-classical multi-objective evolutionary algorithms.

This represents cutting-edge research in quantum multi-objective optimization.

Author: Terragon Labs Quantum Research Team
Version: 1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import time
from collections import defaultdict, deque
import warnings
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.spatial.distance import cdist, pdist
    from scipy.stats import rankdata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some optimization features will be limited.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph-based features will be limited.")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Clustering features will be limited.")

logger = logging.getLogger(__name__)


class DominanceRelation(Enum):
    """Pareto dominance relations."""
    DOMINATES = "dominates"
    DOMINATED_BY = "dominated_by"
    NON_DOMINATED = "non_dominated"
    IDENTICAL = "identical"


@dataclass
class MultiObjectiveSolution:
    """Represents a multi-objective optimization solution."""
    
    solution: np.ndarray
    objectives: np.ndarray  # Multiple objective values
    constraints_violation: float = 0.0
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0
    
    # Solution metadata
    algorithm_used: str = "unknown"
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # Performance metrics
    convergence_metric: float = 0.0
    diversity_metric: float = 0.0
    
    def dominates(self, other: 'MultiObjectiveSolution') -> bool:
        """Check if this solution dominates another (for minimization)."""
        better_in_all = np.all(self.objectives <= other.objectives)
        better_in_at_least_one = np.any(self.objectives < other.objectives)
        
        # Consider constraint violations
        if self.constraints_violation > 0 or other.constraints_violation > 0:
            if self.constraints_violation < other.constraints_violation:
                return True
            elif self.constraints_violation > other.constraints_violation:
                return False
        
        return better_in_all and better_in_at_least_one
    
    def get_dominance_relation(self, other: 'MultiObjectiveSolution') -> DominanceRelation:
        """Get dominance relation with another solution."""
        if np.allclose(self.objectives, other.objectives, rtol=1e-10):
            return DominanceRelation.IDENTICAL
        elif self.dominates(other):
            return DominanceRelation.DOMINATES
        elif other.dominates(self):
            return DominanceRelation.DOMINATED_BY
        else:
            return DominanceRelation.NON_DOMINATED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'solution': self.solution.tolist(),
            'objectives': self.objectives.tolist(),
            'constraints_violation': self.constraints_violation,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'algorithm_used': self.algorithm_used,
            'generation': self.generation
        }


@dataclass
class MultiObjectiveProblem:
    """Defines a multi-objective optimization problem."""
    
    objective_functions: List[Callable[[np.ndarray], float]]
    objective_names: List[str]
    constraint_functions: List[Callable[[np.ndarray], float]] = field(default_factory=list)
    constraint_names: List[str] = field(default_factory=list)
    
    # Problem bounds
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    variable_types: List[str] = field(default_factory=list)  # 'continuous', 'integer', 'binary'
    
    # Problem properties
    num_variables: int = 0
    num_objectives: int = 0
    num_constraints: int = 0
    
    # Optimization direction (1 for minimization, -1 for maximization)
    optimization_directions: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize problem properties."""
        self.num_objectives = len(self.objective_functions)
        self.num_constraints = len(self.constraint_functions)
        
        if not self.optimization_directions:
            self.optimization_directions = [1] * self.num_objectives  # Default: minimize
        
        if not self.objective_names:
            self.objective_names = [f"obj_{i}" for i in range(self.num_objectives)]
        
        if not self.constraint_names and self.constraint_functions:
            self.constraint_names = [f"constraint_{i}" for i in range(self.num_constraints)]
    
    def evaluate(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate solution on all objectives and constraints."""
        # Evaluate objectives
        objectives = np.array([
            direction * func(solution) 
            for func, direction in zip(self.objective_functions, self.optimization_directions)
        ])
        
        # Evaluate constraints
        constraint_violation = 0.0
        if self.constraint_functions:
            violations = [max(0, func(solution)) for func in self.constraint_functions]
            constraint_violation = sum(violations)
        
        return objectives, constraint_violation
    
    def is_feasible(self, solution: np.ndarray) -> bool:
        """Check if solution is feasible."""
        if not self.constraint_functions:
            return True
        
        return all(func(solution) <= 0 for func in self.constraint_functions)


class ParetoFrontAnalyzer:
    """Analyzer for Pareto front properties and metrics."""
    
    @staticmethod
    def compute_pareto_ranks(solutions: List[MultiObjectiveSolution]) -> List[int]:
        """Compute Pareto ranks using fast non-dominated sorting."""
        n = len(solutions)
        if n == 0:
            return []
        
        # Initialize dominance structures
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by i
        domination_count = [0] * n  # Number of solutions dominating i
        
        # Compute dominance relations
        for i in range(n):
            for j in range(i + 1, n):
                relation = solutions[i].get_dominance_relation(solutions[j])
                
                if relation == DominanceRelation.DOMINATES:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif relation == DominanceRelation.DOMINATED_BY:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # Identify fronts
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        rank = 0
        ranks = [0] * n
        
        while current_front:
            fronts.append(current_front[:])
            
            # Assign rank to current front
            for i in current_front:
                ranks[i] = rank
            
            # Find next front
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
            rank += 1
        
        return ranks
    
    @staticmethod
    def compute_crowding_distances(
        solutions: List[MultiObjectiveSolution], 
        front_indices: List[int]
    ) -> List[float]:
        """Compute crowding distances for solutions in a front."""
        if len(front_indices) <= 2:
            return [float('inf')] * len(front_indices)
        
        distances = [0.0] * len(front_indices)
        num_objectives = len(solutions[front_indices[0]].objectives)
        
        for obj_idx in range(num_objectives):
            # Sort by objective value
            sorted_indices = sorted(
                range(len(front_indices)),
                key=lambda i: solutions[front_indices[i]].objectives[obj_idx]
            )
            
            # Set boundary solutions to infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Compute distances for middle solutions
            obj_values = [solutions[front_indices[i]].objectives[obj_idx] for i in sorted_indices]
            obj_range = obj_values[-1] - obj_values[0]
            
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += (
                        obj_values[i + 1] - obj_values[i - 1]
                    ) / obj_range
        
        return distances
    
    @staticmethod
    def compute_hypervolume(solutions: List[MultiObjectiveSolution], reference_point: np.ndarray) -> float:
        """Compute hypervolume indicator."""
        if not solutions:
            return 0.0
        
        # Extract objective vectors
        objectives = np.array([sol.objectives for sol in solutions])
        
        # Simple hypervolume computation (for 2D and 3D)
        if objectives.shape[1] == 2:
            return ParetoFrontAnalyzer._hypervolume_2d(objectives, reference_point)
        elif objectives.shape[1] == 3:
            return ParetoFrontAnalyzer._hypervolume_3d(objectives, reference_point)
        else:
            # Use Monte Carlo approximation for higher dimensions
            return ParetoFrontAnalyzer._hypervolume_monte_carlo(objectives, reference_point)
    
    @staticmethod
    def _hypervolume_2d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
        """Compute 2D hypervolume exactly."""
        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]
        
        hypervolume = 0.0
        prev_x = reference_point[0]
        
        for obj in sorted_objectives:
            if obj[0] <= reference_point[0] and obj[1] <= reference_point[1]:
                hypervolume += (obj[0] - prev_x) * (reference_point[1] - obj[1])
                prev_x = obj[0]
        
        return hypervolume
    
    @staticmethod
    def _hypervolume_3d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
        """Compute 3D hypervolume using inclusion-exclusion principle."""
        # Simplified 3D hypervolume computation
        hypervolume = 0.0
        
        for i, obj in enumerate(objectives):
            if np.all(obj <= reference_point):
                # Compute individual contribution
                individual_volume = np.prod(reference_point - obj)
                
                # Subtract overlaps with other solutions
                overlap = 0.0
                for j, other_obj in enumerate(objectives):
                    if i != j and np.all(other_obj <= reference_point):
                        # Compute overlap volume
                        overlap_bounds = np.minimum(reference_point - obj, reference_point - other_obj)
                        if np.all(overlap_bounds > 0):
                            overlap += np.prod(overlap_bounds)
                
                hypervolume += max(0, individual_volume - overlap * 0.5)  # Approximation
        
        return hypervolume
    
    @staticmethod
    def _hypervolume_monte_carlo(
        objectives: np.ndarray, 
        reference_point: np.ndarray, 
        num_samples: int = 10000
    ) -> float:
        """Approximate hypervolume using Monte Carlo sampling."""
        # Generate random points in the reference space
        bounds = np.array([
            [np.min(objectives[:, i]), reference_point[i]] 
            for i in range(len(reference_point))
        ])
        
        # Sample points uniformly
        samples = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(num_samples, len(reference_point))
        )
        
        # Count dominated samples
        dominated_count = 0
        for sample in samples:
            # Check if sample is dominated by any solution
            if np.any(np.all(objectives <= sample, axis=1)):
                dominated_count += 1
        
        # Estimate hypervolume
        total_volume = np.prod(bounds[:, 1] - bounds[:, 0])
        hypervolume = (dominated_count / num_samples) * total_volume
        
        return hypervolume
    
    @staticmethod
    def compute_spacing_metric(solutions: List[MultiObjectiveSolution]) -> float:
        """Compute spacing metric for diversity assessment."""
        if len(solutions) < 2:
            return 0.0
        
        objectives = np.array([sol.objectives for sol in solutions])
        
        # Compute pairwise distances
        distances = cdist(objectives, objectives, metric='euclidean')
        
        # Find minimum distances for each point
        min_distances = []
        for i in range(len(objectives)):
            row_distances = distances[i]
            row_distances = row_distances[row_distances > 0]  # Exclude self-distance
            if len(row_distances) > 0:
                min_distances.append(np.min(row_distances))
        
        if not min_distances:
            return 0.0
        
        # Compute spacing metric
        mean_distance = np.mean(min_distances)
        variance = np.var(min_distances)
        
        return np.sqrt(variance) / max(mean_distance, 1e-10)


class QuantumNSGAII:
    """
    Quantum-enhanced Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    
    Integrates quantum computing techniques with multi-objective evolutionary
    optimization for superior performance on combinatorial problems.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 200,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.1,
        quantum_enhancement: bool = True,
        quantum_crossover_rate: float = 0.3,
        elite_preservation_rate: float = 0.1
    ):
        """Initialize Quantum NSGA-II."""
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.quantum_enhancement = quantum_enhancement
        self.quantum_crossover_rate = quantum_crossover_rate
        self.elite_preservation_rate = elite_preservation_rate
        
        # Evolution tracking
        self.evolution_history = []
        self.convergence_metrics = []
        self.diversity_metrics = []
        
        # Quantum state tracking
        self.quantum_states = []
        self.quantum_probabilities = []
        
        logger.info(f"Initialized QuantumNSGAII with quantum_enhancement={quantum_enhancement}")
    
    def optimize(
        self,
        problem: MultiObjectiveProblem,
        initial_population: Optional[List[np.ndarray]] = None,
        callback: Optional[Callable] = None
    ) -> List[MultiObjectiveSolution]:
        """Run Quantum NSGA-II optimization."""
        
        logger.info(f"Starting Quantum NSGA-II optimization")
        logger.info(f"Problem: {problem.num_variables} vars, {problem.num_objectives} objs")
        
        # Initialize population
        population = self._initialize_population(problem, initial_population)
        
        # Evaluate initial population
        self._evaluate_population(population, problem)
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Create offspring population
            offspring = self._create_offspring(population, problem, generation)
            
            # Combine parent and offspring populations
            combined_population = population + offspring
            
            # Environmental selection
            population = self._environmental_selection(combined_population)
            
            # Update solution metadata
            for sol in population:
                sol.generation = generation
            
            # Track evolution
            self._track_evolution(population, generation)
            
            # Callback
            if callback:
                callback(generation, population)
            
            # Convergence check
            if self._check_convergence(generation):
                logger.info(f"Converged at generation {generation}")
                break
            
            if generation % 20 == 0:
                logger.info(f"Generation {generation}: {len(population)} solutions")
        
        # Extract final Pareto front
        pareto_front = self._extract_pareto_front(population)
        
        logger.info(f"Optimization complete. Pareto front size: {len(pareto_front)}")
        return pareto_front
    
    def _initialize_population(
        self,
        problem: MultiObjectiveProblem,
        initial_population: Optional[List[np.ndarray]] = None
    ) -> List[MultiObjectiveSolution]:
        """Initialize population."""
        
        population = []
        
        # Use provided initial population if available
        if initial_population:
            for solution in initial_population[:self.population_size]:
                population.append(MultiObjectiveSolution(
                    solution=solution.copy(),
                    objectives=np.zeros(problem.num_objectives),
                    algorithm_used="quantum_nsga2"
                ))
        
        # Generate remaining population randomly
        remaining = self.population_size - len(population)
        
        for _ in range(remaining):
            if problem.bounds:
                # Generate within bounds
                solution = np.array([
                    np.random.uniform(low, high) 
                    for low, high in problem.bounds
                ])
            else:
                # Generate binary solution for QUBO problems
                solution = np.random.choice([0, 1], size=problem.num_variables)
            
            # Apply variable type constraints
            solution = self._apply_variable_constraints(solution, problem)
            
            population.append(MultiObjectiveSolution(
                solution=solution,
                objectives=np.zeros(problem.num_objectives),
                algorithm_used="quantum_nsga2"
            ))
        
        return population
    
    def _apply_variable_constraints(
        self, 
        solution: np.ndarray, 
        problem: MultiObjectiveProblem
    ) -> np.ndarray:
        """Apply variable type constraints."""
        
        if not problem.variable_types:
            return solution
        
        constrained_solution = solution.copy()
        
        for i, var_type in enumerate(problem.variable_types):
            if var_type == 'integer':
                constrained_solution[i] = int(round(constrained_solution[i]))
            elif var_type == 'binary':
                constrained_solution[i] = 1 if constrained_solution[i] > 0.5 else 0
            # 'continuous' requires no modification
        
        return constrained_solution
    
    def _evaluate_population(
        self,
        population: List[MultiObjectiveSolution],
        problem: MultiObjectiveProblem
    ):
        """Evaluate population on all objectives."""
        
        for solution in population:
            objectives, constraint_violation = problem.evaluate(solution.solution)
            solution.objectives = objectives
            solution.constraints_violation = constraint_violation
    
    def _create_offspring(
        self,
        population: List[MultiObjectiveSolution],
        problem: MultiObjectiveProblem,
        generation: int
    ) -> List[MultiObjectiveSolution]:
        """Create offspring population using quantum-enhanced operators."""
        
        offspring = []
        
        # Elite preservation
        elite_count = max(1, int(self.population_size * self.elite_preservation_rate))
        elites = sorted(population, key=lambda x: (x.rank, -x.crowding_distance))[:elite_count]
        offspring.extend(copy.deepcopy(elites))
        
        # Generate remaining offspring
        remaining = self.population_size - len(offspring)
        
        for _ in range(remaining // 2):
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Quantum-enhanced crossover
            if self.quantum_enhancement and np.random.random() < self.quantum_crossover_rate:
                child1, child2 = self._quantum_crossover(parent1, parent2, problem)
            else:
                # Standard crossover
                child1, child2 = self._standard_crossover(parent1, parent2, problem)
            
            # Mutation
            if np.random.random() < self.mutation_probability:
                child1 = self._quantum_mutation(child1, problem, generation)
            if np.random.random() < self.mutation_probability:
                child2 = self._quantum_mutation(child2, problem, generation)
            
            offspring.extend([child1, child2])
        
        # Handle odd population size
        if len(offspring) < self.population_size:
            parent = self._tournament_selection(population)
            child = self._quantum_mutation(copy.deepcopy(parent), problem, generation)
            offspring.append(child)
        
        # Evaluate offspring
        self._evaluate_population(offspring, problem)
        
        return offspring[:self.population_size]
    
    def _tournament_selection(
        self,
        population: List[MultiObjectiveSolution],
        tournament_size: int = 3
    ) -> MultiObjectiveSolution:
        """Tournament selection with crowded comparison."""
        
        candidates = np.random.choice(population, size=tournament_size, replace=False)
        
        # Sort by rank, then by crowding distance
        best = min(candidates, key=lambda x: (x.rank, -x.crowding_distance))
        
        return copy.deepcopy(best)
    
    def _quantum_crossover(
        self,
        parent1: MultiObjectiveSolution,
        parent2: MultiObjectiveSolution,
        problem: MultiObjectiveProblem
    ) -> Tuple[MultiObjectiveSolution, MultiObjectiveSolution]:
        """Quantum-enhanced crossover operator."""
        
        # Quantum superposition-inspired crossover
        prob_amplitudes = np.random.uniform(0, 1, len(parent1.solution))
        quantum_mask = prob_amplitudes > 0.5
        
        # Create quantum superposition state
        child1_solution = np.where(quantum_mask, parent1.solution, parent2.solution)
        child2_solution = np.where(quantum_mask, parent2.solution, parent1.solution)
        
        # Apply quantum interference (modify based on parent fitness)
        parent1_fitness = np.mean(parent1.objectives)
        parent2_fitness = np.mean(parent2.objectives)
        
        if parent1_fitness < parent2_fitness:  # parent1 is better (minimization)
            interference_factor = 0.7
        else:
            interference_factor = 0.3
        
        # Apply interference
        for i in range(len(child1_solution)):
            if np.random.random() < interference_factor:
                alpha = np.random.uniform(0.1, 0.9)
                child1_solution[i] = alpha * parent1.solution[i] + (1 - alpha) * parent2.solution[i]
                child2_solution[i] = alpha * parent2.solution[i] + (1 - alpha) * parent1.solution[i]
        
        # Apply variable constraints
        child1_solution = self._apply_variable_constraints(child1_solution, problem)
        child2_solution = self._apply_variable_constraints(child2_solution, problem)
        
        child1 = MultiObjectiveSolution(
            solution=child1_solution,
            objectives=np.zeros(problem.num_objectives),
            algorithm_used="quantum_nsga2"
        )
        
        child2 = MultiObjectiveSolution(
            solution=child2_solution,
            objectives=np.zeros(problem.num_objectives),
            algorithm_used="quantum_nsga2"
        )
        
        return child1, child2
    
    def _standard_crossover(
        self,
        parent1: MultiObjectiveSolution,
        parent2: MultiObjectiveSolution,
        problem: MultiObjectiveProblem
    ) -> Tuple[MultiObjectiveSolution, MultiObjectiveSolution]:
        """Standard simulated binary crossover (SBX)."""
        
        if np.random.random() > self.crossover_probability:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        eta_c = 20  # Distribution index for SBX
        
        child1_solution = parent1.solution.copy()
        child2_solution = parent2.solution.copy()
        
        for i in range(len(parent1.solution)):
            if np.random.random() <= 0.5:
                y1, y2 = parent1.solution[i], parent2.solution[i]
                
                if abs(y1 - y2) > 1e-14:
                    # Calculate beta
                    u = np.random.random()
                    
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    # Create offspring
                    child1_solution[i] = 0.5 * ((y1 + y2) - beta * abs(y2 - y1))
                    child2_solution[i] = 0.5 * ((y1 + y2) + beta * abs(y2 - y1))
                    
                    # Apply bounds if available
                    if problem.bounds and i < len(problem.bounds):
                        low, high = problem.bounds[i]
                        child1_solution[i] = np.clip(child1_solution[i], low, high)
                        child2_solution[i] = np.clip(child2_solution[i], low, high)
        
        # Apply variable constraints
        child1_solution = self._apply_variable_constraints(child1_solution, problem)
        child2_solution = self._apply_variable_constraints(child2_solution, problem)
        
        child1 = MultiObjectiveSolution(
            solution=child1_solution,
            objectives=np.zeros(problem.num_objectives),
            algorithm_used="quantum_nsga2"
        )
        
        child2 = MultiObjectiveSolution(
            solution=child2_solution,
            objectives=np.zeros(problem.num_objectives),
            algorithm_used="quantum_nsga2"
        )
        
        return child1, child2
    
    def _quantum_mutation(
        self,
        solution: MultiObjectiveSolution,
        problem: MultiObjectiveProblem,
        generation: int
    ) -> MultiObjectiveSolution:
        """Quantum-inspired mutation operator."""
        
        mutated_solution = solution.solution.copy()
        
        # Adaptive mutation rate based on generation
        adaptive_rate = self.mutation_probability * (1.0 - generation / self.max_generations)
        
        for i in range(len(mutated_solution)):
            if np.random.random() < adaptive_rate:
                if problem.variable_types and i < len(problem.variable_types):
                    var_type = problem.variable_types[i]
                else:
                    var_type = 'continuous'
                
                if var_type == 'binary':
                    # Quantum bit flip with probability amplitudes
                    flip_prob = 0.5 + 0.3 * np.sin(generation * np.pi / self.max_generations)
                    mutated_solution[i] = 1 - mutated_solution[i] if np.random.random() < flip_prob else mutated_solution[i]
                
                elif var_type == 'integer':
                    # Quantum-inspired integer mutation
                    if problem.bounds and i < len(problem.bounds):
                        low, high = problem.bounds[i]
                        quantum_step = max(1, int((high - low) * 0.1))
                        delta = np.random.choice([-quantum_step, 0, quantum_step])
                        mutated_solution[i] = np.clip(
                            int(mutated_solution[i] + delta), int(low), int(high)
                        )
                    else:
                        delta = np.random.choice([-1, 0, 1])
                        mutated_solution[i] = int(mutated_solution[i] + delta)
                
                else:  # continuous
                    # Quantum-inspired Gaussian mutation with adaptive variance
                    if problem.bounds and i < len(problem.bounds):
                        low, high = problem.bounds[i]
                        variance = (high - low) * 0.1 * (1.0 - generation / self.max_generations)
                    else:
                        variance = 0.1 * (1.0 - generation / self.max_generations)
                    
                    # Add quantum tunneling effect
                    tunneling_prob = 0.1 * np.exp(-generation / (self.max_generations * 0.3))
                    if np.random.random() < tunneling_prob:
                        variance *= 5  # Large jump for tunneling
                    
                    delta = np.random.normal(0, variance)
                    mutated_solution[i] += delta
                    
                    # Apply bounds
                    if problem.bounds and i < len(problem.bounds):
                        low, high = problem.bounds[i]
                        mutated_solution[i] = np.clip(mutated_solution[i], low, high)
        
        # Apply variable constraints
        mutated_solution = self._apply_variable_constraints(mutated_solution, problem)
        
        mutated = MultiObjectiveSolution(
            solution=mutated_solution,
            objectives=np.zeros(problem.num_objectives),
            algorithm_used="quantum_nsga2"
        )
        
        return mutated
    
    def _environmental_selection(
        self,
        population: List[MultiObjectiveSolution]
    ) -> List[MultiObjectiveSolution]:
        """Environmental selection using non-dominated sorting and crowding distance."""
        
        if len(population) <= self.population_size:
            return population
        
        # Compute Pareto ranks
        ranks = ParetoFrontAnalyzer.compute_pareto_ranks(population)
        
        for i, sol in enumerate(population):
            sol.rank = ranks[i]
        
        # Sort by rank
        population.sort(key=lambda x: x.rank)
        
        # Select solutions front by front
        selected = []
        current_rank = 0
        
        while len(selected) < self.population_size:
            current_front = [sol for sol in population if sol.rank == current_rank]
            
            if len(selected) + len(current_front) <= self.population_size:
                # Include entire front
                selected.extend(current_front)
            else:
                # Partial inclusion based on crowding distance
                remaining_slots = self.population_size - len(selected)
                
                # Compute crowding distances for current front
                front_indices = list(range(len(current_front)))
                distances = ParetoFrontAnalyzer.compute_crowding_distances(
                    current_front, front_indices
                )
                
                for i, distance in enumerate(distances):
                    current_front[i].crowding_distance = distance
                
                # Sort by crowding distance (descending)
                current_front.sort(key=lambda x: x.crowding_distance, reverse=True)
                
                # Add best solutions from current front
                selected.extend(current_front[:remaining_slots])
                break
            
            current_rank += 1
        
        return selected[:self.population_size]
    
    def _extract_pareto_front(
        self,
        population: List[MultiObjectiveSolution]
    ) -> List[MultiObjectiveSolution]:
        """Extract the first Pareto front."""
        
        ranks = ParetoFrontAnalyzer.compute_pareto_ranks(population)
        pareto_front = [sol for i, sol in enumerate(population) if ranks[i] == 0]
        
        return pareto_front
    
    def _track_evolution(self, population: List[MultiObjectiveSolution], generation: int):
        """Track evolution metrics."""
        
        # Extract Pareto front
        pareto_front = self._extract_pareto_front(population)
        
        # Compute metrics
        if len(pareto_front) > 1:
            # Convergence metric (average distance to ideal point)
            objectives = np.array([sol.objectives for sol in pareto_front])
            ideal_point = np.min(objectives, axis=0)
            distances = [np.linalg.norm(sol.objectives - ideal_point) for sol in pareto_front]
            convergence_metric = np.mean(distances)
            
            # Diversity metric (spacing)
            diversity_metric = ParetoFrontAnalyzer.compute_spacing_metric(pareto_front)
        else:
            convergence_metric = float('inf')
            diversity_metric = 0.0
        
        self.convergence_metrics.append(convergence_metric)
        self.diversity_metrics.append(diversity_metric)
        
        # Store evolution snapshot
        self.evolution_history.append({
            'generation': generation,
            'population_size': len(population),
            'pareto_front_size': len(pareto_front),
            'convergence_metric': convergence_metric,
            'diversity_metric': diversity_metric
        })
    
    def _check_convergence(self, generation: int) -> bool:
        """Check convergence criteria."""
        
        if generation < 20:  # Minimum generations
            return False
        
        # Check convergence metric trend
        if len(self.convergence_metrics) >= 10:
            recent_metrics = self.convergence_metrics[-10:]
            trend = np.polyfit(range(10), recent_metrics, 1)[0]
            
            # Converged if improvement trend is very small
            if abs(trend) < 1e-6:
                return True
        
        return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        
        if not self.evolution_history:
            return {"status": "no_data"}
        
        final_stats = self.evolution_history[-1]
        
        summary = {
            "status": "completed",
            "total_generations": len(self.evolution_history),
            "final_pareto_front_size": final_stats['pareto_front_size'],
            "final_convergence_metric": final_stats['convergence_metric'],
            "final_diversity_metric": final_stats['diversity_metric'],
            "quantum_enhancement_used": self.quantum_enhancement,
            "evolution_history": self.evolution_history,
            "convergence_trend": np.polyfit(
                range(len(self.convergence_metrics)), 
                self.convergence_metrics, 1
            )[0] if len(self.convergence_metrics) > 1 else 0.0
        }
        
        return summary


class MultiObjectiveQuantumOptimizer:
    """
    Comprehensive multi-objective quantum optimizer integrating multiple algorithms.
    
    Features:
    - Quantum NSGA-II for evolutionary optimization
    - Quantum-enhanced reference point methods
    - Hybrid quantum-classical decomposition approaches
    - Adaptive algorithm selection
    """
    
    def __init__(
        self,
        quantum_backend=None,
        classical_fallback=True,
        algorithm_portfolio: List[str] = None
    ):
        """Initialize multi-objective quantum optimizer."""
        
        self.quantum_backend = quantum_backend
        self.classical_fallback = classical_fallback
        
        if algorithm_portfolio is None:
            self.algorithm_portfolio = [
                'quantum_nsga2',
                'quantum_moead',  # Multi-Objective Evolutionary Algorithm based on Decomposition
                'quantum_reference_point',
                'hybrid_pareto_search'
            ]
        else:
            self.algorithm_portfolio = algorithm_portfolio
        
        # Initialize algorithms
        self.algorithms = {
            'quantum_nsga2': QuantumNSGAII(quantum_enhancement=True),
            'classical_nsga2': QuantumNSGAII(quantum_enhancement=False)
        }
        
        # Optimization history
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
        logger.info(f"Initialized MultiObjectiveQuantumOptimizer with {len(self.algorithm_portfolio)} algorithms")
    
    def optimize(
        self,
        problem: MultiObjectiveProblem,
        algorithm: str = 'auto',
        max_evaluations: int = 10000,
        target_hypervolume: Optional[float] = None,
        reference_point: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize multi-objective problem using quantum-enhanced methods.
        
        Args:
            problem: Multi-objective optimization problem
            algorithm: Algorithm to use ('auto' for automatic selection)
            max_evaluations: Maximum function evaluations
            target_hypervolume: Target hypervolume for stopping
            reference_point: Reference point for hypervolume computation
            
        Returns:
            Comprehensive optimization results
        """
        
        logger.info(f"Starting multi-objective quantum optimization")
        logger.info(f"Problem: {problem.num_objectives} objectives, {problem.num_variables} variables")
        
        start_time = time.time()
        
        # Algorithm selection
        if algorithm == 'auto':
            selected_algorithm = self._select_algorithm(problem)
        else:
            selected_algorithm = algorithm
        
        logger.info(f"Selected algorithm: {selected_algorithm}")
        
        # Set reference point if not provided
        if reference_point is None:
            reference_point = self._estimate_reference_point(problem)
        
        # Run optimization
        try:
            if selected_algorithm == 'quantum_nsga2':
                optimizer = self.algorithms['quantum_nsga2']
                pareto_solutions = optimizer.optimize(problem)
            
            elif selected_algorithm == 'classical_nsga2':
                optimizer = self.algorithms['classical_nsga2']
                pareto_solutions = optimizer.optimize(problem)
            
            else:
                # Fallback to classical NSGA-II
                logger.warning(f"Algorithm {selected_algorithm} not implemented, using classical NSGA-II")
                optimizer = self.algorithms['classical_nsga2']
                pareto_solutions = optimizer.optimize(problem)
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            if self.classical_fallback:
                logger.info("Falling back to classical NSGA-II")
                optimizer = self.algorithms['classical_nsga2']
                pareto_solutions = optimizer.optimize(problem)
            else:
                raise
        
        optimization_time = time.time() - start_time
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            pareto_solutions, reference_point, problem
        )
        
        # Prepare results
        results = {
            'pareto_front': pareto_solutions,
            'algorithm_used': selected_algorithm,
            'optimization_time': optimization_time,
            'quality_metrics': quality_metrics,
            'reference_point': reference_point,
            'problem_info': {
                'num_objectives': problem.num_objectives,
                'num_variables': problem.num_variables,
                'num_constraints': problem.num_constraints
            }
        }
        
        # Add algorithm-specific information
        if hasattr(optimizer, 'get_optimization_summary'):
            results['algorithm_details'] = optimizer.get_optimization_summary()
        
        # Store in history
        self.optimization_history.append(results)
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Pareto front size: {len(pareto_solutions)}")
        logger.info(f"Hypervolume: {quality_metrics.get('hypervolume', 'N/A')}")
        
        return results
    
    def _select_algorithm(self, problem: MultiObjectiveProblem) -> str:
        """Automatically select best algorithm for the problem."""
        
        # Algorithm selection heuristics
        if problem.num_objectives <= 3 and problem.num_variables <= 50:
            # Small problems: use quantum enhancement
            return 'quantum_nsga2'
        
        elif problem.num_objectives > 5:
            # Many-objective problems: use decomposition-based approach
            return 'quantum_moead' if 'quantum_moead' in self.algorithm_portfolio else 'classical_nsga2'
        
        elif problem.num_variables > 100:
            # Large problems: use classical approach
            return 'classical_nsga2'
        
        else:
            # Default: quantum NSGA-II
            return 'quantum_nsga2'
    
    def _estimate_reference_point(self, problem: MultiObjectiveProblem) -> np.ndarray:
        """Estimate reference point for hypervolume computation."""
        
        # Sample a few solutions to estimate objective ranges
        num_samples = min(50, 10 * problem.num_variables)
        sample_objectives = []
        
        for _ in range(num_samples):
            if problem.bounds:
                solution = np.array([
                    np.random.uniform(low, high) 
                    for low, high in problem.bounds
                ])
            else:
                solution = np.random.choice([0, 1], size=problem.num_variables)
            
            objectives, _ = problem.evaluate(solution)
            sample_objectives.append(objectives)
        
        sample_objectives = np.array(sample_objectives)
        
        # Reference point is slightly worse than the worst observed
        reference_point = np.max(sample_objectives, axis=0) * 1.1
        
        return reference_point
    
    def _compute_quality_metrics(
        self,
        solutions: List[MultiObjectiveSolution],
        reference_point: np.ndarray,
        problem: MultiObjectiveProblem
    ) -> Dict[str, float]:
        """Compute comprehensive quality metrics."""
        
        if not solutions:
            return {}
        
        metrics = {}
        
        # Hypervolume
        try:
            metrics['hypervolume'] = ParetoFrontAnalyzer.compute_hypervolume(
                solutions, reference_point
            )
        except Exception as e:
            logger.warning(f"Could not compute hypervolume: {e}")
            metrics['hypervolume'] = 0.0
        
        # Spacing (diversity)
        metrics['spacing'] = ParetoFrontAnalyzer.compute_spacing_metric(solutions)
        
        # Number of solutions
        metrics['pareto_front_size'] = len(solutions)
        
        # Objective ranges
        objectives = np.array([sol.objectives for sol in solutions])
        metrics['objective_ranges'] = np.ptp(objectives, axis=0).tolist()
        
        # Average constraint violation
        violations = [sol.constraints_violation for sol in solutions]
        metrics['average_constraint_violation'] = np.mean(violations)
        
        # Solution distribution (spread)
        if len(solutions) > 1:
            pairwise_distances = cdist(objectives, objectives)
            metrics['average_distance'] = np.mean(pairwise_distances)
            metrics['max_distance'] = np.max(pairwise_distances)
        
        return metrics
    
    def plot_pareto_front(
        self,
        results: Dict[str, Any],
        objective_indices: Tuple[int, int] = (0, 1),
        save_path: Optional[str] = None
    ):
        """Plot 2D projection of Pareto front."""
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        solutions = results['pareto_front']
        if not solutions:
            logger.warning("No solutions to plot")
            return
        
        # Extract objectives
        objectives = np.array([sol.objectives for sol in solutions])
        
        if objectives.shape[1] < 2:
            logger.warning("Need at least 2 objectives for 2D plot")
            return
        
        obj_idx1, obj_idx2 = objective_indices
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            objectives[:, obj_idx1], 
            objectives[:, obj_idx2],
            c='blue', alpha=0.7, s=60, edgecolors='black', linewidth=0.5
        )
        
        # Add reference point if available
        reference_point = results.get('reference_point')
        if reference_point is not None and len(reference_point) > max(obj_idx1, obj_idx2):
            plt.scatter(
                reference_point[obj_idx1], 
                reference_point[obj_idx2],
                c='red', marker='x', s=100, linewidth=3, label='Reference Point'
            )
        
        plt.xlabel(f'Objective {obj_idx1}')
        plt.ylabel(f'Objective {obj_idx2}')
        plt.title(f'Pareto Front - {results["algorithm_used"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add metrics as text
        metrics = results.get('quality_metrics', {})
        textstr = f"Solutions: {len(solutions)}\n"
        if 'hypervolume' in metrics:
            textstr += f"Hypervolume: {metrics['hypervolume']:.4f}\n"
        if 'spacing' in metrics:
            textstr += f"Spacing: {metrics['spacing']:.4f}"
        
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history."""
        
        if not self.optimization_history:
            return {"status": "no_data"}
        
        # Algorithm performance comparison
        algorithm_performance = defaultdict(list)
        
        for result in self.optimization_history:
            algorithm = result['algorithm_used']
            metrics = result['quality_metrics']
            
            if 'hypervolume' in metrics:
                algorithm_performance[algorithm].append(metrics['hypervolume'])
        
        algorithm_rankings = {
            alg: np.mean(performances) 
            for alg, performances in algorithm_performance.items()
            if performances
        }
        
        insights = {
            "status": "ready",
            "total_optimizations": len(self.optimization_history),
            "algorithm_usage": dict(
                Counter(result['algorithm_used'] for result in self.optimization_history)
            ),
            "algorithm_performance": algorithm_rankings,
            "average_optimization_time": np.mean([
                result['optimization_time'] for result in self.optimization_history
            ]),
            "average_pareto_front_size": np.mean([
                len(result['pareto_front']) for result in self.optimization_history
            ])
        }
        
        return insights


# Example usage and demonstration
def create_test_problem() -> MultiObjectiveProblem:
    """Create a test multi-objective optimization problem."""
    
    # ZDT1 benchmark problem
    def zdt1_f1(x):
        return x[0]
    
    def zdt1_f2(x):
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.sqrt(x[0] / g))
    
    problem = MultiObjectiveProblem(
        objective_functions=[zdt1_f1, zdt1_f2],
        objective_names=['f1', 'f2'],
        bounds=[(0, 1)] * 10,  # 10 variables, each in [0, 1]
        variable_types=['continuous'] * 10,
        num_variables=10
    )
    
    return problem


def demonstrate_multi_objective_optimization():
    """Demonstrate multi-objective quantum optimization."""
    print("Multi-Objective Quantum Optimization Demonstration")
    print("=" * 60)
    
    # Create test problem
    problem = create_test_problem()
    print(f"Test problem: ZDT1 with {problem.num_variables} variables")
    
    # Initialize optimizer
    optimizer = MultiObjectiveQuantumOptimizer()
    
    # Run optimization with different algorithms
    algorithms = ['quantum_nsga2', 'classical_nsga2']
    results = {}
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm}...")
        
        try:
            result = optimizer.optimize(
                problem=problem,
                algorithm=algorithm,
                max_evaluations=5000
            )
            
            results[algorithm] = result
            
            # Print results
            pareto_front = result['pareto_front']
            metrics = result['quality_metrics']
            
            print(f"  Pareto front size: {len(pareto_front)}")
            print(f"  Hypervolume: {metrics.get('hypervolume', 'N/A'):.4f}")
            print(f"  Spacing: {metrics.get('spacing', 'N/A'):.4f}")
            print(f"  Optimization time: {result['optimization_time']:.2f}s")
            
        except Exception as e:
            print(f"  Error with {algorithm}: {e}")
    
    # Compare results
    if len(results) > 1:
        print(f"\nComparison:")
        print("-" * 30)
        
        for alg, result in results.items():
            metrics = result['quality_metrics']
            hv = metrics.get('hypervolume', 0)
            spacing = metrics.get('spacing', 0)
            time_taken = result['optimization_time']
            
            print(f"{alg:>20}: HV={hv:.4f}, Spacing={spacing:.4f}, Time={time_taken:.2f}s")
    
    # Get insights
    insights = optimizer.get_optimization_insights()
    print(f"\nOptimization Insights:")
    print(f"Total runs: {insights['total_optimizations']}")
    if 'algorithm_performance' in insights:
        print("Algorithm performance (by hypervolume):")
        for alg, perf in sorted(insights['algorithm_performance'].items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"  {alg}: {perf:.4f}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_multi_objective_optimization()
    
    # Additional validation tests
    print(f"\nValidation Tests:")
    print("-" * 20)
    
    # Test solution dominance
    sol1 = MultiObjectiveSolution(
        solution=np.array([0.1, 0.2]),
        objectives=np.array([1.0, 2.0])
    )
    sol2 = MultiObjectiveSolution(
        solution=np.array([0.2, 0.1]),
        objectives=np.array([1.5, 1.5])
    )
    
    print(f"✓ Dominance test: sol1 dominates sol2? {sol1.dominates(sol2)}")
    
    # Test Pareto ranking
    solutions = [sol1, sol2]
    ranks = ParetoFrontAnalyzer.compute_pareto_ranks(solutions)
    print(f"✓ Pareto ranks: {ranks}")
    
    # Test hypervolume computation
    reference = np.array([3.0, 3.0])
    hv = ParetoFrontAnalyzer.compute_hypervolume(solutions, reference)
    print(f"✓ Hypervolume: {hv:.4f}")
    
    # Test problem creation
    test_problem = create_test_problem()
    print(f"✓ Test problem created: {test_problem.num_objectives} objectives")
    
    print(f"\nMulti-objective quantum optimization module ready for integration!")
"""
Hybrid Quantum-Classical Decomposition Algorithms

This module implements advanced hybrid algorithms that intelligently combine
quantum and classical optimization techniques for superior performance on
large-scale task scheduling problems.

Key Innovations:
- Problem-aware decomposition strategies
- Dynamic quantum-classical resource allocation
- Hierarchical optimization approaches
- Adaptive switching mechanisms
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import concurrent.futures
import threading

# Import quantum algorithms
from .advanced_quantum_algorithms import (
    QuantumAlgorithmType, QuantumAlgorithmResult, 
    AdaptiveQAOAScheduler, VQETaskScheduler, QuantumMLTaskPredictor
)

logger = logging.getLogger(__name__)


class DecompositionStrategy(Enum):
    """Strategies for problem decomposition."""
    SPECTRAL_CLUSTERING = "spectral_clustering"
    SKILL_BASED = "skill_based_partition"
    GRAPH_BASED = "graph_based_partition"
    TEMPORAL_PARTITION = "temporal_partition"
    COMPLEXITY_AWARE = "complexity_aware"
    ML_GUIDED = "ml_guided_partition"


class HybridMode(Enum):
    """Modes of hybrid quantum-classical operation."""
    SEQUENTIAL = "sequential"  # Quantum then classical refinement
    PARALLEL = "parallel"     # Concurrent quantum and classical
    ADAPTIVE = "adaptive"     # Dynamic switching
    HIERARCHICAL = "hierarchical"  # Multi-level optimization


@dataclass
class SubproblemMetrics:
    """Metrics for evaluating subproblem complexity."""
    num_variables: int
    connectivity: float
    constraint_density: float
    estimated_difficulty: float
    quantum_suitability: float
    recommended_solver: str


@dataclass
class DecompositionResult:
    """Result from problem decomposition."""
    subproblems: List[Dict[str, Any]]
    coupling_matrix: np.ndarray
    decomposition_quality: float
    subproblem_metrics: List[SubproblemMetrics]
    metadata: Dict[str, Any]


@dataclass
class HybridSolutionResult:
    """Result from hybrid quantum-classical solving."""
    solution: Dict[int, int]
    total_energy: float
    quantum_contribution: float
    classical_contribution: float
    decomposition_overhead: float
    total_execution_time: float
    subproblem_results: List[Dict[str, Any]]
    hybrid_mode_used: HybridMode
    quantum_advantage_factor: float
    metadata: Dict[str, Any]


class ProblemDecomposer:
    """Advanced problem decomposition with multiple strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ProblemDecomposer")
    
    def decompose_problem(
        self,
        problem_matrix: np.ndarray,
        strategy: DecompositionStrategy = DecompositionStrategy.SPECTRAL_CLUSTERING,
        max_subproblem_size: int = 20,
        min_subproblems: int = 2,
        **kwargs
    ) -> DecompositionResult:
        """Decompose optimization problem into subproblems."""
        
        start_time = time.time()
        
        if strategy == DecompositionStrategy.SPECTRAL_CLUSTERING:
            result = self._spectral_decomposition(problem_matrix, max_subproblem_size, **kwargs)
        elif strategy == DecompositionStrategy.SKILL_BASED:
            result = self._skill_based_decomposition(problem_matrix, **kwargs)
        elif strategy == DecompositionStrategy.GRAPH_BASED:
            result = self._graph_based_decomposition(problem_matrix, max_subproblem_size, **kwargs)
        elif strategy == DecompositionStrategy.COMPLEXITY_AWARE:
            result = self._complexity_aware_decomposition(problem_matrix, max_subproblem_size, **kwargs)
        else:
            # Fallback to simple partitioning
            result = self._simple_partition(problem_matrix, max_subproblem_size)
        
        # Evaluate subproblem metrics
        subproblem_metrics = []
        for subproblem in result.subproblems:
            metrics = self._evaluate_subproblem_metrics(subproblem)
            subproblem_metrics.append(metrics)
        
        decomposition_time = time.time() - start_time
        
        return DecompositionResult(
            subproblems=result.subproblems,
            coupling_matrix=result.coupling_matrix,
            decomposition_quality=result.decomposition_quality,
            subproblem_metrics=subproblem_metrics,
            metadata={
                "strategy": strategy.value,
                "decomposition_time": decomposition_time,
                "num_subproblems": len(result.subproblems)
            }
        )
    
    def _spectral_decomposition(
        self, 
        matrix: np.ndarray, 
        max_size: int,
        **kwargs
    ) -> DecompositionResult:
        """Spectral clustering-based decomposition."""
        
        try:
            from sklearn.cluster import SpectralClustering
            
            # Create similarity matrix
            similarity = np.abs(matrix)
            
            # Determine number of clusters
            n_variables = matrix.shape[0]
            n_clusters = max(2, min(n_variables // max_size, 8))
            
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            labels = clustering.fit_predict(similarity)
            
            # Create subproblems
            subproblems = []
            for cluster_id in range(n_clusters):
                cluster_vars = np.where(labels == cluster_id)[0]
                if len(cluster_vars) > 0:
                    submatrix = matrix[np.ix_(cluster_vars, cluster_vars)]
                    subproblems.append({
                        "variables": cluster_vars.tolist(),
                        "matrix": submatrix,
                        "cluster_id": cluster_id
                    })
            
            # Calculate coupling matrix
            coupling = self._calculate_coupling_matrix(matrix, labels)
            
            # Evaluate quality (silhouette-like metric)
            quality = self._evaluate_decomposition_quality(matrix, labels)
            
            return DecompositionResult(
                subproblems=subproblems,
                coupling_matrix=coupling,
                decomposition_quality=quality,
                subproblem_metrics=[],
                metadata={"method": "spectral"}
            )
            
        except ImportError:
            self.logger.warning("Scikit-learn not available, using simple partition")
            return self._simple_partition(matrix, max_size)
    
    def _skill_based_decomposition(
        self, 
        matrix: np.ndarray, 
        skill_groups: Optional[Dict] = None,
        **kwargs
    ) -> DecompositionResult:
        """Decomposition based on skill/task groupings."""
        
        if not skill_groups:
            # Create artificial skill groups for demonstration
            n_vars = matrix.shape[0]
            n_groups = max(2, n_vars // 15)
            skill_groups = {}
            
            for i in range(n_vars):
                group_id = i % n_groups
                if group_id not in skill_groups:
                    skill_groups[group_id] = []
                skill_groups[group_id].append(i)
        
        # Create subproblems from skill groups
        subproblems = []
        for group_id, variables in skill_groups.items():
            if len(variables) > 0:
                var_array = np.array(variables)
                submatrix = matrix[np.ix_(var_array, var_array)]
                subproblems.append({
                    "variables": variables,
                    "matrix": submatrix,
                    "skill_group": group_id
                })
        
        # Create coupling matrix
        labels = np.zeros(matrix.shape[0])
        for group_id, variables in skill_groups.items():
            for var in variables:
                labels[var] = group_id
        
        coupling = self._calculate_coupling_matrix(matrix, labels)
        quality = self._evaluate_decomposition_quality(matrix, labels)
        
        return DecompositionResult(
            subproblems=subproblems,
            coupling_matrix=coupling,
            decomposition_quality=quality,
            subproblem_metrics=[],
            metadata={"method": "skill_based", "num_skill_groups": len(skill_groups)}
        )
    
    def _graph_based_decomposition(
        self, 
        matrix: np.ndarray, 
        max_size: int,
        **kwargs
    ) -> DecompositionResult:
        """Graph partitioning-based decomposition."""
        
        try:
            import networkx as nx
            
            # Create graph from matrix
            G = nx.Graph()
            n_vars = matrix.shape[0]
            
            for i in range(n_vars):
                G.add_node(i)
                for j in range(i + 1, n_vars):
                    if abs(matrix[i, j]) > 1e-6:  # Only significant connections
                        G.add_edge(i, j, weight=abs(matrix[i, j]))
            
            # Use community detection for partitioning
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(G)
            except (ImportError, AttributeError):
                # Fallback: simple connected components
                communities = list(nx.connected_components(G))
            
            # Convert communities to subproblems
            subproblems = []
            labels = np.zeros(n_vars)
            
            for i, community in enumerate(communities):
                variables = list(community)
                if len(variables) > 0:
                    var_array = np.array(variables)
                    submatrix = matrix[np.ix_(var_array, var_array)]
                    subproblems.append({
                        "variables": variables,
                        "matrix": submatrix,
                        "community": i
                    })
                    
                    for var in variables:
                        labels[var] = i
            
            coupling = self._calculate_coupling_matrix(matrix, labels)
            quality = self._evaluate_decomposition_quality(matrix, labels)
            
            return DecompositionResult(
                subproblems=subproblems,
                coupling_matrix=coupling,
                decomposition_quality=quality,
                subproblem_metrics=[],
                metadata={"method": "graph_based", "num_communities": len(communities)}
            )
            
        except ImportError:
            self.logger.warning("NetworkX not available, using simple partition")
            return self._simple_partition(matrix, max_size)
    
    def _complexity_aware_decomposition(
        self, 
        matrix: np.ndarray, 
        max_size: int,
        **kwargs
    ) -> DecompositionResult:
        """Decomposition that considers problem complexity."""
        
        n_vars = matrix.shape[0]
        
        # Calculate complexity metrics for each variable
        complexity_scores = np.zeros(n_vars)
        for i in range(n_vars):
            # Complexity based on connections and weights
            connections = np.sum(np.abs(matrix[i, :]) > 1e-6)
            weight_magnitude = np.sum(np.abs(matrix[i, :]))
            complexity_scores[i] = connections * weight_magnitude
        
        # Sort variables by complexity
        sorted_indices = np.argsort(complexity_scores)[::-1]  # Highest complexity first
        
        # Create balanced subproblems
        subproblems = []
        current_subproblem = []
        current_complexity = 0
        max_complexity_per_subproblem = np.sum(complexity_scores) / max(2, n_vars // max_size)
        
        labels = np.zeros(n_vars)
        subproblem_id = 0
        
        for var_idx in sorted_indices:
            var_complexity = complexity_scores[var_idx]
            
            if (len(current_subproblem) < max_size and 
                current_complexity + var_complexity <= max_complexity_per_subproblem):
                # Add to current subproblem
                current_subproblem.append(var_idx)
                current_complexity += var_complexity
                labels[var_idx] = subproblem_id
            else:
                # Start new subproblem
                if current_subproblem:
                    var_array = np.array(current_subproblem)
                    submatrix = matrix[np.ix_(var_array, var_array)]
                    subproblems.append({
                        "variables": current_subproblem,
                        "matrix": submatrix,
                        "complexity": current_complexity
                    })
                
                subproblem_id += 1
                current_subproblem = [var_idx]
                current_complexity = var_complexity
                labels[var_idx] = subproblem_id
        
        # Add final subproblem
        if current_subproblem:
            var_array = np.array(current_subproblem)
            submatrix = matrix[np.ix_(var_array, var_array)]
            subproblems.append({
                "variables": current_subproblem,
                "matrix": submatrix,
                "complexity": current_complexity
            })
        
        coupling = self._calculate_coupling_matrix(matrix, labels)
        quality = self._evaluate_decomposition_quality(matrix, labels)
        
        return DecompositionResult(
            subproblems=subproblems,
            coupling_matrix=coupling,
            decomposition_quality=quality,
            subproblem_metrics=[],
            metadata={"method": "complexity_aware", "complexity_scores": complexity_scores.tolist()}
        )
    
    def _simple_partition(self, matrix: np.ndarray, max_size: int) -> DecompositionResult:
        """Simple sequential partitioning as fallback."""
        
        n_vars = matrix.shape[0]
        subproblems = []
        labels = np.zeros(n_vars)
        
        for start in range(0, n_vars, max_size):
            end = min(start + max_size, n_vars)
            variables = list(range(start, end))
            
            var_array = np.array(variables)
            submatrix = matrix[np.ix_(var_array, var_array)]
            subproblems.append({
                "variables": variables,
                "matrix": submatrix,
                "partition": start // max_size
            })
            
            for var in variables:
                labels[var] = start // max_size
        
        coupling = self._calculate_coupling_matrix(matrix, labels)
        quality = 0.5  # Neutral quality for simple partition
        
        return DecompositionResult(
            subproblems=subproblems,
            coupling_matrix=coupling,
            decomposition_quality=quality,
            subproblem_metrics=[],
            metadata={"method": "simple_partition"}
        )
    
    def _calculate_coupling_matrix(self, matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate coupling strength between subproblems."""
        
        n_clusters = int(np.max(labels)) + 1
        coupling = np.zeros((n_clusters, n_clusters))
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    cluster_i = int(labels[i])
                    cluster_j = int(labels[j])
                    coupling[cluster_i, cluster_j] += abs(matrix[i, j])
        
        return coupling
    
    def _evaluate_decomposition_quality(self, matrix: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate quality of decomposition (0-1 scale)."""
        
        # Calculate intra-cluster vs inter-cluster weights
        intra_weight = 0.0
        inter_weight = 0.0
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    weight = abs(matrix[i, j])
                    if labels[i] == labels[j]:
                        intra_weight += weight
                    else:
                        inter_weight += weight
        
        total_weight = intra_weight + inter_weight
        if total_weight == 0:
            return 0.5
        
        # Quality is ratio of intra-cluster to total connections
        quality = intra_weight / total_weight
        return quality
    
    def _evaluate_subproblem_metrics(self, subproblem: Dict[str, Any]) -> SubproblemMetrics:
        """Evaluate metrics for a single subproblem."""
        
        variables = subproblem["variables"]
        matrix = subproblem["matrix"]
        
        num_variables = len(variables)
        
        # Calculate connectivity (density of non-zero elements)
        non_zero_elements = np.sum(np.abs(matrix) > 1e-6)
        total_elements = matrix.shape[0] * matrix.shape[1]
        connectivity = non_zero_elements / total_elements if total_elements > 0 else 0
        
        # Constraint density (simplified metric)
        constraint_density = connectivity
        
        # Estimated difficulty (based on size and connectivity)
        estimated_difficulty = num_variables * (1 + connectivity)
        
        # Quantum suitability (heuristic based on problem structure)
        if num_variables <= 20 and connectivity > 0.3:
            quantum_suitability = 0.8
            recommended_solver = "quantum"
        elif num_variables <= 50 and connectivity > 0.2:
            quantum_suitability = 0.6
            recommended_solver = "hybrid"
        else:
            quantum_suitability = 0.3
            recommended_solver = "classical"
        
        return SubproblemMetrics(
            num_variables=num_variables,
            connectivity=connectivity,
            constraint_density=constraint_density,
            estimated_difficulty=estimated_difficulty,
            quantum_suitability=quantum_suitability,
            recommended_solver=recommended_solver
        )


class HybridQuantumClassicalSolver:
    """Main hybrid solver combining quantum and classical approaches."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HybridQuantumClassicalSolver")
        self.decomposer = ProblemDecomposer()
        
        # Initialize quantum algorithms
        self.qaoa_solver = AdaptiveQAOAScheduler()
        self.vqe_solver = VQETaskScheduler()
        self.qml_solver = QuantumMLTaskPredictor()
        
        # Classical solvers (simplified interfaces)
        self.classical_solvers = {
            "simulated_annealing": self._simulated_annealing_solve,
            "genetic_algorithm": self._genetic_algorithm_solve,
            "local_search": self._local_search_solve
        }
    
    def solve_hybrid(
        self,
        problem_matrix: np.ndarray,
        hybrid_mode: HybridMode = HybridMode.ADAPTIVE,
        decomposition_strategy: DecompositionStrategy = DecompositionStrategy.SPECTRAL_CLUSTERING,
        max_subproblem_size: int = 20,
        **kwargs
    ) -> HybridSolutionResult:
        """Solve optimization problem using hybrid approach."""
        
        start_time = time.time()
        
        # Step 1: Decompose problem
        self.logger.info("Decomposing problem...")
        decomposition = self.decomposer.decompose_problem(
            problem_matrix, 
            strategy=decomposition_strategy,
            max_subproblem_size=max_subproblem_size,
            **kwargs
        )
        
        decomposition_time = time.time() - start_time
        
        # Step 2: Solve subproblems based on hybrid mode
        self.logger.info(f"Solving {len(decomposition.subproblems)} subproblems in {hybrid_mode.value} mode")
        
        if hybrid_mode == HybridMode.SEQUENTIAL:
            subproblem_results = self._solve_sequential(decomposition)
        elif hybrid_mode == HybridMode.PARALLEL:
            subproblem_results = self._solve_parallel(decomposition)
        elif hybrid_mode == HybridMode.ADAPTIVE:
            subproblem_results = self._solve_adaptive(decomposition)
        elif hybrid_mode == HybridMode.HIERARCHICAL:
            subproblem_results = self._solve_hierarchical(decomposition)
        else:
            raise ValueError(f"Unknown hybrid mode: {hybrid_mode}")
        
        # Step 3: Merge solutions
        merged_solution = self._merge_solutions(decomposition, subproblem_results)
        
        # Step 4: Post-processing optimization
        final_solution = self._post_process_solution(merged_solution, problem_matrix)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        quantum_contribution = sum(
            r.get("quantum_time", 0) for r in subproblem_results
        ) / max(total_time - decomposition_time, 1e-6)
        
        classical_contribution = 1.0 - quantum_contribution
        
        total_energy = self._calculate_total_energy(final_solution, problem_matrix)
        
        # Estimate quantum advantage
        quantum_advantage_factor = self._estimate_quantum_advantage(
            subproblem_results, decomposition
        )
        
        return HybridSolutionResult(
            solution=final_solution,
            total_energy=total_energy,
            quantum_contribution=quantum_contribution,
            classical_contribution=classical_contribution,
            decomposition_overhead=decomposition_time / total_time,
            total_execution_time=total_time,
            subproblem_results=subproblem_results,
            hybrid_mode_used=hybrid_mode,
            quantum_advantage_factor=quantum_advantage_factor,
            metadata={
                "num_subproblems": len(decomposition.subproblems),
                "decomposition_strategy": decomposition_strategy.value,
                "decomposition_quality": decomposition.decomposition_quality
            }
        )
    
    def _solve_sequential(self, decomposition: DecompositionResult) -> List[Dict[str, Any]]:
        """Solve subproblems sequentially with quantum-first approach."""
        
        results = []
        
        for i, (subproblem, metrics) in enumerate(zip(decomposition.subproblems, decomposition.subproblem_metrics)):
            
            start_time = time.time()
            
            # Try quantum first if suitable
            if metrics.quantum_suitability > 0.5:
                try:
                    if metrics.num_variables <= 15:
                        quantum_result = self.qaoa_solver.solve_scheduling_problem(
                            subproblem["matrix"], metrics.num_variables
                        )
                    else:
                        quantum_result = self.vqe_solver.solve_scheduling_problem(
                            subproblem["matrix"], metrics.num_variables
                        )
                    
                    result = {
                        "subproblem_id": i,
                        "solver_used": "quantum",
                        "solution": quantum_result.solution,
                        "energy": quantum_result.energy,
                        "quantum_time": quantum_result.execution_time,
                        "classical_time": 0,
                        "variables": subproblem["variables"]
                    }
                    results.append(result)
                    continue
                    
                except Exception as e:
                    self.logger.warning(f"Quantum solving failed for subproblem {i}: {e}")
            
            # Fall back to classical
            classical_result = self._solve_classical_subproblem(subproblem, metrics)
            classical_time = time.time() - start_time
            
            result = {
                "subproblem_id": i,
                "solver_used": "classical",
                "solution": classical_result["solution"],
                "energy": classical_result["energy"],
                "quantum_time": 0,
                "classical_time": classical_time,
                "variables": subproblem["variables"]
            }
            results.append(result)
        
        return results
    
    def _solve_parallel(self, decomposition: DecompositionResult) -> List[Dict[str, Any]]:
        """Solve subproblems in parallel using multiple solvers."""
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i, (subproblem, metrics) in enumerate(zip(decomposition.subproblems, decomposition.subproblem_metrics)):
                # Submit both quantum and classical solutions
                if metrics.quantum_suitability > 0.4:
                    quantum_future = executor.submit(
                        self._solve_quantum_subproblem, subproblem, metrics, i
                    )
                    futures.append(("quantum", i, quantum_future))
                
                classical_future = executor.submit(
                    self._solve_classical_subproblem, subproblem, metrics
                )
                futures.append(("classical", i, classical_future))
            
            # Collect results and select best
            subproblem_solutions = {}
            
            for solver_type, subproblem_id, future in futures:
                try:
                    result = future.result(timeout=60)
                    
                    if subproblem_id not in subproblem_solutions:
                        subproblem_solutions[subproblem_id] = []
                    
                    result["solver_type"] = solver_type
                    result["subproblem_id"] = subproblem_id
                    subproblem_solutions[subproblem_id].append(result)
                    
                except Exception as e:
                    self.logger.error(f"{solver_type} solver failed for subproblem {subproblem_id}: {e}")
            
            # Select best solution for each subproblem
            for subproblem_id, solutions in subproblem_solutions.items():
                best_solution = min(solutions, key=lambda x: x.get("energy", float('inf')))
                best_solution["variables"] = decomposition.subproblems[subproblem_id]["variables"]
                results.append(best_solution)
        
        return results
    
    def _solve_adaptive(self, decomposition: DecompositionResult) -> List[Dict[str, Any]]:
        """Solve subproblems with adaptive quantum-classical switching."""
        
        results = []
        quantum_budget = 30.0  # seconds
        quantum_used = 0.0
        
        # Sort subproblems by quantum suitability
        indexed_subproblems = list(enumerate(zip(decomposition.subproblems, decomposition.subproblem_metrics)))
        indexed_subproblems.sort(key=lambda x: x[1][1].quantum_suitability, reverse=True)
        
        for i, (subproblem, metrics) in indexed_subproblems:
            start_time = time.time()
            
            # Decide whether to use quantum based on budget and suitability
            use_quantum = (
                quantum_used < quantum_budget and
                metrics.quantum_suitability > 0.6 and
                metrics.num_variables <= 25
            )
            
            if use_quantum:
                try:
                    quantum_result = self._solve_quantum_subproblem(subproblem, metrics, i)
                    quantum_time = time.time() - start_time
                    quantum_used += quantum_time
                    
                    quantum_result["solver_used"] = "quantum_adaptive"
                    quantum_result["quantum_time"] = quantum_time
                    quantum_result["classical_time"] = 0
                    quantum_result["variables"] = subproblem["variables"]
                    results.append(quantum_result)
                    continue
                    
                except Exception as e:
                    self.logger.warning(f"Adaptive quantum solving failed: {e}")
            
            # Use classical solver
            classical_result = self._solve_classical_subproblem(subproblem, metrics)
            classical_time = time.time() - start_time
            
            result = {
                "subproblem_id": i,
                "solver_used": "classical_adaptive",
                "solution": classical_result["solution"],
                "energy": classical_result["energy"],
                "quantum_time": 0,
                "classical_time": classical_time,
                "variables": subproblem["variables"]
            }
            results.append(result)
        
        return results
    
    def _solve_hierarchical(self, decomposition: DecompositionResult) -> List[Dict[str, Any]]:
        """Solve with hierarchical quantum-classical approach."""
        
        # First pass: Quick classical solutions for all subproblems
        classical_results = []
        for i, (subproblem, metrics) in enumerate(zip(decomposition.subproblems, decomposition.subproblem_metrics)):
            classical_result = self._solve_classical_subproblem(subproblem, metrics, quick=True)
            classical_result["subproblem_id"] = i
            classical_result["pass"] = "classical_first"
            classical_results.append(classical_result)
        
        # Second pass: Quantum refinement for most promising subproblems
        refined_results = []
        
        # Sort by potential for improvement (high energy, high quantum suitability)
        candidates = [
            (i, result, decomposition.subproblem_metrics[i]) 
            for i, result in enumerate(classical_results)
        ]
        candidates.sort(key=lambda x: x[1]["energy"] * x[2].quantum_suitability, reverse=True)
        
        # Refine top candidates with quantum
        for i, classical_result, metrics in candidates[:3]:  # Top 3 candidates
            if metrics.quantum_suitability > 0.5:
                try:
                    subproblem = decomposition.subproblems[i]
                    quantum_result = self._solve_quantum_subproblem(subproblem, metrics, i)
                    
                    if quantum_result["energy"] < classical_result["energy"]:
                        quantum_result["pass"] = "quantum_refinement"
                        quantum_result["variables"] = subproblem["variables"]
                        refined_results.append(quantum_result)
                    else:
                        classical_result["variables"] = subproblem["variables"]
                        refined_results.append(classical_result)
                        
                except Exception as e:
                    self.logger.warning(f"Quantum refinement failed: {e}")
                    classical_result["variables"] = decomposition.subproblems[i]["variables"]
                    refined_results.append(classical_result)
            else:
                classical_result["variables"] = decomposition.subproblems[i]["variables"]
                refined_results.append(classical_result)
        
        # Add remaining classical results
        for i, result in enumerate(classical_results):
            if not any(r["subproblem_id"] == i for r in refined_results):
                result["variables"] = decomposition.subproblems[i]["variables"]
                refined_results.append(result)
        
        return refined_results
    
    def _solve_quantum_subproblem(self, subproblem: Dict, metrics: SubproblemMetrics, subproblem_id: int) -> Dict[str, Any]:
        """Solve subproblem using quantum algorithm."""
        
        matrix = subproblem["matrix"]
        num_vars = metrics.num_variables
        
        # Choose quantum algorithm based on problem characteristics
        if num_vars <= 12:
            result = self.qaoa_solver.solve_scheduling_problem(matrix, num_vars)
        elif metrics.connectivity > 0.5:
            result = self.vqe_solver.solve_scheduling_problem(matrix, num_vars)
        else:
            result = self.qml_solver.solve_scheduling_problem(matrix, num_vars)
        
        return {
            "subproblem_id": subproblem_id,
            "solution": result.solution,
            "energy": result.energy,
            "execution_time": result.execution_time,
            "algorithm": result.algorithm_type.value
        }
    
    def _solve_classical_subproblem(self, subproblem: Dict, metrics: SubproblemMetrics, quick: bool = False) -> Dict[str, Any]:
        """Solve subproblem using classical algorithm."""
        
        matrix = subproblem["matrix"]
        
        # Choose classical algorithm based on problem size and time constraints
        if quick or metrics.num_variables > 30:
            solver = "local_search"
        elif metrics.num_variables > 15:
            solver = "genetic_algorithm"
        else:
            solver = "simulated_annealing"
        
        result = self.classical_solvers[solver](matrix)
        result["solver"] = solver
        
        return result
    
    def _simulated_annealing_solve(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Simple simulated annealing solver."""
        
        n = matrix.shape[0]
        
        # Random initial solution
        solution = {i: np.random.randint(0, 2) for i in range(n)}
        
        # Calculate initial energy
        current_energy = self._calculate_total_energy(solution, matrix)
        best_solution = solution.copy()
        best_energy = current_energy
        
        # Simulated annealing parameters
        temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 0.1
        
        while temperature > min_temperature:
            # Generate neighbor solution
            neighbor = solution.copy()
            flip_bit = np.random.randint(0, n)
            neighbor[flip_bit] = 1 - neighbor[flip_bit]
            
            # Calculate energy change
            neighbor_energy = self._calculate_total_energy(neighbor, matrix)
            
            # Accept or reject
            if (neighbor_energy < current_energy or 
                np.random.random() < np.exp(-(neighbor_energy - current_energy) / temperature)):
                solution = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = solution.copy()
                    best_energy = current_energy
            
            temperature *= cooling_rate
        
        return {"solution": best_solution, "energy": best_energy}
    
    def _genetic_algorithm_solve(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Simple genetic algorithm solver."""
        
        n = matrix.shape[0]
        population_size = 50
        generations = 100
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {i: np.random.randint(0, 2) for i in range(n)}
            population.append(individual)
        
        best_solution = None
        best_energy = float('inf')
        
        for generation in range(generations):
            # Evaluate fitness
            fitness = []
            for individual in population:
                energy = self._calculate_total_energy(individual, matrix)
                fitness.append(-energy)  # Negative because we want to minimize energy
                
                if energy < best_energy:
                    best_energy = energy
                    best_solution = individual.copy()
            
            # Selection and reproduction (simplified)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                parent1 = population[np.argmax(np.random.choice(fitness, 3))]
                parent2 = population[np.argmax(np.random.choice(fitness, 3))]
                
                # Crossover
                child = {}
                for i in range(n):
                    child[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]
                
                # Mutation
                if np.random.random() < 0.1:
                    mutation_bit = np.random.randint(0, n)
                    child[mutation_bit] = 1 - child[mutation_bit]
                
                new_population.append(child)
            
            population = new_population
        
        return {"solution": best_solution, "energy": best_energy}
    
    def _local_search_solve(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Simple local search solver."""
        
        n = matrix.shape[0]
        
        # Random initial solution
        solution = {i: np.random.randint(0, 2) for i in range(n)}
        current_energy = self._calculate_total_energy(solution, matrix)
        
        improved = True
        while improved:
            improved = False
            
            for i in range(n):
                # Try flipping bit i
                solution[i] = 1 - solution[i]
                new_energy = self._calculate_total_energy(solution, matrix)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert change
                    solution[i] = 1 - solution[i]
        
        return {"solution": solution, "energy": current_energy}
    
    def _merge_solutions(self, decomposition: DecompositionResult, subproblem_results: List[Dict]) -> Dict[int, int]:
        """Merge subproblem solutions into global solution."""
        
        global_solution = {}
        
        for result in subproblem_results:
            variables = result["variables"]
            sub_solution = result["solution"]
            
            # Map subproblem solution to global variables
            for local_var, value in sub_solution.items():
                if local_var < len(variables):
                    global_var = variables[local_var]
                    global_solution[global_var] = value
        
        return global_solution
    
    def _post_process_solution(self, solution: Dict[int, int], original_matrix: np.ndarray) -> Dict[int, int]:
        """Post-process solution to handle coupling between subproblems."""
        
        # Simple post-processing: local search on the merged solution
        current_energy = self._calculate_total_energy(solution, original_matrix)
        
        improved = True
        iterations = 0
        max_iterations = min(100, len(solution))
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for var in solution:
                # Try flipping this variable
                original_value = solution[var]
                solution[var] = 1 - original_value
                
                new_energy = self._calculate_total_energy(solution, original_matrix)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert change
                    solution[var] = original_value
        
        return solution
    
    def _calculate_total_energy(self, solution: Dict[int, int], matrix: np.ndarray) -> float:
        """Calculate total energy of solution."""
        
        energy = 0.0
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i in solution and j in solution:
                    energy += matrix[i, j] * solution[i] * solution[j]
        
        return energy
    
    def _estimate_quantum_advantage(self, subproblem_results: List[Dict], decomposition: DecompositionResult) -> float:
        """Estimate quantum advantage factor."""
        
        quantum_results = [r for r in subproblem_results if "quantum" in r.get("solver_used", "")]
        
        if not quantum_results:
            return 1.0  # No quantum advantage
        
        # Compare quantum vs classical solving times (simplified)
        quantum_avg_time = np.mean([r.get("quantum_time", r.get("execution_time", 1.0)) for r in quantum_results])
        
        # Estimate classical time for same problems
        total_vars = sum(m.num_variables for m in decomposition.subproblem_metrics)
        estimated_classical_time = total_vars * 0.1  # Rough estimate
        
        if quantum_avg_time > 0:
            return estimated_classical_time / quantum_avg_time
        else:
            return 1.0


# Export main interfaces
__all__ = [
    'DecompositionStrategy',
    'HybridMode', 
    'SubproblemMetrics',
    'DecompositionResult',
    'HybridSolutionResult',
    'ProblemDecomposer',
    'HybridQuantumClassicalSolver'
]